from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    PNDMScheduler,
    DDIMScheduler,
    StableDiffusionControlNetPipeline,
)
from diffusers.utils.import_utils import is_xformers_available
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
from mvdream.camera_utils import get_camera, convert_opengl_to_blender, normalize_camera
from diffusers import DDIMScheduler


# suppress partial model loading warning
logging.set_verbosity_error()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


class StableDiffusionControlNet(nn.Module):
    def __init__(
        self,
        device,
        fp16=True,
        vram_O=True,
        t_range=[0.02, 0.98],
    ):
        super().__init__()

        self.device = device
        self.dtype = torch.float16 if fp16 else torch.float32

        # Create model
        controlnet_path = "./pretrained_models/controlnet.pt"
        sd_path = "./pretrained_models/sd.pt"

        controlnet = torch.load(controlnet_path, map_location=self.device)
        sd = torch.load(sd_path, map_location=self.device)

        pipe = StableDiffusionControlNetPipeline(
            controlnet=controlnet,
            sd=sd,
            device=self.device,
            torch_dtype=self.dtype,
        )

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        self.embeddings = {}

    @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts):
        pos_embeds = self.encode_text(prompts).repeat(4, 1, 1)  # [1, 77, 768]
        neg_embeds = self.encode_text(negative_prompts).repeat(4, 1, 1)
        self.embeddings = torch.cat([neg_embeds, pos_embeds], dim=0)  # [2, 77, 768]

    def encode_text(self, prompt):
        # prompt: [str]
        embeddings = self.model.get_learned_conditioning(prompt).to(self.device)
        return embeddings

    @torch.no_grad()
    def refine(
        self,
        pred_rgb,
        camera,
        guidance_scale=100,
        steps=50,
        strength=0.8,
    ):
        batch_size = pred_rgb.shape[0]
        pred_rgb_256 = F.interpolate(
            pred_rgb, (256, 256), mode="bilinear", align_corners=False
        )
        latents = self.encode_imgs(pred_rgb_256.to(self.dtype))
        # latents = torch.randn((1, 4, 64, 64), device=self.device, dtype=self.dtype)

        self.scheduler.set_timesteps(steps)
        init_step = int(steps * strength)
        latents = self.scheduler.add_noise(
            latents, torch.randn_like(latents), self.scheduler.timesteps[init_step]
        )

        camera = camera[:, [0, 2, 1, 3]]  # to blender convention (flip y & z axis)
        camera[:, 1] *= -1
        camera = normalize_camera(camera).view(batch_size, 16)
        camera = camera.repeat(2, 1)
        context = {"context": self.embeddings, "camera": camera, "num_frames": 4}

        for i, t in enumerate(self.scheduler.timesteps[init_step:]):
            latent_model_input = torch.cat([latents] * 2)

            tt = torch.cat([t.unsqueeze(0).repeat(batch_size)] * 2).to(self.device)

            noise_pred = self.model.apply_model(latent_model_input, tt, context)

            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]
        return imgs

    def train_step(
        self,
        pred_rgb,  # [B, C, H, W], B is multiples of 4
        camera,  # [B, 4, 4]
        step_ratio=None,
        guidance_scale=50,
        as_latent=False,
    ):
        batch_size = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.dtype)

        if as_latent:
            latents = (
                F.interpolate(pred_rgb, (32, 32), mode="bilinear", align_corners=False)
                * 2
                - 1
            )
        else:
            # interp to 256x256 to be fed into vae.
            pred_rgb_256 = F.interpolate(
                pred_rgb, (256, 256), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_256)

        if step_ratio is not None:
            # dreamtime-like
            # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
            t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(
                self.min_step, self.max_step
            )
            t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
        else:
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                (batch_size,),
                dtype=torch.long,
                device=self.device,
            )

        # camera = convert_opengl_to_blender(camera)
        # flip_yz = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]).unsqueeze(0)
        # camera = torch.matmul(flip_yz.to(camera), camera)
        camera = camera[:, [0, 2, 1, 3]]  # to blender convention (flip y & z axis)
        camera[:, 1] *= -1
        # TODO:Base on the camera to generate the openpose images
        camera = normalize_camera(camera).view(batch_size, 16)

        camera = camera.repeat(2, 1)
        # TODO: change context to match the input of ControlNet
        context = {"context": self.embeddings, "camera": camera, "num_frames": 4}

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)

            # import kiui
            # kiui.lo(latent_model_input, t, context['context'], context['camera'])

            noise_pred = self.unet(
                latent_model_input, tt, encoder_hidden_states=context
            ).sample

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_pos - noise_pred_uncond
            )

        grad = noise_pred - noise
        grad = torch.nan_to_num(grad)

        # seems important to avoid NaN...
        # grad = grad.clamp(-1, 1)

        target = (latents - grad).detach()
        loss = (
            0.5
            * F.mse_loss(latents.float(), target, reduction="sum")
            / latents.shape[0]
        )

        return loss

    def decode_latents(self, latents):
        imgs = self.model.decode_first_stage(latents)
        imgs = ((imgs + 1) / 2).clamp(0, 1)
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, 256, 256]
        imgs = 2 * imgs - 1
        latents = self.model.get_first_stage_encoding(
            self.model.encode_first_stage(imgs)
        )
        return latents  # [B, 4, 32, 32]

    @torch.no_grad()
    def prompt_to_img(
        self,
        prompts,
        negative_prompts="",
        height=256,
        width=256,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
        elevation=0,
        azimuth_start=0,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        batch_size = len(prompts) * 4

        # Text embeds -> img latents
        sampler = DDIMSampler(self.model)
        shape = [4, height // 8, width // 8]
        c_ = {"context": self.encode_text(prompts).repeat(4, 1, 1)}
        uc_ = {"context": self.encode_text(negative_prompts).repeat(4, 1, 1)}

        camera = get_camera(4, elevation=elevation, azimuth_start=azimuth_start)
        camera = camera.repeat(batch_size // 4, 1).to(self.device)

        c_["camera"] = uc_["camera"] = camera
        c_["num_frames"] = uc_["num_frames"] = 4

        latents, _ = sampler.sample(
            S=num_inference_steps,
            conditioning=c_,
            batch_size=batch_size,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=uc_,
            eta=0,
            x_T=None,
        )

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [4, 3, 256, 256]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype("uint8")

        return imgs


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str)
    parser.add_argument("--negative", default="", type=str)
    parser.add_argument("--steps", type=int, default=30)
    opt = parser.parse_args()

    device = torch.device("cuda")

    sd = StableDiffusionControlNet(device)

    while True:
        imgs = sd.prompt_to_img(opt.prompt, opt.negative, num_inference_steps=opt.steps)

        grid = np.concatenate(
            [
                np.concatenate([imgs[0], imgs[1]], axis=1),
                np.concatenate([imgs[2], imgs[3]], axis=1),
            ],
            axis=0,
        )

        # visualize image
        plt.imshow(grid)
        plt.show()
