from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    PNDMScheduler,
    DDIMScheduler,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
)
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import randn_tensor
from diffusers.image_processor import VaeImageProcessor
from openpose_utils import *
from cam_utils import orbit_camera
from gs_renderer import MiniCam
from openpose_utils import *

import numpy as np

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
        self.controlnet = pipe.controlnet
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )
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

    # @torch.no_grad()
    # def refine(
    #     self,
    #     pred_rgb,
    #     camera,
    #     guidance_scale=100,
    #     steps=50,
    #     strength=0.8,
    # ):
    #     batch_size = pred_rgb.shape[0]
    #     pred_rgb_256 = F.interpolate(
    #         pred_rgb, (256, 256), mode="bilinear", align_corners=False
    #     )
    #     latents = self.encode_imgs(pred_rgb_256.to(self.dtype))
    #     # latents = torch.randn((1, 4, 64, 64), device=self.device, dtype=self.dtype)

    #     self.scheduler.set_timesteps(steps)
    #     init_step = int(steps * strength)
    #     latents = self.scheduler.add_noise(
    #         latents, torch.randn_like(latents), self.scheduler.timesteps[init_step]
    #     )

    #     camera = camera[:, [0, 2, 1, 3]]  # to blender convention (flip y & z axis)
    #     camera[:, 1] *= -1
    #     camera = normalize_camera(camera).view(batch_size, 16)
    #     camera = camera.repeat(2, 1)
    #     context = {"context": self.embeddings, "camera": camera, "num_frames": 4}

    #     for i, t in enumerate(self.scheduler.timesteps[init_step:]):
    #         latent_model_input = torch.cat([latents] * 2)

    #         tt = torch.cat([t.unsqueeze(0).repeat(batch_size)] * 2).to(self.device)

    #         noise_pred = self.model.apply_model(latent_model_input, tt, context)

    #         noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
    #         noise_pred = noise_pred_uncond + guidance_scale * (
    #             noise_pred_cond - noise_pred_uncond
    #         )

    #         latents = self.scheduler.step(noise_pred, t, latents).prev_sample

    #     imgs = self.decode_latents(latents)  # [1, 3, 512, 512]
    #     return imgs

    def train_step(
        self,
        pred_rgb,  # TODO:This can be [B,C,H,W] or [C,H*N_1,W*N_2], modify the code accordingly
        camera,  # TODO: [B, 4, 4], modify the code accordingly
        cur_cam,
        step_ratio=None,
        guidance_scale=7.5,
        as_latent=False,
        hors=None,
    ):
        batch_size = pred_rgb.shape[0]
        # TODO: get the width and height of the generate image
        width = 512
        height = 512

        pred_rgb = pred_rgb.to(self.dtype)

        if as_latent:
            latents = (
                F.interpolate(pred_rgb, (64, 64), mode="bilinear", align_corners=False)
                * 2
                - 1
            )
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(
                pred_rgb, (512, 512), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        with torch.no_grad():
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

            # 7.1 Create tensor stating which controlnets to keep
            # controlnet_keep = []
            # for i in range(len(timesteps)):
            #     keeps = [
            #         1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
            #         for s, e in zip(control_guidance_start, control_guidance_end)
            #     ]
            #     controlnet_keep.append(
            #         keeps[0] if isinstance(controlnet, ControlNetModel) else keeps
            #     )

            # camera = convert_opengl_to_blender(camera)
            # flip_yz = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]).unsqueeze(0)
            # camera = torch.matmul(flip_yz.to(camera), camera)
            T_pose_keypoints = np.array(
                [
                    [0, 158, 14],
                    [0, 138, 0],
                    [-17, 138, 0],
                    [-17, 113, 0],
                    [-17, 88, 0],
                    [17, 138, 0],
                    [17, 113, 0],
                    [17, 88, 0],
                    [-10, 92, 0],
                    [-10, 52, 0],
                    [-10, 16, 0],
                    [10, 92, 0],
                    [10, 52, 0],
                    [10, 16, 0],
                    [-3, 161, 11],
                    [3, 161, 11],
                    [-7, 158, 3],
                    [7, 158, 3],
                ]
            )

            normalized_keypoints = mid_and_scale(T_pose_keypoints)
            w2c = np.linalg.inv(camera)
            w2c[1:3, :3] *= -1
            w2c[:3, 3] *= -1

            K = cur_cam.K()
            RT = w2c[:3, :]
            # TODO:Base on the camera to generate the openpose images, blender convention!!!

            if hors is None:
                embeddings = torch.cat(
                    [
                        self.embeddings["pos"].expand(batch_size, -1, -1),
                        self.embeddings["neg"].expand(batch_size, -1, -1),
                    ]
                )
            else:

                def _get_dir_ind(h):
                    if abs(h) < 60:
                        return "front"
                    elif abs(h) < 120:
                        return "side"
                    else:
                        return "back"

                embeddings = torch.cat(
                    [self.embeddings[_get_dir_ind(h)] for h in hors]
                    + [self.embeddings["neg"].expand(batch_size, -1, -1)]
                )

            image = draw_openpose_human_pose(
                normalized_keypoints,
                (512, 512),
                K,
                RT,
            )
            # predict the noise residual with unet, NO grad!

            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)
            w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)
            # Run controlnet

            image = self.prepare_image(
                image=image,
                width=width,
                height=height,
                batch_size=batch_size * 1,
                num_images_per_prompt=1,
                device=device,
                dtype=self.controlnet.dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                guess_mode=False,
            )
            height, width = image.shape[-2:]

            control_model_input = latent_model_input
            controlnet_prompt_embeds = embeddings
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                control_model_input,
                tt,
                encoder_hidden_states=controlnet_prompt_embeds,
                controlnet_cond=image,
                # conditioning_scale=cond_scale,
                guess_mode=False,
                return_dict=False,
            )
            # import kiui
            # kiui.lo(latent_model_input, t, context['context'], context['camera'])

            # noise_pred = self.unet(
            #     latent_model_input, tt, encoder_hidden_states=context
            # ).sample
            noise_pred = self.unet(
                latent_model_input,
                tt,
                encoder_hidden_states=embeddings,
                # timestep_cond=timestep_cond,
                # cross_attention_kwargs=cross_attention_kwargs,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_pos - noise_pred_uncond
            )

        grad = w * (noise_pred - noise)
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

    @torch.no_grad()
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        image = self.control_image_processor.preprocess(
            image, height=height, width=width
        ).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    def decode_latents(self, latents):
        imgs = self.model.decode_first_stage(latents)
        imgs = ((imgs + 1) / 2).clamp(0, 1)
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    @torch.no_grad()
    def prompt_to_img(
        self,
        prompts,
        negative_prompts="",
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        self.get_text_embeds(prompts, negative_prompts)

        # Text embeds -> img latents
        latents = self.produce_latents(
            height=height,
            width=width,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )  # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

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
