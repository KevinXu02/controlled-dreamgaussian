from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    PNDMScheduler,
    DDIMScheduler,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
)

from diffusers.utils import load_image
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import randn_tensor
from diffusers.image_processor import VaeImageProcessor

import kiui
import numpy as np
from PIL import Image

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


MODEL_CARDS = {
    "pose": "lllyasviel/sd-controlnet-openpose",
    "depth": "lllyasviel/sd-controlnet-depth",
    "canny": "lllyasviel/sd-controlnet-canny",
    "seg": "lllyasviel/sd-controlnet-seg",
    "normal": "lllyasviel/sd-controlnet-normal",
    # 'pose': "fusing/stable-diffusion-v1-5-controlnet-openpose",
    # 'depth': "fusing/stable-diffusion-v1-5-controlnet-depth",
    # 'canny': "fusing/stable-diffusion-v1-5-controlnet-canny",
    # 'seg': "fusing/stable-diffusion-v1-5-controlnet-seg",
    # 'normal': "fusing/stable-diffusion-v1-5-controlnet-normal",
}


class ControlNet(nn.Module):
    def __init__(
        self,
        device,
        fp16=True,
        vram_O=False,
        t_range=[0.02, 0.98],
    ):
        super().__init__()

        self.device = device
        self.dtype = torch.float16 if fp16 else torch.float32

        # # Create model
        # controlnet_path = "./pretrained_models/control_v11p_sd15_openpose.pth"
        # sd_path = "./pretrained_models/v1-5-pruned-emaonly.ckpt"

        # controlnet = ControlNetModel.from_single_file(controlnet_path)

        # pipe = StableDiffusionControlNetPipeline.from_single_file(
        #     sd_path,
        #     controlnet=controlnet,
        #     torch_dtype=self.dtype,
        # )

        # if is_xformers_available():
        #     pipe.enable_xformers_memory_efficient_attention()

        cond_type = "pose"

        controlnet = ControlNetModel.from_pretrained(
            MODEL_CARDS[cond_type], torch_dtype=torch.float16 if fp16 else torch.float32
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16 if fp16 else torch.float32,
        )

        if vram_O:
            self.pipe.enable_sequential_cpu_offload()
            self.pipe.enable_vae_slicing()
            self.pipe.unet.to(memory_format=torch.channels_last)
            self.pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device)

        self.tokenizer = self.pipe.tokenizer
        self.scheduler = self.pipe.scheduler
        self.text_encoder = self.pipe.text_encoder
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.controlnet = self.pipe.controlnet
        # self.cond_processors = cond_processors

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=False,
            do_normalize=False,
        )

        self.scheduler = DDIMScheduler.from_config(
            self.pipe.scheduler.config, torch_dtype=self.dtype
        )
        del self.pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        self.embeddings = {}

    @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts):
        pos_embeds = self.encode_text(prompts)  # [1, 77, 768]
        neg_embeds = self.encode_text(negative_prompts)
        self.embeddings["pos"] = pos_embeds
        self.embeddings["neg"] = neg_embeds

        # directional embeddings
        for d in ["front", "side", "back"]:
            embeds = self.encode_text([f"{p}, {d} view" for p in prompts])
            self.embeddings[d] = embeds

    def encode_text(self, prompt):
        # prompt: [str]
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings

    def train_step(
        self,
        pred_rgb,  # TODO:This can be [B,C,H,W] or [C,H*N_1,W*N_2], modify the code accordingly
        cond_img=None,
        step_ratio=None,
        guidance_scale=10,
        as_latent=False,
        hors=None,
        debug=False,
    ):
        batch_size = 1
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
            # decode and visualize latents for debug
        if debug:
            imgs = self.decode_latents(latents)
            kiui.vis.plot_image(imgs[0].detach().cpu().permute(1, 2, 0).numpy())
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

        # predict the noise residual with unet, NO grad!
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
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)

            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)

            # Run controlnet

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

            image = self.prepare_image(
                image=cond_img,
                width=width,
                height=height,
                batch_size=batch_size * 1,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                guess_mode=False,
                device=self.device,
                dtype=self.dtype,
            )
            height, width = image.shape[-2:]

            # visualize image for debug

            # kiui.lo(embeddings, camera)
            # kiui.vis.plot_image(image[0].cpu().permute(1, 2, 0).numpy())

            # controlnet(s) inference
            control_model_input = latent_model_input
            controlnet_prompt_embeds = embeddings

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                sample=control_model_input,
                timestep=tt,
                encoder_hidden_states=controlnet_prompt_embeds,
                controlnet_cond=image,
                conditioning_scale=0,
                guess_mode=False,
                return_dict=False,
            )

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

            ### visualize the denoised image for debug. BUG HERE!!!IDK WHY
            if debug:
                alpha_t = self.scheduler.alphas.to(self.device)[t]
                beta_t = self.scheduler.betas.to(self.device)[t]
                alpha_comp_t = self.scheduler.alphas_cumprod.to(self.device)[t]
                latents_denoise = (
                    latents_noisy.detach()
                    - noise_pred.detach() * beta_t / torch.sqrt(1 - alpha_comp_t)
                ) / torch.sqrt(alpha_t)
                imgs = self.decode_latents(latents_denoise)
                kiui.vis.plot_image(imgs[0].detach().cpu().permute(1, 2, 0).numpy())

        grad = w * (noise_pred - noise).float()
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
        do_classifier_free_guidance=True,
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
        # print(image.shape)
        return image

    @torch.no_grad()
    def produce_latents(
        self,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        cond_img=None,
        latents=None,
    ):
        if latents is None:
            latents = torch.randn(
                (
                    1,
                    self.unet.in_channels,
                    height // 8,
                    width // 8,
                ),
                device=self.device,
                dtype=self.dtype,
            )

        batch_size = latents.shape[0]
        self.scheduler.set_timesteps(num_inference_steps)
        embeddings = torch.cat(
            [
                self.embeddings["pos"].expand(batch_size, -1, -1),
                self.embeddings["neg"].expand(batch_size, -1, -1),
            ],
        )

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            control_model_input = latent_model_input
            controlnet_prompt_embeds = embeddings

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                sample=control_model_input,
                timestep=t,
                encoder_hidden_states=controlnet_prompt_embeds,
                controlnet_cond=cond_img,
                conditioning_scale=1,
                guess_mode=False,
                return_dict=False,
            )

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=embeddings,
                # timestep_cond=timestep_cond,
                # cross_attention_kwargs=cross_attention_kwargs,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample

            # perform guidance
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

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
        cond_img=None,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        self.get_text_embeds(prompts, negative_prompts)

        cond_img = self.prepare_image(
            image=cond_img,
            width=width,
            height=height,
            batch_size=1,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            guess_mode=False,
            device=self.device,
            dtype=self.dtype,
        )

        # Text embeds -> img latents
        latents = self.produce_latents(
            height=height,
            width=width,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            cond_img=cond_img,
        )  # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype("uint8")

        return imgs


if __name__ == "__main__":
    from openpose_utils import *
    from cam_utils import orbit_camera
    from gs_renderer import MiniCam

    pose = orbit_camera(30, 180, 2.5)
    w2c = np.linalg.inv(pose)
    w2c[1:3, :3] *= -1
    w2c[:3, 3] *= -1
    fovy = 49
    # print(pose)
    cur_cam = MiniCam(
        pose,
        512,
        512,
        np.deg2rad(fovy),
        np.deg2rad(fovy),
        0.01,
        100,
    )
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
    w2c = np.linalg.inv(pose)
    w2c[1:3, :3] *= -1
    w2c[:3, 3] *= -1
    K = cur_cam.K()
    RT = w2c[:3, :]
    is_back = False

    openpose_image = draw_openpose_human_pose(
        K,
        RT,
        keypoints=normalized_keypoints,
        is_back=is_back,
    )

    # plt.imshow(image)
    # plt.show()
    openpose_image = Image.fromarray(openpose_image)

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str)
    parser.add_argument("--negative", default="", type=str)
    parser.add_argument("--fp16", action="store_true", help="use float16 for training")
    parser.add_argument(
        "--vram_O", action="store_true", help="optimization for low VRAM usage"
    )
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device("cuda")

    sd = ControlNet(device)

    imgs = sd.prompt_to_img(
        opt.prompt, opt.negative, opt.H, opt.W, opt.steps, cond_img=openpose_image
    )

    # visualize image
    plt.imshow(imgs[0])
    plt.show()
