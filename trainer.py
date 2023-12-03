import os
import cv2
import time
import tqdm
import numpy as np
import dearpygui.dearpygui as dpg
from configs.train_config import GuideConfig

import torch
import torch.nn.functional as F

import rembg

from cam_utils import orbit_camera, OrbitCamera
from gs_renderer import Renderer, MiniCam

from grid_put import mipmap_linear_grid_put_2d
import wandb
from openpose_utils import *


class Trainer:
    def __init__(self, opt):
        # init wandb

        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.gui = opt.gui  # enable gui
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.mode = "image"
        self.seed = "random"

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # update buffer_image

        # models
        self.device = torch.device("cuda")
        self.bg_remover = None

        self.guidance_sd = None
        self.guidance_zero123 = None

        self.enable_sd = False
        self.enable_zero123 = False

        # renderer
        self.renderer = Renderer(sh_degree=self.opt.sh_degree)
        self.gaussain_scale_factor = 1

        # input image
        self.input_img = None
        self.input_mask = None
        self.input_img_torch = None
        self.input_mask_torch = None
        self.overlay_input_img = False
        self.overlay_input_img_ratio = 0.5

        # input text
        self.prompt = ""
        self.negative_prompt = ""

        # training stuff
        self.training = False
        self.optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop

        # logging
        if opt.wandb:
            wandb.init(project="DreamGaussian", config=opt)
            wandb.watch(self.renderer.gaussians, log="all")

        # load input data from cmdline
        if self.opt.input is not None:
            self.load_input(self.opt.input)

        # override prompt from cmdline
        if self.opt.prompt is not None:
            self.prompt = self.opt.prompt

        # override if provide a checkpoint
        if self.opt.load is not None:
            self.renderer.initialize(self.opt.load, opt=self.opt)
        else:
            # initialize gaussians to a blob
            self.renderer.initialize(num_pts=self.opt.num_pts)

    def __del__(self):
        if self.gui:
            dpg.destroy_context()

    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.last_seed = seed

    def prepare_train(self):
        self.step = 0

        # setup training
        self.renderer.gaussians.training_setup(self.opt)
        # do not do progressive sh-level
        self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
        self.optimizer = self.renderer.gaussians.optimizer

        # default camera
        pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        self.fixed_cam = MiniCam(
            pose,
            self.opt.ref_size,
            self.opt.ref_size,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
        )

        self.enable_sd = self.opt.lambda_sd > 0 and self.prompt != ""
        self.enable_zero123 = self.opt.lambda_zero123 > 0 and self.input_img is not None

        # lazy load guidance model
        if self.guidance_sd is None and self.enable_sd:
            if self.opt.sdcn:
                print(f"[INFO] loading SDCN...")
                from guidance.sdcn_utils_from_avatar import (
                    ControllableScoreDistillationSampling,
                )

                self.guidance_sd = ControllableScoreDistillationSampling(
                    self.device, guide_cfg=GuideConfig()
                )
                print(f"[INFO] loaded SDCN!")

            elif self.opt.mvdream:
                print(f"[INFO] loading MVDream...")
                from guidance.mvdream_utils import MVDream

                self.guidance_sd = MVDream(self.device)
                print(f"[INFO] loaded MVDream!")
            else:
                print(f"[INFO] loading SD...")
                from guidance.sd_utils import StableDiffusion

                self.guidance_sd = StableDiffusion(
                    self.device,
                    load_from_local=opt.load_from_local,
                    local_path=opt.local_path,
                )
                print(f"[INFO] loaded SD!")

        # prepare embeddings
        with torch.no_grad():
            if self.enable_sd:
                self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])

            if self.enable_zero123:
                self.guidance_zero123.get_img_embeds(self.input_img_torch)

    def train_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        for _ in range(self.train_steps):
            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters)

            # update lr
            self.renderer.gaussians.update_learning_rate(self.step)

            loss = 0
            ###if step=200: add negative prompt
            if self.enable_sd:
                if self.step == 200:
                    extra_prompt = "unrealistic, blurry, low quality, out of focus,ugly, low contrast, dull, dark, low-resolution, gloomy"
                    self.negative_prompt += extra_prompt
                    self.guidance_sd.get_text_embeds(
                        [self.prompt], [self.negative_prompt]
                    )

            ### known view
            if self.input_img_torch is not None:
                cur_cam = self.fixed_cam
                out = self.renderer.render(cur_cam)

                # rgb loss
                image = out["image"].unsqueeze(0)  # [1, 3, H, W] in [0, 1]
                loss = loss + 10000 * step_ratio * F.mse_loss(
                    image, self.input_img_torch
                )

                # mask loss
                mask = out["alpha"].unsqueeze(0)  # [1, 1, H, W] in [0, 1]
                loss = loss + 1000 * step_ratio * F.mse_loss(
                    mask, self.input_mask_torch
                )

            ### novel view (manual batch)
            render_resolution = (
                128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)
            )
            images = []
            poses = []
            vers, hors, radii = [], [], []
            # avoid too large elevation (> 80 or < -80), and make sure it always cover [-30, 30]
            min_ver = max(min(-30, -30 - self.opt.elevation), -75 - self.opt.elevation)
            max_ver = min(max(30, 30 - self.opt.elevation), 45 - self.opt.elevation)

            for _ in range(self.opt.batch_size):
                # render random view
                ver = np.random.randint(min_ver, max_ver)
                hor = np.random.randint(-180, 180)
                # dynamic radius to avoid too small radius
                radius = 0 if step_ratio < 0.3 else (0.4 if step_ratio < 0.6 else 0.8)

                vers.append(ver)
                hors.append(hor)
                radii.append(radius)

                pose = orbit_camera(
                    self.opt.elevation + ver, hor, self.opt.radius + radius
                )
                poses.append(pose)

                cur_cam = MiniCam(
                    pose,
                    render_resolution,
                    render_resolution,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                )

                bg_color = torch.tensor(
                    [1, 1, 1]
                    if np.random.rand() > self.opt.invert_bg_prob
                    else [0, 0, 0],
                    dtype=torch.float32,
                    device="cuda",
                )
                out = self.renderer.render(cur_cam, bg_color=bg_color)

                image = out["image"].unsqueeze(0)  # [1, 3, H, W] in [0, 1]
                images.append(image)

                # enable mvdream training
                if self.opt.mvdream:
                    for view_i in range(1, 4):
                        pose_i = orbit_camera(
                            self.opt.elevation + ver,
                            hor + 90 * view_i,
                            self.opt.radius + radius,
                        )
                        poses.append(pose_i)

                        cur_cam_i = MiniCam(
                            pose_i,
                            render_resolution,
                            render_resolution,
                            self.cam.fovy,
                            self.cam.fovx,
                            self.cam.near,
                            self.cam.far,
                        )

                        # bg_color = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device="cuda")
                        out_i = self.renderer.render(cur_cam_i, bg_color=bg_color)

                        image = out_i["image"].unsqueeze(0)  # [1, 3, H, W] in [0, 1]
                        images.append(image)

            images = torch.cat(images, dim=0)
            poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)

            if self.step % self.opt.save_interval == 0:
                from PIL import Image

                front_pose = orbit_camera(-15, 15, self.opt.radius + radius)
                front_cam = MiniCam(
                    front_pose,
                    512,
                    512,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                )
                bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
                front_out = self.renderer.render(front_cam, bg_color=bg_color)
                img = front_out["image"].unsqueeze(0)[0]
                # save image
                img = img.detach().permute(1, 2, 0).cpu().numpy()
                img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img)
                output_dir = f"output/{self.opt.save_path}"
                os.makedirs(output_dir, exist_ok=True)
                img.save(f"{output_dir}/{self.step}.png")

            # guidance loss
            if self.enable_sd:
                if self.opt.sdcn:
                    cur_cam = MiniCam(
                        pose,
                        512,
                        512,
                        self.cam.fovy,
                        self.cam.fovx,
                        self.cam.near,
                        self.cam.far,
                    )
                    w2c = np.linalg.inv(pose)
                    w2c[1:3, :3] *= -1
                    w2c[:3, 3] *= -1
                    K = cur_cam.K()
                    RT = w2c[:3, :]

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

                    # TODO:Base on the camera to generate the openpose images, blender convention!!!
                    openpose_image = draw_openpose_human_pose(
                        K,
                        RT,
                        keypoints=normalized_keypoints,
                    )
                    from PIL import Image

                    openpose_image = Image.fromarray(openpose_image)

                    loss = self.opt.lambda_sd * self.guidance_sd.train_step(
                        images, poses, step_ratio, cond_img=openpose_image
                    )
                elif self.opt.mvdream:
                    loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(
                        images, poses, step_ratio
                    )
                else:
                    loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(
                        images, step_ratio, hors=hors, vers=vers
                    )

            # logging loss and render_resuluion
            if self.opt.wandb:
                wandb.log({"loss": loss.item()})
                wandb.log({"render_resolution": render_resolution})
            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # densify and prune
            if (
                self.step >= self.opt.density_start_iter
                and self.step <= self.opt.density_end_iter
            ):
                viewspace_point_tensor, visibility_filter, radii = (
                    out["viewspace_points"].to(self.device),
                    out["visibility_filter"],
                    out["radii"],
                )
                self.renderer.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.renderer.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.renderer.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )

                if self.step % self.opt.densification_interval == 0:
                    # original 0.01, 4, 1.0
                    # tuning this for better quality
                    self.renderer.gaussians.densify_and_prune(
                        self.opt.densify_grad_threshold,
                        min_opacity=self.opt.min_opacity,
                        extent=self.opt.extent,
                        max_screen_size=self.opt.max_screen_size,
                    )
                    # print number of gaussians
                    print(
                        f"[INFO] num gaussians: {self.renderer.gaussians.num_points()}"
                    )

                if self.step % self.opt.opacity_reset_interval == 0:
                    self.renderer.gaussians.reset_opacity()

        if self.step >= self.opt.density_end_iter:
            # prune every 1000 iters
            if self.step % 500 == 0:
                # min_opacity, extent, max_screen_size
                self.renderer.gaussians.prune(
                    min_opacity=self.opt.min_opacity,
                    extent=0.5,
                    max_screen_size=0,
                )

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = True
        # dynamic train steps (no need for now)
        # max allowed train time per-frame is 500 ms
        # full_t = t / self.train_steps * 16
        # train_steps = min(16, max(4, int(16 * 500 / full_t)))
        # if train_steps > self.train_steps * 1.2 or train_steps < self.train_steps * 0.8:
        #     self.train_steps = train_steps

    @torch.no_grad()
    def save_model(self, mode="ply", iter=None):
        save_dir = os.path.join(self.opt.outdir, self.opt.save_path)
        os.makedirs(save_dir, exist_ok=True)
        if mode == "ply":
            if iter is None:
                path = os.path.join(save_dir, self.opt.save_path + "_model.ply")
            else:
                path = os.path.join(
                    save_dir,
                    self.opt.save_path + "_model_" + str(iter) + ".ply",
                )
            self.renderer.gaussians.save_ply(path)
            print(f"[INFO] save model to {path}.")
        elif mode == "ckpt":
            if iter is None:
                path = os.path.join(save_dir, self.opt.save_path + "_model.ckpt")
            else:
                path = os.path.join(
                    save_dir,
                    self.opt.save_path + "_model_" + str(iter) + ".ckpt",
                )
            self.renderer.gaussians.save_ckpt(path)
            print(f"[INFO] save model to {path}.")

    # no gui mode
    def train(self, iters=500):
        if iters > 0:
            self.prepare_train()
            for iter in tqdm.trange(iters):
                self.train_step()
                # save ply per 500 iters
                if iter % self.opt.save_interval == 0 and iter != 0:
                    self.save_model(iter=iter, mode="ckpt")

            # do a last prune
            self.renderer.gaussians.prune(min_opacity=0.01, extent=1, max_screen_size=1)

        # save
        self.save_model(mode="ply")
        # self.save_model(mode="geo+tex")
        # zip the images in output folder, ensure unique name
        import shutil

        shutil.make_archive(f"./training_imgs/{self.prompt}_output", "zip", "output")


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    trainer = Trainer(opt)
    trainer.train(opt.iters)
