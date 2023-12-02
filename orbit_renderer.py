import os
import cv2
import time
import tqdm
import numpy as np
import dearpygui.dearpygui as dpg

import torch
import torch.nn.functional as F

import rembg

from cam_utils import orbit_camera, OrbitCamera
from gs_renderer import Renderer, MiniCam

from grid_put import mipmap_linear_grid_put_2d
from mesh import Mesh, safe_normalize
import wandb
from PIL import Image

import image_utils


def render_orbit_imgs(load_path):
    renderer = Renderer(sh_degree=0)
    renderer.initialize(load_path)
    poses = []
    # render from hor -180 to 180 per 20 degree
    for hor in range(-180, 180, 40):
        pose = orbit_camera(-15, hor, 3)
        poses.append(pose)

    fovy = 50
    for i, pose in enumerate(poses):
        w2c = np.linalg.inv(pose)
        w2c[1:3, :3] *= -1
        w2c[:3, 3] *= -1
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
        bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        out = renderer.render(cur_cam, bg_color=bg_color)
        img = out["image"].unsqueeze(0)[0]
        img = img.detach().permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img.save(f"renders/{i}.png")


# main
if __name__ == "__main__":
    load_path = "logs\darthvader_sd_neg_model.ply"
    render_orbit_imgs(load_path)
    grid_image = image_utils.resize_and_fit_images("renders", "renders/gird_sd_neg.png")
