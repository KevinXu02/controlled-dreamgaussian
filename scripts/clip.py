from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import numpy as np
import path_utils
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch32")


def calculate_clip_score(images, prompts):
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=0)
    images_int = images.astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2).to('cuda'), prompts).detach()
    return round(float(clip_score), 6)


def get_dir_clip_score(dir_path):
    # load darth_vader prompts and images
    prompts, images = path_utils.load_prompt_images(dir_path)
    images = np.array(images)
    # print(prompts)

    return calculate_clip_score(images, prompts)

for dir in os.listdir('./data/images/loras_out'):
    dir_path = os.path.join('./data/images/loras_out', dir)
    print(dir, get_dir_clip_score(dir_path))


