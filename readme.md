# ControlledDreamGaussian

This repository contains an unofficial implementation for [DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation](https://arxiv.org/abs/2309.16653). Deleted mesh stuffs and adding support for loss tracking, OpenPose, SMPL, more render methods and integrates ControlNet into the pipeline. Hyperparameters are also tuned for better results. Trainer is splitted from visualization.

## Install

conda environment:
python 3.8.18
```bash
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -U xformers --index-url https://download.pytorch.org/whl/cu118 (optional) (if not installed, you can change pytorch cuda version to the one you have)

pip install -r requirements.txt

# a modified gaussian splatting (+ depth, alpha rendering)
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

# kiuikit
pip install git+https://github.com/ashawkey/kiuikit
```

## Usage
Text-to-3D:

```bash
### training gaussian stage
python trainer.py --config configs/text.yaml prompt="a photo of an icecream" save_path=icecream

### training from checkpoint
python trainer.py --config configs/text.yaml prompt="a photo of an icecream" save_path=icecream load={path_to_icecream_model.ckpt}

### loading and visualize gaussian stage model
python vis.py --config configs/text.yaml load={path_to_icecream_model}
```

Please check `./configs/text.yaml` for more options.

Text-to-3D (With ControlNet OpenPose):

```bash
### training gaussian stage
python trainer.py --config configs/text_sdcn.yaml prompt="iron man" save_path=iron_man

### training from checkpoint
python trainer.py --config configs/text_sdcn.yaml prompt="iron man" save_path=iron_man

### loading and visualize gaussian stage model
python vis.py --config configs/text_sdcn.yaml load={path_to_iron_man_model}
```

Please check `./configs/text_sdcn.yaml` for more options. You may use your own pose from https://zhuyu1997.github.io/open-pose-editor/. Just right click and copy the key points into configs\t_pose_keypoints.py. And change the pose name in configs\text_sdcn.yaml.

Text-to-3D (With ControlNet OpenPose and Depth):
THIS REQUIRES PYRENDERER TO BE INSTALLED.

```bash
### training gaussian stage
python trainer.py --config configs/text_sdcn_depth.yaml prompt="iron man" save_path=iron_man

### training from checkpoint
python trainer.py --config configs/text_sdcn_depth.yaml prompt="iron man" save_path=iron_man

### loading and visualize gaussian stage model
python vis.py --config configs/text_sdcn_depth.yaml load={path_to_iron_man_model}
```

Please check `./configs/text_sdcn.yaml` for more options. You may use your own pose from https://zhuyu1997.github.io/open-pose-editor/. Just right click and copy the key points into config/text_sdcn.yaml.

## Loading LoRA models
As diffuser doesn't support loading LoRA models to the text encoder, we have to merge the base SD model with the LoRA model in the A1111WebUI. https://github.com/AUTOMATIC1111/stable-diffusion-webui. And you can load the merged model by modifying the config file.

Helper scripts:
    
    ```bash
    # visualize the gaussian splatting results
    see orbit_renderer.py
    
    ```

    ```bash
```

## Acknowledgement

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [threestudio](https://github.com/threestudio-project/threestudio)
- [dearpygui](https://github.com/hoffstadt/DearPyGui)

