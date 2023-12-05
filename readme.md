# ControlledDreamGaussian

This repository contains a unofficial implementation for [DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation](https://arxiv.org/abs/2309.16653). Adding support for loss tracking, openpose ,more render methods and integrates ControlNet into the pipeline.

## TODOs
- [x] Add support for LoRA (???)
- [x] Add support for sd-turbo (???)
- [x] Add support for controlnet reference mode (???)
- [x] Hyperparameter tuning and 3d content generation (gsgen/gspalt codebases may help) (Alex)
- [x] Add more logging controls in config (all)
- [x] Fully separate training and visualization (Kevin)
- [x] MORE IDEAS!!! (all)


## IMPORTANT!!!
Please look at ./configs for the most up-to-date config options. Please add new options to the config files if needed.

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

# simple-knn
pip install ./simple-knn

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
python main.py --config configs/text.yaml load={path_to_icecream_model}
```

Please check `./configs/text.yaml` for more options.

Text-to-3D (With ControlNet):

```bash
### training gaussian stage
python trainer.py --config configs/text_sdcn.yaml prompt="a plush toy of a corgi nurse" save_path=corgi_nurse

### training from checkpoint
python trainer.py --config configs/text.yaml prompt="a photo of an icecream" save_path=icecream load={path_to_icecream_model.ckpt}

### loading and visualize gaussian stage model
python main.py --config configs/text_sdcn.yaml load={path_to_corgi_nurse_model}
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
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast)
- [dearpygui](https://github.com/hoffstadt/DearPyGui)

