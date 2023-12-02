# ControlledDreamGaussian

This repository contains a unofficial implementation for [DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation](https://arxiv.org/abs/2309.16653). Adding support for loss tracking, openpose ,more render methods and integrates ControlNet into the pipeline.

## TODO
- [x] Debugging ControlNet
- [x] Add support for LoRA
- [x] Hyperparameter tuning (gsgen/gspalt codebases may help)
- [x] Run more tests

Kevin:
- [x] Add more logging controls in config
- [x] Fully separate training and visualization
- [x] Add support for sd-turbo
- [x] Add support for save/load checkpoints

## IMPORTANT!!!
Please look at ./configs for the most up-to-date config options. Please add new options to the config files if needed.

## Install

conda environment:
python 3.8.18
```bash
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -U xformers --index-url https://download.pytorch.org/whl/cu118

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

### loading and visualize gaussian stage model
python main.py --config configs/text.yaml load={path_to_icecream_model}
```

Please check `./configs/text.yaml` for more options.

Text-to-3D (MVDream):

```bash
### training gaussian stage
python main.py --config configs/text_mv.yaml prompt="a plush toy of a corgi nurse" save_path=corgi_nurse

### loading gaussian stage model
python main.py --config configs/text_mv.yaml load={path_to_corgi_nurse_model}
```

Please check `./configs/text_mv.yaml` for more options.

Helper scripts:
    
    ```bash
    # visualize the gaussian splatting results
    run orbit_renderer.py
    
    ```

    ```bash

Gradio Demo:

```bash
python gradio_app.py
```

## Acknowledgement

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [threestudio](https://github.com/threestudio-project/threestudio)
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast)
- [dearpygui](https://github.com/hoffstadt/DearPyGui)

