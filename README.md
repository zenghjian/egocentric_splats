<div align="center">

<h2> Photoreal Scene Reconstruction from an Egocentric Device </h2>

<a href="#"><img src="https://img.shields.io/badge/arxiv-red" alt="arXiv"></a>
<a href="https://www.projectaria.com/photoreal-reconstruction/">
  <img src="https://img.shields.io/badge/Photoreal_Reconstruction-project_page-blue" alt="Project Page">
</a>
<a href="https://explorer.projectaria.com/aria-scenes"><img src="https://img.shields.io/badge/Aria_Scene_Dataset-dataset_download-purple" alt="Aria Dataset"></a>

Zhaoyang Lv, Maurizio Monge, Ka Chen, Yufeng Zhu, Michael Goesele, Jakob Engel, Zhao Dong, Richard Newcombe

Reality Labs Research, Meta

ACM SIGGRAPH Conference 2025

</div>

## Overview

This repository focuses on reconstructing photorealistic 3D scenes captured from an egocentric device. In contrast to off-the-shelf Gaussian Splatting reconstruction pipelines that use videos as input from structure-from-motion, we highlight two major innovations that are crucial for improving the reconstruction quality:

- **Visual-inertial bundle adjustment (VIBA)**: Unlike the mainstream approach of treating an RGB camera as a frame-rate camera, VIBA allows us to calibrate the precise timestamps and movements of an RGB camera in a high-frequency trajectory format. This supports the system in precisely modeling the online RGB camera calibrations and the pixel movements of a rolling-shutter camera.

- **Gaussian Splatting model**: We incorporate a physical image formation model based on the Gaussian Splatting algorithm, which effectively addresses sensor characteristics, including the rolling-shutter effect of RGB cameras and the dynamic ranges measured by the sensors. This formulation is general to other variants of rasterization-based techniques.

In this repository, we provide comprehensive guidelines for using the data recorded by the Aria Gen 1 device. We acquire the VIBA input from the [machine perception services](https://facebookresearch.github.io/projectaria_tools/docs/ARK/mps) provided by the Project Aria platform. Below, we offer detailed guidance on preprocessing the recordings and reconstructing them using several major variants of the Gaussian Splatting algorithms. In addition to reconstructing scenes using RGB sensors, we also provide examples of using SLAM cameras or combining all cameras together.

```
@inproceedings{lv2025egosplats,
    title={Photoreal Scene Reconstruction from an Egocentric Device},
    author={Lv, Zhaoyang and Monge, Maurizio and Chen, Ka and Zhu, Yufeng and Goesele, Michael and Engel, Jakob and Dong, Zhao and Newcombe, Richard},
    booktitle={ACM SIGGRAPH}
    year={2025}
}
```

## Quick start

``` bash
conda create -n ego_splats python=3.10
conda activate ego_splats

# Install pytorch (tested version). Choose a version that is compatible with your system
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

## Download exemplar Aria scene recording

Register on project aria dataset and get your download file at [Photoreal Reconstruction Project Aria](https://www.projectaria.com/photoreal-reconstruction/).

``` bash
# the path to the downloadable cdn file
DOWNLOAD_CDN_FILE=AriaScenes_download_urls.json
# download one of the exemplar sequence (this will take about 16GB disk space)
python scripts/downloader.py --cdn_file $DOWNLOAD_CDN_FILE -o data/aria_scenes --sequence_names livingroom

# download all the sequences (this will take about 114GB disk space)
python scripts/downloader.py --cdn_file $DOWNLOAD_CDN_FILE -o data/aria_scenes
```

You can browse the scene using the [Aria Scene Explorer](https://explorer.projectaria.com/aria-scenes).

## Run on a Project Aria recording

### Preprocess the Aria VRS recording with machine perception tool

We provided an exemplar script to preprocess the Aria VRS recording with the machine perception tool (with input of the semidense point cloud, closed loop trajectories and the online calibration files). Assume you have the examplar scene "livingroom" downloaded according to the previous step in the "data/aria_scenes" path, you can run

``` bash
bash scripts/bash_local/run_vrs_preprocessing.sh
```

For more details that happened during the preprocessing, check [Preprocess Aria video](docs/preprocess_aria_video.md)

### Run Gaussian-splatting algorithm

We provided an exemplar training script following the above preprocessing script. This is the standard setting we used in the paper.
``` bash
# Run 3D GS reconstruction using RGB camera only
bash scripts/bash_local/run_aria_rgb_camera.sh
```

During training, the model wil launch an online visualizer at "http://0.0.0.0:8080". Open browser to check the reconstruction results interactively.

In addition, we also provided a few settings that we did not use in the paper, but leveraged the capabilities of Aria videos, including using the SLAM (monochrome) camera inputs, using multi-modal cameras inputs, and other variants of Gaussian Splatting algorithm.

#### Option 1: training using aria SLAM camera only

We can run the same reconstruction process on the SLAM cameras, which are global shutter monochrome cameras. There are two of them on Project Aria Gen1 devices, which offer better field of view coverage (with limited overlap between them though). For certain applications, e.g. geometry reconstruction, this may offer more advantage over the RGB cameras.
``` bash
# Run 3D GS reconstruction using (two) SLAM cameras only
# --train_model: choose 3dgs or 2dgs.
# --strategy: default or MCMC.
# example:
bash scripts/bash_local/run_aria_slam_camera.sh --train_model 3dgs --strategy default

# or using 2d-gs
bash scripts/bash_local/run_aria_slam_camera.sh --train_model 2dgs --strategy default
```

#### Option 2: Using both RGB and SLAM modalities jointly
In addition, we can combine the RGB camera and SLAM cameras jointly in the reconstruction. We will reconstruct a RGBM radiance field with shared geometry structure.
``` bash
# has not been checked-in or tested
# --train_model: choose 3dgs or 2dgs.
# --strategy: default or MCMC.
bash scripts/local/run_aria_all_cameras.sh
```

Note: with fixed number of training iterations, this does not necessarily offer better view synthesis results over RGB or SLAM channel quantitatively, but you may find it provides reconstruction with less floaters by using all the cameras from different views.

## Visualize the reconstruction via interactive viewer

We provide an interactive viewer to visualize the trained models. For example, after launching the training scripts above, you can visualize all the models using
``` bash
python launch_viewer.py model_root=output/recording/camera-rgb-rectified-1200-h2000/
```

In default, it will show the visualizer at "http://0.0.0.0:8080". Open browser to check the results interactively.

## Render video 

We provide an example script to render the video from the trained model. Check the script for more details.

``` bash
bash scripts/bash_local/run_aria_render.sh 
```

## Capture your own videos 

Please refer to [Project Aria Docs](https://facebookresearch.github.io/projectaria_tools/docs/intro) and [Aria Research Kit](https://www.projectaria.com/research-kit/) for more details on capturing videos, running machine perception services to get the calibration and location metadata. 

To capture the videos, we used the Profile 31 which supports full resolution RGB camera with maximum exposure capped at 3ms. For outdoor scenes, we should generally support all variants of profiles. For indoor videos, this might lead to relatively darker video input if the scene is not sufficiently illuminated, but you may have a chance to recover the full dynamic range of the scene after reconstruction. If you have questions about how to get the best practice for a specific scenario, feel free to make an issue request and we will be happy to help providing some inputs. 

## License

This implementation is Creative Commons licensed, as found in the LICENSE file.

The work built in this repository benefits from the great work in the following open-source projects: 

* [Project Aria tool](https://github.com/facebookresearch/projectaria_tools): Apache 2.0 
* [EgoLifter](https://github.com/facebookresearch/egolifter), Apache 2.0
* [gsplats](https://github.com/nerfstudio-project/gsplat), Apache 2.0
* [viser](https://github.com/nerfstudio-project/viser), Apache 2.0
* [nerfview](https://github.com/nerfstudio-project/nerfview), Apache 2.0