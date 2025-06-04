# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import hydra

from omegaconf import DictConfig
from pathlib import Path

from scene import initialize_render_info, initialize_eval_info, get_data_loader
import lightning as L

def create_video(input_folder, output_video, framerate, rotate: bool=False, quality: int=10):
    """
    Create a video from a folder of images using FFmpeg.
    Args:
    input_folder (str): Path to the folder containing input images.
    output_video (str): Path for the output video file.
    framerate (int): Frames per second in the output video.
    quality (int): Constant Rate Factor (CRF) for quality (0-51, where 0 is lossless and 51 is worst quality).
    """
    # Construct the FFmpeg command to convert images to video
    command = [
        'ffmpeg', '-y',
        '-framerate', str(framerate),  # Frames per second
        '-pattern_type', 'glob',
        '-i', f"'{os.path.join(input_folder, '*.png')}'",  # Input file pattern
        '-c:v', 'libx264',  # Codec video using x264
        '-preset', 'slow',  # Preset for compression (trade-off between speed and quality)
        '-crf', str(quality),  # Constant Rate Factor
        '-pix_fmt', 'yuv420p',  # Pixel format
    ]

    if rotate: 
        command += [
            '-vf', 'transpose=1',  # Video filter for 90 degrees clockwise rotation
        ]

    command.append(output_video)
    command = " ".join(command)
    print(command)
    # Execute the command
    os.system(command)

    print(f"Generate rendering video at {output_video}")

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    
    if cfg.render.render_json.endswith("transforms.json") or cfg.render.render_json.endswith("transforms_with_sparse_depth.json"):
        print("Render original trajectory")
        cfg.scene.source_path = Path(cfg.render.render_json).parent
        scene_info = initialize_eval_info(cfg)
    else:
        print("Render novel view.")
        scene_info = initialize_render_info(cfg)

    if "3dgs" in cfg.train_model:
        from model.vanilla_gsplat import VanillaGSplat
        module = VanillaGSplat(cfg=cfg, scene_info=scene_info)
    elif "2dgs" in cfg.train_model:
        from model.GS2D_gsplat import Gaussians2D
        module = Gaussians2D(cfg=cfg, scene_info=scene_info)
    else: 
        raise RuntimeError(f"cannot recognize the train model {cfg.train_model}")

    # initilaize the Gaussian splats in scene model
    assert (os.path.exists(cfg.scene.load_ply)), f"Need to have one valid ply file to load from! {cfg.scene.load_ply} does not exist!"
    module.load_ply(cfg.scene.load_ply)    
    
    render_loader = get_data_loader(scene_info.test_cameras, subset='test', render_only=True, shuffle=False)
    
    trainer = L.Trainer() 
    trainer.test(
        model=module,
        dataloaders=render_loader,
    )    

    output = cfg.render.render_output

    # use ffmpeg to automatically generate the rendering images into a video 
    if cfg.render.render_image:
        create_video(
            input_folder=f"{output}/images", 
            output_video=f"{output}/images.mp4", \
            framerate=10 if cfg.render.render_fps < 0 else cfg.render.render_fps
        )

    if cfg.render.render_depth: 
        create_video(
            input_folder=f"{output}/depth", 
            output_video=f"{output}/depth.mp4", \
            framerate=10 if cfg.render.render_fps < 0 else cfg.render.render_fps
        )

    if cfg.render.render_normal: 
        create_video(
            input_folder=f"{output}/normal", 
            output_video=f"{output}/normal.mp4", \
            framerate=10 if cfg.render.render_fps < 0 else cfg.render.render_fps
        )

    if cfg.render.render_gt: 
        create_video(
            input_folder=f"{output}/gt", 
            output_video=f"{output}/gt.mp4", \
            framerate=10 if cfg.render.render_fps < 0 else cfg.render.render_fps
        )

    



if __name__ == "__main__":
    main()