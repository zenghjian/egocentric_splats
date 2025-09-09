
#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

results_folder=output/synthetic_video/camera-rgb-rectified-600-h1000/aria_rgb_camera_synthetic_video/camera-rgb-rectified-600-h1000_3dgs_default

gain_amplify=1  # amplify the analog gain by 2, compared to original video
render_fps=10   # render the 10 fps video. Will interpolate the poses from the keyframes

ply_file=$results_folder/point_cloud/iteration_30000/point_cloud.ply
json_file=$results_folder/cameras.json
render_output=$results_folder/traj_render_fps_"$render_fps"_gain_"$gain_amplify"

python render_lightning.py \
    train_model=3dgs \
    scene.load_ply=$ply_file \
    render.render_only=true \
    render.render_json=$json_file \
    render.render_output=$render_output \
    render.render_fps=$render_fps \
    render.gain_amplify=$gain_amplify