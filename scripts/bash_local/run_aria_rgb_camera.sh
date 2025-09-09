#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# An example to run the reconstruction scripts

# data_root=data/aria_scenes/livingroom
# scene_name="recording/camera-rgb-rectified-1200-h2400"

data_root=data/Apartment_release_golden_skeleton_seq100_10s_sample_M1292
scene_name="synthetic_video/camera-rgb-rectified-600-h1000"
output_dir="output"/$scene_name

train_model="3dgs"
opt_config="simple_gsplat_30K"
strategy="default"

# Dense point cloud initialization options
use_dense_pointcloud="true"  # Use dense point cloud from depth maps
dense_skip_pixels=20  # Skip every N pixels when generating dense point cloud
dense_downsample_images=10  # Use every Nth frame for point cloud generation
dense_max_frames=100  # Maximum frames to use for dense point cloud

# Dense depth supervision options
use_dense_depth="true"  # Use dense depth supervision during training
dense_depth_lambda=1.0  # Weight for dense depth loss
dense_depth_loss_type="huber"  # Loss type: huber, l1, or l2

wandb_project_name="aria_scene_benchmark"
wandb_exp_name="apartment_rgb"

# Use the recording from the start to end
start_timestamp_ns=-1
end_timestamp_ns=-1

# Process arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --train_model)
      train_model="$2"
      shift # past argument
      shift # past value
      ;;
    --opt_config)
      opt_config="$2"
      shift # past argument
      shift # past value
      ;;
    --strategy)
      strategy="$2"
      shift # past argument
      shift # past value
      ;;
    --steps_scaler)
      steps_scaler="$2"
      shift # past argument
      shift # past value
      ;;
    --use_dense_pointcloud)
      use_dense_pointcloud="$2"
      shift # past argument
      shift # past value
      ;;
    --use_dense_depth)
      use_dense_depth="$2"
      shift # past argument
      shift # past value
      ;;
    *)    # unknown option
      shift # past argument
      ;;
  esac
done

echo "run 3D reconstruction for $data_root."
echo "train model: $train_model"
echo "optimization config: $opt_config"
echo "training strategy: $strategy"
echo "use dense pointcloud: $use_dense_pointcloud"
echo "use dense depth supervision: $use_dense_depth"

wandb_exp_name="aria_rgb_camera_$scene_name"_"$train_model"_"$strategy"

python train_lightning.py \
    train_model=$train_model \
    opt=$opt_config \
    opt.densification_strategy=$strategy \
    opt.batch_size=1 \
    opt.depth_loss=false \
    opt.use_dense_depth=$use_dense_depth \
    opt.dense_depth_lambda=$dense_depth_lambda \
    opt.dense_depth_loss_type=$dense_depth_loss_type \
    opt.use_inverse_depth=true \
    opt.combine_depth_supervision=false \
    opt.steps_scaler=1 \
    opt.handle_rolling_shutter=true \
    scene.data_root=$data_root\
    scene.scene_name=$scene_name \
    scene.input_format="aria" \
    scene.start_timestamp_ns=$start_timestamp_ns \
    scene.end_timestamp_ns=$end_timestamp_ns \
    scene.use_dense_pointcloud=$use_dense_pointcloud \
    scene.dense_skip_pixels=$dense_skip_pixels \
    scene.dense_downsample_images=$dense_downsample_images \
    scene.dense_max_frames=$dense_max_frames \
    wandb.project="$wandb_project_name" \
    wandb.use_wandb=false \
    exp_name=$wandb_exp_name \
    output_root=$output_dir \
    viewer.use_trainer_viewer=true
