#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Script to run 3D reconstruction with two modes:
# - improved: Uses ADT depth maps, dense point cloud, and static mask (recommended for ADT data)
# - original: Uses only sparse SLAM points, no ADT enhancements (traditional 3DGS)
#
# Usage: ./run_aria_rgb_camera.sh [--mode improved|original] [other options...]

# data_root=data/aria_scenes/livingroom
# scene_name="recording/camera-rgb-rectified-1200-h2400"

# data_root=data/Apartment_release_golden_skeleton_seq100_10s_sample_M1292
data_root=data/Apartment_release_work_seq136_M1292
# scene_name="synthetic_video/camera-rgb-rectified-600-h1000"
# data_root=data/Apartment_release_clean_seq131_M1292
# data_root=data/Apartment_release_decoration_skeleton_seq131_M1292
scene_name="video/camera-rgb-rectified-600-h1000"
output_dir="output"/$scene_name

train_model="3dgs"
opt_config="simple_gsplat_30K"
strategy="default"

# Mode selection: "improved" or "original"
# improved: Uses ADT depth maps, dense point cloud, and static mask
# original: Uses only sparse SLAM points (traditional 3DGS)
mode="improved"  # Default to improved mode

# Dense point cloud initialization options
dense_skip_pixels=20  # Skip every N pixels when generating dense point cloud
dense_downsample_images=10  # Use every Nth frame for point cloud generation
dense_max_frames=100  # Maximum frames to use for dense point cloud

# Dense depth supervision options (from ADT depth maps)
dense_depth_lambda=1.0  # Weight for dense depth loss from ADT
dense_depth_loss_type="huber"  # Loss type: huber, l1, or l2

# Sparse depth supervision options (from SLAM points)
sparse_depth_lambda=2e-4  # Weight for sparse depth loss (improved: not used, original: 2e-4)

# These will be set automatically based on mode
use_dense_pointcloud=""
use_dense_depth=""
use_static_mask_loss=""
use_sparse_depth=""

wandb_project_name="aria_scene_benchmark"
wandb_exp_name=""  # Will be set based on mode

# Use the recording from the start to end
start_timestamp_ns=-1
end_timestamp_ns=-1

# Process arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --mode)
      mode="$2"
      shift # past argument
      shift # past value
      ;;
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
    --use_static_mask_loss)
      use_static_mask_loss="$2"
      shift # past argument
      shift # past value
      ;;
    *)    # unknown option
      shift # past argument
      ;;
  esac
done

# Configure settings based on mode
if [ "$mode" = "improved" ]; then
    echo "================================================"
    echo "Using IMPROVED mode with ADT enhancements"
    echo "================================================"
    use_dense_pointcloud="true"   # Use dense point cloud from ADT depth maps
    use_dense_depth="true"        # Use dense depth supervision from ADT
    use_static_mask_loss="true"   # Use static mask to filter dynamic objects
    use_sparse_depth="false"      # Don't use sparse SLAM points (we have dense)
    wandb_exp_name="apartment_rgb_improved"
elif [ "$mode" = "original" ]; then
    echo "================================================"
    echo "Using ORIGINAL mode (traditional 3DGS)"
    echo "================================================"
    use_dense_pointcloud="false"  # Don't use dense point cloud
    use_dense_depth="false"       # Don't use dense depth supervision
    use_static_mask_loss="false"  # Don't use static mask
    use_sparse_depth="true"       # Use sparse SLAM points for depth supervision
    sparse_depth_lambda=2e-4      # Use original sparse depth weight
    wandb_exp_name="apartment_rgb_original"
else
    echo "ERROR: Invalid mode '$mode'. Must be 'improved' or 'original'"
    echo "Usage: $0 [--mode improved|original] [other options...]"
    exit 1
fi

echo "Configuration Summary:"
echo "  Data root: $data_root"
echo "  Scene: $scene_name"
echo "  Train model: $train_model"
echo "  Optimization config: $opt_config"
echo "  Training strategy: $strategy"
echo "------------------------------------------------"
echo "Mode-specific settings:"
echo "  Use dense pointcloud: $use_dense_pointcloud"
echo "  Use dense depth: $use_dense_depth"
echo "  Use sparse depth: $use_sparse_depth"
echo "  Use static mask loss: $use_static_mask_loss"
echo "  Sparse depth lambda: $sparse_depth_lambda"
echo "  Dense depth lambda: $dense_depth_lambda"
echo "================================================"

wandb_exp_name="${wandb_exp_name}_${train_model}_${strategy}"

python train_lightning.py \
    train_model=$train_model \
    opt=$opt_config \
    opt.densification_strategy=$strategy \
    opt.batch_size=1 \
    opt.depth_loss=$use_sparse_depth \
    opt.depth_lambda=$sparse_depth_lambda \
    opt.use_dense_depth=$use_dense_depth \
    opt.dense_depth_lambda=$dense_depth_lambda \
    opt.dense_depth_loss_type=$dense_depth_loss_type \
    opt.use_inverse_depth=true \
    opt.combine_depth_supervision=false \
    opt.use_static_mask_loss=$use_static_mask_loss \
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
    wandb.use_wandb=true \
    exp_name=$wandb_exp_name \
    output_root=$output_dir \
    viewer.use_trainer_viewer=true