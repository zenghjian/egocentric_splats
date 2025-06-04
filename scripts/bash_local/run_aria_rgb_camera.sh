#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# An example to run the reconstruction scripts

data_root=data/aria_scenes/livingroom
scene_name="recording/camera-rgb-rectified-1200-h2400"
output_dir="output"/$scene_name

train_model="3dgs"
opt_config="simple_gsplat_30K"
strategy="default"

wandb_project_name="aria_scene_benchmark"
wandb_exp_name="livingroom_rgb"

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
    *)    # unknown option
      shift # past argument
      ;;
  esac
done

echo "run 3D reconstruction for $data_root."
echo "train model: $train_model"
echo "optimization config: $opt_config"
echo "training strategy: $strategy"

wandb_exp_name="aria_rgb_camera_$scene_name"_"$train_model"_"$strategy"

python train_lightning.py \
    train_model=$train_model \
    opt=$opt_config \
    opt.densification_strategy=$strategy \
    opt.batch_size=1 \
    opt.depth_loss=false \
    opt.steps_scaler=1 \
    opt.handle_rolling_shutter=true \
    scene.data_root=$data_root\
    scene.scene_name=$scene_name \
    scene.input_format="aria" \
    scene.start_timestamp_ns=$start_timestamp_ns \
    scene.end_timestamp_ns=$end_timestamp_ns \
    wandb.project="$PROJECT" \
    wandb.use_wandb=true \
    exp_name=$wandb_exp_name \
    output_root=$output_dir \
    viewer.use_trainer_viewer=true
