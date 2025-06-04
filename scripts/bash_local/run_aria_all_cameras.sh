#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# An example to run the reconstruction scripts

data_root=data/aria_scenes/livingroom
# use the right folder name accordingly
# RGB folder name: camera-rgb-rectified-1200-h2400
# SLAM folder names: slam-*-rectified-180-h480
scene_name="recording/camera-rgb-rectified-1200-h2400+camera-slam-*-rectified-180-h480"

wandb_project_name="aria_scene_benchmark"
output_dir=./output/$scene_name

train_model="3dgs" # choose between "3dgs", "2dgs"
opt_config="simple_gsplat_30K" # currently only supporting this
strategy="default" # choose between "default" or "MCMC"

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

#run Gaussian-splatting reconstruction for SLAM cameras first
wandb_exp_name="aria_all_camreas_$scene_name"_"$train_model"_"$strategy"

python train_lightning.py \
    train_model=$train_model \
    opt=$opt_config \
    opt.batch_size=1 \
    opt.densification_strategy=$strategy \
    opt.handle_rolling_shutter=true \
    scene.data_root=$data_root\
    scene.scene_name=$scene_name \
    scene.input_format="aria" \
    scene.start_timestamp_ns=$start_timestamp_ns \
    scene.end_timestamp_ns=$end_timestamp_ns \
    wandb.project=$wandb_project_name \
    exp_name=$wandb_exp_name \
    output_root=$output_dir \
    wandb.use_wandb=true \
    viewer.use_trainer_viewer=true
