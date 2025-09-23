#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# VRS preprocessing script with frame quality filtering

data_root="data/Apartment_release_golden_skeleton_seq100_10s_sample_M1292"
# data_root="data/Apartment_release_work_seq136_M1292"
vrs_file="video.vrs"
# vrs_file="synthetic_video.vrs"
# MPS output files
trajectory_file="${data_root}/mps/slam/closed_loop_trajectory.csv"
online_calib_file="${data_root}/mps/slam/online_calibration.jsonl"
semidense_points_file="${data_root}/mps/slam/semidense_points.csv.gz"
semidense_observation_file="${data_root}/mps/slam/semidense_observations.csv.gz"

output_root="${data_root}"

# Rectification parameters
rectified_rgb_focal=600
rectified_rgb_size=1000
rectified_monochrome_focal=250
rectified_monochrome_height=640

# Auto-detect video type and set appropriate filtering parameters
# Default parameters for regular video
filter_frames="true"  # Enable frame filtering
blur_threshold=15.0   # Minimum Laplacian variance for sharp images
trans_threshold=0.10  # Minimum translation between frames (meters)
rot_threshold=2.0     # Minimum rotation between frames (degrees)
max_angular_velocity=120.0  # Maximum angular velocity (deg/s)

# Check if using synthetic video and adjust parameters accordingly
if [[ "$vrs_file" == *"synthetic"* ]]; then
    echo "Detected synthetic video - using optimized parameters for synthetic content"
    blur_threshold=0.0    # Lower threshold for synthetic images (less texture detail)
    trans_threshold=0.00  # Lower translation threshold (smoother motion)
    rot_threshold=0.0     # Lower rotation threshold (more uniform motion)
    max_angular_velocity=100.0  # Higher tolerance for angular velocity
else
    echo "Detected regular video - using standard parameters for real footage"
    # Keep default parameters for regular video
fi

# Process arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --data_root)
      data_root="$2"
      shift 2
      ;;
    --filter_frames)
      filter_frames="$2"
      shift 2
      ;;
    --blur_threshold)
      blur_threshold="$2"
      shift 2
      ;;
    --trans_threshold)
      trans_threshold="$2"
      shift 2
      ;;
    --rot_threshold)
      rot_threshold="$2"
      shift 2
      ;;
    --max_angular_velocity)
      max_angular_velocity="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

echo "==================================================="
echo "VRS Preprocessing with Frame Quality Filtering"
echo "==================================================="
echo "Data root: $data_root"
echo "VRS file: $vrs_file"
echo ""
echo "Frame Filtering Settings:"
echo "  Enabled: $filter_frames"
echo "  Blur threshold: $blur_threshold"
echo "  Translation threshold: $trans_threshold m"
echo "  Rotation threshold: $rot_threshold°"
echo "  Max angular velocity: $max_angular_velocity°/s"
echo "==================================================="

# Build the command
cmd="python scripts/extract_aria_vrs.py \
    --input_root $data_root \
    --vrs_file $vrs_file \
    --trajectory_file $trajectory_file \
    --online_calib_file $online_calib_file \
    --semi_dense_points_file $semidense_points_file \
    --semi_dense_observation_file $semidense_observation_file \
    --output_root $output_root \
    --rectified_rgb_focal $rectified_rgb_focal \
    --rectified_rgb_size $rectified_rgb_size \
    --rectified_monochrome_focal $rectified_monochrome_focal \
    --rectified_monochrome_height $rectified_monochrome_height \
    --overwrite"

# Add frame filtering arguments if enabled
if [ "$filter_frames" = "true" ]; then
    cmd="$cmd \
    --filter_frames \
    --blur_threshold $blur_threshold \
    --trans_threshold $trans_threshold \
    --rot_threshold $rot_threshold \
    --max_angular_velocity $max_angular_velocity"
fi

# Add other options
cmd="$cmd \
    --filter_dynamic_objects"

echo ""
echo "Running command:"
echo "$cmd"
echo ""

# Execute the command
eval $cmd

echo ""
echo "==================================================="
echo "Preprocessing complete!"
echo "==================================================="