#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# VRS preprocessing script with frame quality filtering

# data_root="data/Apartment_release_golden_skeleton_seq100_10s_sample_M1292"
# data_root="data/Apartment_release_clean_seq134_M1292"
data_root="data/Apartment_release_work_seq136_M1292"
# vrs_file="video.vrs"
vrs_file="synthetic_video.vrs"

# MPS output files
trajectory_file="${data_root}/mps/slam/closed_loop_trajectory.csv"
online_calib_file="${data_root}/mps/slam/online_calibration.jsonl"
semidense_points_file="${data_root}/mps/slam/semidense_points.csv.gz"
semidense_observation_file="${data_root}/mps/slam/semidense_observations.csv.gz"

# Allow custom output root (e.g., for external drives)
# output_root="${data_root}"
scene_name=$(basename "${data_root}")
output_root="data_preprocessed/${scene_name}"

# Rectification parameters
rectified_rgb_focal=600
rectified_rgb_size=1000
rectified_monochrome_focal=250
rectified_monochrome_height=640

# Frame filtering parameters (NEW)
filter_frames="true"  # Enable frame filtering
blur_threshold=10.0   # Minimum Laplacian variance for sharp images (RGB ~15-20, SLAM ~250-300)
trans_threshold=0.05  # Minimum translation between frames (meters)
rot_threshold=1.0     # Minimum rotation between frames (degrees)
max_angular_velocity=120.0  # Maximum angular velocity (deg/s)

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