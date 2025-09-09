#!/usr/bin/env python3
"""
Generate point cloud from rectified depth maps.
"""

import numpy as np
import json
from pathlib import Path
from PIL import Image
import glob
from tqdm import tqdm


def generate_pointcloud_from_depth_maps(
    depth_maps_folder: Path,
    transforms_json_path: Path,
    skip_n_pixels: int = 20,
    downsample_images: int = 10,
    max_frames: int = 100,
):
    """
    Generate dense point cloud from rectified depth maps.
    
    Args:
        depth_maps_folder: Path to folder containing depth .npy files
        transforms_json_path: Path to transforms.json or transforms_with_sparse_depth.json
        skip_n_pixels: Skip every N pixels when sampling points
        downsample_images: Use every Nth frame
        max_frames: Maximum number of frames to use
    
    Returns:
        points_world: Nx3 array of 3D points in world space
        colors: Nx3 array of RGB colors (0-1 range)
    """
    
    # Load transforms.json to get camera parameters and poses
    with open(transforms_json_path, 'r') as f:
        transforms_data = json.load(f)
    
    frames = transforms_data['frames']
    
    # Get list of depth files
    depth_files = sorted(glob.glob(str(depth_maps_folder / "depth_*.npy")))
    print(f"Found {len(depth_files)} depth maps")
    
    if len(depth_files) == 0:
        print("No depth maps found!")
        return None, None
    
    # Limit number of frames
    depth_files = depth_files[::downsample_images][:max_frames]
    print(f"Using {len(depth_files)} depth maps after downsampling")
    
    all_points = []
    all_colors = []
    
    for depth_file in tqdm(depth_files, desc="Processing depth maps"):
        # Extract timestamp from filename
        timestamp = Path(depth_file).stem.replace("depth_", "")
        
        # Find corresponding frame in transforms.json
        frame = None
        for f in frames:
            if timestamp in f['image_path']:
                frame = f
                break
        
        if frame is None:
            continue
        
        # Load depth map
        depth = np.load(depth_file)
        h, w = depth.shape
        
        # Get camera parameters
        fx = frame['fx']
        fy = frame['fy']
        cx = frame['cx']
        cy = frame['cy']
        
        # Get camera-to-world transform
        c2w = np.array(frame['transform_matrix'])
        
        # Create pixel grid
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        
        # Sample pixels
        xx_sampled = xx[::skip_n_pixels, ::skip_n_pixels]
        yy_sampled = yy[::skip_n_pixels, ::skip_n_pixels]
        depth_sampled = depth[::skip_n_pixels, ::skip_n_pixels]
        
        # Filter out invalid depths
        valid_mask = (depth_sampled > 0.1) & (depth_sampled < 100.0)  # 0.1m to 100m range
        xx_valid = xx_sampled[valid_mask]
        yy_valid = yy_sampled[valid_mask]
        depth_valid = depth_sampled[valid_mask]
        
        if len(depth_valid) == 0:
            continue
        
        # Unproject to camera space
        x_cam = (xx_valid - cx) * depth_valid / fx
        y_cam = (yy_valid - cy) * depth_valid / fy
        z_cam = depth_valid
        
        # Stack into points
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
        
        # Transform to world space
        points_cam_homo = np.concatenate([points_cam, np.ones((len(points_cam), 1))], axis=-1)
        points_world = (c2w @ points_cam_homo.T).T[:, :3]
        
        all_points.append(points_world)
        
        # Load corresponding RGB image for colors
        image_path = transforms_json_path.parent / frame['image_path']
        if image_path.exists():
            img = Image.open(image_path)
            img_array = np.array(img)
            
            # Sample colors at the same positions
            colors_sampled = img_array[::skip_n_pixels, ::skip_n_pixels]
            colors_valid = colors_sampled[valid_mask]
            
            # Convert to 0-1 range
            colors_valid = colors_valid.astype(np.float32) / 255.0
            all_colors.append(colors_valid)
        else:
            # Use default gray color if image not found
            gray_colors = np.ones((len(points_world), 3)) * 0.5
            all_colors.append(gray_colors)
    
    if len(all_points) == 0:
        print("No valid points found!")
        return None, None
    
    # Concatenate all points and colors
    points_world = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    
    print(f"Generated {len(points_world)} points from depth maps")
    
    # Remove outliers (points too far from the center)
    center = np.median(points_world, axis=0)
    distances = np.linalg.norm(points_world - center, axis=1)
    percentile_95 = np.percentile(distances, 95)
    inlier_mask = distances < percentile_95 * 1.5
    
    points_world = points_world[inlier_mask]
    colors = colors[inlier_mask]
    
    print(f"Kept {len(points_world)} points after outlier removal")
    
    return points_world, colors