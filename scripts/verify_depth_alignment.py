#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Script to verify if ADT depth maps are aligned with rectified RGB images.
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import sys
sys.path.append(str(Path(__file__).parent.parent))

from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.projects.adt import (
    AriaDigitalTwinDataProvider,
    AriaDigitalTwinDataPathsProvider,
)


def load_rectified_rgb_frame(rectified_folder, frame_index=0):
    """Load a rectified RGB frame and its calibration."""
    
    # Load transforms.json
    transform_path = rectified_folder / "transforms.json"
    with open(transform_path, 'r') as f:
        transforms = json.load(f)
    
    # Get frame at index
    frame = transforms["frames"][frame_index]
    
    # Load image
    image_path = rectified_folder / frame["image_path"]
    image = np.array(Image.open(image_path))
    
    # Get calibration
    fx = frame["fx"]
    fy = frame["fy"]
    cx = frame["cx"]
    cy = frame["cy"]
    w = frame["w"]
    h = frame["h"]
    timestamp = frame["timestamp"]
    
    print(f"Loaded rectified RGB frame:")
    print(f"  Timestamp: {timestamp}")
    print(f"  Size: {w}x{h}")
    print(f"  Focal: ({fx:.1f}, {fy:.1f})")
    print(f"  Principal point: ({cx:.1f}, {cy:.1f})")
    
    return image, frame, timestamp


def load_rectified_depth_frame(rectified_folder, timestamp_ns):
    """Load rectified depth map from depth_maps folder."""
    
    # Look for depth file with matching timestamp
    depth_folder = rectified_folder / "depth_maps"
    depth_file = depth_folder / f"depth_{timestamp_ns}.npy"
    
    if not depth_file.exists():
        # Try to find any depth file as fallback
        depth_files = sorted(depth_folder.glob("depth_*.npy"))
        if depth_files:
            print(f"Exact timestamp not found, using first available depth: {depth_files[0].name}")
            depth_file = depth_files[0]
        else:
            print(f"No depth files found in {depth_folder}")
            return None
    
    # Load depth map
    depth_m = np.load(depth_file)
    
    print(f"Loaded rectified depth map:")
    print(f"  File: {depth_file.name}")
    print(f"  Shape: {depth_m.shape}")
    valid_depth = depth_m[depth_m > 0.1]  # Use small threshold for valid depth
    if len(valid_depth) > 0:
        print(f"  Range: [{valid_depth.min():.2f}, {valid_depth.max():.2f}] meters")
        print(f"  Valid pixels: {len(valid_depth)} / {depth_m.size} ({100*len(valid_depth)/depth_m.size:.1f}%)")
    else:
        print(f"  WARNING: No valid depth values found!")
    
    return depth_m


def project_depth_to_3d(depth_map, fx, fy, cx, cy):
    """Project depth map to 3D points using pinhole camera model."""
    
    H, W = depth_map.shape
    
    # Create pixel grid
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    
    # Valid depth mask
    valid = depth_map > 0.1
    
    # Get valid pixels
    u = xx[valid]
    v = yy[valid]
    z = depth_map[valid]
    
    # Project to 3D (pinhole model)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    points_3d = np.stack([x, y, z], axis=1)
    pixels_2d = np.stack([u, v], axis=1)
    
    return points_3d, pixels_2d, z


def check_alignment(rectified_folder, frame_index=0):
    """Check alignment between rectified RGB and rectified depth."""
    
    # Load rectified RGB
    rgb_image, rgb_frame, timestamp = load_rectified_rgb_frame(
        rectified_folder, frame_index
    )
    
    # Load rectified depth
    depth_map = load_rectified_depth_frame(
        rectified_folder, int(timestamp)
    )
    
    if depth_map is None:
        print("Failed to load depth map")
        return
    
    # Check if sizes match
    rgb_h, rgb_w = rgb_image.shape[:2]
    depth_h, depth_w = depth_map.shape
    
    print(f"\nSize comparison:")
    print(f"  Rectified RGB: {rgb_w}x{rgb_h}")
    print(f"  ADT depth: {depth_w}x{depth_h}")
    
    if (rgb_w, rgb_h) != (depth_w, depth_h):
        print("⚠️  WARNING: Sizes don't match!")
        print(f"  This should not happen with rectified data - both should be {rgb_w}x{rgb_h}")
        return
    
    # Project depth to 3D using rectified camera parameters
    fx = rgb_frame["fx"]
    fy = rgb_frame["fy"]
    cx = rgb_frame["cx"]
    cy = rgb_frame["cy"]
    
    points_3d, pixels_2d, depths = project_depth_to_3d(
        depth_map, fx, fy, cx, cy
    )
    
    # Visualize alignment
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original RGB
    axes[0, 0].imshow(rgb_image)
    axes[0, 0].set_title("Rectified RGB")
    axes[0, 0].axis('off')
    
    # Depth map with proper scaling
    valid_mask = depth_map > 0.1
    if valid_mask.any():
        vmin = depth_map[valid_mask].min()
        vmax = depth_map[valid_mask].max()
    else:
        vmin, vmax = 0, 1
    im = axes[0, 1].imshow(depth_map, cmap='turbo', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f"Rectified Depth Map\n{depth_map.shape}\nRange: [{vmin:.2f}, {vmax:.2f}]m")
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Overlay depth edges on RGB
    axes[0, 2].imshow(rgb_image)
    # Sample some depth points
    if len(pixels_2d) > 0:
        sample_idx = np.random.choice(len(pixels_2d), min(1000, len(pixels_2d)), replace=False)
        sampled_pixels = pixels_2d[sample_idx]
        sampled_depths = depths[sample_idx]
        
        # Color by depth
        scatter = axes[0, 2].scatter(
            sampled_pixels[:, 0], sampled_pixels[:, 1],
            c=sampled_depths, cmap='turbo', s=1, alpha=0.5
        )
        plt.colorbar(scatter, ax=axes[0, 2])
    axes[0, 2].set_title("Depth Points on RGB")
    axes[0, 2].axis('off')
    
    # Check edge alignment
    # Compute depth gradients
    from scipy import ndimage
    depth_edges = ndimage.sobel(depth_map)
    depth_edges = np.abs(depth_edges)
    depth_edges = depth_edges / (depth_edges.max() + 1e-6)
    
    # Convert RGB to grayscale for edge detection
    rgb_gray = np.mean(rgb_image, axis=2)
    rgb_edges = ndimage.sobel(rgb_gray)
    rgb_edges = np.abs(rgb_edges)
    rgb_edges = rgb_edges / (rgb_edges.max() + 1e-6)
    
    axes[1, 0].imshow(rgb_edges, cmap='gray')
    axes[1, 0].set_title("RGB Edges")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(depth_edges, cmap='gray')
    axes[1, 1].set_title("Depth Edges")
    axes[1, 1].axis('off')
    
    # Overlay edges
    axes[1, 2].imshow(rgb_edges, cmap='gray', alpha=0.5)
    axes[1, 2].imshow(depth_edges, cmap='hot', alpha=0.5)
    axes[1, 2].set_title("Edge Overlay\n(Gray=RGB, Hot=Depth)")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('depth_alignment_check.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to depth_alignment_check.png")
    
    # Compute correlation between edges
    valid_mask = (depth_edges > 0.1) & (rgb_edges > 0.1)
    if valid_mask.sum() > 0:
        correlation = np.corrcoef(
            rgb_edges[valid_mask].flatten(),
            depth_edges[valid_mask].flatten()
        )[0, 1]
        print(f"\nEdge correlation: {correlation:.3f}")
        if correlation > 0.5:
            print("✓ Good alignment detected")
        elif correlation > 0.3:
            print("⚠️  Moderate alignment - may need adjustment")
        else:
            print("✗ Poor alignment - depth may need rectification")
    
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify alignment between rectified RGB and rectified depth")
    parser.add_argument(
        "--rectified_folder",
        type=str,
        default="data/Apartment_release_golden_skeleton_seq100_10s_sample_M1292/synthetic_video/camera-rgb-rectified-600-h1000",
        help="Path to rectified folder containing both images/ and depth_maps/"
    )
    parser.add_argument(
        "--frame_index",
        type=int,
        default=10,
        help="Frame index to check"
    )
    
    args = parser.parse_args()
    
    rectified_folder = Path(args.rectified_folder)
    
    if not rectified_folder.exists():
        print(f"Error: Rectified folder not found: {rectified_folder}")
        return
    
    if not (rectified_folder / "depth_maps").exists():
        print(f"Error: No depth_maps folder found in {rectified_folder}")
        print("Please run preprocessing with depth rectification first")
        return
    
    print("Checking alignment between rectified RGB and rectified depth maps...")
    print(f"Rectified folder: {rectified_folder}")
    print(f"Frame index: {args.frame_index}")
    print("-" * 50)
    
    check_alignment(rectified_folder, args.frame_index)


if __name__ == "__main__":
    main()