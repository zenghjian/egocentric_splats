#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Script to visualize point clouds (dense or sparse) using rerun or matplotlib.
"""

import argparse
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

try:
    import rerun as rr
    HAS_RERUN = True
except ImportError:
    HAS_RERUN = False
    print("Rerun not available, will use matplotlib for visualization")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_with_rerun(points, colors=None, name="pointcloud"):
    """Visualize point cloud using rerun."""
    rr.init(f"Point Cloud Viewer - {name}", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP)
    
    if colors is not None:
        # Ensure colors are in 0-255 range for rerun
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        rr.log("world/points", rr.Points3D(points, colors=colors))
    else:
        rr.log("world/points", rr.Points3D(points))
    
    print("Rerun viewer launched. Press Ctrl+C to exit.")
    input("Press Enter to close...")


def visualize_with_matplotlib(points, colors=None, subsample=10000):
    """Visualize point cloud using matplotlib."""
    # Subsample if too many points
    if len(points) > subsample:
        print(f"Subsampling {len(points)} points to {subsample} for visualization")
        indices = np.random.choice(len(points), subsample, replace=False)
        points = points[indices]
        if colors is not None:
            colors = colors[indices]
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if colors is not None:
        # Ensure colors are in 0-1 range for matplotlib
        if colors.max() > 1.0:
            colors = colors / 255.0
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c=colors, s=1, alpha=0.6)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c='blue', s=1, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Point Cloud ({len(points)} points)')
    
    # Set equal aspect ratio
    max_range = np.array([
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.show()


def load_pointcloud(path):
    """Load point cloud from various formats."""
    path = Path(path)
    
    if path.suffix == '.npz':
        data = np.load(path)
        points = data['points']
        colors = data.get('colors', None)
        return points, colors
    
    elif path.suffix == '.ply':
        from plyfile import PlyData
        plydata = PlyData.read(str(path))
        
        # Extract points
        x = plydata['vertex']['x']
        y = plydata['vertex']['y']
        z = plydata['vertex']['z']
        points = np.stack([x, y, z], axis=1)
        
        # Extract colors if available
        colors = None
        if 'red' in plydata['vertex'].dtype.names:
            r = plydata['vertex']['red']
            g = plydata['vertex']['green']
            b = plydata['vertex']['blue']
            colors = np.stack([r, g, b], axis=1)
        
        return points, colors
    
    elif path.suffix == '.gz':
        # Assume it's sparse SLAM points
        import projectaria_tools.core.mps as mps
        points_data = mps.read_global_point_cloud(str(path))
        
        points = []
        for point in points_data:
            if point.inverse_distance_std < 0.01 and point.distance_std < 0.02:
                points.append(point.position_world)
        
        if len(points) == 0:
            print("No valid points found in sparse point cloud")
            return np.array([]), None
        
        return np.array(points), None
    
    else:
        raise ValueError(f"Unknown file format: {path.suffix}")


def compare_pointclouds(dense_path, sparse_path):
    """Compare dense and sparse point clouds side by side."""
    # Load point clouds
    print(f"Loading dense point cloud from {dense_path}")
    dense_points, dense_colors = load_pointcloud(dense_path)
    
    print(f"Loading sparse point cloud from {sparse_path}")
    sparse_points, _ = load_pointcloud(sparse_path)
    
    print(f"Dense points: {len(dense_points):,}")
    print(f"Sparse points: {len(sparse_points):,}")
    
    if HAS_RERUN:
        rr.init("Point Cloud Comparison", spawn=True)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP)
        
        # Log dense points in blue/colored
        if dense_colors is not None:
            if dense_colors.max() <= 1.0:
                dense_colors = (dense_colors * 255).astype(np.uint8)
            rr.log("world/dense", rr.Points3D(dense_points, colors=dense_colors))
        else:
            rr.log("world/dense", rr.Points3D(dense_points, colors=[0, 0, 255]))
        
        # Log sparse points in red
        rr.log("world/sparse", rr.Points3D(sparse_points, colors=[255, 0, 0]))
        
        print("Rerun viewer launched. Blue/colored = dense, Red = sparse")
        print("Viewer is running. Check http://localhost:9876 in your browser")
        try:
            input("Press Enter to close...")
        except EOFError:
            print("Running in non-interactive mode. Viewer will stay open.")
    else:
        # Use matplotlib with subplots
        fig = plt.figure(figsize=(16, 8))
        
        # Subsample for visualization
        max_points = 10000
        if len(dense_points) > max_points:
            dense_idx = np.random.choice(len(dense_points), max_points, replace=False)
            dense_points_vis = dense_points[dense_idx]
            dense_colors_vis = dense_colors[dense_idx] if dense_colors is not None else None
        else:
            dense_points_vis = dense_points
            dense_colors_vis = dense_colors
        
        if len(sparse_points) > max_points:
            sparse_idx = np.random.choice(len(sparse_points), max_points, replace=False)
            sparse_points_vis = sparse_points[sparse_idx]
        else:
            sparse_points_vis = sparse_points
        
        # Dense subplot
        ax1 = fig.add_subplot(121, projection='3d')
        if dense_colors_vis is not None:
            if dense_colors_vis.max() > 1.0:
                dense_colors_vis = dense_colors_vis / 255.0
            ax1.scatter(dense_points_vis[:, 0], dense_points_vis[:, 1], 
                       dense_points_vis[:, 2], c=dense_colors_vis, s=1)
        else:
            ax1.scatter(dense_points_vis[:, 0], dense_points_vis[:, 1], 
                       dense_points_vis[:, 2], c='blue', s=1)
        ax1.set_title(f'Dense ({len(dense_points):,} points)')
        
        # Sparse subplot
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(sparse_points_vis[:, 0], sparse_points_vis[:, 1], 
                   sparse_points_vis[:, 2], c='red', s=1)
        ax2.set_title(f'Sparse ({len(sparse_points):,} points)')
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize point clouds")
    parser.add_argument(
        "pointcloud_path",
        type=str,
        help="Path to point cloud file (.npz, .ply, or .csv.gz)",
    )
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help="Path to second point cloud for comparison",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "rerun", "matplotlib"],
        help="Visualization backend (default: auto)",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=10000,
        help="Subsample points for matplotlib visualization (default: 10000)",
    )
    
    args = parser.parse_args()
    
    path = Path(args.pointcloud_path)
    if not path.exists():
        print(f"Error: {path} does not exist")
        return
    
    if args.compare:
        compare_path = Path(args.compare)
        if not compare_path.exists():
            print(f"Error: {compare_path} does not exist")
            return
        compare_pointclouds(path, compare_path)
    else:
        # Load single point cloud
        print(f"Loading point cloud from {path}")
        points, colors = load_pointcloud(path)
        
        if len(points) == 0:
            print("No points to visualize")
            return
        
        # Print statistics
        print(f"Point cloud statistics:")
        print(f"  Number of points: {len(points):,}")
        print(f"  Bounding box min: {points.min(axis=0)}")
        print(f"  Bounding box max: {points.max(axis=0)}")
        print(f"  Has colors: {'Yes' if colors is not None else 'No'}")
        
        # Visualize
        use_rerun = HAS_RERUN and args.backend != "matplotlib"
        if args.backend == "matplotlib":
            use_rerun = False
        
        if use_rerun:
            visualize_with_rerun(points, colors, name=path.stem)
        else:
            visualize_with_matplotlib(points, colors, subsample=args.subsample)


if __name__ == "__main__":
    main()