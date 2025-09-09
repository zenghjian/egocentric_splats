#!/usr/bin/env python3
"""
Convert 3D Gaussian Splatting PLY file to SPLAT format for web viewer.
Compatible with https://antimatter15.com/splat/
"""

import numpy as np
import struct
import argparse
from plyfile import PlyData
from pathlib import Path


def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])


def compute_covariance_3d(scale, rotation_q):
    """Compute 3D covariance matrix from scale and rotation."""
    # Convert log scale to actual scale
    S = np.diag(np.exp(scale))
    
    # Convert quaternion to rotation matrix
    R = quaternion_to_rotation_matrix(rotation_q)
    
    # Compute covariance: RSS^TR^T
    cov = R @ S @ S @ R.T
    
    return cov


def ply_to_splat(ply_path, output_path=None):
    """
    Convert PLY file to SPLAT format.
    
    The SPLAT format is a binary format with the following structure for each Gaussian:
    - 3 floats: position (x, y, z)
    - 3 floats: scale (sx, sy, sz) 
    - 4 bytes: color (r, g, b, a) as uint8
    - 4 bytes: quaternion as uint8 (remapped from [-1,1] to [0,255])
    """
    
    # Read PLY file
    print(f"Loading PLY file: {ply_path}")
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    
    num_gaussians = len(vertex)
    print(f"Number of Gaussians: {num_gaussians:,}")
    
    # Prepare output array
    splat_data = []
    
    for i in range(num_gaussians):
        if i % 10000 == 0:
            print(f"Processing Gaussian {i}/{num_gaussians}")
        
        v = vertex[i]
        
        # Position (3 floats)
        pos = [v['x'], v['y'], v['z']]
        
        # Scale - convert from log scale to actual scale
        scale = [
            np.exp(v['scale_0']),
            np.exp(v['scale_1']),
            np.exp(v['scale_2'])
        ]
        
        # Color from spherical harmonics DC component
        # Convert from SH to RGB [0,1] then to uint8
        SH_C0 = 0.28209479177387814  # Constant for SH degree 0
        color_r = 0.5 + SH_C0 * v['f_dc_0']
        color_g = 0.5 + SH_C0 * v['f_dc_1']
        color_b = 0.5 + SH_C0 * v['f_dc_2']
        
        # Clamp to [0,1]
        color_r = np.clip(color_r, 0, 1)
        color_g = np.clip(color_g, 0, 1)
        color_b = np.clip(color_b, 0, 1)
        
        # Convert to uint8
        r = int(color_r * 255)
        g = int(color_g * 255)
        b = int(color_b * 255)
        
        # Opacity - convert from log space and to uint8
        opacity = sigmoid(v['opacity'])
        a = int(opacity * 255)
        
        # Quaternion - normalize and remap to [0,255]
        quat = np.array([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']])
        quat = quat / np.linalg.norm(quat)  # Normalize
        
        # Remap from [-1,1] to [0,255]
        quat_uint8 = [int((q + 1) * 127.5) for q in quat]
        
        # Pack data for this Gaussian
        # Format: 3 floats (pos) + 3 floats (scale) + 4 bytes (rgba) + 4 bytes (quat)
        gaussian_data = struct.pack(
            '<3f3f4B4B',  # Little-endian: 3 floats, 3 floats, 4 bytes, 4 bytes
            *pos,
            *scale,
            r, g, b, a,
            *quat_uint8
        )
        
        splat_data.append(gaussian_data)
    
    # Combine all data
    full_data = b''.join(splat_data)
    
    # Write to file
    if output_path is None:
        output_path = Path(ply_path).with_suffix('.splat')
    
    print(f"Writing SPLAT file: {output_path}")
    with open(output_path, 'wb') as f:
        f.write(full_data)
    
    print(f"Conversion complete! File size: {len(full_data) / (1024*1024):.2f} MB")
    print(f"Output saved to: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Convert PLY to SPLAT format')
    parser.add_argument(
        'input',
        type=str,
        help='Path to input PLY file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to output SPLAT file (default: same as input with .splat extension)'
    )
    
    args = parser.parse_args()
    
    # Convert
    ply_to_splat(args.input, args.output)


if __name__ == '__main__':
    main()