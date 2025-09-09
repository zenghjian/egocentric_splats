# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.projects.adt import (
    AriaDigitalTwinDataProvider,
    AriaDigitalTwinDataPathsProvider,
)


class ADTDepthLoader:
    """
    Loader for ADT (Aria Digital Twin) dense depth maps.
    This class handles loading and preprocessing of depth images from ADT dataset.
    """
    
    def __init__(self, data_root: Path, downsample_factor: int = 1):
        """
        Initialize ADT depth loader.
        
        Args:
            data_root: Path to ADT data directory containing depth_images.vrs
            downsample_factor: Factor to downsample depth maps (1, 2, 4, or 8)
        """
        self.data_root = Path(data_root)
        self.downsample_factor = downsample_factor
        
        # Check if depth_images.vrs exists
        self.depth_vrs_path = self.data_root / "depth_images.vrs"
        if not self.depth_vrs_path.exists():
            raise FileNotFoundError(f"Depth VRS file not found: {self.depth_vrs_path}")
        
        # Initialize ADT data provider
        self.paths_provider = AriaDigitalTwinDataPathsProvider(str(self.data_root))
        self.data_paths = self.paths_provider.get_datapaths()
        self.gt_provider = AriaDigitalTwinDataProvider(self.data_paths)
        
        # Get RGB camera stream ID
        self.rgb_stream_id = StreamId("214-1")  # RGB camera stream
        
        # Get all available timestamps
        self.timestamps_ns = self.gt_provider.get_aria_device_capture_timestamps_ns(
            self.rgb_stream_id
        )
        
        print(f"ADT Depth Loader initialized with {len(self.timestamps_ns)} frames")
        print(f"Depth maps will be downsampled by factor: {self.downsample_factor}")
    
    def get_depth_by_timestamp(
        self, 
        timestamp_ns: int,
        return_mask: bool = True
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get depth map for a specific timestamp.
        
        Args:
            timestamp_ns: Timestamp in nanoseconds
            return_mask: Whether to return valid depth mask
            
        Returns:
            depth_map: Depth map in meters as torch tensor (H, W)
            valid_mask: Boolean mask of valid depth values (H, W) if return_mask=True
        """
        # Check if timestamp is within valid range
        if (timestamp_ns < self.gt_provider.get_start_time_ns() or 
            timestamp_ns > self.gt_provider.get_end_time_ns()):
            return None, None
        
        try:
            # Get depth image from provider
            depth_data = self.gt_provider.get_depth_image_by_timestamp_ns(
                timestamp_ns, self.rgb_stream_id
            )
            
            if depth_data is None or not depth_data.is_valid:
                return None, None
            
            # Convert to numpy array (depth in millimeters)
            depth_mm = depth_data.data().to_numpy_array()
            
            # Convert to meters
            depth_m = depth_mm.astype(np.float32) / 1000.0
            
            # Downsample if needed
            if self.downsample_factor > 1:
                H, W = depth_m.shape
                new_H = H // self.downsample_factor
                new_W = W // self.downsample_factor
                
                # Use average pooling for downsampling
                depth_m_tensor = torch.from_numpy(depth_m).unsqueeze(0).unsqueeze(0)
                depth_m_downsampled = torch.nn.functional.avg_pool2d(
                    depth_m_tensor, 
                    kernel_size=self.downsample_factor,
                    stride=self.downsample_factor
                )
                depth_m = depth_m_downsampled.squeeze().numpy()
            
            # Convert to torch tensor
            depth_tensor = torch.from_numpy(depth_m)
            
            # Create valid mask (depth > 0)
            valid_mask = None
            if return_mask:
                valid_mask = depth_tensor > 0
            
            return depth_tensor, valid_mask
            
        except Exception as e:
            print(f"Error loading depth for timestamp {timestamp_ns}: {e}")
            return None, None
    
    def get_depth_by_index(
        self, 
        index: int,
        return_mask: bool = True
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
        """
        Get depth map by frame index.
        
        Args:
            index: Frame index
            return_mask: Whether to return valid depth mask
            
        Returns:
            depth_map: Depth map in meters as torch tensor (H, W)
            valid_mask: Boolean mask of valid depth values (H, W) if return_mask=True
            timestamp_ns: Timestamp in nanoseconds
        """
        if index < 0 or index >= len(self.timestamps_ns):
            return None, None, -1
        
        timestamp_ns = self.timestamps_ns[index]
        depth_map, valid_mask = self.get_depth_by_timestamp(timestamp_ns, return_mask)
        
        return depth_map, valid_mask, timestamp_ns
    
    def get_camera_calibration(self):
        """
        Get camera calibration for RGB camera.
        
        Returns:
            Camera calibration object
        """
        return self.gt_provider.get_aria_camera_calibration(self.rgb_stream_id)
    
    def get_aria_pose(self, timestamp_ns: int):
        """
        Get Aria device pose at specific timestamp.
        
        Args:
            timestamp_ns: Timestamp in nanoseconds
            
        Returns:
            Transform matrix (4x4) from device to scene coordinates
        """
        pose_with_dt = self.gt_provider.get_aria_3d_pose_by_timestamp_ns(timestamp_ns)
        
        if not pose_with_dt.is_valid:
            return None
        
        return pose_with_dt.data().transform_scene_device
    
    def preprocess_depth_for_training(
        self,
        depth_map: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        max_depth: float = 10.0,
        min_depth: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess depth map for training.
        
        Args:
            depth_map: Raw depth map in meters
            valid_mask: Valid depth mask
            max_depth: Maximum depth threshold
            min_depth: Minimum depth threshold
            
        Returns:
            processed_depth: Processed depth map
            processed_mask: Updated valid mask
        """
        # Clone to avoid modifying original
        processed_depth = depth_map.clone()
        
        # Create or update valid mask
        if valid_mask is None:
            processed_mask = (processed_depth > min_depth) & (processed_depth < max_depth)
        else:
            processed_mask = valid_mask & (processed_depth > min_depth) & (processed_depth < max_depth)
        
        # Clamp depth values
        processed_depth = torch.clamp(processed_depth, min=min_depth, max=max_depth)
        
        # Set invalid regions to 0
        processed_depth[~processed_mask] = 0
        
        return processed_depth, processed_mask