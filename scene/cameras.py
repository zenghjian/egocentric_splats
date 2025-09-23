# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import math
from dataclasses import dataclass

from pathlib import Path
from typing import Dict, List, Literal

import einops
import numpy as np
import projectaria_tools.core.mps as mps
import pyquaternion

import torch
from PIL import Image
from projectaria_tools.core.sophus import interpolate, SE3
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.aria_utils import interpolate_aria_pose

from utils.point_utils import project


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def PILtoTorch(input_image, resolution: tuple = None):
    if resolution is not None:
        input_image = input_image.resize(resolution)

    updated_image = np.array(input_image)
    updated_image = torch.from_numpy(updated_image) / 255.0
    updated_image = updated_image.clamp(0.0, 1.0)

    if len(updated_image.shape) == 3:
        return updated_image.permute(2, 0, 1)
    else:
        return updated_image[None]


@dataclass
class SubImageMask:
    """
    a sub-image ROI mask information
    """

    index: int  # the mask index in the original image
    mask: (
        torch.Tensor
    )  # mask with non-zero indicating the corresponding ROI in this image.
    y_min: int  # min index of y of the mask in the original image that is non-zero
    y_max: int  # max index of y of the mask in the original image that is non-zero


class Camera:

    # The following are shared across all cameras.
    # The vignette image determines the falloff lens shading ratio for each pixel.
    _vignette_image: Dict[str, torch.Tensor] = {}
    # The mask image determines an overall thresholded region for each pixel.
    _mask_image: Dict[str, torch.Tensor] = {}

    # the mapping of camera models supported in Aria camera to GSplat render camera model
    _camera_model_aria2gsplats = {
        "linear": "pinhole",
        "spherical": "fisheye",
    }

    max_exposure_s: float = 2e-3

    # map the irradiance in a nonlinear (gamma) space
    use_nonlinear_irradiance: bool = True

    def __init__(
        self,
        uid: int,
        w2c: np.ndarray,  
        FoVx: float,
        FoVy: float,
        image_width: int,
        image_height: int,
        cx: int = -1,
        cy: int = -1,
        image_name: str = None,
        image_path: str = None,
        mask_path: str = None,
        camera_name: Literal[
            "camera-rgb", "camera-slam-left", "camera-slam-right"
        ] = "camera-rgb",
        camera_projection_model: Literal["linear", "spherical"] = "linear",
        camera_modality: Literal["rgb", "monochrome"] = "rgb",
        exposure_duration_s: float = 1.0,
        gain: float = 1.0,
        radiance_weight: float = 1.0,
        scene_name: str = "",
        cache_image: bool = False,
    ):
        super(Camera, self).__init__()

        # This is mostly used as the index of the camera among all cameras.
        self._uid = uid  

        self._w2c = w2c

        self._fov_x = FoVx
        self._fov_y = FoVy
        self._image_width = image_width
        self._image_height = image_height

        if cx < 0:
            self._cx = self._image_width // 2
        else:
            self._cx = cx

        if cy < 0:
            self._cy = self._image_height // 2
        else:
            self._cy = cy

        # retain a copy for reset purpose
        self._fov_x_initial = FoVx
        self._fov_y_initial = FoVy
        self._image_width_initial = image_width
        self._image_height_initial = image_height

        self._image_path = image_path
        self._mask_path = mask_path
        self._exposure_duration_s = exposure_duration_s
        self._gain = gain
        self._camera_name = camera_name
        self._camera_modality = camera_modality
        self._camera_projection_model = camera_projection_model

        # for a standard camera without explicit camera model. We will use the uid.
        self._timestamp_ns = uid

        self._rs_info = []

        # self.image_name = image_name
        self.scene_name = scene_name

        self.zfar = 100.0
        self.znear = 0.01

        # We will rescale the dynamic range of the observed irradiance image.
        # The denominator to compensate the weight of exposure & gain
        self._radiance_weight = radiance_weight

        # The observed image (if available)
        self._observed_image: torch.Tensor = None
        # An associated mask image (if available)
        self._alpha_mask: torch.Tensor = None
        # sparse depth input (if available)
        self._sparse_points: torch.Tensor = None  # (N, 5)

        # The estimated image (cached from most recent rendering if available)
        self._est_image: torch.Tensor = None
        # The estimated depth (cached from most recent rendering if available)
        self._est_depth: torch.Tensor = None
        # The estimated normal (cached from most recent rendering if available)
        self._est_normal: torch.Tensor = None

        # cache the loaded image (could lead to memory issue if turns out to be too many of them)
        self.cache_image: bool = cache_image

    def set_vignette_image(self, vignette_image_path: str, camera_name: str):
        """
        This will set the vignette image for a specific camera_name
        """
        if camera_name != self._camera_name:
            raise RuntimeError(
                f"Vignette image camera name {camera_name} is inconsistent with current name {self.camera_name}"
            )

        if camera_name in self._vignette_image.keys():
            raise Warning(
                "The provided vignette image has already been set! Overwrite existing vignette image path!"
            )

        assert (
            vignette_image_path.exists()
        ), f"Could not find the vignette image for for camera-rgb at {vignette_image_path}"
        vignette_image = np.array(Image.open(str(vignette_image_path))) / 255.0

        # only choose one channel for slam camera
        if camera_name.startswith("camera-slam"):
            vignette = torch.from_numpy(vignette_image[..., 0]).float()
        else:
            if vignette_image.ndim == 2:
                vignette = torch.from_numpy(vignette_image)[None].float()
            else:
                vignette = einops.rearrange(
                    torch.from_numpy(vignette_image), "h w c -> c h w"
                ).float()

        self._vignette_image[camera_name] = vignette

    def set_valid_mask(self, mask_image_path: str, camera_name: str):
        """
        This will set the valid mask for the camera.
        For example, in Aria RGB camera, the mask exclude periphery regions where pixels are too noisy
        """
        if camera_name != self._camera_name:
            raise RuntimeError(
                f"Mask image camera name {camera_name} is inconsistent with current name {self.camera_name}"
            )

        if camera_name in self._mask_image.keys():
            raise Warning(
                "The provided vignette image has already been set! Overwrite existing vignette image path!"
            )

        assert (
            mask_image_path.exists()
        ), f"Could not find vignette for for camera-rgb at {mask_image_path}"
        mask_image = np.array(Image.open(str(mask_image_path))) / 255

        if mask_image.ndim == 2:
            self._mask_image[camera_name] = torch.from_numpy(mask_image)[None]
        else:
            self._mask_image[camera_name] = einops.rearrange(
                torch.from_numpy(mask_image), "h w c -> c h w"
            )

    @property
    def camera_id(self):
        return self._uid

    @property
    def is_rgb(self) -> bool:
        return self.camera_modality == "rgb"

    @property
    def exposure_multiplier(self):
        """
        This aggregates the exposure and analog gain as a single value
        """
        return self._exposure_duration_s * self._gain

    @property
    def exposure_s(self):
        return self._exposure_duration_s

    @property
    def gain(self):
        return self._gain

    def amplify_gain(self, ratio):
        self._gain *= ratio

    @property
    def vignette_image(self):
        if self._camera_name in self._vignette_image.keys():
            return self._vignette_image[self._camera_name]
        else:
            return None

    @property
    def valid_mask(self):
        if self._camera_name in self._mask_image.keys():
            return self._mask_image[self._camera_name]
        else:
            return None

    @property
    def radiance_weight(self):
        return self._radiance_weight

    @radiance_weight.setter
    def radiance_weight(self, v: float):
        self._radiance_weight = v

    @property
    def fov_x(self):
        return self._fov_x

    @property
    def fov_y(self):
        return self._fov_y

    @property
    def fx(self):
        return fov2focal(self.fov_x, self._image_width)

    @property
    def fy(self):
        return fov2focal(self.fov_y, self._image_height)

    @property
    def cx(self):
        return self._cx

    @property
    def cy(self):
        return self._cy

    @property
    def image_width(self):
        return self._image_width

    @property
    def image_height(self):
        return self._image_height

    @image_height.setter
    def image_height(self, value):
        """
        This will only reset the image height while keeping other sensor config the same
        """
        scale_ratio = value / self._image_height

        self._image_width = int(value * self.aspect_ratio)
        self._image_height = value

        self._cx *= scale_ratio
        self._cy *= scale_ratio

    def zoom(self, ratio, keep_width: bool = False, keep_height: bool = False):
        """
        simulate a digital zoom by adjusting the focal
        """
        if not keep_width:
            fx = ratio * self.fx
            fov_x = focal2fov(fx, self._image_width)
        else:
            fov_x = self._fov_x

        if not keep_height:
            fy = ratio * self.fy
            fov_y = focal2fov(fy, self._image_height)
        else:
            fov_y = self._fov_y

        self._set_fov(fov_x, fov_y)

    def reset(self):
        """
        reset the parameters to the initialized camera config
        """
        scale_ratio_x = self._image_height_initial / self._image_height
        scale_ratio_y = self._image_width_initial / self._image_width

        self._image_width = self._image_width_initial
        self._image_height = self._image_height_initial

        self._cx *= scale_ratio_x
        self._cy *= scale_ratio_y

        self._set_fov(self._fov_x_initial, self._fov_y_initial)

    @property
    def intrinsic(self):
        return torch.FloatTensor(
            [
                [self.fx, 0, self._cx],
                [0, self.fy, self._cy],
                [0, 0, 1],
            ]
        )

    @property
    def intrinsic_np(self):
        return np.array(
            [
                [self.fx, 0, self._cx],
                [0, self.fy, self._cy],
                [0, 0, 1],
            ]
        )

    @property
    def c2w_44(self):
        """
        a 4x4 camera to world matrix in torch tensor.
        This is in default in the COLMAP coordinate system (Y down, Z forward)
        """
        return torch.from_numpy(self.c2w_44_np).float()

    @property
    def c2w_44_np(self):
        """
        a 4x4 camera to world matrix in numpy array
        """
        c2w = np.linalg.inv(self.w2c_44_np)
        return c2w

    @property
    def c2w_44_opengl(self):
        """
        a 4x4 camera to world matrix in OpenGL/Blender camera axes

        change from OpenGL/Blender camera axes (Y up, Z back)
        """
        c2w = self.c2w_44
        c2w[:3, 1:3] *= -1
        return c2w

    @property
    def w2c_44(self):
        return torch.from_numpy(self.w2c_44_np).float()

    @property
    def w2c_44_np(self):
        return self._w2c

    @property
    def R_w2c(self):
        return torch.FloatTensor(self._w2c[:3, :3])

    @property
    def T_w2c(self):
        return torch.FloatTensor(self._w2c[:3, 3:])

    @property
    def aspect_ratio(self):
        return float(self._image_width) / self._image_height

    @aspect_ratio.setter
    def aspect_ratio(self, value):
        # existing_ratio = self.aspect_ratio
        # reset fov horizontally
        new_width = int(self._image_height * value)
        scale_ratio_x = new_width / self._image_width

        fov_x = focal2fov(self.fx, new_width)
        self._image_width = new_width
        self._set_fov(fov_x, self._fov_y)

        self._cx *= scale_ratio_x

    @property
    def camera_name(self):
        return self._camera_name

    @property
    def camera_modality(self):
        return self._camera_modality

    @property
    def camera_projection_model(self):
        return self._camera_projection_model

    @property
    def camera_projection_model_gsplat(self):
        return self._camera_model_aria2gsplats[self._camera_projection_model]

    @property
    def is_moving_camera(self):
        return False

    def get_exposure_viewmatrices(self, max_exposure_s: float = 2e-3):
        """
        currently it is only simulating the pose for the center row
        """
        w2c_SE3_exposure_start = SE3.from_matrix(self._w2c)
        w2c_SE3_exposure_end = SE3.from_matrix(self._w2c_exposure_end)

        w2c_array = []
        exposure_step_size = int(self.exposure_s / max_exposure_s)
        for idx in range(0, exposure_step_size + 1):
            w2c_interp = interpolate(
                w2c_SE3_exposure_start, w2c_SE3_exposure_end, idx / exposure_step_size
            )
            w2c_array.append(w2c_interp.to_matrix())
        w2c_array = np.stack(w2c_array)

        return w2c_array

    def expose_image(self, irradiance: torch.Tensor, clamp: bool = True):

        if self.use_nonlinear_irradiance:
            return self.expose_image_gamma(irradiance, clamp)
        else:
            return self.expose_image_linear(irradiance, clamp)

    def expose_image_gamma(self, irradiance: torch.Tensor, clamp: bool = True):
        """
        We assume the scene irradiance is in a gamma space instead of linear space.
        When adding linear operations, we will convert it to a gamma space and then convert it back.
        """
        gamma = 2.2
        mask_linear = irradiance < 0.04045
        gamma_reciprocal = 1.0 / gamma
        irradiance_linear = irradiance[mask_linear] / 12.92
        irradiance_gamma = ((irradiance[~mask_linear] + 0.055) / 1.055) ** gamma

        irradiance_linear *= self.exposure_multiplier * self._radiance_weight
        irradiance_gamma *= self.exposure_multiplier * self._radiance_weight
        # exposed_image = image * self.exposure_multiplier * self._radiance_weight

        if self.vignette_image is not None:
            v = self.vignette_image.expand_as(irradiance).to(irradiance.device)
            irradiance_linear *= v[mask_linear]
            irradiance_gamma *= v[~mask_linear]

        if self.is_rgb:
            # apply gamma compression in the image space
            exposed_image = torch.zeros_like(irradiance)
            irradiance_linear = 12.92 * irradiance_linear
            irradiance_gamma = (
                1.055 * torch.pow(irradiance_gamma, gamma_reciprocal) - 0.055
            )
            exposed_image[mask_linear] = irradiance_linear
            exposed_image[~mask_linear] = irradiance_gamma
        else:
            exposed_image = torch.zeros_like(irradiance)
            exposed_image[mask_linear] = irradiance_linear
            exposed_image[~mask_linear] = irradiance_gamma

        if self.valid_mask is not None:
            exposed_image *= self.valid_mask.to(irradiance.device)

        if clamp:
            return exposed_image.clamp(0, 1).float()
        else:
            return exposed_image.float()

    @property
    def image_name(self):
        if self._image_path is not None:
            return self._image_path.split("/")[-1]
        else:
            return f"{self._timestamp_ns:03f}.png"

    @property
    def image_path(self):
        return self._image_path

    def get_image(self):

        if self._observed_image is not None:
            return self._observed_image, self._alpha_mask

        image = Image.open(self._image_path)
        image_array = PILtoTorch(image)

        # If the image is RGBA image, extract the alpha channel as the mask
        alpha_mask = None
        gt_image = image_array[:3, ...]
        if image_array.shape[0] == 4:
            alpha_mask = image_array[3:4, ...]

        # If there is an additional mask associated with it to further crop the ROI
        if alpha_mask is None and self._mask_path is not None:
            alpha_mask = Image.open(self._mask_path)
            # Mask should already be rectified to correct size
            # If size mismatch, it's likely an error in the pipeline
            if alpha_mask.size != (self.image_width, self.image_height):
                print(f"Warning: Mask size {alpha_mask.size} doesn't match image size ({self.image_width}, {self.image_height})")
                # Fallback to resize as emergency measure
                alpha_mask = alpha_mask.resize((self.image_width, self.image_height), Image.NEAREST)
            alpha_mask = PILtoTorch(alpha_mask).squeeze()

        if self.cache_image:
            self._observed_image = gt_image
            self._alpha_mask = alpha_mask

        return gt_image, alpha_mask

    @property
    def image(self):
        return Image.open(self._image_path)

    @property
    def sparse_depth(self):
        if self._sparse_points is not None:
            return self._sparse_points[:, 2]
        else:
            return None

    @property
    def sparse_inv_depth(self):
        if self._sparse_points is not None:
            depth = self.sparse_depth
            # will suppress that is too close
            invdepth = torch.where(depth > 1e-1, 1.0 / depth, torch.zeros_like(depth))
            return invdepth
        else:
            return None

    @property
    def sparse_point2d(self):
        if self._sparse_points is not None:
            return self._sparse_points[:, :2]
        else:
            return None

    @property
    def sparse_point_inv_distance_std(self):
        """
        Standard deviation of the distance estimate, in meters.
        Could be used for determining the quality of the 3D point position
        """
        if self._sparse_points is not None:
            return self._sparse_points[:, 3]
        else:
            return None

    @property
    def sparse_point_distance_std(self):
        """
        Standard deviation of the inverse distance estimate, in meter^-1.
        Could be used for determining the quality of the 3D point position
        """
        if self._sparse_points is not None:
            return self._sparse_points[:, 4]
        else:
            return None

    def read_sparse_depth(self, json_file: str):
        """
        read sparse depth from the sparse depth json file
        """
        with open(json_file, "r") as f:
            data = json.load(f)

        u = torch.FloatTensor(data["u"])
        v = torch.FloatTensor(data["v"])
        z = torch.FloatTensor(data["z"])
        invd_std = torch.FloatTensor(data["inverseDistanceStd"])
        d_std = torch.FloatTensor(data["distanceStd"])

        if len(u) > 0:
            self._sparse_points = torch.stack([u, v, z, invd_std, d_std], dim=1)
    
    def load_segmentation(self, static_ids=None, dynamic_ids=None):
        """
        Load segmentation data and generate static/dynamic masks (EgoLifter style).
        
        Args:
            static_ids: List of instance IDs that are static
            dynamic_ids: List of instance IDs that are dynamic
            
        Returns:
            Tuple of (static_mask, dynamic_mask, segmentation_data)
        """
        if self.segmentation_path is None:
            return None, None, None
            
        try:
            import pickle
            import gzip
            import numpy as np
            
            # Load segmentation data from pickle file
            with gzip.open(self.segmentation_path, 'rb') as f:
                segmentation_data = pickle.load(f)
            
            # Convert to tensor
            segmentation_data = torch.from_numpy(segmentation_data.astype(np.int64))
            
            # Generate masks if IDs are provided
            static_mask = None
            dynamic_mask = None
            
            if static_ids is not None:
                # Create static mask (use bool type for bitwise operations)
                static_mask = torch.zeros_like(segmentation_data, dtype=torch.bool)
                for instance_id in static_ids:
                    static_mask |= (segmentation_data == instance_id)
                static_mask = static_mask.float().unsqueeze(0)  # Convert to float and add channel dimension
            
            if dynamic_ids is not None:
                # Create dynamic mask (use bool type for bitwise operations)
                dynamic_mask = torch.zeros_like(segmentation_data, dtype=torch.bool)
                for instance_id in dynamic_ids:
                    dynamic_mask |= (segmentation_data == instance_id)
                dynamic_mask = dynamic_mask.float().unsqueeze(0)  # Convert to float and add channel dimension
            
            return static_mask, dynamic_mask, segmentation_data
            
        except Exception as e:
            print(f"Warning: Failed to load segmentation from {self.segmentation_path}: {e}")
            return None, None, None
    
    def load_dense_depth(self):
        """
        Load rectified depth map from depth_maps folder if available.
        Returns depth map and valid mask.
        """
        if self.dense_depth_path is None:
            return None, None
        
        if self.dense_depth_map is not None:
            # Already loaded, return cached version
            return self.dense_depth_map, self.dense_depth_mask
        
        # Load from depth_maps folder
        try:
            import numpy as np
            
            # Construct depth file path based on timestamp
            # Expected format: depth_maps/depth_{timestamp}.npy
            if isinstance(self.dense_depth_path, Path):
                depth_maps_folder = self.dense_depth_path
            else:
                depth_maps_folder = Path(self.dense_depth_path)
            
            depth_file = depth_maps_folder / f"depth_{self._timestamp_ns}.npy"
            
            if depth_file.exists():
                # Load depth map
                depth_map = np.load(depth_file)
                depth_map = torch.from_numpy(depth_map).float()
                
                # Create valid mask (non-zero depth values)
                valid_mask = (depth_map > 0.1) & (depth_map < 100.0)  # Valid range: 0.1-100m
                
                # Cache the loaded depth
                self.dense_depth_map = depth_map
                self.dense_depth_mask = valid_mask
                
                return depth_map, valid_mask
            
        except Exception as e:
            print(f"Failed to load dense depth: {e}")
        
        return None, None

    @property
    def render_image(self):
        return self._est_image

    @render_image.setter
    def render_image(self, value: torch.Tensor):
        self._est_image = value

    @property
    def render_depth(self):
        return self._est_depth

    @render_depth.setter
    def render_depth(self, value: torch.Tensor):
        self._est_depth = value

    @property
    def render_depth_min(self):
        """
        The closest distance of the rendered depth
        """
        return self._est_depth_min

    @render_depth_min.setter
    def render_depth_min(self, value: float):
        self._est_depth_min = value

    @property
    def render_normal(self) -> torch.Tensor:
        return self._est_normal

    @property
    def sample_interval_min(self):
        """
        a sample interval depth / focal, eq(6) defined in mip-splatting
        """
        if self.render_depth is None:
            return 0.0  # we should densely sample as much as possible

        return self.render_depth.min() / max(self.fx, self.fy)

    @render_normal.setter
    def render_normal(self, value: torch.Tensor):
        self._est_normal = value

    def _set_image_size(self, new_width, new_height, keep_fovy=True):
        old_width = self._image_width
        old_height = self._image_height
        self._image_width = new_width
        self._image_height = new_height

        # Adjust the FoV according to the new image size
        if keep_fovy:
            ratio = new_width / new_height / (old_width / old_height)
            self._fov_x = np.arctan(np.tan(self._fov_x / 2) * ratio) * 2
        else:
            ratio = new_height / new_width / (old_height / old_width)
            self._fov_y = np.arctan(np.tan(self._fov_y / 2) * ratio) * 2

        self._set_fov(self._fov_x, self._fov_y)

    def _set_fov(self, FoVx, FoVy):
        self._fov_x = FoVx
        self._fov_y = FoVy

    def reset_extrinsic(self, w2c):
        self._w2c = w2c

    def copy(self):
        new_cam = copy.deepcopy(self)
        return new_cam

    def to_json(self):
        return camera_to_JSON(self._uid, self)

    @property
    def time_s(self):
        return self._timestamp_ns

    def update_time(self, value):
        self._timestamp_ns = value


class AriaCamera(Camera):
    """
    A physical moving camera (for Aria Gen 1 camera)
    """

    _rs_row_index_image: Dict[str, torch.Tensor] = {}
    _rs_row_index_masks: List[SubImageMask] = []

    # the read out time for an image at different resolution
    _readout_time_calib = {
        "full": int(16.26 * 1e6),  # The full resolution for Aria Gen 1 RGB camera 2880x2880
        "half": int(5 * 1e6),  # The half resolution for Aria Gen 1 RGB camera 1408x1408
    }

    # the sampling frequency in time (ns) we use to calculate the potential offset
    _motion_sample_max = 2 * 1e6  

    _max_rolling_shutter_sample = 8   # maximum motion samples for a rolling shutter camera
    _max_exposure_sample = 1  # adjust this value to allow more exposure sample per bracket

    def __init__(
        self,
        uid: int,
        closed_loop_traj: List[mps.ClosedLoopTrajectoryPose],
        camera2device: np.ndarray,
        timestamp_ns: int,
        camera_name: str,
        camera_modality: str,
        is_rolling_shutter: bool,
        FoVx: float,
        FoVy: float,
        image_width: int,
        image_height: int,
        cx: int = -1,
        cy: int = -1,
        image_name: str = None,
        image_path: str = None,
        mask_path: str = None,
        rolling_shutter_index_image_path: Path = None,
        sparse_depth_path: Path = None,
        dense_depth_path: Path = None,  # ADT dense depth map path
        segmentation_path: str = None,  # EgoLifter-style segmentation path
        camera_projection_model: Literal["linear", "spherical"] = "linear",
        exposure_duration_s: float = 1.0,
        gain: float = 1.0,
        radiance_weight: float = 1.0,
        scene_name: str = "",
        cache_image: bool = False,
        readout_time_ns: int = 0,
    ):
        super().__init__(
            uid=uid,
            w2c=None,
            FoVx=FoVx,
            FoVy=FoVy,
            image_width=image_width,
            image_height=image_height,
            cx=cx,
            cy=cy,
            image_name=image_name,
            image_path=image_path,
            mask_path=mask_path,
            camera_name=camera_name,
            camera_projection_model=camera_projection_model,
            camera_modality=camera_modality,
            exposure_duration_s=exposure_duration_s,
            gain=gain,
            radiance_weight=radiance_weight,
            scene_name=scene_name,
            cache_image=cache_image,
        )

        camera2device = SE3.from_matrix(camera2device)
        self._is_rolling_shutter = is_rolling_shutter

        exposure_ns = exposure_duration_s * 1e9
        exp_offset_ns = exposure_ns // 2

        if sparse_depth_path is not None:
            self.read_sparse_depth(sparse_depth_path)
            # we use the sparse point to calculate the possible pixel offset during 1ms, and use this value to compensate the motion
            sampled_pixel_offset = self._estimate_pixel_motion_offset(
                closed_loop_traj,
                timestamp_ns,
                exp_offset_ns=self._motion_sample_max // 2,
            )
        else:
            # we will use an imaginary value if this does not exist!
            sampled_pixel_offset = 1.0
        
        # Store dense depth path for later loading
        self.dense_depth_path = dense_depth_path
        self.dense_depth_map = None
        self.dense_depth_mask = None
        
        # Store segmentation path for later loading (EgoLifter style)
        self.segmentation_path = segmentation_path
        self.segmentation_data = None

        if camera_name in ["camera-rgb", "camera_rgb"]:  # rolling shutter camera
            assert is_rolling_shutter, f"{camera_name} should be a rolling shutter camera!"

            readout_time = readout_time_ns

            # We choose the quantization range, we expect the maximum pixel offset is within 1 pixel if possible.
            # We use a maximum of 16 samples.
            rs_row_quantization_range = int(
                min(
                    max(
                        readout_time / self._motion_sample_max * sampled_pixel_offset, 1
                    ),
                    self._max_rolling_shutter_sample,
                )
            )
            self.rs_row_quantization_range = rs_row_quantization_range

            if (
                rs_row_quantization_range > 1
            ):  # is rolling shutter camera and require multiple samples across time
                print(
                    f"There are {sampled_pixel_offset:02f} pixels within {self._motion_sample_max/1e6} ms. Will take rs step: {rs_row_quantization_range}"
                )

            rs_step_ns = readout_time / rs_row_quantization_range
            # sampled rolling_shutter time
            if rs_row_quantization_range > 1:
                # We will use the center rolling-shutter time within each rolling-shutter bracket
                rs_read_start_ns = timestamp_ns - readout_time // 2
                rs_bracket_timestamps = [
                    rs_read_start_ns + idx * rs_step_ns
                    for idx in range(rs_row_quantization_range + 1)
                ]
                rs_sampled_timestamps = [
                    (rs_bracket_timestamps[i] + rs_bracket_timestamps[i + 1]) / 2
                    for i in range(rs_row_quantization_range)
                ]
            else:  # this indicate the frame is static!
                rs_sampled_timestamps = [timestamp_ns]

            if self._camera_name not in self._rs_row_index_image.keys():
                self._rs_row_index_image[self._camera_name] = (
                    np.array(Image.open(str(rolling_shutter_index_image_path))) / 255.0
                )

            # we will use center row timestamp to approximate
            self._timestamp_ns = timestamp_ns

        else:  # global shutter camera
            rs_sampled_timestamps = [timestamp_ns]
            self._timestamp_ns = timestamp_ns
            self.rs_row_quantization_range = 1

        # sampled motion blur time for each camera. 
        # Note: this is not currently used in default.
        exposure_step_range = int(
            min(
                max(
                    round(exposure_ns / self._motion_sample_max * sampled_pixel_offset),
                    1,
                ),
                self._max_exposure_sample,
            )
        )
        if exposure_step_range > 1:
            # Require multiple samples within the exposure range
            # We will sample the mid-exposure time within a evenly divided exposure bracket.
            # This allows us to average the exposed images at the given exposure time and integrate them using the full exposure time value.
            exposure_step_ns = exposure_ns / exposure_step_range
            sampled_timestamps = []
            for t in rs_sampled_timestamps:
                t_exp_start = t - exp_offset_ns
                exposure_brackets_timestamps = [
                    t_exp_start + idx * exposure_step_ns
                    for idx in range(exposure_step_range + 1)
                ]
                exposure_sample_timestamps = [
                    (
                        exposure_brackets_timestamps[i]
                        + exposure_brackets_timestamps[i + 1]
                    )
                    / 2.0
                    for i in range(exposure_step_range)
                ]
                sampled_timestamps.append(exposure_sample_timestamps)
            print(f"camera {uid} has multiple exposure samples {exposure_step_range}")
        else:
            sampled_timestamps = [[t] for t in rs_sampled_timestamps]

        # generate an array of poses (M N), where M is the samples of the exposure bracket, and N is the rolling-shutter bracket
        self.sampled_timestamps = np.array(sampled_timestamps)

        # sampled the corresponding poses for the corresponding timestamp
        self._w2c_array = self.sample_viewmatrices(
            closed_loop_traj, self.sampled_timestamps, camera2device
        )

        # store the center view transform pose. Used if not using the pose array
        center_row_pose_info = interpolate_aria_pose(closed_loop_traj, timestamp_ns)
        c2w_SE3 = center_row_pose_info.transform_world_device @ camera2device
        w2c_SE3 = c2w_SE3.inverse()
        self._w2c_center = w2c_SE3.to_matrix()

    def sample_viewmatrices(
        self,
        closed_loop_traj: List[mps.ClosedLoopTrajectoryPose],
        sampled_timestamps_ns: np.ndarray,
        camera2device,
    ):
        """
        Will return a set of viewmatrix correspond to the sampled motion timestamps.

        This is currently assuming exposure time is within limit.
        """
        timestamps = sampled_timestamps_ns.flatten()

        w2c_array = []
        for timestamp in timestamps:
            pose_info = interpolate_aria_pose(closed_loop_traj, timestamp)

            # if pose_info.quality_score < 0.9:
            #     print(f"pose quality score below 1.0: {pose_info.quality_score}!")

            c2w_SE3 = pose_info.transform_world_device @ camera2device
            w2c_SE3 = c2w_SE3.inverse()
            w2c_array.append(w2c_SE3.to_matrix())

        w2c_array = np.stack(w2c_array)

        N_rs, N_exp = sampled_timestamps_ns.shape
        return w2c_array.reshape(N_rs, N_exp, 4, 4)

    def _estimate_pixel_motion_offset(
        self,
        closed_loop_traj: List[mps.ClosedLoopTrajectoryPose],
        timestamp: int,
        exp_offset_ns: int = 1e6,
        threshold_percentile: float = 50,
    ):
        """
        Calculate the moved pixel range given a time-window (2 * exp_offset_ns).

        threshold_percentile: the thresholding percentile to determinte the projection offset. 
        """

        uv = self.sparse_point2d
        if uv is None:
            return 1e-3  # we almost cannot treat it as moving camera

        u = uv[:, 0]
        v = uv[:, 1]
        z = self.sparse_depth

        fx, fy, cx, cy = self.fx, self.fy, self._cx, self._cy
        x = (u - cx) / fx * z
        y = (v - cy) / fy * z

        t_start = timestamp - exp_offset_ns
        t_end = timestamp + exp_offset_ns
        pose_start = interpolate_aria_pose(closed_loop_traj, t_start)
        pose_end = interpolate_aria_pose(closed_loop_traj, t_end)

        w2c_rel = (
            pose_end.transform_world_device.inverse()
            @ pose_start.transform_world_device
        )

        pt3d = np.stack([x, y, z])

        u_proj, v_proj, z_proj, mask = project(
            pt3d,
            T_w2c=w2c_rel.to_matrix(),
            calibK=self.intrinsic_np,
            frame_h=self.image_height,
            frame_w=self.image_width,
        )

        # calculate the follow vector
        proj_offset = np.stack(
            [(u[mask] - u_proj).abs(), (v[mask] - v_proj).abs()]
        ).max(axis=0)
        # exposure_flow_det = abs(u[mask] - u_proj)  abs(v[mask] - v_proj)

        # print(f"timestamp: {timestamp_start}")
        # print(f"maximum pixel offset during exposure time: {exposure_flow_det.max()}")
        # print(f"75 percentile offset during exposure time: {np.percentile(exposure_flow_det, 75)}")
        # print(f"50 percentile offset during exposure time: {np.percentile(exposure_flow_det, 50)}")
        return np.percentile(proj_offset, threshold_percentile)

    @property
    def is_moving_camera(self):
        return self._w2c_array.shape[0] > 1 or self._w2c_array.shape[1] > 1

    @property
    def w2c_44_np(self):
        """
        choose the center pixel poses among all
        """
        return self._w2c_center

    def reset_extrinsic(self, w2c):
        self._w2c_center = w2c

    @property
    def motion_w2c_array_tensor(self):
        return torch.from_numpy(self._w2c_array).float()

    @property
    def is_rolling_shutter(self):
        return self._is_rolling_shutter

    @property
    def rolling_shutter_index_image(self):
        return torch.from_numpy(
            np.round(
                self._rs_row_index_image[self._camera_name]
                * (self.rs_row_quantization_range - 1)
            )
        ).long()

    @property
    def time_s(self) -> float:
        return self._timestamp_ns / 1e9

    def update_time(self, value):
        self._timestamp_ns = value * 1e9


def to_quat(rot_3x3):
    try:
        quat = pyquaternion.Quaternion(matrix=rot_3x3)
    except:
        U, _, Vt = np.linalg.svd(rot_3x3)
        rot_orthognal = np.dot(U, Vt)
        quat = pyquaternion.Quaternion(matrix=rot_orthognal)
    return quat


def interpolate_piecewise(cam1: Camera, cam2: Camera, w: float):
    """
    Generate an interpolated camera given two existing cameras.
    It will interpolate properties between the two
    """
    cam_interp = copy.deepcopy(cam1)

    # interpolate using c2w
    c2w_src = cam1.c2w_44_np
    c2w_dst = cam2.c2w_44_np

    q_src = to_quat(c2w_src[:3, :3])
    q_dst = to_quat(c2w_dst[:3, :3])

    q = pyquaternion.Quaternion.slerp(q_src, q_dst, w)

    c2w_interp = np.eye(4)
    c2w_interp[:3, :3] = q.rotation_matrix
    c2w_interp[:3, 3] = (1 - w) * c2w_src[:3, 3] + w * c2w_dst[:3, 3]

    w2c_interp = np.linalg.inv(c2w_interp)

    cam_interp.reset_extrinsic(w2c=w2c_interp)
    return cam_interp


def interpolate_fps_piecewise(cameras: List[Camera], fps: int):
    """
    Interpolate the cameras using a piecewise function of provided cameras.
    """
    assert len(cameras) > 1, "Require >=2 cameras to generate interpolated trajectory"

    last_cam = cameras[0]
    interp_cameras = []

    # we assume generating 60 fps frames between every two sampled camera poses.
    sample_rate = 1.0 / fps * (cameras[1].time_s - cameras[0].time_s)

    print("Generate interpolated trajectories for rendering...")
    for idx in tqdm(range(1, len(cameras))):

        cam = cameras[idx]
        interp_cameras.append(copy.copy(last_cam))

        # skip the first sample which we have added
        samples_t = np.arange(
            last_cam.time_s + sample_rate, cam.time_s, sample_rate
        ).tolist()

        for _, ts in enumerate(samples_t):

            t_normalized = (ts - last_cam.time_s) / (cam.time_s - last_cam.time_s)

            # generate interpolated pose in between
            cam_interp = interpolate_piecewise(last_cam, cam, t_normalized)

            cam_interp.update_time(ts)

            interp_cameras.append(cam_interp)

        last_cam = cam

    return interp_cameras


class CameraDataset(Dataset):

    def __init__(
        self,
        cameras: List[Camera],
        name: str = "train",
        render_only: bool = False,
    ):
        self.cameras = cameras
        self.name = name
        self.render_only = render_only

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        cam = self.cameras[idx]

        if self.render_only:
            return {"idx": idx, "image_id": cam.camera_id, "subset": self.name}
        else:
            image, mask = cam.get_image()

            output = {
                "idx": idx,
                "subset": self.name,
                "scene_name": cam.scene_name,
                "image_id": cam.camera_id,
                "image_name": cam.image_name,
                "camera_model": cam.camera_projection_model_gsplat,
                "image": image,
                "K": cam.intrinsic,
            }

            # combine the exposure estimate into a per-pixel irradiance multiplier in the screenspace.
            # if a pixel is not visible, the corresponding irradiance will be zero.
            irradiance_multiplier = cam.exposure_multiplier * cam.radiance_weight
            if cam.vignette_image is not None:
                irradiance_multiplier *= cam.vignette_image
            if cam.valid_mask is not None:
                irradiance_multiplier *= cam.valid_mask
                if mask is not None:

                    if mask.ndim == 2:
                        mask = mask[None]

                    mask = mask * cam.valid_mask

            output["irradiance_multiplier"] = irradiance_multiplier

            # if sparse depth is available
            if cam.sparse_depth is not None:
                assert (
                    cam.sparse_point2d is not None
                ), "require to load 2d positions for 3d depth"
                assert (
                    cam.sparse_inv_depth is not None
                ), "require to load the inverse depth"
                output["sparse_depth"] = cam.sparse_depth
                output["sparse_point2d"] = cam.sparse_point2d
                output["sparse_inv_depth"] = cam.sparse_inv_depth
                output["sparse_inv_distance_std"] = cam.sparse_point_inv_distance_std
            
            # Load dense depth if available (for AriaCamera with ADT data)
            if hasattr(cam, 'load_dense_depth'):
                dense_depth, dense_depth_mask = cam.load_dense_depth()
                if dense_depth is not None:
                    output["dense_depth"] = dense_depth
                    if dense_depth_mask is not None:
                        output["dense_depth_mask"] = dense_depth_mask

            if mask is not None:
                output["mask"] = mask
            
            # Load segmentation and generate static/dynamic masks (EgoLifter style)
            if hasattr(cam, 'segmentation_path') and cam.segmentation_path is not None:
                # Try to load instance info if available
                instance_info_path = Path(cam.segmentation_path).parent.parent / "instance_info.json"
                static_ids = None
                dynamic_ids = None
                
                if instance_info_path.exists():
                    import json
                    # Try to load refined instance info first (from intelligent motion detection)
                    refined_path = instance_info_path.parent / "instance_info_refined.json"
                    if refined_path.exists():
                        with open(refined_path, 'r') as f:
                            instance_info = json.load(f)
                            static_ids = instance_info.get("static_ids", None)
                            dynamic_ids = instance_info.get("dynamic_ids", None)
                            # Log that we're using refined detection
                            if hasattr(instance_info, 'metadata'):
                                falsely_dynamic = instance_info['metadata'].get('falsely_dynamic_count', 0)
                                if falsely_dynamic > 0:
                                    print(f"Using refined mask: {falsely_dynamic} objects reclassified as static")
                    else:
                        # Fall back to original instance info
                        with open(instance_info_path, 'r') as f:
                            instance_info = json.load(f)
                            static_ids = instance_info.get("static_ids", None)
                            dynamic_ids = instance_info.get("dynamic_ids", None)
                
                # Load segmentation and generate masks
                static_mask, dynamic_mask, seg_data = cam.load_segmentation(static_ids, dynamic_ids)
                
                if static_mask is not None:
                    output["static_mask"] = static_mask
                if dynamic_mask is not None:
                    output["dynamic_mask"] = dynamic_mask
                if seg_data is not None:
                    output["segmentation"] = seg_data

        return output


@dataclass
class CameraPropertyBasicRGB:
    camera_name: str = "camera-rgb"
    apply_tonemapping: bool = False


@dataclass
class CameraPropertyAriaRGB:
    """
    Camera property for Aria RGB camera
    """

    camera_name: str = "camera-rgb"
    vignette_image: torch.Tensor = None
    mask_image: torch.Tensor = None
    apply_tonemapping: bool = False
    gamma4sRGB: float = 2.2
    exposure_gain_ratio_ref: float = 4e-2
    shutter_type: str = "rollingshutter"


@dataclass
class CameraPropertyAriaSLAM:
    """
    Camera property for Aria SLAM camera
    """

    camera_name: str = "camera-slam"
    vignette_image: torch.Tensor = None
    apply_tonemapping: bool = False
    gamma4sRGB: float = 2.0
    exposure_gain_ratio_ref: float = 4e-2
    shutter_type: str = "globalshutter"


def camera_to_JSON(id, camera: Camera):
    camera_entry = {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.image_width,
        "height": camera.image_height,
        "c2w": camera.c2w_44.tolist(),
        "fov_x": camera.fov_x,
        "fov_y": camera.fov_y,
        "fx": camera.fx,
        "fy": camera.fy,
        "exposure_multiplier": camera.exposure_multiplier,
        "radiance_weight": camera.radiance_weight,
        "camera_name": camera.camera_name,
        "camera_modality": camera.camera_modality,
    }
    return camera_entry


def create_camera_from_JSON(data):

    c2w = np.array(data["c2w"])
    w2c = np.linalg.inv(c2w)
    camera = Camera(
        uid=data["id"],
        w2c=w2c,
        FoVx=data["fov_x"],
        FoVy=data["fov_y"],
        image_width=data["width"],
        image_height=data["height"],
        image_name=data["img_name"],
        camera_name=data["camera_name"],
        camera_modality=data["camera_modality"],
        exposure_duration_s=data["exposure_multiplier"],
        gain=1.0,
        radiance_weight=data["radiance_weight"],
    )

    return camera
