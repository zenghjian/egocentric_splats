# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import projectaria_tools.core.mps as mps
from scene.cameras import (
    AriaCamera,
    Camera,
    focal2fov,
    interpolate_aria_pose,
    interpolate_fps_piecewise,
)

from utils.point_utils import BasicPointCloud, project


def get_scene_info(scene_cfg):

    source_path = Path(scene_cfg.source_path)

    assert source_path.exists, f"Source path {source_path} does not exist!"

    if scene_cfg.input_format == "aria":
        scene_info = readAriaSceneInfo(input_folder=source_path, scene_cfg=scene_cfg)
    else:
        raise RuntimeError("cannot recognize the input format!!!")

    return scene_info


class SceneType(Enum):
    COLMAP = 1
    ARIA = 2


@dataclass
class SceneInfo:
    point_cloud: Optional[BasicPointCloud] = None
    point_source_path: Optional[str] = None
    scene_type: Optional[SceneType] = None
    all_cameras: Optional[SceneType] = None
    train_cameras: Optional[list] = None
    valid_cameras: Optional[list] = None
    test_cameras: Optional[list] = None
    scene_scale: Optional[int] = 1.0
    camera_labels: Optional[set] = None
    overwrite_sh_degree: Optional[int] = None

    @property
    def subset_to_cameras(self):
        return {
            "train": self.train_cameras,
            "valid": self.valid_cameras,
            "test": self.test_cameras,
        }


def estimate_scene_camera_scale(cam_infos):
    """
    Estimate the scale of the scene according to the camera distributions
    """
    c2ws = np.stack([cam.c2w_44_np for cam in cam_infos])

    camera_locations = c2ws[:, :3, 3]
    scene_center = np.mean(camera_locations, axis=0)
    dists = np.linalg.norm(camera_locations - scene_center, axis=1)
    scene_scale = np.max(dists)
    return scene_scale


def pinhole_camera_rectify(
    image: np.array, opencv_distortion_params: np.array, downsample: int = 1
):

    assert (
        len(opencv_distortion_params) == 8
    ), "currently only consumes distortion model with 4 radial distortions"

    fx, fy, cx, cy, k1, k2, k3, k4 = opencv_distortion_params
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist_coeff = np.array([k1, k2, k3, k4])

    image_height, image_width = image.shape[:2]

    K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
        K, dist_coeff, (image_width, image_height), 0
    )
    mapx, mapy = cv2.initUndistortRectifyMap(
        K, dist_coeff, None, K_undist, (image_width, image_height), cv2.CV_32FC1
    )

    full_image_mask = np.ones((image_height, image_width))
    image_undistorted = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
    mask_undistorted = cv2.remap(full_image_mask, mapx, mapy, cv2.INTER_LINEAR) * 255

    image_undistorted_rgba = np.concatenate(
        [image_undistorted, mask_undistorted[..., None]], axis=-1
    )

    return image_undistorted_rgba.astype(np.uint8)


def equidistant_camera_rectify(
    image: np.array, opencv_distortion_params: np.array, downsample: int = 1
):
    """
    Undistort opencv fisheye images to a equidistant fisheye projection
    https://github.com/zmliao/Fisheye-GS/blob/697cd86359efaae853a52f0bef9758b700390dc7/prepare_scannetpp.py#L58
    """
    assert (
        len(opencv_distortion_params) == 8
    ), "currently only consumes distortion model with 4 radial distortions"

    fx, fy, cx, cy, k1, k2, k3, k4 = opencv_distortion_params
    fx = fx // downsample
    fy = fy // downsample
    cx = cx // downsample
    cy = cy // downsample

    H, W = image.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32),
        indexing="xy",
    )
    x1 = (grid_x - cx) / fx
    y1 = (grid_y - cy) / fy
    theta = np.sqrt(x1**2 + y1**2)
    # theta = np.arctan(radius)
    r = 1.0 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8
    mapx = fx * x1 * r + cx
    mapy = fy * y1 * r + cy

    full_image_mask = np.ones((H, W))

    image_undistorted = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
    mask_undistorted = cv2.remap(full_image_mask, mapx, mapy, cv2.INTER_LINEAR) * 255

    image_undistorted_rgba = np.concatenate(
        [image_undistorted, mask_undistorted[..., None]], axis=-1
    )

    return image_undistorted_rgba.astype(np.uint8)


def visualize_cameras_aria(
    cameras: List[Camera],
    point_cloud: BasicPointCloud,
    close_loop_traj,
    readout_time=16e6,
):
    """
    visualize points and cameras using a rerun viewer
    """
    import rerun as rr

    rr.init(f"Visualize the cameras", spawn=True)

    points = point_cloud.points
    colors = point_cloud.colors
    scale = 1000
    rr.log(
        f"world/points_3D",
        rr.Points3D(points * scale, colors=colors, radii=0.005 * scale),
        timeless=True,
    )

    for frame_idx, camera in enumerate(cameras):

        w, h = camera.image_width, camera.image_height
        c2w = camera.c2w_44

        # mask = (u_proj > 0) & (u_proj < w) & (v_proj > 0) & (v_proj < h) & (z > 0)
        points2d = (
            camera.sparse_point2d.numpy()
        )  # np.stack([u_proj[mask], v_proj[mask]]).T

        if points2d is None:
            return 1e-3  # we almost cannot treat it as moving camera

        u = points2d[:, 0]
        v = points2d[:, 1]
        z = camera.sparse_depth.numpy()

        fx, fy, cx, cy = camera.fx, camera.fy, camera.cx, camera.cy
        x = (u - cx) / fx * z
        y = (v - cy) / fy * z

        t_start = camera._timestamp_ns - readout_time / 2.0
        t_end = camera._timestamp_ns + readout_time / 2.0
        pose_start = interpolate_aria_pose(close_loop_traj, t_start)
        pose_end = interpolate_aria_pose(close_loop_traj, t_end)

        w2c_rel = (
            pose_end.transform_world_device.inverse()
            @ pose_start.transform_world_device
        )

        pt3d = np.stack([x, y, z])

        u_proj, v_proj, z_proj, mask = project(
            pt3d,
            T_w2c=w2c_rel.to_matrix(),
            calibK=camera.intrinsic_np,
            frame_h=camera.image_height,
            frame_w=camera.image_width,
        )

        reproj_origin = points2d[mask]  # np.stack((u_proj, v_proj), axis=1)
        reproj_vector2d = np.stack([u_proj - u[mask], v_proj - v[mask]], axis=1)

        proj_offset = np.absolute(reproj_vector2d).max(axis=1)

        rr.set_time_sequence("frame_idx", frame_idx)

        rr.log(
            f"world/device/rgb",
            rr.Pinhole(resolution=[w, h], focal_length=camera.fx, fov_y=camera.fov_y),
        )

        image = camera.image
        rr.log(
            f"world/device/rgb/image",
            rr.Image(image).compress(jpeg_quality=75),
        )

        rr.log(
            f"world/device",
            rr.Transform3D(
                translation=c2w[:3, 3] * scale,
                mat3x3=c2w[:3, :3],
            ),
        )

        rr.log(
            f"world/device/rgb/points_2D",
            rr.Points2D(points2d.astype(np.int32), colors=[0, 200, 0], radii=2),
        )
        rr.log(
            f"world/device/rgb/arrows",
            rr.Arrows2D(
                origins=reproj_origin,
                vectors=reproj_vector2d,
                colors=[255, 0, 0],
                radii=3,
            ),
        )
        rr.log(
            f"world/device/reprojection_75_percentile",
            rr.Scalar(np.percentile(proj_offset, 75)),
        )
        rr.log(
            f"world/device/reprojection_50_percentile",
            rr.Scalar(np.percentile(proj_offset, 50)),
        )
        rr.log(
            f"world/device/reprojection_25_percentile",
            rr.Scalar(np.percentile(proj_offset, 25)),
        )


def visualize_cameras(cameras: List[Camera], point_cloud: BasicPointCloud):
    """
    visualize points projected to the cameras
    """
    import rerun as rr

    rr.init(f"Visualize the cameras", spawn=True)

    points = point_cloud.points
    colors = point_cloud.colors
    if len(points) > 5e6:
        points_subsampled = points[::5]
    else:
        points_subsampled = points
    scale = 1000
    rr.log(
        f"world/points_3D",
        rr.Points3D(points * scale, colors=colors, radii=0.005 * scale),
        timeless=True,
    )

    for frame_idx, camera in enumerate(cameras):

        w, h = camera.image_width, camera.image_height
        c2w = camera.c2w_44

        calibK = camera.intrinsic
        w2c = np.linalg.inv(c2w)

        point3d_cam = w2c[:3, :3] @ points_subsampled.T + w2c[:3, 3:4]
        point3d_proj = calibK @ point3d_cam

        u_proj = point3d_proj[0] / point3d_proj[2]
        v_proj = point3d_proj[1] / point3d_proj[2]
        z = point3d_proj[2]

        mask = (u_proj > 0) & (u_proj < w) & (v_proj > 0) & (v_proj < h) & (z > 0)
        points2d = np.stack([u_proj[mask], v_proj[mask]]).T

        rr.set_time_sequence("frame_idx", frame_idx)

        rr.log(
            f"world/device/rgb",
            rr.Pinhole(resolution=[w, h], focal_length=camera.fx, fov_y=camera.fov_y),
        )

        image = camera.image
        rr.log(
            f"world/device/rgb/image",
            rr.Image(image).compress(jpeg_quality=75),
        )

        rr.log(
            f"world/device",
            rr.Transform3D(
                translation=c2w[:3, 3] * scale,
                mat3x3=c2w[:3, :3],
            ),
        )

        rr.log(
            f"world/device/rgb/points_2D",
            rr.Points2D(points2d.astype(np.int32), colors=[0, 200, 0], radii=2),
        )


def read_Aria_transform_json(
    transform_paths: List[Path],
    high_freq_trajectory: List[mps.ClosedLoopTrajectoryPose],
    input_folder: Path = None,
    scene_name: str = "none",
    start_timestamp_ns: int = -1,
    end_timestamp_ns: int = -1,
    read_vignette: bool = True,
    read_mask: bool = True,
    sample_interval_s: float = 0,
) -> List[Camera]:
    frames = []
    for transform_path in transform_paths:
        with open(transform_path) as json_file:
            transforms = json.loads(json_file.read())
        frames += transforms["frames"]

    camera_names = set()

    time_start_with_pose = (
        high_freq_trajectory[0].tracking_timestamp.total_seconds() * 1e9
    )
    time_end_with_pose = (
        high_freq_trajectory[-1].tracking_timestamp.total_seconds() * 1e9
    )

    # This sorts the frame list by first camera_name, then by capture time.
    frames.sort(key=lambda f: f["image_path"])

    all_cam_list = []
    last_sampled_timestamp_ns = 0
    for idx, frame in enumerate(frames):

        if input_folder is None:
            image_path_full = None
            image_name = None
        else:
            image_path_full = str(input_folder / frame["image_path"])
            image_name = image_path_full.split("/")[-1].split(".")[0]

            if not os.path.exists(image_path_full):
                print(f"{image_path_full} does not exist. Will skip!")
                continue

        # The center row timestamp for the camera. The 1e3 is a temporary fix for arcata camera
        timestamp_center = frame["timestamp"]

        if start_timestamp_ns > 0 and timestamp_center < start_timestamp_ns:
            # print(f"skip frames that before time {start_timestamp_ns}")
            continue

        if timestamp_center < time_start_with_pose + 1e6:  # skip first 100 ms
            print(
                f"skip frames that before time {start_timestamp_ns} in trajectory with valid pose"
            )
            continue

        if end_timestamp_ns > 0 and timestamp_center >= end_timestamp_ns:
            # print(f"skip frame that after time {end_timestamp_ns}")
            continue

        if timestamp_center > time_end_with_pose - 1e6:  # skip last 100 ms
            print(
                f"skip frames that after time {time_end_with_pose} in trajectory with valid pose"
            )
            continue

        if transforms["camera_label"].startswith("camera-slam"):
            camera_modality = "monochrome"
            camera_names.update(["camera-slam"])
            is_rolling_shutter = False
            rolling_shutter_index_image_path = None
        elif transforms["camera_label"].startswith("camera-rgb"):
            camera_modality = "rgb"
            camera_names.update(["camera-rgb"])
            is_rolling_shutter = True
            rolling_shutter_index_image_path = input_folder / "image_index.png"
        elif transforms["camera_label"].startswith(
            "camera_rgb"
        ):  # used for quest camera only
            camera_modality = "rgb"
            camera_names.update((["camera-rgb"]))
            is_rolling_shutter = True
            rolling_shutter_index_image_path = input_folder / "image_index.png"
        else:
            raise NotImplementedError(
                f"Unrecognized cameras labels {transforms['camera_label']}"
            )

        # Check for mask_path in frame
        if (
            "mask_path" in frame.keys()
            and frame["mask_path"] != ""
            and input_folder is not None
        ):
            mask_path_full = input_folder / frame["mask_path"]
            assert mask_path_full.exists(), "mask file in the transform does not exist!"
            mask_path_full = str(mask_path_full)
        else:
            mask_path_full = None
        
        # Check for segmentation data (EgoLifter style - in masks folder)
        segmentation_path = None
        masks_folder = input_folder / "masks"
        if masks_folder.exists():
            # Get timestamp from frame
            timestamp_ns = frame.get("timestamp", idx)
            seg_filename = f"seg_{idx:06d}.pkl.gz"
            seg_path = masks_folder / seg_filename
            if seg_path.exists():
                segmentation_path = str(seg_path)
                print(f"Found segmentation for frame {idx}: {seg_filename}") if idx == 0 else None

        fx, fy, cx, cy = frame["fx"], frame["fy"], frame["cx"], frame["cy"]
        width, height = frame["w"], frame["h"]
        fovx = focal2fov(fx, width)
        fovy = focal2fov(fy, height)

        camera2device = np.asarray(frame["camera2device"])

        if "sparse_depth" in frame.keys():
            sparse_depth_path = input_folder / frame["sparse_depth"]
        else:
            sparse_depth_path = None
        
        # Check if depth_maps folder exists for dense depth
        depth_maps_folder = input_folder / "depth_maps"
        dense_depth_path = depth_maps_folder if depth_maps_folder.exists() else None

        cam = AriaCamera(
            uid=idx,
            closed_loop_traj=high_freq_trajectory,
            camera2device=camera2device,
            camera_name=transforms["camera_label"],
            camera_modality=camera_modality,
            is_rolling_shutter=is_rolling_shutter,
            rolling_shutter_index_image_path=rolling_shutter_index_image_path,
            timestamp_ns=timestamp_center,
            FoVx=fovx,
            FoVy=fovy,
            image_width=width,
            image_height=height,
            cx=cx,
            cy=cy,
            image_name=image_name,
            image_path=image_path_full,
            mask_path=mask_path_full,
            sparse_depth_path=sparse_depth_path,
            dense_depth_path=dense_depth_path,
            segmentation_path=segmentation_path,  # Add segmentation path
            camera_projection_model=transforms["camera_model"],
            exposure_duration_s=frame["exposure_duration_s"],
            gain=frame["gain"],
            scene_name=scene_name,
            readout_time_ns=frame["timestamp_read_end"] - frame["timestamp_read_start"],
        )

        if cam.vignette_image is None and read_vignette:
            cam.set_vignette_image(
                vignette_image_path=input_folder / "vignette.png",
                camera_name=transforms["camera_label"],
            )
        if (
            transforms["camera_label"] == "camera-rgb"
            and cam.valid_mask is None
            and read_mask
        ):
            cam.set_valid_mask(
                mask_image_path=input_folder / "mask.png",
                camera_name=transforms["camera_label"],
            )

        all_cam_list.append(cam)

        last_sampled_timestamp_ns = frame["timestamp"]

    print(
        f"Found {len(all_cam_list)} cameras given criterion among {len(frames)} frames"
    )

    # calculate the average the exposure & gain ratio
    exp_all = np.array([cam.exposure_multiplier for cam in all_cam_list])
    exp_median = np.median(exp_all)

    for cam in all_cam_list:
        cam.radiance_weight = 1.0 / exp_median

    return all_cam_list, camera_names


def read_render_transform_json(
    transform: dict,
) -> List[Camera]:
    """
    read it from the transform file generated from interactive visualize tools.
    """
    camera_names = set()

    all_cam_list = []
    for idx, frame in enumerate(transform):
        width = frame["width"]
        height = frame["height"]
        camera_names.update([frame["camera_name"]])

        c2w = np.asarray(frame["c2w"]).reshape(4, 4)
        w2c = np.linalg.inv(c2w)

        cam = Camera(
            uid=frame["id"],
            w2c=w2c,
            FoVx=frame["fov_x"],
            FoVy=frame["fov_y"],
            image_width=width,
            image_height=height,
            camera_name=frame["camera_name"],
            exposure_duration_s=frame["exposure_multiplier"],
            radiance_weight=frame["radiance_weight"],
            gain=1.0,
        )

        all_cam_list.append(cam)

    return all_cam_list, camera_names


def readAriaSceneInfo(
    input_folder: Path,
    scene_cfg,
    visualize: bool = False,
):
    scene_name = scene_cfg.scene_name
    data_format = scene_cfg.data_format
    train_split = scene_cfg.train_split
    start_timestamp_ns = scene_cfg.start_timestamp_ns
    end_timestamp_ns = scene_cfg.end_timestamp_ns

    trajectory_file = input_folder / "closed_loop_trajectory.csv"
    closed_loop_traj = mps.read_closed_loop_trajectory(str(trajectory_file))

    # Go through the transforms and create the camera infos
    if train_split == "fixed":
        train_transform_paths = glob.glob(str(input_folder / "transforms_train.json"))
        train_cam_list, camera_names = read_Aria_transform_json(
            transform_paths=train_transform_paths,
            high_freq_trajectory=closed_loop_traj,
            input_folder=input_folder,
            scene_name=scene_name,
            start_timestamp_ns=start_timestamp_ns,
            end_timestamp_ns=end_timestamp_ns,
        )

        transform_valid_paths = glob.glob(str(input_folder / "transforms_valid.json"))
        assert len(transform_valid_paths) > 0, "No transform_valid.json found"
        valid_cam_list, valid_camera_names = read_Aria_transform_json(
            transform_paths=transform_valid_paths,
            high_freq_trajectory=closed_loop_traj,
            input_folder=input_folder,
            scene_name=scene_name,
        )
        camera_names.update(valid_camera_names)
    else:
        transform_paths = glob.glob(str(input_folder / data_format))
        all_cam_list, camera_names = read_Aria_transform_json(
            transform_paths=transform_paths,
            high_freq_trajectory=closed_loop_traj,
            input_folder=input_folder,
            scene_name=scene_name,
            start_timestamp_ns=start_timestamp_ns,
            end_timestamp_ns=end_timestamp_ns,
        )

    # Read the test set cameras if there are available
    test_view_folder = input_folder / "test_views"
    if test_view_folder.exists():
        test_json_paths = glob.glob(str(test_view_folder / "**/transforms_test.json"))
        print(f"Read test cameras from transforms_test.json within {test_view_folder}")

        test_cam_list, test_camera_names = [], set()
        for test_json_path in test_json_paths:
            test_input_folder = Path(test_json_path).parent
            test_view_closed_loop_traj_path = (
                test_input_folder / "closed_loop_trajectory.csv"
            )
            assert (
                test_view_closed_loop_traj_path.exists()
            ), f"cannot find test view closed loop trajectory! {test_view_closed_loop_traj_path}"

            test_closed_loop_traj = mps.read_closed_loop_trajectory(
                str(test_view_closed_loop_traj_path)
            )
            test_cam, test_camera_name = read_Aria_transform_json(
                transform_paths=[test_json_path],
                high_freq_trajectory=test_closed_loop_traj,
                input_folder=test_input_folder,
                scene_name=scene_name,
            )
            test_cam_list += test_cam
            test_camera_names.update(test_camera_name)

        # rest the test camera radiance weight to be consistent with train views

        for cam in test_cam_list:
            cam.radiance_weight = all_cam_list[0].radiance_weight

    elif (input_folder / "transforms_test.json").exists():
        transform_test_path = input_folder / "transforms_test.json"
        print("Read test cameras from transforms_test.json")
        test_cam_list, test_camera_names = read_Aria_transform_json(
            transform_paths=[transform_test_path],
            high_freq_trajectory=closed_loop_traj,
            input_folder=input_folder,
            scene_name=scene_name,
        )
        camera_names.update(test_camera_names)
    else:
        test_cam_list, test_camera_names = None, None

    print(f"Using cameras: {camera_names}")

    if train_split == "all":
        print("will use all the cameras for the training & test split")
        train_camera_infos = all_cam_list
        valid_camera_infos = all_cam_list
        test_camera_infos = all_cam_list
    elif train_split == "fixed":
        train_camera_infos = train_cam_list
        valid_camera_infos = valid_cam_list
        test_camera_infos = test_cam_list
    else:
        if train_split == "4-1":
            print(
                "will use 4/5 of split for training, and 1/5 for validation and testing."
            )
            valid_interval = 5
        elif train_split == "7-1":
            print(
                "will use 7/8 of split for training, and 1/8 for validation and testing."
            )
            valid_interval = 8
        else:
            raise RuntimeError(f"Cannot recognize train_split: {train_split}")

        valid_idx = np.arange(0, len(all_cam_list), valid_interval)
        train_idx = np.setdiff1d(np.arange(0, len(all_cam_list)), valid_idx)

        train_camera_infos = [all_cam_list[i] for i in train_idx]
        valid_camera_infos = [all_cam_list[i] for i in valid_idx]

    if test_cam_list is not None:
        test_camera_infos = test_cam_list
    else:
        print("Did not find test cameras. Use validation cameras as test cameras")
        test_camera_infos = valid_camera_infos

    scene_scale = estimate_scene_camera_scale(all_cam_list)

    # Try to use dense point cloud from depth maps if available, otherwise fall back to sparse
    # Check if scene_cfg has the attribute (it's a config object, not a dict)
    use_dense_pointcloud = getattr(scene_cfg, "use_dense_pointcloud", False)
    depth_maps_folder = input_folder / "depth_maps"
    
    if use_dense_pointcloud and depth_maps_folder.exists():
        print(f"Generating dense point cloud from rectified depth maps in {depth_maps_folder}")
        # Generate dense point cloud from rectified depth maps
        from utils.depth_maps_to_pointcloud import generate_pointcloud_from_depth_maps
        
        points_world, colors = generate_pointcloud_from_depth_maps(
            depth_maps_folder=depth_maps_folder,
            transforms_json_path=input_folder / "transforms_with_sparse_depth.json",
            skip_n_pixels=getattr(scene_cfg, "dense_skip_pixels", 20),
            downsample_images=getattr(scene_cfg, "dense_downsample_images", 10),
            max_frames=getattr(scene_cfg, "dense_max_frames", 100),
        )
        
        # Set points_path to None since we're not saving to file
        points_path = None
    else:
        # Fall back to sparse point cloud
        print("Using sparse SLAM point cloud")
        # Get the real path if the input is a symbolic path
        points_path = (input_folder / "semidense_points.csv.gz").resolve()
        
        if not points_path.exists():
            print(f"Warning: No point cloud found at {points_path}")
            # Create minimal point cloud to avoid crash
            points_world = np.random.randn(100, 3) * 5  # Random points
            colors = None
        else:
            # read pointcloud
            points = mps.read_global_point_cloud(str(points_path))
            # filter the point cloud by inverse depth and depth
            filtered_points = []
            for point in points:
                if point.inverse_distance_std < 0.01 and point.distance_std < 0.02:
                    filtered_points.append(point)
            
            # example: get position of this point in the world coordinate frame
            points_world = []
            colors = None  # Sparse points don't have color
            for point in filtered_points:
                position_world = point.position_world
                points_world.append(position_world)

    # Handle both dense and sparse point cloud formats
    if isinstance(points_world, np.ndarray):
        xyz = points_world  # Already in the right format for dense
    else:
        xyz = np.stack(points_world, axis=0)  # Convert list to array for sparse
    
    point_cloud = BasicPointCloud(points=xyz, colors=colors, normals=None)

    if visualize:
        visualize_cameras(train_camera_infos, point_cloud)

    # save it together with with train info
    scene_info = SceneInfo(
        point_cloud=point_cloud,
        point_source_path=points_path,
        all_cameras=all_cam_list,
        train_cameras=train_camera_infos,
        valid_cameras=valid_camera_infos,
        test_cameras=test_camera_infos,
        scene_scale=scene_scale,
        scene_type=SceneType.ARIA,
        camera_labels=camera_names,
    )

    return scene_info


def readRenderInfo(
    render_cfg: dict,
    split: str = "valid",
):
    # Go through the transforms and create the camera infos
    with open(render_cfg.render_json) as json_file:
        transform = json.loads(json_file.read())

    assert (
        split in transform.keys()
    ), f"{split} is not in transform split of {transform.keys()}"

    render_cameras, render_names = read_render_transform_json(
        transform=transform[split],
    )

    if not np.isclose(render_cfg.gain_amplify, 1.0):
        for cam in render_cameras:
            cam.amplify_gain(render_cfg.gain_amplify)

    if render_cfg.render_fps > 1:
        render_cameras = interpolate_fps_piecewise(
            render_cameras, render_cfg.render_fps
        )

    # reset the render camera if needed
    for cam in render_cameras:
        # digital zoom
        if (render_cfg.zoom - 1.0) > 1e-1:
            cam.zoom(render_cfg.zoom)

        if (
            render_cfg.render_height > 0
            and render_cfg.render_height != cam.image_height
        ):
            cam.image_height = render_cfg.render_height

        if render_cfg.aspect_ratio > 0 and render_cfg.aspect_ratio != cam.aspect_ratio:
            cam.aspect_ratio = render_cfg.aspect_ratio

    # it will be used to normalize the Gaussian Points.
    scene_info = SceneInfo(
        point_cloud=None,
        all_cameras=render_cameras,
        train_cameras=[],
        valid_cameras=[],
        test_cameras=render_cameras,
        scene_scale=1.0,
        scene_type=SceneType.ARIA,
        camera_labels=render_names,
    )

    return scene_info


def aggregate_scene_infos(scene_infos):
    """
    Aggregate multiple camera and scenes into one
    """
    points_source_path = []
    points_agg = []
    colors_agg = []
    normals_agg = []
    all_cameras = []
    train_cameras = []
    valid_cameras = []
    test_cameras = []
    scene_type = None
    camera_labels = set()

    for scene_info in scene_infos:
        skip_merge = False
        for existing_source_path in points_source_path:
            if scene_info.point_source_path.samefile(existing_source_path):
                print(
                    f"There are duplicate point clouds read from {scene_info.point_source_path}. Skip merging."
                )
                skip_merge = True
                break

        if not skip_merge:
            points_source_path.append(scene_info.point_source_path)
            points_agg.append(scene_info.point_cloud.points)
            if scene_info.point_cloud.colors:
                colors_agg.append(scene_info.point_cloud.colors)
            if scene_info.point_cloud.normals:
                normals_agg.append(scene_info.point_cloud.normals)

        all_cameras += scene_info.all_cameras
        train_cameras += scene_info.train_cameras
        valid_cameras += scene_info.valid_cameras
        test_cameras += scene_info.test_cameras

        if scene_type is None:
            scene_type = scene_info.scene_type
        else:
            assert (
                scene_type == scene_info.scene_type
            ), "aggregated scene infos need to have the same scene type"

        camera_labels.update(scene_info.camera_labels)

    # Limit the number of validation images for a single scene to 5000
    if len(valid_cameras) > 5000:
        raise Warning(
            f"The validation camera number is huge! It will be very slow {len(valid_cameras)}"
        )

    all_cameras = sorted(all_cameras, key=lambda camera: camera.time_s)
    train_cameras = sorted(train_cameras, key=lambda camera: camera.time_s)
    valid_cameras = sorted(valid_cameras, key=lambda camera: camera.time_s)
    test_cameras = sorted(test_cameras, key=lambda camera: camera.time_s)

    # recalculate the exposure multiplier
    # exp_all = np.array([cam.exposure_multiplier for cam in all_cameras])
    # exp_median = np.median(exp_all)
    # global_radiance_weight = 1.0 / exp_median
    # for cam in all_cameras:
    #     cam.radiance_weight = global_radiance_weight
    # for cam in train_cameras:
    #     cam.radiance_weight = global_radiance_weight
    # for cam in valid_cameras:
    #     cam.radiance_weight = global_radiance_weight
    # for cam in test_cameras:
    #     cam.radiance_weight = global_radiance_weight

    points_agg = np.concatenate(points_agg, axis=0)
    if len(colors_agg) > 0:
        colors_agg = np.concatenate(colors_agg, axis=0)
        assert len(colors_agg) == len(
            points_agg
        ), "number of points that have colors doest not match point number"
    else:
        colors_agg = None

    if len(normals_agg) > 0:
        normals_agg = np.concatenate(normals_agg, axis=0)
        assert len(normals_agg) == len(
            points_agg
        ), "number of points that have normals doest not match point number"
    else:
        normals_agg = None

    pcd = BasicPointCloud(points=points_agg, colors=colors_agg, normals=normals_agg)

    scene_scale = estimate_scene_camera_scale(all_cameras)

    scene_info_agg = SceneInfo(
        point_cloud=pcd,
        point_source_path=points_source_path,
        all_cameras=all_cameras,
        train_cameras=train_cameras,
        valid_cameras=valid_cameras,
        test_cameras=test_cameras,
        scene_scale=scene_scale,
        scene_type=scene_type,
        camera_labels=camera_labels,
    )

    return scene_info_agg
