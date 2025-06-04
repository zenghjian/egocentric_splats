# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass

import numpy as np
import projectaria_tools.core.mps as mps

from projectaria_tools.core import calibration
from projectaria_tools.core.sophus import interpolate, SE3


@dataclass
class PoseSE3:
    transform_world_device: SE3
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    quality_score: float
    time_ns: int


def interpolate_closeloop_trajectory_pose(
    start_pose: mps.ClosedLoopTrajectoryPose,
    end_pose: mps.ClosedLoopTrajectoryPose,
    time_ns: int,
):
    start_time = start_pose.tracking_timestamp.total_seconds() * 1e9
    end_time = end_pose.tracking_timestamp.total_seconds() * 1e9
    ratio = (time_ns - start_time) / (end_time - start_time)

    assert (
        ratio >= 0.0 and ratio <= 1.0
    ), f"interpolation ratio {ratio} is not within [0.0, 1.0]"
    interp_pose_SE3 = interpolate(
        start_pose.transform_world_device, end_pose.transform_world_device, ratio
    )
    interp_linear_velocity = (
        start_pose.device_linear_velocity_device * (1 - ratio)
        + end_pose.device_linear_velocity_device * ratio
    )
    interp_angular_velocity = (
        start_pose.angular_velocity_device * (1 - ratio)
        + end_pose.angular_velocity_device * ratio
    )
    interp_pose_score = (
        start_pose.quality_score * (1 - ratio) + end_pose.quality_score * ratio
    )

    return PoseSE3(
        transform_world_device=interp_pose_SE3,
        linear_velocity=interp_linear_velocity,
        angular_velocity=interp_angular_velocity,
        quality_score=interp_pose_score,
        time_ns=int(time_ns),
    )


def interpolate_aria_pose(closed_loop_traj, capture_time_ns) -> PoseSE3:

    start_idx = mps.utils.bisection_timestamp_search(closed_loop_traj, capture_time_ns)
    if start_idx is None:
        return None

    if (
        closed_loop_traj[start_idx].tracking_timestamp.total_seconds() * 1e9
        > capture_time_ns
    ):
        start_idx -= 1

    start_pose = closed_loop_traj[start_idx]

    if start_idx + 1 >= len(closed_loop_traj):
        return closed_loop_traj[start_pose]

    end_pose = closed_loop_traj[start_idx + 1]

    interp_pose = interpolate_closeloop_trajectory_pose(
        start_pose, end_pose, time_ns=capture_time_ns
    )

    return interp_pose
