# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Joint constants and interpolation utilities for G1 deployment.

No SDK dependency — safe to import on any machine.
"""

import numpy as np

CONTROL_HZ: int = 500
MOTION_FPS: int = 30
NUM_JOINTS: int = 29

# unitree_sdk2 motor index order for G1 (0-based):
# 0-5:   left leg  (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
# 6-11:  right leg (same order)
# 12-14: waist     (yaw, roll, pitch)
# 15-21: left arm  (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
#                   wrist_roll, wrist_pitch, wrist_yaw)
# 22-28: right arm (same order)
JOINT_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
    "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
    "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]

# (motor_index_start, motor_index_end_exclusive, kp, kd, max_delta_q_per_control_step)
# max_delta_q is in radians per 2ms step (500Hz).
# e.g. 0.005 rad/step = 2.5 rad/s — conservative for motion playback.
JOINT_GROUPS: dict[str, tuple[int, int, float, float, float]] = {
    "left_leg":  (0,  6,  100.0, 5.0, 0.005),
    "right_leg": (6,  12, 100.0, 5.0, 0.005),
    "waist":     (12, 15,  80.0, 4.0, 0.004),
    "left_arm":  (15, 22,  40.0, 2.0, 0.008),
    "right_arm": (22, 29,  40.0, 2.0, 0.008),
}


class JointMapper:
    """Converts Kimodo qpos arrays to per-motor arrays and performs interpolation.

    Stateless after construction — thread-safe.
    """

    def __init__(self) -> None:
        self.kp_default = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.kd_default = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.max_dq = np.zeros(NUM_JOINTS, dtype=np.float32)
        for _name, (lo, hi, kp, kd, dq) in JOINT_GROUPS.items():
            self.kp_default[lo:hi] = kp
            self.kd_default[lo:hi] = kd
            self.max_dq[lo:hi] = dq

    def qpos_to_joints(self, qpos_frame: np.ndarray) -> np.ndarray:
        """Extract 29 joint angles from a single qpos row.

        Args:
            qpos_frame: shape (36,) — output of MujocoQposConverter.to_qpos
                        with mujoco_rest_zero=True.
                        Cols [0:3]=root_pos, [3:7]=root_quat, [7:36]=joint angles.
        Returns:
            shape (29,) float32, motor-index order 0-28, radians from T-pose zero.
        """
        assert qpos_frame.shape == (36,), f"Expected (36,), got {qpos_frame.shape}"
        return qpos_frame[7:].astype(np.float32)

    def interpolate(
        self,
        q_start: np.ndarray,
        q_end: np.ndarray,
        phase: float,
    ) -> np.ndarray:
        """Linear interpolation between two joint angle vectors.

        Args:
            q_start: (29,) joints at frame N
            q_end:   (29,) joints at frame N+1
            phase:   float in [0.0, 1.0]
        Returns:
            (29,) interpolated joint angles
        """
        return (q_start + phase * (q_end - q_start)).astype(np.float32)

    def cosine_blend(
        self,
        q_a: np.ndarray,
        q_b: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """Cosine-eased blend between two joint configurations.

        Args:
            q_a: (29,) source configuration
            q_b: (29,) target configuration
            alpha: float in [0.0, 1.0]; 0 = all a, 1 = all b
        Returns:
            (29,) blended joint angles
        """
        t = 0.5 * (1.0 - np.cos(np.pi * alpha))
        return (q_a * (1.0 - t) + q_b * t).astype(np.float32)
