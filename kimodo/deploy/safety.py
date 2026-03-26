# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Safety layer: velocity clamping, linear ramp-up, watchdog timer, E-stop.

Applied once per 500Hz control step before publishing LowCmd.
"""

import time
from typing import Optional

import numpy as np

from .joint_map import JointMapper, NUM_JOINTS


class SafetyLayer:
    """Per-step safety filter between MotionBuffer and the SDK publish call.

    Usage:
        layer = SafetyLayer(mapper)
        # When starting playback of a new motion:
        layer.begin_ramp(q_current_from_robot, first_frame_of_motion)
        # Each 500Hz step:
        layer.tick()
        q_cmd, kp, kd = layer.filter(q_target, q_current, kp_default, kd_default)
    """

    def __init__(
        self,
        joint_mapper: JointMapper,
        ramp_duration_sec: float = 3.0,
        watchdog_ms: float = 20.0,
        control_hz: float = 500.0,
    ) -> None:
        self._mapper = joint_mapper
        self._watchdog_threshold = watchdog_ms / 1000.0
        self._ramp_total: int = max(1, int(ramp_duration_sec * control_hz))

        # State
        self._q_prev: Optional[np.ndarray] = None
        self._estopped: bool = False
        self._ramp_active: bool = False
        self._ramp_step: int = 0
        self._ramp_q_start: Optional[np.ndarray] = None
        self._ramp_q_end: Optional[np.ndarray] = None
        self._last_tick_time: float = time.monotonic()

    # ------------------------------------------------------------------
    # Ramp
    # ------------------------------------------------------------------

    def begin_ramp(self, q_current: np.ndarray, q_first_frame: np.ndarray) -> None:
        """Start a gradual ramp from the robot's current pose to the first motion frame.

        Args:
            q_current:    (29,) current joint positions read from LowState.
            q_first_frame: (29,) first frame of the incoming motion.
        """
        self._ramp_q_start = q_current.astype(np.float32)
        self._ramp_q_end = q_first_frame.astype(np.float32)
        self._ramp_step = 0
        self._ramp_active = True

    def in_ramp(self) -> bool:
        return self._ramp_active

    # ------------------------------------------------------------------
    # Watchdog
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Call once per control step to reset the watchdog timer."""
        self._last_tick_time = time.monotonic()

    def watchdog_triggered(self) -> bool:
        """True if more than watchdog_ms have elapsed since the last tick()."""
        return (time.monotonic() - self._last_tick_time) > self._watchdog_threshold

    # ------------------------------------------------------------------
    # E-stop
    # ------------------------------------------------------------------

    def trigger_estop(self) -> None:
        """Engage E-stop: zeroes kp, damping-only on all subsequent filter() calls."""
        self._estopped = True
        self._ramp_active = False

    def is_estopped(self) -> bool:
        return self._estopped

    def clear_estop(self) -> None:
        """Clear E-stop (call only when safe to resume)."""
        self._estopped = False

    # ------------------------------------------------------------------
    # Main filter
    # ------------------------------------------------------------------

    def filter(
        self,
        q_target: np.ndarray,
        q_current: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply safety checks and return (q_cmd, kp_out, kd_out).

        Priority: E-stop > watchdog > ramp > normal.

        Args:
            q_target:  (29,) desired joint positions this step.
            q_current: (29,) current joint positions from LowState.
            kp:        (29,) proportional gains (default from JointMapper).
            kd:        (29,) derivative gains.
        Returns:
            q_cmd:  (29,) final commanded positions (velocity-clamped).
            kp_out: (29,) gains to use (may be zeroed by safety modes).
            kd_out: (29,) derivative gains (always present for damping).
        """
        kd_out = kd.copy()

        # --- E-stop: hold, zero kp ---
        if self._estopped:
            q_cmd = (self._q_prev if self._q_prev is not None else q_current).copy()
            return q_cmd, np.zeros(NUM_JOINTS, dtype=np.float32), kd_out

        # --- Watchdog: hold, zero kp ---
        if self.watchdog_triggered():
            q_cmd = (self._q_prev if self._q_prev is not None else q_current).copy()
            return q_cmd, np.zeros(NUM_JOINTS, dtype=np.float32), kd_out

        # --- Ramp: interpolate from current pose to first motion frame ---
        if self._ramp_active:
            alpha = self._ramp_step / self._ramp_total
            q_cmd = (
                self._ramp_q_start + alpha * (self._ramp_q_end - self._ramp_q_start)
            ).astype(np.float32)
            kp_out = (kp * alpha).astype(np.float32)
            self._ramp_step += 1
            if self._ramp_step >= self._ramp_total:
                self._ramp_active = False
            self._q_prev = q_cmd
            return q_cmd, kp_out, kd_out

        # --- Normal: velocity clamp ---
        if self._q_prev is None:
            self._q_prev = q_current.copy()

        delta = q_target - self._q_prev
        max_dq = self._mapper.max_dq
        delta = np.clip(delta, -max_dq, max_dq)
        q_cmd = (self._q_prev + delta).astype(np.float32)
        self._q_prev = q_cmd
        return q_cmd, kp.copy(), kd_out
