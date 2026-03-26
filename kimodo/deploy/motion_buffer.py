# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Thread-safe double buffer for Kimodo motion playback.

One thread (generator) writes (T, 29) motion arrays.
One thread (controller) reads one (29,) frame per 500Hz step.
Transitions between clips use cosine blending.
"""

import threading
from typing import Optional

import numpy as np

from .joint_map import JointMapper


class MotionBuffer:
    """Double buffer: slot A plays, slot B queues.

    Thread safety:
    - A single threading.Condition guards _play_buf, _next_buf, _cursor.
    - The controller holds the lock only to swap buffers — never during interpolation.
    - push_motion() blocks if slot B is already occupied (backpressure).
    """

    def __init__(
        self,
        blend_duration_sec: float = 0.5,
        motion_fps: float = 30.0,
        control_hz: float = 500.0,
    ) -> None:
        self._mapper = JointMapper()
        self._blend_frames = int(blend_duration_sec * motion_fps)
        self._step = motion_fps / control_hz          # frames advanced per control step

        self._lock = threading.Condition()
        self._play_buf: Optional[np.ndarray] = None   # (T, 29)
        self._next_buf: Optional[np.ndarray] = None   # (T, 29)
        self._cursor: float = 0.0                     # continuous frame index in play_buf
        self._last_frame: Optional[np.ndarray] = None # held when play_buf exhausted

        # Blend state (set during swap)
        self._blending: bool = False
        self._blend_step: int = 0

    # ------------------------------------------------------------------
    # Writer API (generator thread)
    # ------------------------------------------------------------------

    def push_motion(self, joints: np.ndarray) -> None:
        """Push a (T, 29) float32 motion into slot B.

        Blocks until slot B is free (i.e., the controller has consumed slot A
        and swapped B→A).
        """
        assert joints.ndim == 2 and joints.shape[1] == 29, (
            f"Expected (T, 29), got {joints.shape}"
        )
        joints = joints.astype(np.float32)
        with self._lock:
            while self._next_buf is not None:
                self._lock.wait()
            self._next_buf = joints
            self._lock.notify_all()

    # ------------------------------------------------------------------
    # Reader API (controller thread — called at 500Hz)
    # ------------------------------------------------------------------

    def get_next_joints(self) -> Optional[np.ndarray]:
        """Return the next (29,) frame, or None if no motion is loaded.

        Non-blocking. When None is returned the controller should apply
        damping-only mode (kp=0) and hold position.
        """
        with self._lock:
            if self._play_buf is None:
                # No motion loaded at all — try to pick up next_buf
                if self._next_buf is not None:
                    self._play_buf = self._next_buf
                    self._next_buf = None
                    self._cursor = 0.0
                    self._blending = False
                    self._lock.notify_all()
                else:
                    return None

            T = len(self._play_buf)

            # --- Normal playback ---
            frame_idx = int(self._cursor)

            if frame_idx < T - 1:
                phase = self._cursor - frame_idx
                if self._blending:
                    # During blend: override with cosine blend output
                    alpha = self._blend_step / max(self._blend_frames, 1)
                    result = self._mapper.cosine_blend(
                        self._last_frame, self._play_buf[0], alpha
                    )
                    self._blend_step += 1
                    if self._blend_step >= self._blend_frames:
                        self._blending = False
                else:
                    result = self._mapper.interpolate(
                        self._play_buf[frame_idx],
                        self._play_buf[frame_idx + 1],
                        phase,
                    )
                self._cursor += self._step
                self._last_frame = result
                return result

            # --- End of clip ---
            self._last_frame = self._play_buf[-1]

            if self._next_buf is not None:
                # Swap: B becomes A, start blending
                self._play_buf = self._next_buf
                self._next_buf = None
                self._cursor = 0.0
                self._lock.notify_all()

                if self._blend_frames > 0:
                    self._blending = True
                    self._blend_step = 0
                    result = self._last_frame  # first blend frame = tail of A
                else:
                    self._blending = False
                    result = self._play_buf[0]
                self._cursor += self._step
                return result

            # No next motion — hold last frame, signal None so controller damps
            self._play_buf = None
            return None

    def is_playing(self) -> bool:
        with self._lock:
            return self._play_buf is not None or self._next_buf is not None

    def clear(self) -> None:
        """Clear all buffers immediately (E-stop helper)."""
        with self._lock:
            self._play_buf = None
            self._next_buf = None
            self._cursor = 0.0
            self._blending = False
            self._last_frame = None
            self._lock.notify_all()
