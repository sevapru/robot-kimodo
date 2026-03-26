# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Background generation thread: text prompt → (T, 29) joint array → MotionBuffer.

Environment variables (set by run_g1.py before calling start()):
    TEXT_ENCODER_MODE=api
    TEXT_ENCODER_URL=http://<gpu-server>:<port>/
"""

import queue
import threading
from typing import Optional

import numpy as np
import torch

from .motion_buffer import MotionBuffer


class G1Generator:
    """Generates G1 motions from text prompts in a background thread.

    One prompt is processed at a time (FIFO). Additional prompts queue up.
    Model is loaded once when start() is called.

    Args:
        motion_buffer:   MotionBuffer to push generated motions into.
        model_name:      Kimodo model short-key (default "kimodo-g1-rp").
        device:          Torch device string. None = auto (cuda if available).
        duration_sec:    Length of each generated motion in seconds.
        diffusion_steps: Number of DDIM denoising steps (50 = fast, 100 = quality).
        cfg_weight:      CFG guidance weights [text_weight, constraint_weight].
    """

    def __init__(
        self,
        motion_buffer: MotionBuffer,
        model_name: str = "kimodo-g1-rp",
        device: Optional[str] = None,
        duration_sec: float = 5.0,
        diffusion_steps: int = 50,
        cfg_weight: Optional[list] = None,
    ) -> None:
        self._buf = motion_buffer
        self._model_name = model_name
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._duration_sec = duration_sec
        self._diffusion_steps = diffusion_steps
        self._cfg_weight = cfg_weight or [2.0, 2.0]

        self._prompt_queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._generating = threading.Event()

        self._model = None       # loaded in start()
        self._converter = None   # MujocoQposConverter, created after model load

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Load the Kimodo model (blocking) then start the generation thread."""
        print(f"[generator] Loading model {self._model_name!r} on {self._device}...")
        from kimodo.model.load_model import load_model
        from kimodo.exports.mujoco import MujocoQposConverter

        self._model = load_model(self._model_name, device=self._device)
        self._model.eval()
        self._converter = MujocoQposConverter(self._model.skeleton)
        print("[generator] Model loaded.")

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._generation_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the generation thread to stop and join."""
        self._stop_event.set()
        self._prompt_queue.put(None)   # unblock queue.get()
        if self._thread is not None:
            self._thread.join(timeout=10.0)

    def submit(self, prompt: str) -> None:
        """Add a text prompt to the generation queue. Non-blocking."""
        self._prompt_queue.put(prompt)

    def is_generating(self) -> bool:
        """True if a generation is currently in progress."""
        return self._generating.is_set()

    def queue_size(self) -> int:
        return self._prompt_queue.qsize()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _generation_loop(self) -> None:
        while not self._stop_event.is_set():
            prompt = self._prompt_queue.get()
            if prompt is None or self._stop_event.is_set():
                break
            self._generating.set()
            try:
                print(f"[generator] Generating: {prompt!r}")
                joints = self._generate_one(prompt)
                self._buf.push_motion(joints)
                print(f"[generator] Done: {joints.shape[0]} frames pushed to buffer.")
            except Exception as exc:
                print(f"[generator] ERROR during generation: {exc}")
            finally:
                self._generating.clear()

    def _generate_one(self, prompt: str) -> np.ndarray:
        """Run one generation. Returns (T, 29) float32 joint array."""
        from kimodo.exports.mujoco import apply_g1_real_robot_projection
        from kimodo.skeleton import global_rots_to_local_rots

        fps = self._model.fps if hasattr(self._model, "fps") else 30
        num_frames = int(self._duration_sec * fps)

        with torch.no_grad():
            output = self._model(
                [prompt],
                [num_frames],
                num_denoising_steps=self._diffusion_steps,
                multi_prompt=True,
                post_processing=False,
                return_numpy=False,
                cfg_weight=self._cfg_weight,
            )

        # Project to 1-DoF per hinge, clamp to XML joint limits
        joints_pos, joints_rot = apply_g1_real_robot_projection(
            self._model.skeleton,
            output["posed_joints"],     # (B, T, J, 3)
            output["global_rot_mats"],  # (B, T, J, 3, 3)
            clamp_to_limits=True,
        )

        # Reconstruct local rotation matrices from projected global rots
        local_rot_mats = global_rots_to_local_rots(joints_rot, self._model.skeleton)
        root_positions = joints_pos[..., self._model.skeleton.root_idx, :]

        # Convert to qpos with mechanical-zero convention (T-pose = q=0)
        qpos = self._converter.to_qpos(
            local_rot_mats,     # ensure_batched handles (T,J,3,3) → (1,T,J,3,3)
            root_positions,
            mujoco_rest_zero=True,
        )
        # qpos shape: (1, T, 36) — take batch 0, columns [7:] = 29 joints
        return qpos[0, :, 7:].cpu().numpy().astype(np.float32)
