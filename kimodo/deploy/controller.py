# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""500Hz LowCmd control loop for Unitree G1.

unitree_sdk2py is an optional dependency — guarded at import time.
When dry_run=True, joint angles are printed to stdout and no SDK is needed.
"""

import threading
import time
from typing import Optional

import numpy as np

from .joint_map import JointMapper, NUM_JOINTS, CONTROL_HZ
from .motion_buffer import MotionBuffer
from .safety import SafetyLayer

try:
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelPublisher,
        ChannelSubscriber,
    )
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC

    UNITREE_AVAILABLE = True
except ImportError:
    UNITREE_AVAILABLE = False


_SERVO_MODE = 0x01   # PD position control
_DAMP_MODE = 0x00    # passive / damping only


class G1Controller:
    """500Hz LowCmd loop for Unitree G1.

    Args:
        motion_buffer: MotionBuffer instance (shared with generator).
        safety_layer:  SafetyLayer instance.
        joint_mapper:  JointMapper instance.
        network_interface: Ethernet interface name (e.g. "eth0").
        dry_run: If True, skip SDK and print joint angles to stdout.
        control_hz: Control loop frequency (default 500).
    """

    def __init__(
        self,
        motion_buffer: MotionBuffer,
        safety_layer: SafetyLayer,
        joint_mapper: JointMapper,
        network_interface: str = "eth0",
        dry_run: bool = False,
        control_hz: float = CONTROL_HZ,
    ) -> None:
        if not dry_run and not UNITREE_AVAILABLE:
            raise ImportError(
                "unitree_sdk2py is not installed. Install it or use --dry-run."
            )
        self._buf = motion_buffer
        self._safety = safety_layer
        self._mapper = joint_mapper
        self._iface = network_interface
        self._dry_run = dry_run
        self._period = 1.0 / control_hz

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Latest joint positions from LowState (protected by _state_lock)
        self._q_current = np.zeros(NUM_JOINTS, dtype=np.float32)
        self._state_lock = threading.Lock()

        # SDK objects (None in dry-run)
        self._pub = None
        self._sub = None
        self._crc = None
        self._step_count = 0
        self._motion_active = False  # True while a motion is being played

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Initialize DDS (if not dry-run) and start the 500Hz loop thread."""
        if not self._dry_run:
            ChannelFactoryInitialize(0, self._iface)
            self._crc = CRC()

            self._pub = ChannelPublisher("rt/lowcmd", LowCmd_)
            self._pub.Init()

            self._sub = ChannelSubscriber("rt/lowstate", LowState_)
            self._sub.Init(self._on_low_state, 10)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the control loop to stop and join the thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def get_current_joints(self) -> np.ndarray:
        """Return latest (29,) joint positions from LowState. Thread-safe."""
        with self._state_lock:
            return self._q_current.copy()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_low_state(self, msg: "LowState_") -> None:
        """LowState subscriber callback — updates q_current."""
        q = np.array(
            [msg.motor_state[i].q for i in range(NUM_JOINTS)], dtype=np.float32
        )
        with self._state_lock:
            self._q_current = q

    def _control_loop(self) -> None:
        """Main 500Hz loop. Runs in a daemon thread."""
        kp = self._mapper.kp_default.copy()
        kd = self._mapper.kd_default.copy()

        while not self._stop_event.is_set():
            t0 = time.monotonic()

            self._safety.tick()
            q_current = self.get_current_joints()

            if self._safety.is_estopped():
                q_cmd, kp_cmd, kd_cmd = self._safety.filter(q_current, q_current, kp, kd)
                mode = _DAMP_MODE
            elif self._safety.watchdog_triggered():
                q_cmd, kp_cmd, kd_cmd = self._safety.filter(q_current, q_current, kp, kd)
                mode = _DAMP_MODE
            else:
                q_target = self._buf.get_next_joints()
                if q_target is None:
                    # No motion — hold position with damping; reset so next motion ramps
                    self._motion_active = False
                    q_cmd = q_current
                    kp_cmd = np.zeros(NUM_JOINTS, dtype=np.float32)
                    kd_cmd = kd
                    mode = _DAMP_MODE
                else:
                    # Trigger ramp on the first frame of each new motion
                    if not self._motion_active:
                        self._safety.begin_ramp(q_current, q_target)
                        self._motion_active = True
                    q_cmd, kp_cmd, kd_cmd = self._safety.filter(
                        q_target, q_current, kp, kd
                    )
                    mode = _SERVO_MODE

            self._publish_cmd(q_cmd, kp_cmd, kd_cmd, mode)
            self._step_count += 1

            elapsed = time.monotonic() - t0
            sleep_time = self._period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _publish_cmd(
        self,
        q_cmd: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        mode: int,
    ) -> None:
        """Build and publish one LowCmd message (or print in dry-run)."""
        if self._dry_run:
            if self._step_count % 500 == 0:  # print once per second
                vals = ", ".join(f"{v:.3f}" for v in q_cmd[:6])
                print(f"[dry-run step={self._step_count}] legs[0:6]=[{vals}]")
            return

        cmd = unitree_go_msg_dds__LowCmd_()
        cmd.head[0] = 0xFE
        cmd.head[1] = 0xEF
        cmd.level_flag = 0xFF
        cmd.gpio = 0

        for i in range(NUM_JOINTS):
            cmd.motor_cmd[i].mode = mode
            cmd.motor_cmd[i].q = float(q_cmd[i])
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kp = float(kp[i])
            cmd.motor_cmd[i].kd = float(kd[i])
            cmd.motor_cmd[i].tau = 0.0

        # Motors 29-34: set to passive/safe
        for i in range(NUM_JOINTS, 35):
            cmd.motor_cmd[i].mode = _DAMP_MODE
            cmd.motor_cmd[i].q = 0.0
            cmd.motor_cmd[i].kp = 0.0
            cmd.motor_cmd[i].kd = 2.0
            cmd.motor_cmd[i].tau = 0.0

        cmd.crc = self._crc.Crc(cmd)
        self._pub.Write(cmd)
