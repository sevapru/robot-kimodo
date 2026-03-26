# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI entry point: stream Kimodo motions to a physical Unitree G1.

Usage:
    kimodo_deploy_g1 --server-ip 192.168.1.10 --network-interface eth0
    kimodo_deploy_g1 --dry-run   # no SDK, print joint values
"""

import argparse
import os
import signal
import sys
import threading
import time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stream Kimodo motion generation to a physical Unitree G1."
    )
    p.add_argument("--server-ip", default="127.0.0.1",
                   help="IP of the GPU server running kimodo_textencoder (default: 127.0.0.1)")
    p.add_argument("--server-port", type=int, default=9550,
                   help="Port of the kimodo_textencoder Gradio server (default: 9550)")
    p.add_argument("--network-interface", default="eth0",
                   help="Ethernet interface for unitree_sdk2 DDS (default: eth0)")
    p.add_argument("--model", default="kimodo-g1-rp",
                   help="Kimodo model name (default: kimodo-g1-rp)")
    p.add_argument("--duration", type=float, default=5.0,
                   help="Generated motion duration in seconds (default: 5.0)")
    p.add_argument("--diffusion-steps", type=int, default=50,
                   help="DDIM denoising steps (default: 50; use 20 for fast testing)")
    p.add_argument("--ramp-duration", type=float, default=3.0,
                   help="Seconds to ramp kp from 0 to default at playback start (default: 3.0)")
    p.add_argument("--blend-duration", type=float, default=0.5,
                   help="Cosine blend window between motion clips in seconds (default: 0.5)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print joint angles to stdout; skip all SDK calls.")
    return p.parse_args()


def _status_thread(generator, buffer, stop_event):
    """Print status once per second to stderr."""
    while not stop_event.is_set():
        gen = "generating" if generator.is_generating() else "idle"
        buf = "playing" if buffer.is_playing() else "waiting"
        q = generator.queue_size()
        print(f"\r[gen:{gen}  buf:{buf}  queue:{q}]   ", end="", flush=True, file=sys.stderr)
        time.sleep(1.0)


def main() -> None:
    args = parse_args()

    # Set text encoder to API mode pointing at the GPU server
    os.environ["TEXT_ENCODER_MODE"] = "api"
    os.environ["TEXT_ENCODER_URL"] = f"http://{args.server_ip}:{args.server_port}/"

    from kimodo.deploy.joint_map import JointMapper
    from kimodo.deploy.motion_buffer import MotionBuffer
    from kimodo.deploy.safety import SafetyLayer
    from kimodo.deploy.generator import G1Generator
    from kimodo.deploy.controller import G1Controller

    mapper = JointMapper()
    buffer = MotionBuffer(blend_duration_sec=args.blend_duration)
    safety = SafetyLayer(mapper, ramp_duration_sec=args.ramp_duration)
    generator = G1Generator(
        buffer,
        model_name=args.model,
        duration_sec=args.duration,
        diffusion_steps=args.diffusion_steps,
    )
    controller = G1Controller(
        buffer,
        safety,
        mapper,
        network_interface=args.network_interface,
        dry_run=args.dry_run,
    )

    # Load model (blocking — takes ~10-30s on Jetson)
    generator.start()

    # Start 500Hz loop
    controller.start()

    # Graceful shutdown on Ctrl+C
    stop_event = threading.Event()

    def _shutdown(sig, frame):
        print("\n[run_g1] Shutting down...", file=sys.stderr)
        safety.trigger_estop()
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Status display thread
    st = threading.Thread(
        target=_status_thread, args=(generator, buffer, stop_event), daemon=True
    )
    st.start()

    print("Ready. Type a motion prompt and press Enter. Type 'q' to quit.")
    print("Example prompts: 'robot walks forward', 'robot raises both arms', 'robot turns left'\n")

    try:
        while not stop_event.is_set():
            try:
                line = input("prompt> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if line.lower() in ("q", "quit", "exit"):
                break
            if line == "":
                continue
            status = "generating" if generator.is_generating() else "queued"
            print(f"[{status}] {line!r}")
            generator.submit(line)
    finally:
        safety.trigger_estop()
        generator.stop()
        controller.stop()
        print("\n[run_g1] Stopped.", file=sys.stderr)


if __name__ == "__main__":
    main()
