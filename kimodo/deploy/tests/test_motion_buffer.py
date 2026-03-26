import threading
import time
import numpy as np
import pytest
from kimodo.deploy.motion_buffer import MotionBuffer


def make_motion(T: int = 150, val: float = 0.0) -> np.ndarray:
    """(T, 29) motion filled with a constant value."""
    return np.full((T, 29), val, dtype=np.float32)


def test_not_playing_initially():
    buf = MotionBuffer()
    assert not buf.is_playing()


def test_get_next_returns_none_when_empty():
    buf = MotionBuffer()
    assert buf.get_next_joints() is None


def test_push_makes_buffer_playing():
    buf = MotionBuffer()
    buf.push_motion(make_motion(150, 1.0))
    assert buf.is_playing()


def test_get_next_returns_array_after_push():
    buf = MotionBuffer()
    buf.push_motion(make_motion(150, 1.0))
    result = buf.get_next_joints()
    assert result is not None
    assert result.shape == (29,)


def test_drain_150_frames_at_500hz():
    """150 frames at 30fps = 5 seconds. At 500Hz that's 2500 steps."""
    buf = MotionBuffer(blend_duration_sec=0.0)
    buf.push_motion(make_motion(150, 1.0))
    non_none = 0
    for _ in range(2500):
        r = buf.get_next_joints()
        if r is not None:
            non_none += 1
    # 150 frames × 149 interpolation intervals × (500/30) ≈ 2483 steps
    assert non_none >= 2480


def test_hold_last_frame_when_buffer_empty_at_end():
    """After clip ends with no next motion, returns None (controller goes damping)."""
    buf = MotionBuffer(blend_duration_sec=0.0)
    buf.push_motion(make_motion(1))   # single frame clip
    # Drain well past end
    results = [buf.get_next_joints() for _ in range(100)]
    # Should have gotten None at some point after clip ends
    assert any(r is None for r in results)


def test_clear_stops_playback():
    buf = MotionBuffer()
    buf.push_motion(make_motion(150, 1.0))
    buf.clear()
    assert not buf.is_playing()
    assert buf.get_next_joints() is None


def test_two_motions_transition_without_deadlock():
    """Generator pushes motion B while controller is draining motion A."""
    buf = MotionBuffer(blend_duration_sec=0.1)
    results = []

    def consumer():
        for _ in range(1200):   # drain ~2.4s worth
            r = buf.get_next_joints()
            results.append(r)
            time.sleep(1.0 / 500)

    buf.push_motion(make_motion(150, 0.0))   # motion A

    t = threading.Thread(target=consumer)
    t.start()
    time.sleep(0.1)  # let consumer start

    # Push motion B while A is playing
    buf.push_motion(make_motion(150, 1.0))   # blocks until slot B available

    t.join(timeout=5.0)
    assert not t.is_alive(), "Consumer thread deadlocked"

    non_none = [r for r in results if r is not None]
    assert len(non_none) > 500, f"Too few frames: {len(non_none)}"


def test_values_transition_from_a_to_b():
    """After transition, joint values should move toward motion B's value."""
    buf = MotionBuffer(blend_duration_sec=0.5, motion_fps=30, control_hz=500)
    buf.push_motion(make_motion(150, 0.0))   # motion A: all zeros

    frames = []
    done = threading.Event()

    def consumer():
        for _ in range(2500 + 500):  # drain A + some of B
            r = buf.get_next_joints()
            frames.append(r)
            time.sleep(1.0 / 500)
        done.set()

    t = threading.Thread(target=consumer, daemon=True)
    t.start()
    time.sleep(0.5)

    buf.push_motion(make_motion(150, 2.0))   # motion B: all twos

    done.wait(timeout=8.0)
    t.join(timeout=1.0)

    # Find last non-None frame — should be near 2.0
    last_val = next(r[0] for r in reversed(frames) if r is not None)
    assert last_val > 1.5, f"Expected value near 2.0 after transition, got {last_val}"
