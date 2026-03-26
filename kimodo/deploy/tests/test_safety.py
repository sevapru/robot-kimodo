import time
import numpy as np
import pytest
from kimodo.deploy.joint_map import JointMapper
from kimodo.deploy.safety import SafetyLayer


@pytest.fixture
def mapper():
    return JointMapper()


@pytest.fixture
def safety(mapper):
    return SafetyLayer(mapper, ramp_duration_sec=0.1, watchdog_ms=20.0, control_hz=500.0)


def test_not_estopped_initially(safety):
    assert not safety.is_estopped()


def test_not_in_ramp_initially(safety):
    assert not safety.in_ramp()


def test_watchdog_not_triggered_immediately(safety):
    safety.tick()
    assert not safety.watchdog_triggered()


def test_watchdog_triggers_after_delay(safety):
    safety.tick()
    time.sleep(0.025)  # 25ms > 20ms threshold
    assert safety.watchdog_triggered()


def test_watchdog_resets_on_tick(safety):
    safety.tick()
    time.sleep(0.025)
    assert safety.watchdog_triggered()
    safety.tick()
    assert not safety.watchdog_triggered()


def test_ramp_starts_in_ramp_state(safety):
    q0 = np.zeros(29, dtype=np.float32)
    q1 = np.ones(29, dtype=np.float32)
    safety.begin_ramp(q0, q1)
    assert safety.in_ramp()


def test_ramp_kp_starts_near_zero(safety, mapper):
    q0 = np.zeros(29, dtype=np.float32)
    q1 = np.ones(29, dtype=np.float32)
    safety.begin_ramp(q0, q1)
    _, kp_out, _ = safety.filter(q1, q0, mapper.kp_default, mapper.kd_default)
    # First step: kp should be very small (nearly zero)
    assert kp_out[0] < 1.0


def test_ramp_reaches_target_kp_at_end(safety, mapper):
    q0 = np.zeros(29, dtype=np.float32)
    q1 = np.ones(29, dtype=np.float32)
    safety.begin_ramp(q0, q1)
    ramp_steps = safety._ramp_total
    for _ in range(ramp_steps):
        q_cmd, kp_out, _ = safety.filter(q1, q0, mapper.kp_default, mapper.kd_default)
        safety.tick()
    # After ramp: kp should be at default
    assert not safety.in_ramp()
    _, kp_final, _ = safety.filter(q1, q0, mapper.kp_default, mapper.kd_default)
    np.testing.assert_array_almost_equal(kp_final, mapper.kp_default)


def test_velocity_clamp_limits_delta(safety, mapper):
    safety._q_prev = np.zeros(29, dtype=np.float32)
    q_large_step = np.full(29, 10.0, dtype=np.float32)  # huge jump
    q_current = np.zeros(29, dtype=np.float32)
    q_cmd, _, _ = safety.filter(q_large_step, q_current, mapper.kp_default, mapper.kd_default)
    delta = np.abs(q_cmd - safety._q_prev)
    np.testing.assert_array_less(delta, mapper.max_dq + 1e-6)


def test_estop_zeroes_kp(safety, mapper):
    safety._q_prev = np.zeros(29, dtype=np.float32)
    safety.trigger_estop()
    assert safety.is_estopped()
    q = np.zeros(29, dtype=np.float32)
    _, kp_out, _ = safety.filter(q, q, mapper.kp_default, mapper.kd_default)
    np.testing.assert_array_equal(kp_out, np.zeros(29))


def test_estop_clear_restores_normal(safety, mapper):
    safety._q_prev = np.zeros(29, dtype=np.float32)
    safety.trigger_estop()
    safety.clear_estop()
    assert not safety.is_estopped()
    q = np.zeros(29, dtype=np.float32)
    _, kp_out, _ = safety.filter(q, q, mapper.kp_default, mapper.kd_default)
    # After clear, kp should be non-zero
    assert kp_out[0] > 0.0


def test_watchdog_mode_zeroes_kp(safety, mapper):
    safety._q_prev = np.zeros(29, dtype=np.float32)
    safety.tick()
    time.sleep(0.025)
    q = np.zeros(29, dtype=np.float32)
    _, kp_out, _ = safety.filter(q, q, mapper.kp_default, mapper.kd_default)
    np.testing.assert_array_equal(kp_out, np.zeros(29))
