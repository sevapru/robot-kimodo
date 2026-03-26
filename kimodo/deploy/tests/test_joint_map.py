import numpy as np
import pytest
from kimodo.deploy.joint_map import JointMapper, NUM_JOINTS, JOINT_GROUPS, CONTROL_HZ, MOTION_FPS


def test_num_joints():
    assert NUM_JOINTS == 29


def test_joint_groups_cover_all_joints():
    covered = set()
    for name, (lo, hi, kp, kd, dq) in JOINT_GROUPS.items():
        for i in range(lo, hi):
            covered.add(i)
    assert covered == set(range(NUM_JOINTS))


def test_qpos_to_joints_extracts_correct_slice():
    mapper = JointMapper()
    qpos = np.arange(36, dtype=np.float32)
    joints = mapper.qpos_to_joints(qpos)
    assert joints.shape == (29,)
    np.testing.assert_array_equal(joints, qpos[7:])


def test_qpos_to_joints_wrong_shape_raises():
    mapper = JointMapper()
    with pytest.raises(AssertionError):
        mapper.qpos_to_joints(np.zeros(35))


def test_interpolate_at_zero_returns_start():
    mapper = JointMapper()
    q0 = np.zeros(29, dtype=np.float32)
    q1 = np.ones(29, dtype=np.float32)
    result = mapper.interpolate(q0, q1, 0.0)
    np.testing.assert_array_almost_equal(result, q0)


def test_interpolate_at_one_returns_end():
    mapper = JointMapper()
    q0 = np.zeros(29, dtype=np.float32)
    q1 = np.ones(29, dtype=np.float32)
    result = mapper.interpolate(q0, q1, 1.0)
    np.testing.assert_array_almost_equal(result, q1)


def test_interpolate_midpoint():
    mapper = JointMapper()
    q0 = np.zeros(29, dtype=np.float32)
    q1 = np.full(29, 2.0, dtype=np.float32)
    result = mapper.interpolate(q0, q1, 0.5)
    np.testing.assert_array_almost_equal(result, np.ones(29))


def test_cosine_blend_at_zero_returns_a():
    mapper = JointMapper()
    qa = np.zeros(29, dtype=np.float32)
    qb = np.ones(29, dtype=np.float32)
    result = mapper.cosine_blend(qa, qb, 0.0)
    np.testing.assert_array_almost_equal(result, qa)


def test_cosine_blend_at_one_returns_b():
    mapper = JointMapper()
    qa = np.zeros(29, dtype=np.float32)
    qb = np.ones(29, dtype=np.float32)
    result = mapper.cosine_blend(qa, qb, 1.0)
    np.testing.assert_array_almost_equal(result, qb)


def test_cosine_blend_is_smooth():
    """Cosine blend output should be monotone for monotone inputs."""
    mapper = JointMapper()
    qa = np.zeros(29, dtype=np.float32)
    qb = np.ones(29, dtype=np.float32)
    alphas = np.linspace(0, 1, 20)
    values = [mapper.cosine_blend(qa, qb, a)[0] for a in alphas]
    for i in range(1, len(values)):
        assert values[i] >= values[i - 1]


def test_default_kp_kd_shapes():
    mapper = JointMapper()
    assert mapper.kp_default.shape == (29,)
    assert mapper.kd_default.shape == (29,)
    assert mapper.max_dq.shape == (29,)


def test_leg_kp_higher_than_arm_kp():
    mapper = JointMapper()
    leg_kp = mapper.kp_default[0]   # left_hip_pitch
    arm_kp = mapper.kp_default[15]  # left_shoulder_pitch
    assert leg_kp > arm_kp
