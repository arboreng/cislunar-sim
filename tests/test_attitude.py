"""Attitude dynamics tests: quaternion math, slew control, wheels, saturation taper."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cislunar.physics import (
    Action,
    AttitudeConfig,
    AttitudeDynamicsModel,
)
from cislunar.physics.constants import Body
from cislunar.physics.forces.attitude import (
    attitude_error_quat,
    body_to_inertial,
    cubesat_inertia_12u,
    inertial_to_body,
    quat_conjugate,
    quat_multiply,
    quat_normalize,
    quat_to_dcm,
)


# ══════════════════════════════════════════════════════════════════════════════
# Section 11 — Attitude dynamics
# ══════════════════════════════════════════════════════════════════════════════
class TestAttitudeDynamics:
    """Inertia tensor, quaternion kinematics, slew dynamics, gravity-gradient."""

    def test_inertia_is_3x3(self):
        I = cubesat_inertia_12u(12.0)
        assert I.shape == (3, 3)

    def test_inertia_is_diagonal(self):
        I = cubesat_inertia_12u(12.0)
        assert np.allclose(I, np.diag(np.diag(I)))

    def test_all_moments_positive(self):
        I = cubesat_inertia_12u(12.0)
        assert np.all(np.diag(I) > 0), f"diag(I)={np.diag(I)}"

    def test_long_axis_has_smallest_inertia(self):
        """A 12U CubeSat's long axis (z) is the most slender → smallest Izz."""
        I = cubesat_inertia_12u(12.0)
        assert np.diag(I)[2] < np.diag(I)[0], f"diag(I)={np.diag(I)}"

    def test_identity_times_q_is_q(self):
        q_id = np.array([1.0, 0.0, 0.0, 0.0])
        q_90x = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
        assert np.allclose(quat_multiply(q_id, q_90x), q_90x, atol=1e-9)

    def test_dcm_of_90deg_x_rotation(self):
        """A 90° rotation about x maps body-z to inertial -y."""
        q_90x = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
        dcm = quat_to_dcm(q_90x)
        assert np.allclose(dcm[:, 2], [0.0, -1.0, 0.0], atol=1e-6), f"dcm[:,2]={dcm[:, 2]}"

    def test_quat_conjugate_times_self_is_identity(self):
        """q ⊗ q* = [1,0,0,0] for any unit quaternion."""
        q = quat_normalize(np.array([0.5, 0.3, 0.7, 0.2]))
        result = quat_multiply(q, quat_conjugate(q))
        assert np.allclose(result, [1.0, 0.0, 0.0, 0.0], atol=1e-12), f"q⊗q*={result}"

    def test_body_to_inertial_roundtrip(self):
        """inertial_to_body(body_to_inertial(v, q), q) = v."""
        q = quat_normalize(np.array([0.5, 0.3, 0.7, 0.2]))
        v = np.array([1.0, 2.0, 3.0])
        v_rt = inertial_to_body(body_to_inertial(v, q), q)
        np.testing.assert_allclose(v_rt, v, atol=1e-12)

    def test_body_to_inertial_identity_quat(self):
        """With identity quaternion, frame rotation is a no-op."""
        q_id = np.array([1.0, 0.0, 0.0, 0.0])
        v = np.array([1.0, 2.0, 3.0])
        assert np.allclose(body_to_inertial(v, q_id), v, atol=1e-12)

    def test_initial_sail_normal_is_plus_x(self):
        att = AttitudeDynamicsModel()
        assert np.allclose(att.sail_normal_inertial, [1.0, 0.0, 0.0], atol=1e-9)

    def test_wheels_start_at_zero_momentum(self):
        att = AttitudeDynamicsModel()
        assert np.allclose(att.h_wheels, 0.0), f"h={att.h_wheels}"

    def test_sail_normal_to_quaternion_returns_unit(self):
        att = AttitudeDynamicsModel()
        q = att.sail_normal_to_quaternion(np.array([0.0, 1.0, 0.0]))
        assert abs(np.linalg.norm(q) - 1.0) < 1e-9

    def test_sail_normal_moves_toward_target_in_first_step(self):
        att = AttitudeDynamicsModel()
        target = np.array([0.0, 1.0, 0.0])
        q_target = att.sail_normal_to_quaternion(target)
        n_before = att.sail_normal_inertial.copy()
        att.step(q_target, dt=60.0)
        movement = np.linalg.norm(att.sail_normal_inertial - n_before)
        assert movement > 1e-4, f"moved={movement:.6f} rad"

    def test_sail_makes_progress_over_one_hour(self):
        """After 1 h, residual slew error < π/2 (meaningful progress toward target)."""
        att = AttitudeDynamicsModel()
        target = np.array([0.0, 1.0, 0.0])
        q_target = att.sail_normal_to_quaternion(target)
        for _ in range(60):
            att.step(q_target, dt=60.0)
        err = np.linalg.norm(att.sail_normal_inertial - target)
        assert err < math.pi / 2, f"err={err:.4f} rad"

    def test_large_slew_has_more_residual_error_than_small(self):
        """After 20 min, the big-slew case has a larger attitude error."""
        att_small = AttitudeDynamicsModel()
        att_large = AttitudeDynamicsModel()
        tgt_small = np.array([0.98, 0.2, 0.0]) / np.linalg.norm([0.98, 0.2, 0.0])
        tgt_large = np.array([0.0, 0.0, 1.0])
        q_small = att_small.sail_normal_to_quaternion(tgt_small)
        q_large = att_large.sail_normal_to_quaternion(tgt_large)
        for _ in range(20):
            att_small.step(q_small, dt=60.0)
            att_large.step(q_large, dt=60.0)
        err_small = np.linalg.norm(att_small.sail_normal_inertial - tgt_small)
        err_large = np.linalg.norm(att_large.sail_normal_inertial - tgt_large)
        assert err_large > err_small, f"small={err_small:.3f}  large={err_large:.3f}"

    def test_gravity_gradient_nonzero_for_inclined_orbit(self):
        att = AttitudeDynamicsModel()
        pos_inclined = np.array([5.5e6, 0.0, 3.8e6])
        tau = att._gravity_gradient_torque(pos_inclined, Body.MU_EARTH)
        assert np.linalg.norm(tau) > 0

    def test_gravity_gradient_magnitude_physically_reasonable(self):
        att = AttitudeDynamicsModel()
        pos_inclined = np.array([5.5e6, 0.0, 3.8e6])
        tau_mag = np.linalg.norm(att._gravity_gradient_torque(pos_inclined, Body.MU_EARTH))
        assert 1e-10 < tau_mag < 1e-3, f"|τ_gg|={tau_mag:.2e} N·m"

    def test_make_lunar_craft_has_attitude_model(self, leo_spacecraft):
        assert leo_spacecraft.attitude is not None

    def test_10min_step_with_attitude_completes(self, leo_spacecraft):
        act = Action(attitude_dir_cmd=np.array([0.0, 1.0, 0.0]), throttle=0.0)
        leo_spacecraft.step(act, dt_requested=600.0)
        assert np.all(np.isfinite(leo_spacecraft.state.position))

    def test_sail_normal_changes_under_command(self, leo_spacecraft):
        act = Action(attitude_dir_cmd=np.array([0.0, 1.0, 0.0]), throttle=0.0)
        before = leo_spacecraft.state.sail_normal.copy()
        leo_spacecraft.step(act, dt_requested=600.0)
        assert np.linalg.norm(leo_spacecraft.state.sail_normal - before) > 0


# ══════════════════════════════════════════════════════════════════════════════
# Section 12 — Wheel desaturation & reward
# ══════════════════════════════════════════════════════════════════════════════
class TestWheelDesaturation:
    """Momentum accumulation, desaturate() burn, Action field, and reward penalty."""

    @pytest.fixture
    def saturated_att(self):
        """4-hour attitude hold on an inclined orbit produces partial saturation."""
        att = AttitudeDynamicsModel()
        target_q = att.sail_normal_to_quaternion(np.array([-1.0, 0.0, 0.0]))
        pos_inclined = np.array([5.5e6, 0.0, 3.8e6])
        for _ in range(240):
            att.step(target_q, dt=60.0, position=pos_inclined)
        return att

    def test_4h_hold_accumulates_saturation(self, saturated_att):
        sat = saturated_att.wheel_saturation_fraction
        assert sat > 0, f"sat={sat:.4f}"

    def test_4h_hold_not_fully_saturated(self, saturated_att):
        assert saturated_att.wheel_saturation_fraction < 1.0

    def test_desaturate_returns_positive_dv_cost(self, saturated_att):
        dv = saturated_att.desaturate()
        assert dv > 0, f"Δv={dv:.4f} m/s"

    def test_desaturate_reduces_wheel_momentum(self, saturated_att):
        H_before = np.linalg.norm(saturated_att.A_rw @ saturated_att.h_wheels)
        saturated_att.desaturate()
        H_after = np.linalg.norm(saturated_att.A_rw @ saturated_att.h_wheels)
        assert H_after <= H_before + 1e-12, f"H_before={H_before:.4e}  H_after={H_after:.4e}"

    def test_desaturate_dv_cost_under_10_mm_per_s(self, saturated_att):
        dv = saturated_att.desaturate()
        assert dv < 0.010, f"Δv={dv * 1e3:.3f} mm/s"

    def test_repeated_desaturation_drives_momentum_down(self, saturated_att):
        H0 = np.linalg.norm(saturated_att.A_rw @ saturated_att.h_wheels)
        for _ in range(50):
            saturated_att.desaturate()
        H_final = np.linalg.norm(saturated_att.A_rw @ saturated_att.h_wheels)
        assert H_final < H0, f"H0={H0 * 1e3:.4f}  H_final={H_final * 1e3:.4f} mN·m·s"

    def test_action_has_rcs_trigger_field(self):
        a = Action(attitude_dir_cmd=np.array([-1.0, 0.0, 0.0]), throttle=0.0, rcs_trigger=True)
        assert hasattr(a, "rcs_trigger")

    def test_action_rcs_trigger_true(self):
        a = Action(attitude_dir_cmd=np.array([-1.0, 0.0, 0.0]), throttle=0.0, rcs_trigger=True)
        assert a.rcs_trigger is True

    def test_action_rcs_trigger_default_false(self):
        a = Action(attitude_dir_cmd=np.array([-1.0, 0.0, 0.0]), throttle=0.0)
        assert a.rcs_trigger is False

    def test_desaturation_burn_costs_propellant(self, leo_spacecraft):
        h_max = leo_spacecraft.attitude.cfg.wheel_max_momentum_nms
        leo_spacecraft.attitude.h_wheels = 0.8 * h_max * np.array([1.0, 1.0, 1.0, 1.0])

        act = Action(attitude_dir_cmd=np.array([-1.0, 0.0, 0.0]), throttle=0.0, rcs_trigger=True)
        prop_before = leo_spacecraft.state.propellant_kg
        leo_spacecraft.step(act, dt_requested=60.0)
        dprop = prop_before - leo_spacecraft.state.propellant_kg
        assert dprop > 0, f"Δprop={dprop * 1e9:.3f} ng"


# ══════════════════════════════════════════════════════════════════════════════
# Section 16 — Wheel saturation smooth taper
# ══════════════════════════════════════════════════════════════════════════════
class TestWheelSaturationTaper:
    """Authority tapers smoothly between 85–100 % saturation (not hard-clipped)."""

    @pytest.fixture(scope="class")
    def taper_setup(self):
        cfg = AttitudeConfig()
        h_max = cfg.wheel_max_momentum_nms
        q_target = np.array([math.cos(math.pi / 8), 0.0, 0.0, math.sin(math.pi / 8)])

        ref = AttitudeDynamicsModel(config=cfg)
        ref.q = np.array([1.0, 0.0, 0.0, 0.0])
        ref.omega = np.zeros(3)
        qe = attitude_error_quat(ref.q, quat_normalize(q_target))
        s = 1.0 if qe[0] >= 0 else -1.0
        raw = np.clip(
            -(ref.A_rw_pinv @ (-cfg.kp * s * qe[1:4])),
            -cfg.wheel_max_torque_nm,
            cfg.wheel_max_torque_nm,
        )
        dh_sign = np.sign(raw)
        return dict(cfg=cfg, h_max=h_max, q_target=q_target, dh_sign=dh_sign)

    @staticmethod
    def _tau_at(frac: float, setup) -> float:
        m = AttitudeDynamicsModel(config=setup["cfg"])
        m.q = np.array([1.0, 0.0, 0.0, 0.0])
        m.omega = np.zeros(3)
        m.h_wheels = frac * setup["h_max"] * setup["dh_sign"]
        return float(np.linalg.norm(m._pd_torque(setup["q_target"])))

    def test_full_authority_below_85_percent(self, taper_setup):
        t0 = self._tau_at(0.0, taper_setup)
        t80 = self._tau_at(0.80, taper_setup)
        assert abs(t80 - t0) < t0 * 1e-6, f"|τ|@80%={t80:.3e}  |τ|@0%={t0:.3e}"

    def test_taper_begins_above_85_percent(self, taper_setup):
        t80 = self._tau_at(0.80, taper_setup)
        t87 = self._tau_at(0.87, taper_setup)
        assert t87 < t80, f"|τ|@87%={t87:.3e}  |τ|@80%={t80:.3e}"

    def test_taper_continues_toward_saturation(self, taper_setup):
        t87 = self._tau_at(0.87, taper_setup)
        t94 = self._tau_at(0.94, taper_setup)
        assert t94 < t87, f"|τ|@94%={t94:.3e}  |τ|@87%={t87:.3e}"

    def test_authority_reaches_zero_at_full_saturation(self, taper_setup):
        t0 = self._tau_at(0.0, taper_setup)
        t100 = self._tau_at(1.00, taper_setup)
        assert t100 < t0 * 0.01, f"|τ|@100%={t100:.3e} — must be <1% of {t0:.3e}"

    def test_opposing_direction_retains_authority_when_saturated(self, taper_setup):
        """Saturation in the +dir does not impair torque in the −dir."""
        cfg = taper_setup["cfg"]
        h_max = taper_setup["h_max"]
        dh_sign = taper_setup["dh_sign"]
        q_opp = np.array([math.cos(math.pi / 8), 0.0, 0.0, -math.sin(math.pi / 8)])

        m_sat = AttitudeDynamicsModel(config=cfg)
        m_sat.q = np.array([1.0, 0.0, 0.0, 0.0])
        m_sat.omega = np.zeros(3)
        m_sat.h_wheels = h_max * dh_sign

        m_free = AttitudeDynamicsModel(config=cfg)
        m_free.q = np.array([1.0, 0.0, 0.0, 0.0])
        m_free.omega = np.zeros(3)

        tau_sat = float(np.linalg.norm(m_sat._pd_torque(q_opp)))
        tau_free = float(np.linalg.norm(m_free._pd_torque(q_opp)))
        assert abs(tau_sat - tau_free) < tau_free * 1e-6, (
            f"|τ_sat|={tau_sat:.3e}  |τ_free|={tau_free:.3e}"
        )

    def test_wheel_momentum_never_exceeds_h_max(self, taper_setup):
        """Over 200 × 5 s large-dt steps, no wheel momentum exceeds h_max."""
        cfg = taper_setup["cfg"]
        m = AttitudeDynamicsModel(config=cfg)
        m.q = np.array([1.0, 0.0, 0.0, 0.0])
        m.h_wheels = np.zeros(4)
        for _ in range(200):
            m.step(taper_setup["q_target"], dt=5.0)
        assert not np.any(np.abs(m.h_wheels) > taper_setup["h_max"] * 1.001), (
            f"max|h|={np.max(np.abs(m.h_wheels)):.5f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Gap-closing additions
# ══════════════════════════════════════════════════════════════════════════════
class TestSailNormalFromAttitude:
    def test_identity_quaternion_returns_body_axis(self):
        from cislunar.physics.forces.attitude import sail_normal_from_attitude

        q = np.array([1.0, 0.0, 0.0, 0.0])
        axis = np.array([1.0, 0.0, 0.0])
        n = sail_normal_from_attitude(q, axis)
        np.testing.assert_allclose(n, axis, atol=1e-12)

    def test_90deg_rotation_transforms_axis(self):
        from cislunar.physics.forces.attitude import sail_normal_from_attitude

        # 90° around z: x → y
        q = np.array([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])
        axis = np.array([1.0, 0.0, 0.0])
        n = sail_normal_from_attitude(q, axis)
        np.testing.assert_allclose(n, [0.0, 1.0, 0.0], atol=1e-12)


class TestAttitudeDynamicsReset:
    def test_reset_zeros_angular_velocity(self):
        m = AttitudeDynamicsModel()
        m.omega = np.array([0.1, 0.2, 0.3])
        m.reset()
        np.testing.assert_array_equal(m.omega, np.zeros(3))

    def test_reset_zeros_wheels(self):
        m = AttitudeDynamicsModel()
        m.h_wheels[:] = 0.5
        m.reset()
        np.testing.assert_array_equal(m.h_wheels, np.zeros(4))

    def test_reset_with_q_init_sets_quaternion(self):
        m = AttitudeDynamicsModel()
        q = np.array([0.0, 1.0, 0.0, 0.0])
        m.reset(q_init=q)
        np.testing.assert_allclose(np.abs(m.q), np.abs(quat_normalize(q)), atol=1e-12)

    def test_reset_without_q_init_uses_identity(self):
        m = AttitudeDynamicsModel()
        m.reset()
        np.testing.assert_allclose(m.q, [1.0, 0.0, 0.0, 0.0], atol=1e-12)


class TestDesaturation:
    def test_no_momentum_returns_zero_cost(self):
        m = AttitudeDynamicsModel()
        # Fresh model has zero wheels → h_mag < 1e-10 → return 0.0
        cost = m.desaturate()
        assert cost == pytest.approx(0.0)

    def test_saturated_wheels_returns_positive_cost(self):
        m = AttitudeDynamicsModel()
        m.h_wheels[:] = 0.05  # non-zero momentum
        cost = m.desaturate()
        assert cost >= 0.0


class TestRcsCraftType:
    def test_rcs_step_keeps_h_wheels_zero(self):
        """Craft type 'rcs' should not update wheel momentum."""
        m = AttitudeDynamicsModel()
        m.h_wheels[:] = 0.0
        # step() requires a quaternion command, not a unit vector
        q_target = np.array([1.0, 0.0, 0.0, 0.0])
        m.step(q_target, dt=1.0, craft_type="rcs", srp_force_eci=np.zeros(3))
        np.testing.assert_array_equal(m.h_wheels, np.zeros(4))


class TestAlignToTargetAntiParallel:
    def test_anti_aligned_sail_step_does_not_crash(self):
        """180° anti-aligned case: step() with anti-parallel command should not raise."""
        m = AttitudeDynamicsModel()
        # Build a quaternion that rotates the body sail_axis by 180° around z
        q_180_z = np.array([0.0, 0.0, 0.0, 1.0])
        m.step(q_180_z, dt=0.1, craft_type="wheel", srp_force_eci=np.zeros(3))
        assert abs(np.linalg.norm(m.q) - 1.0) < 1e-6
