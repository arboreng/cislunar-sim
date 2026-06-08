"""Thruster tests: ion thruster, plume impingement, startup lag, propellant timing, RCS."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from numpy.typing import NDArray

from cislunar.physics import (
    Action,
    GravityModel,
    IntegratorConfig,
    IonThrusterModel,
    SolarSailModel,
    Spacecraft,
    SpacecraftState,
)
from cislunar.physics.constants import AU, Body
from cislunar.physics.constants import Thruster as ThrConst


# ══════════════════════════════════════════════════════════════════════════════
# Section 4 — Ion thruster
# ══════════════════════════════════════════════════════════════════════════════
class TestIonThruster:
    @pytest.fixture
    def warm_thruster(self):
        """A thruster pre-warmed past its full warmup time."""
        thr = IonThrusterModel()
        for _ in range(int(thr.warmup_time_s) * 2):
            thr.tick(1.0, 1.0)
        return thr

    def test_full_throttle_produces_thrust(self, warm_thruster):
        F, _ = warm_thruster.thrust(1.0, available_power_w=100.0, propellant_kg=0.5)
        assert F > 0, f"F={F * 1e3:.3f} mN"

    def test_mass_flow_positive_at_full_throttle(self, warm_thruster):
        _, mdot = warm_thruster.thrust(1.0, 100.0, 0.5)
        assert mdot > 0, f"mdot={mdot}"

    def test_zero_propellant_gives_zero_thrust(self, warm_thruster):
        F, mdot = warm_thruster.thrust(1.0, 100.0, propellant_kg=0.0)
        assert F == 0.0 and mdot == 0.0, f"F={F}, mdot={mdot}"

    def test_power_limited_below_max(self, warm_thruster):
        """A starved power budget reduces thrust below the rated maximum."""
        F, _ = warm_thruster.thrust(1.0, available_power_w=0.1, propellant_kg=0.5)
        assert warm_thruster.max_thrust_n > F, f"F={F:.3e}"

    def test_delta_v_positive_with_propellant(self, warm_thruster):
        dv = warm_thruster.delta_v_remaining(propellant_kg=0.5, dry_mass_kg=12.0)
        assert dv > 0, f"Δv={dv:.1f} m/s"


# ══════════════════════════════════════════════════════════════════════════════
# Plume impingement
# ══════════════════════════════════════════════════════════════════════════════
class TestPlumeImpingement:
    _src_pos = np.zeros(3)
    _plume_axis = np.array([1.0, 0.0, 0.0])
    _full_F = ThrConst.MAX_THRUST
    _tgt_area = 32.0

    @staticmethod
    def _front_face_toward(src_to_tgt: NDArray) -> NDArray:
        r_hat = src_to_tgt / np.linalg.norm(src_to_tgt)
        return -r_hat

    def test_zero_thrust_gives_zero_force(self):
        from cislunar.physics.forces.plume import plume_impingement_force

        tgt_pos = np.array([100.0, 0.0, 0.0])
        F = plume_impingement_force(
            source_pos=self._src_pos,
            plume_axis=self._plume_axis,
            thrust_mag_n=0.0,
            target_pos=tgt_pos,
            target_normal=self._front_face_toward(tgt_pos - self._src_pos),
            target_area=self._tgt_area,
        )
        assert np.array_equal(F, np.zeros(3))

    def test_target_behind_thruster_is_zero(self):
        """Target at θ > 90° from plume axis receives no force (first gate)."""
        from cislunar.physics.forces.plume import plume_impingement_force

        tgt_pos = np.array([-100.0, 0.0, 0.0])
        F = plume_impingement_force(
            source_pos=self._src_pos,
            plume_axis=self._plume_axis,
            thrust_mag_n=self._full_F,
            target_pos=tgt_pos,
            target_normal=self._front_face_toward(tgt_pos - self._src_pos),
            target_area=self._tgt_area,
        )
        assert np.array_equal(F, np.zeros(3))

    def test_target_back_face_is_zero(self):
        from cislunar.physics.forces.plume import plume_impingement_force

        tgt_pos = np.array([100.0, 0.0, 0.0])
        back_normal = np.array([1.0, 0.0, 0.0])
        F = plume_impingement_force(
            source_pos=self._src_pos,
            plume_axis=self._plume_axis,
            thrust_mag_n=self._full_F,
            target_pos=tgt_pos,
            target_normal=back_normal,
            target_area=self._tgt_area,
        )
        assert np.array_equal(F, np.zeros(3))

    def test_inverse_square_falloff_on_axis(self):
        """On-axis force magnitude falls as 1/r²."""
        from cislunar.physics.forces.plume import plume_impingement_force

        tgt_near = np.array([100.0, 0.0, 0.0])
        tgt_far = np.array([200.0, 0.0, 0.0])
        n_hat = self._front_face_toward(tgt_near - self._src_pos)

        F_near = plume_impingement_force(
            source_pos=self._src_pos,
            plume_axis=self._plume_axis,
            thrust_mag_n=self._full_F,
            target_pos=tgt_near,
            target_normal=n_hat,
            target_area=self._tgt_area,
        )
        F_far = plume_impingement_force(
            source_pos=self._src_pos,
            plume_axis=self._plume_axis,
            thrust_mag_n=self._full_F,
            target_pos=tgt_far,
            target_normal=n_hat,
            target_area=self._tgt_area,
        )
        assert np.linalg.norm(F_near) > 0.0
        assert F_near[0] > 0.0 and F_far[0] > 0.0
        ratio = np.linalg.norm(F_near) / np.linalg.norm(F_far)
        assert ratio == pytest.approx(4.0, rel=1e-12)

    def test_cosine_exponent_angular_falloff(self):
        from cislunar.physics.forces.plume import plume_impingement_force

        r = 100.0
        tgt_on = np.array([r, 0.0, 0.0])
        theta = math.radians(30.0)
        tgt_off = np.array([r * math.cos(theta), r * math.sin(theta), 0.0])

        F_on = plume_impingement_force(
            source_pos=self._src_pos,
            plume_axis=self._plume_axis,
            thrust_mag_n=self._full_F,
            target_pos=tgt_on,
            target_normal=self._front_face_toward(tgt_on - self._src_pos),
            target_area=self._tgt_area,
        )
        F_off = plume_impingement_force(
            source_pos=self._src_pos,
            plume_axis=self._plume_axis,
            thrust_mag_n=self._full_F,
            target_pos=tgt_off,
            target_normal=self._front_face_toward(tgt_off - self._src_pos),
            target_area=self._tgt_area,
        )
        assert np.linalg.norm(F_off) > 0.0
        assert np.linalg.norm(F_off) < np.linalg.norm(F_on)
        ratio = np.linalg.norm(F_off) / np.linalg.norm(F_on)
        expected = math.cos(theta) ** ThrConst.PLUME_DIVERGENCE_EXPONENT
        assert ratio == pytest.approx(expected, rel=1e-12)


# ══════════════════════════════════════════════════════════════════════════════
# Section 21 — Hall thruster throttle startup lag
# ══════════════════════════════════════════════════════════════════════════════
class TestThrottleStartupLag:
    """Cold-start, warmup ramp, shutdown, and re-ignition behaviour."""

    def test_cold_thruster_zero_thrust_at_ignition(self):
        thr = IonThrusterModel()
        F, _ = thr.thrust(1.0, 100.0, 1.0)
        assert F == 0.0, f"F_cold={F:.2e} N"

    def test_half_warmup_gives_about_half_thrust(self):
        """At t = warmup_time_s / 2, thrust ≈ 50 % of steady-state."""
        thr = IonThrusterModel(warmup_time_s=60.0)
        for _ in range(30):
            thr.tick(1.0, 1.0)
        F_half, _ = thr.thrust(1.0, 500.0, 1.0)
        F_nominal = thr.max_thrust_n
        assert 0.40 < F_half / F_nominal < 0.60, (
            f"F/F_nominal = {F_half / F_nominal:.3f} (expect ~0.5)"
        )

    def test_double_warmup_reaches_steady_state(self):
        """After 2 × warmup_time_s, thrust is ≥ 99 % of power-limited peak."""
        thr = IonThrusterModel(warmup_time_s=60.0)
        for _ in range(120):
            thr.tick(1.0, 1.0)
        F_warm, _ = thr.thrust(1.0, 500.0, 1.0)
        F_peak = min(thr.max_thrust_n, thr.max_power_w / thr.power_per_newton)
        assert F_warm / F_peak >= 0.99, f"F/F_peak = {F_warm / F_peak:.3f} (expect ≥ 0.99)"

    def test_throttle_cut_shuts_down_within_5s(self):
        thr = IonThrusterModel(warmup_time_s=60.0)
        for _ in range(120):
            thr.tick(1.0, 1.0)
        for _ in range(5):
            thr.tick(0.0, 1.0)
        assert thr._warmup_frac < 1e-9, f"warmup_frac after cutoff={thr._warmup_frac:.2e}"

    def test_reignition_starts_cold(self):
        """After a full shutdown, re-igniting requires another full warmup."""
        thr = IonThrusterModel(warmup_time_s=60.0)
        for _ in range(120):
            thr.tick(1.0, 1.0)
        for _ in range(10):
            thr.tick(0.0, 1.0)
        F, _ = thr.thrust(1.0, 100.0, 1.0)
        assert F == 0.0, f"F at re-ignition={F:.2e} N"

    def test_single_tick_advances_warmup_fraction(self):
        """A single 30-s tick at full throttle advances warmup_frac to 0.5."""
        thr = IonThrusterModel(warmup_time_s=60.0)
        thr.tick(1.0, 30.0)
        assert abs(thr._warmup_frac - 0.5) < 1e-9, f"warmup_frac={thr._warmup_frac:.4f}"

    def test_warmup_time_constant_in_physical_range(self):
        from cislunar.physics.constants import Thruster as ThrConst

        assert 30.0 <= ThrConst.WARMUP_DELAY_S <= 120.0, (
            f"WARMUP_DELAY_S={ThrConst.WARMUP_DELAY_S} s (expect 30–120)"
        )

    def test_warmup_ramp_is_linear(self):
        """_warmup_frac grows at exactly 1/warmup_time_s per second — linear ramp."""
        warmup_s = 60.0
        thr = IonThrusterModel(warmup_time_s=warmup_s)
        fractions = []
        dt = 6.0  # 10 equal steps across the warmup window
        for _ in range(10):
            thr.tick(1.0, dt)
            fractions.append(thr._warmup_frac)
        expected = [min(1.0, (i + 1) * dt / warmup_s) for i in range(10)]
        for i, (got, exp) in enumerate(zip(fractions, expected)):
            assert abs(got - exp) < 1e-12, f"step {i}: frac={got:.6f} expected={exp:.6f}"

    def test_warmup_clamps_at_one(self):
        """_warmup_frac never exceeds 1.0 regardless of how many ticks are applied."""
        thr = IonThrusterModel(warmup_time_s=60.0)
        for _ in range(300):
            thr.tick(1.0, 1.0)
        assert thr._warmup_frac == 1.0, f"frac={thr._warmup_frac}"


# ══════════════════════════════════════════════════════════════════════════════
# Thruster propellant timing
# ══════════════════════════════════════════════════════════════════════════════
class TestThrusterPropellantTiming:
    """Accepted-step propellant use must match the thrust state that was integrated."""

    @staticmethod
    def _make_thruster_timing_sc() -> tuple[Spacecraft, IonThrusterModel]:
        class _FixedPower:
            def step(self, **kwargs):
                return 100.0

        state = SpacecraftState(
            position=np.array([Body.R_EARTH + 400e3, 0.0, 0.0]),
            velocity=np.zeros(3),
            propellant_kg=0.5,
            mass_dry_kg=12.0,
            power_w=100.0,
        )
        thr = IonThrusterModel(warmup_time_s=60.0)
        sc = Spacecraft(
            initial_state=state,
            sail_model=SolarSailModel(area_m2=0.0),
            thruster_model=thr,
            gravity_model=GravityModel().add_body("Earth", mu=0.0),
            sun_pos_fn=lambda t: np.array([AU, 0.0, 0.0]),
            power_model=_FixedPower(),
            integrator_cfg=IntegratorConfig(
                atol=1e-9,
                rtol=1e-9,
                dt_min_s=30.0,
                dt_max_s=30.0,
                dt_init_s=30.0,
            ),
        )
        return sc, thr

    @staticmethod
    def _burn_action() -> Action:
        return Action(
            attitude_dir_cmd=np.array([1.0, 0.0, 0.0]),
            thrust_dir=np.array([1.0, 0.0, 0.0]),
            throttle=1.0,
        )

    def test_cold_start_step_does_not_consume_propellant_before_impulse(self):
        sc, thr = self._make_thruster_timing_sc()
        prop_before = sc.state.propellant_kg

        sc.step(self._burn_action(), dt_requested=30.0)

        assert sc.state.propellant_kg == pytest.approx(prop_before, abs=1e-15)
        assert thr._warmup_frac == pytest.approx(0.5, abs=1e-15)

    def test_second_step_consumes_propellant_after_warmup_advances(self):
        sc, _ = self._make_thruster_timing_sc()
        action = self._burn_action()

        sc.step(action, dt_requested=30.0)
        prop_after_first = sc.state.propellant_kg
        sc.step(action, dt_requested=30.0)

        assert sc.state.propellant_kg < prop_after_first


# ══════════════════════════════════════════════════════════════════════════════
# ColdGasRCSModel
# ══════════════════════════════════════════════════════════════════════════════
class TestColdGasRCSModel:
    """Unit tests for physics/forces/rcs.py — ColdGasRCSModel."""

    from cislunar.physics.constants import RCS as RCSConst
    from cislunar.physics.forces.rcs import ColdGasRCSModel, RCSConfig

    PROP = 0.15

    def _model(self):
        from cislunar.physics.forces.rcs import ColdGasRCSModel

        return ColdGasRCSModel()

    def test_pitch_command_gives_positive_y_torque(self):
        """d = +y → τ_y > 0, τ_x ≈ 0, τ_z ≈ 0."""
        m = self._model()
        tau, force, _ = m.fire(np.array([0.0, 1.0, 0.0]), True, 1.0, self.PROP)
        assert tau[1] > 0.0, f"τ_y={tau[1]:.2e}"
        assert abs(tau[0]) < 1e-12, f"τ_x leak={tau[0]:.2e}"
        assert abs(tau[2]) < 1e-12, f"τ_z leak={tau[2]:.2e}"

    def test_yaw_command_gives_positive_x_torque(self):
        """d = +x → τ_x > 0, τ_y ≈ 0, τ_z ≈ 0."""
        m = self._model()
        tau, _, _ = m.fire(np.array([1.0, 0.0, 0.0]), True, 1.0, self.PROP)
        assert tau[0] > 0.0, f"τ_x={tau[0]:.2e}"
        assert abs(tau[1]) < 1e-12
        assert abs(tau[2]) < 1e-12

    def test_roll_command_gives_positive_z_torque(self):
        """d = +z → τ_z > 0, τ_x ≈ 0, τ_y ≈ 0."""
        m = self._model()
        tau, force, _ = m.fire(np.array([0.0, 0.0, 1.0]), True, 1.0, self.PROP)
        assert tau[2] > 0.0, f"τ_z={tau[2]:.2e}"
        assert abs(tau[0]) < 1e-12
        assert abs(tau[1]) < 1e-12

    def test_roll_force_is_near_zero(self):
        """Pure roll command: symmetric mid-body pair → net force ≈ 0."""
        m = self._model()
        _, force, _ = m.fire(np.array([0.0, 0.0, 1.0]), True, 1.0, self.PROP)
        assert np.linalg.norm(force) < 1e-12, f"|force|={np.linalg.norm(force):.2e}"

    def test_pitch_force_is_lateral_x(self):
        """Pitch nozzle fires laterally → net force in +x, no y or z."""
        m = self._model()
        _, force, _ = m.fire(np.array([0.0, 1.0, 0.0]), True, 1.0, self.PROP)
        assert force[0] > 0.0, f"F_x={force[0]:.2e}"
        assert abs(force[1]) < 1e-12
        assert abs(force[2]) < 1e-12

    def test_torque_magnitude_matches_geometry(self):
        """Full-authority pitch: τ_y = F_per_nozzle × NOSE_ARM_M."""
        from cislunar.physics.constants import RCS as RCSConst

        m = self._model()
        tau, _, _ = m.fire(np.array([0.0, 1.0, 0.0]), True, 1.0, self.PROP)
        expected = RCSConst.F_PER_NOZZLE_N * RCSConst.NOSE_ARM_M
        assert abs(tau[1] - expected) < 1e-12, f"τ_y={tau[1]:.4e} expected={expected:.4e}"

    def test_roll_torque_uses_two_nozzles(self):
        """Full-authority roll: τ_z = 2 × F_per_nozzle × BODY_ARM_M."""
        from cislunar.physics.constants import RCS as RCSConst

        m = self._model()
        tau, _, _ = m.fire(np.array([0.0, 0.0, 1.0]), True, 1.0, self.PROP)
        expected = 2.0 * RCSConst.F_PER_NOZZLE_N * RCSConst.BODY_ARM_M
        assert abs(tau[2] - expected) < 1e-12, f"τ_z={tau[2]:.4e} expected={expected:.4e}"

    def test_negative_command_gives_negative_torque(self):
        """Direction reversal flips torque sign."""
        m = self._model()
        tau_pos, _, _ = m.fire(np.array([0.0, 1.0, 0.0]), True, 1.0, self.PROP)
        tau_neg, _, _ = m.fire(np.array([0.0, -1.0, 0.0]), True, 1.0, self.PROP)
        assert tau_pos[1] > 0.0
        assert tau_neg[1] < 0.0
        assert abs(tau_pos[1] + tau_neg[1]) < 1e-12

    def test_non_unit_command_normalized(self):
        """Command magnitude does not affect torque beyond direction."""
        m = self._model()
        tau1, _, _ = m.fire(np.array([0.0, 1.0, 0.0]), True, 1.0, self.PROP)
        tau5, _, _ = m.fire(np.array([0.0, 5.0, 0.0]), True, 1.0, self.PROP)
        np.testing.assert_allclose(tau1, tau5, atol=1e-12)

    def test_propellant_consumed_is_positive(self):
        m = self._model()
        _, _, prop = m.fire(np.array([0.0, 1.0, 0.0]), True, 60.0, self.PROP)
        assert prop > 0.0, f"prop_consumed={prop}"

    def test_propellant_consumed_does_not_exceed_available(self):
        """Even with a long dt, consumption is capped at available propellant."""
        m = self._model()
        tiny_prop = 1e-6
        _, _, prop = m.fire(np.array([1.0, 1.0, 1.0]), True, 1000.0, tiny_prop)
        assert prop <= tiny_prop + 1e-15, f"consumed {prop:.2e} > available {tiny_prop:.2e}"

    def test_near_empty_tank_scales_output(self):
        """Output is scaled down proportionally when nearly out of propellant."""
        m = self._model()
        tiny_prop = 1e-8
        tau_full, _, _ = m.fire(np.array([0.0, 1.0, 0.0]), True, 1.0, self.PROP)
        tau_tiny, _, _ = m.fire(np.array([0.0, 1.0, 0.0]), True, 1.0, tiny_prop)
        assert np.linalg.norm(tau_tiny) < np.linalg.norm(tau_full)

    def test_zero_propellant_returns_zero_output(self):
        """Empty tank → zero torque, zero force, zero consumption."""
        m = self._model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            tau, force, prop = m.fire(np.array([0.0, 1.0, 0.0]), True, 1.0, 0.0)
        assert np.all(tau == 0.0)
        assert np.all(force == 0.0)
        assert prop == 0.0

    def test_zero_propellant_emits_warning(self):
        """Trigger with empty tank emits a UserWarning."""
        m = self._model()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m.fire(np.array([0.0, 1.0, 0.0]), True, 1.0, 0.0)
        assert any(issubclass(w.category, UserWarning) for w in caught)

    def test_no_trigger_returns_zero_silently(self):
        """rcs_trigger=False: zero output, no warning even with full tank."""
        m = self._model()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tau, force, prop = m.fire(np.array([0.0, 1.0, 0.0]), False, 1.0, self.PROP)
        assert np.all(tau == 0.0)
        assert np.all(force == 0.0)
        assert prop == 0.0
        assert not any(issubclass(w.category, UserWarning) for w in caught)

    def test_zero_direction_returns_zero(self):
        """Zero command vector → no output (avoids divide-by-zero)."""
        m = self._model()
        tau, force, prop = m.fire(np.zeros(3), True, 1.0, self.PROP)
        assert np.all(tau == 0.0)
        assert np.all(force == 0.0)
        assert prop == 0.0
