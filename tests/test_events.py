"""Event detection tests: checkpoint, surface (terminal), eclipse, bisection."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cislunar.physics import (
    Action,
    CheckpointEvent,
    EclipseEvent,
    GravityModel,
    IntegratorConfig,
    IonThrusterModel,
    PowerBudgetModel,
    SolarSailModel,
    Spacecraft,
    SpacecraftState,
    SurfaceEvent,
)
from cislunar.physics.constants import AU, Body
from cislunar.physics.spacecraft import _bisect_event


def _make_sc_for_events(pos: list[float], vel: list[float]) -> Spacecraft:
    """Minimal Earth-J2 spacecraft used across the event-detection tests."""
    state = SpacecraftState(
        position=np.array(pos, dtype=float),
        velocity=np.array(vel, dtype=float),
        sail_normal=np.array([-1.0, 0.0, 0.0]),
        propellant_kg=0.5,
        mass_dry_kg=12.0,
    )
    return Spacecraft(
        initial_state=state,
        sail_model=SolarSailModel(area_m2=32.0),
        thruster_model=IonThrusterModel(),
        gravity_model=GravityModel().add_body(
            "Earth", mu=Body.MU_EARTH, j2=Body.J2_EARTH, r_body=Body.R_EARTH
        ),
        sun_pos_fn=lambda t: np.array([AU, 0.0, 0.0]),
        power_model=PowerBudgetModel(),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Section 22 — Event detection in integrator (bisection + terminal events)
# ══════════════════════════════════════════════════════════════════════════════
class TestCheckpointEvent:
    """Gate-crossing detection via mid-step bisection."""

    CP_POS = np.array([7e6, 30e3, 0.0])
    CP_R = 25_000.0

    @pytest.fixture
    def gate_event(self):
        return CheckpointEvent("test_gate", self.CP_POS, self.CP_R, tol_s=1.0)

    @pytest.fixture
    def crossing_sc(self):
        """Spacecraft moving along +y at 3 km/s — crosses the gate early."""
        return _make_sc_for_events(pos=[7e6, 0.0, 0.0], vel=[0.0, 3_000.0, 0.0])

    def test_event_triggers_on_crossing(self, crossing_sc, gate_event):
        _, triggered = crossing_sc.step(Action(), dt_requested=60.0, events=[gate_event])
        assert len(triggered) == 1, f"triggered={len(triggered)}"

    def test_crossing_time_within_2s_of_analytical(self, crossing_sc, gate_event):
        """Analytic entry: y = 30 km − 25 km = 5 km → t ≈ 5 000 / 3 000 = 1.67 s."""
        t0 = crossing_sc.state.time_s
        _, triggered = crossing_sc.step(Action(), dt_requested=60.0, events=[gate_event])
        offset = triggered[0].t_event - t0
        assert abs(offset - (5e3 / 3_000)) < 2.0, (
            f"offset={offset:.2f} s  analytic={5e3 / 3_000:.2f} s"
        )

    def test_event_name_propagates(self, crossing_sc, gate_event):
        _, triggered = crossing_sc.step(Action(), dt_requested=60.0, events=[gate_event])
        assert triggered[0].name == "test_gate"

    def test_no_trigger_when_gate_not_crossed(self, gate_event):
        sc = _make_sc_for_events(pos=[7e6 + 100e3, 0.0, 0.0], vel=[0.0, 1.0, 0.0])
        _, triggered = sc.step(Action(), dt_requested=60.0, events=[gate_event])
        assert len(triggered) == 0


class TestSurfaceEvent:
    """Terminal event: integration stops at Earth reentry threshold."""

    R_REENTRY = Body.R_EARTH + 80e3

    @pytest.fixture
    def reentry_sc(self):
        """Spacecraft 500 m above the reentry threshold, falling at 200 m/s."""
        return _make_sc_for_events(
            pos=[self.R_REENTRY + 500.0, 0.0, 0.0],
            vel=[-200.0, 0.0, 0.0],
        )

    @pytest.fixture
    def surface_event(self):
        return SurfaceEvent("reentry", body_radius_m=self.R_REENTRY, tol_s=1.0)

    def test_terminal_event_triggers(self, reentry_sc, surface_event):
        _, triggered = reentry_sc.step(Action(), dt_requested=60.0, events=[surface_event])
        assert len(triggered) == 1, f"triggered={len(triggered)}"

    def test_impact_time_within_2s_of_analytical(self, reentry_sc, surface_event):
        """500 m / 200 m/s = 2.5 s expected."""
        t0 = reentry_sc.state.time_s
        _, triggered = reentry_sc.step(Action(), dt_requested=60.0, events=[surface_event])
        offset = triggered[0].t_event - t0
        assert abs(offset - 2.5) < 2.0, f"offset={offset:.2f} s"

    def test_terminal_flag_set(self, reentry_sc, surface_event):
        _, triggered = reentry_sc.step(Action(), dt_requested=60.0, events=[surface_event])
        assert triggered[0].terminal

    def test_no_overshoot_past_threshold(self, reentry_sc, surface_event):
        """Final radius must be within 1 km of the reentry threshold."""
        reentry_sc.step(Action(), dt_requested=60.0, events=[surface_event])
        r_final = float(np.linalg.norm(reentry_sc.state.position))
        assert abs(r_final - self.R_REENTRY) < 1_000.0, (
            f"overshoot={r_final - self.R_REENTRY:.0f} m"
        )

    def test_terminal_event_only_debits_propellant_for_elapsed_time(self):
        # Abstract propellant-accounting test: no gravity, no sail, spacecraft
        # at fictitious 10 km radius heading toward a 9.5 km surface threshold.
        # Uses a fixed-power stub so the thruster fires at constant 100 W without
        # eclipse or panel geometry interfering with the accounting assertion.
        class _FixedPower:
            def step(self, **kwargs):
                return 100.0

        state = SpacecraftState(
            position=np.array([10_000.0, 0.0, 0.0]),
            velocity=np.array([-200.0, 0.0, 0.0]),
            propellant_kg=0.5,
            mass_dry_kg=12.0,
            power_w=100.0,
        )
        thr = IonThrusterModel()
        for _ in range(120):
            thr.tick(1.0, 1.0)
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
                dt_min_s=60.0,
                dt_max_s=60.0,
                dt_init_s=60.0,
            ),
        )
        ev = SurfaceEvent("impact", body_radius_m=9_500.0, tol_s=1e-6)

        prop_before = sc.state.propellant_kg
        _, triggered = sc.step(
            Action(thrust_dir=np.array([1.0, 0.0, 0.0]), throttle=1.0),
            dt_requested=60.0,
            events=[ev],
        )

        t_impact = triggered[0].t_event
        _, mdot = thr.thrust(1.0, 100.0, prop_before)
        expected_drop = mdot * t_impact
        actual_drop = prop_before - sc.state.propellant_kg
        assert actual_drop == pytest.approx(expected_drop, rel=1e-6)

    def test_terminal_event_only_advances_power_model_for_elapsed_time(self):
        sun_pos = np.array([AU, 0.0, 0.0])
        state = SpacecraftState(
            position=np.array([-10_000.0, 0.0, 0.0]),
            velocity=np.array([200.0, 0.0, 0.0]),
            propellant_kg=0.5,
            mass_dry_kg=12.0,
            power_w=0.0,
        )
        power = PowerBudgetModel()
        sc = Spacecraft(
            initial_state=state,
            sail_model=SolarSailModel(area_m2=0.0),
            thruster_model=IonThrusterModel(),
            gravity_model=GravityModel().add_body("Earth", mu=0.0),
            integrator_cfg=IntegratorConfig(
                atol=1e-9,
                rtol=1e-9,
                dt_min_s=60.0,
                dt_max_s=60.0,
                dt_init_s=60.0,
            ),
            power_model=power,
            sun_pos_fn=lambda _t: sun_pos,
        )
        ev = SurfaceEvent("impact", body_radius_m=9_500.0, tol_s=1e-6)

        batt_before = power.batt_energy_j
        _, triggered = sc.step(Action(throttle=0.0), dt_requested=60.0, events=[ev])

        t_impact = triggered[0].t_event
        expected_drop = power.P_hk * t_impact / power.eta_batt
        actual_drop = batt_before - power.batt_energy_j
        assert actual_drop == pytest.approx(expected_drop, rel=1e-6)


class TestEclipseEvent:
    """Shadow-entry detection with bisection."""

    @staticmethod
    def _sun_fn(t):
        return np.array([AU, 0.0, 0.0])

    def test_shadow_entry_detected_during_half_orbit(self):
        R_orb = Body.R_EARTH + 500e3
        sc = _make_sc_for_events(
            pos=[R_orb, 0.0, 0.0],
            vel=[0.0, math.sqrt(Body.MU_EARTH / R_orb), 0.0],
        )
        sc._sun_pos_fn = self._sun_fn
        ev = EclipseEvent(
            sun_pos_fn=self._sun_fn,
            body_pos_fn=None,
            body_radius_m=Body.R_EARTH,
            detect="entry",
            tol_s=1.0,
        )
        half_T = math.pi * math.sqrt(R_orb**3 / Body.MU_EARTH)
        _, triggered = sc.step(Action(), dt_requested=half_T, events=[ev])
        assert len(triggered) >= 1, f"triggered={len(triggered)}"

    def test_shadow_entry_time_within_60s_of_analytical(self):
        R_orb = Body.R_EARTH + 500e3
        sc = _make_sc_for_events(
            pos=[R_orb, 0.0, 0.0],
            vel=[0.0, math.sqrt(Body.MU_EARTH / R_orb), 0.0],
        )
        sc._sun_pos_fn = self._sun_fn
        ev = EclipseEvent(
            sun_pos_fn=self._sun_fn,
            body_pos_fn=None,
            body_radius_m=Body.R_EARTH,
            detect="entry",
            tol_s=1.0,
        )
        half_T = math.pi * math.sqrt(R_orb**3 / Body.MU_EARTH)
        theta = math.pi - math.asin(Body.R_EARTH / R_orb)
        t_expected = (theta / math.pi) * half_T

        t0 = sc.state.time_s
        _, triggered = sc.step(Action(), dt_requested=half_T, events=[ev])
        offset = triggered[0].t_event - t0
        assert abs(offset - t_expected) < 60.0, (
            f"offset={offset:.1f} s  analytic={t_expected:.1f} s"
        )


class TestBisectEvent:
    """Low-level _bisect_event root-finder on a known linear trajectory."""

    def test_root_time_within_1ms(self):
        """f(y) = y[0] − 5, linear trajectory y(t) = 10·t → root at t = 0.5 s."""
        y0 = np.array([0.0])
        y1 = np.array([10.0])
        fn = lambda y, t: y[0] - 5.0  # noqa: E731
        _, t_ev = _bisect_event(fn, y0, 0.0, y1, 1.0, tol_s=0.001)
        assert abs(t_ev - 0.5) < 0.001, f"t={t_ev:.6f}"

    def test_root_value_within_tolerance(self):
        y0 = np.array([0.0])
        y1 = np.array([10.0])
        fn = lambda y, t: y[0] - 5.0  # noqa: E731
        y_ev, _ = _bisect_event(fn, y0, 0.0, y1, 1.0, tol_s=0.001)
        assert abs(y_ev[0] - 5.0) < 0.01, f"y={y_ev[0]:.4f}"


# ══════════════════════════════════════════════════════════════════════════════
# Section — EclipseEvent direction mapping and degenerate-sun guard
# ══════════════════════════════════════════════════════════════════════════════
class TestEclipseEventDetectModes:
    def test_detect_entry_sets_falling_direction(self):
        from cislunar.physics.forces.events import EclipseEvent

        ev = EclipseEvent(sun_pos_fn=lambda t: np.array([AU, 0.0, 0.0]), detect="entry")
        assert ev.direction == "falling"

    def test_detect_exit_sets_rising_direction(self):
        from cislunar.physics.forces.events import EclipseEvent

        ev = EclipseEvent(sun_pos_fn=lambda t: np.array([AU, 0.0, 0.0]), detect="exit")
        assert ev.direction == "rising"

    def test_detect_both_sets_both_direction(self):
        from cislunar.physics.forces.events import EclipseEvent

        ev = EclipseEvent(sun_pos_fn=lambda t: np.array([AU, 0.0, 0.0]), detect="both")
        assert ev.direction == "both"

    def test_degenerate_sun_returns_positive(self):
        """When sun_pos is at the origin, sun_dist < 1 → return 1.0 (sunlit)."""
        from cislunar.physics.forces.events import EclipseEvent

        ev = EclipseEvent(sun_pos_fn=lambda t: np.zeros(3), detect="both")
        y = np.array([7e6, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        assert ev.fn(y, 0.0) == pytest.approx(1.0)

    def test_eclipse_event_with_body_pos_fn(self):
        """Passing a body_pos_fn (non-Earth body) should be accepted without error."""
        from cislunar.physics.constants import Body
        from cislunar.physics.forces.events import EclipseEvent

        body_pos = np.array([3.844e8, 0.0, 0.0])
        ev = EclipseEvent(
            sun_pos_fn=lambda t: np.array([AU, 0.0, 0.0]),
            body_pos_fn=lambda t: body_pos,
            body_radius_m=Body.R_MOON,
            detect="both",
        )
        y = np.array([body_pos[0] + 200e3, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        val = ev.fn(y, 0.0)
        assert np.isfinite(val)
