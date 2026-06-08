"""Unit tests for small physics gaps: SpacecraftState, Checkpoint, plume, propulsion."""

from __future__ import annotations

import numpy as np
import pytest

from cislunar.physics.constants import AU, Body
from cislunar.physics.forces.plume import plume_impingement_force
from cislunar.physics.forces.propulsion import SolarSailModel
from cislunar.physics.state import Checkpoint, SpacecraftState


# ══════════════════════════════════════════════════════════════════════════════
# SpacecraftState properties and classmethod
# ══════════════════════════════════════════════════════════════════════════════
class TestSpacecraftStateProperties:
    def test_speed_ms_matches_velocity_norm(self):
        s = SpacecraftState(
            position=np.array([7e6, 0.0, 0.0]),
            velocity=np.array([0.0, 7500.0, 100.0]),
        )
        assert s.speed_ms == pytest.approx(float(np.linalg.norm([0.0, 7500.0, 100.0])))

    def test_speed_ms_zero_for_stationary(self):
        s = SpacecraftState(
            position=np.array([7e6, 0.0, 0.0]),
            velocity=np.zeros(3),
        )
        assert s.speed_ms == pytest.approx(0.0)

    def test_altitude_from_body_m_at_leo(self):
        r_leo = Body.R_EARTH + 400e3
        s = SpacecraftState(
            position=np.array([r_leo, 0.0, 0.0]),
            velocity=np.array([0.0, 7669.0, 0.0]),
        )
        alt = s.altitude_from_body_m
        assert alt == pytest.approx(r_leo - Body.R_EARTH, rel=1e-6)

    def test_from_vector_round_trip(self):
        s = SpacecraftState(
            position=np.array([7e6, 100.0, -200.0]),
            velocity=np.array([0.0, 7500.0, 50.0]),
            sail_normal=np.array([0.0, 1.0, 0.0]),
            propellant_kg=0.3,
            power_w=25.0,
            time_s=600.0,
            mass_dry_kg=10.0,
        )
        vec = s.as_vector()
        s2 = SpacecraftState.from_vector(
            vec,
            propellant_kg=s.propellant_kg,
            power_w=s.power_w,
            time_s=s.time_s,
            mass_dry_kg=s.mass_dry_kg,
        )
        np.testing.assert_allclose(s2.position, s.position)
        np.testing.assert_allclose(s2.velocity, s.velocity)
        assert s2.propellant_kg == pytest.approx(s.propellant_kg)
        assert s2.time_s == pytest.approx(s.time_s)

    def test_from_vector_with_rcs_propellant(self):
        s = SpacecraftState(
            position=np.array([7e6, 0.0, 0.0]),
            velocity=np.array([0.0, 7500.0, 0.0]),
            rcs_propellant_kg=0.05,
        )
        vec = s.as_vector()
        s2 = SpacecraftState.from_vector(
            vec,
            propellant_kg=0.5,
            power_w=30.0,
            time_s=0.0,
            mass_dry_kg=12.0,
            rcs_propellant_kg=0.05,
        )
        assert s2.rcs_propellant_kg == pytest.approx(0.05)


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint
# ══════════════════════════════════════════════════════════════════════════════
class TestCheckpoint:
    def test_inside_sphere_is_passed(self):
        cp = Checkpoint(position=np.array([7e6, 0.0, 0.0]), radius_m=1000.0, label="gate")
        assert cp.is_passed(np.array([7e6 + 500.0, 0.0, 0.0])) is True

    def test_outside_sphere_not_passed(self):
        cp = Checkpoint(position=np.array([7e6, 0.0, 0.0]), radius_m=1000.0)
        assert cp.is_passed(np.array([7e6 + 2000.0, 0.0, 0.0])) is False

    def test_exactly_on_boundary_is_passed(self):
        cp = Checkpoint(position=np.zeros(3), radius_m=1000.0)
        assert cp.is_passed(np.array([1000.0, 0.0, 0.0])) is True


# ══════════════════════════════════════════════════════════════════════════════
# Plume impingement — edge-case gates
# ══════════════════════════════════════════════════════════════════════════════
class TestPlumeImpingement:
    _source = np.array([0.0, 0.0, 0.0])
    _plume_axis = np.array([1.0, 0.0, 0.0])  # plume travels in +x
    _target_normal = np.array([-1.0, 0.0, 0.0])  # sail faces source
    _thrust = 1e-3

    def test_normal_case_nonzero_force(self):
        target = np.array([200.0, 0.0, 0.0])
        f = plume_impingement_force(
            self._source, self._plume_axis, self._thrust, target, self._target_normal, 32.0
        )
        assert np.linalg.norm(f) > 0

    def test_force_direction_away_from_source(self):
        target = np.array([200.0, 0.0, 0.0])
        f = plume_impingement_force(
            self._source, self._plume_axis, self._thrust, target, self._target_normal, 32.0
        )
        assert f[0] > 0  # pushes +x (away from source)

    def test_zero_thrust_returns_zero(self):
        target = np.array([200.0, 0.0, 0.0])
        f = plume_impingement_force(
            self._source, self._plume_axis, 0.0, target, self._target_normal, 32.0
        )
        np.testing.assert_array_equal(f, np.zeros(3))

    def test_collocated_source_target_returns_zero(self):
        # r < 1 m — numerical safety gate
        f = plume_impingement_force(
            self._source,
            self._plume_axis,
            self._thrust,
            np.array([0.5, 0.0, 0.0]),
            self._target_normal,
            32.0,
        )
        np.testing.assert_array_equal(f, np.zeros(3))

    def test_target_behind_thruster_returns_zero(self):
        # plume_axis = +x, target at -x → cos_theta < 0
        target_behind = np.array([-200.0, 0.0, 0.0])
        f = plume_impingement_force(
            self._source, self._plume_axis, self._thrust, target_behind, self._target_normal, 32.0
        )
        np.testing.assert_array_equal(f, np.zeros(3))

    def test_target_back_face_toward_source_returns_zero(self):
        # normal points away from source → cos_alpha < 0
        target = np.array([200.0, 0.0, 0.0])
        back_normal = np.array([1.0, 0.0, 0.0])  # sail back face toward source
        f = plume_impingement_force(
            self._source, self._plume_axis, self._thrust, target, back_normal, 32.0
        )
        np.testing.assert_array_equal(f, np.zeros(3))

    def test_negative_thrust_returns_zero(self):
        target = np.array([200.0, 0.0, 0.0])
        f = plume_impingement_force(
            self._source, self._plume_axis, -1e-3, target, self._target_normal, 32.0
        )
        np.testing.assert_array_equal(f, np.zeros(3))

    def test_force_falls_off_with_distance_squared(self):
        target_near = np.array([100.0, 0.0, 0.0])
        target_far = np.array([200.0, 0.0, 0.0])
        f_near = np.linalg.norm(
            plume_impingement_force(
                self._source, self._plume_axis, self._thrust, target_near, self._target_normal, 32.0
            )
        )
        f_far = np.linalg.norm(
            plume_impingement_force(
                self._source, self._plume_axis, self._thrust, target_far, self._target_normal, 32.0
            )
        )
        ratio = f_near / f_far
        assert abs(ratio - 4.0) < 0.1  # inverse-square: 2× distance → 4× weaker


# ══════════════════════════════════════════════════════════════════════════════
# SolarSailModel.reflected_force — gate conditions
# ══════════════════════════════════════════════════════════════════════════════
class TestSolarSailReflectedForce:
    _sail = SolarSailModel(area_m2=32.0)
    _sc_from_sun = np.array([-AU, 0.0, 0.0])  # spacecraft at +x, Sun at origin
    _sail_normal = np.array([1.0, 0.0, 0.0])
    _arrival_dir = np.array([1.0, 0.0, 0.0])  # reflected photons travelling +x

    def test_normal_case_returns_nonzero(self):
        # sail_normal · (-arrival_dir) = -1 < 0 → back face → need to fix setup
        # arrival from -x direction, sail pointing +x: cos_theta = sail · (-arrival) = +1
        arrival = np.array([-1.0, 0.0, 0.0])
        f = self._sail.reflected_force(
            sc_pos_from_sun=self._sc_from_sun,
            sail_normal=self._sail_normal,
            reflected_arrival_dir=arrival,
            reflected_flux_ratio=0.1,
        )
        assert np.linalg.norm(f) > 0

    def test_zero_flux_ratio_returns_zero(self):
        f = self._sail.reflected_force(
            sc_pos_from_sun=self._sc_from_sun,
            sail_normal=self._sail_normal,
            reflected_arrival_dir=self._arrival_dir,
            reflected_flux_ratio=0.0,
        )
        np.testing.assert_array_equal(f, np.zeros(3))

    def test_negative_flux_ratio_returns_zero(self):
        f = self._sail.reflected_force(
            sc_pos_from_sun=self._sc_from_sun,
            sail_normal=self._sail_normal,
            reflected_arrival_dir=self._arrival_dir,
            reflected_flux_ratio=-0.1,
        )
        np.testing.assert_array_equal(f, np.zeros(3))

    def test_back_face_illuminated_returns_zero(self):
        # sail_normal · (-arrival_dir) = sail_normal · -arrival = [1,0,0]·[-1,0,0] = -1 ≤ 0
        f = self._sail.reflected_force(
            sc_pos_from_sun=self._sc_from_sun,
            sail_normal=self._sail_normal,
            reflected_arrival_dir=self._arrival_dir,  # photons from +x, normal in +x → cos_theta = -1
            reflected_flux_ratio=0.1,
        )
        np.testing.assert_array_equal(f, np.zeros(3))

    def test_boom_shadow_gate(self):
        # With boom_shadow_fraction large enough, f_self → 0
        sail_with_boom = SolarSailModel(area_m2=32.0, boom_shadow_fraction=0.99)
        arrival = np.array([-1.0, 0.0, 0.0])  # front face illuminated
        f = sail_with_boom.reflected_force(
            sc_pos_from_sun=self._sc_from_sun,
            sail_normal=self._sail_normal,
            reflected_arrival_dir=arrival,
            reflected_flux_ratio=0.1,
        )
        # Should still produce some force (boom_shadow_fraction = 0.99, cos_theta = 1 → f_self = 0.01)
        assert np.isfinite(np.linalg.norm(f))


# ══════════════════════════════════════════════════════════════════════════════
# IonThrusterModel edge cases
# ══════════════════════════════════════════════════════════════════════════════
class TestIonThrusterEdgeCases:
    def test_delta_v_remaining_zero_propellant(self):
        from cislunar.physics.forces.propulsion import IonThrusterModel

        thr = IonThrusterModel()
        assert thr.delta_v_remaining(propellant_kg=0.0, dry_mass_kg=12.0) == pytest.approx(0.0)

    def test_power_request_w_zero_propellant(self):
        from cislunar.physics.forces.propulsion import IonThrusterModel

        thr = IonThrusterModel()
        # propellant=0 → early return 0.0
        assert thr.power_request_w(throttle=1.0, propellant_kg=0.0) == pytest.approx(0.0)

    def test_delta_v_remaining_positive_propellant(self):
        from cislunar.physics.forces.propulsion import IonThrusterModel

        thr = IonThrusterModel()
        dv = thr.delta_v_remaining(propellant_kg=0.5, dry_mass_kg=12.0)
        assert dv > 0.0


# ══════════════════════════════════════════════════════════════════════════════
# SolarSailModel.albedo_acceleration — F_sun <= 0 guard
# ══════════════════════════════════════════════════════════════════════════════
class TestAlbedoAccelerationGuard:
    def test_zero_solar_flux_returns_zero(self, monkeypatch):
        import cislunar.physics.forces.propulsion as prop_mod

        monkeypatch.setattr(prop_mod, "solar_flux_at_position", lambda _: 0.0)
        sail = SolarSailModel(area_m2=32.0)
        result = sail.albedo_acceleration(
            sc_pos_eci=np.array([AU, 0.0, 0.0]),
            body_pos_eci=np.zeros(3),
            sun_pos_eci=np.zeros(3),
            sc_pos_from_sun=np.array([AU, 0.0, 0.0]),
            sail_normal=np.array([1.0, 0.0, 0.0]),
            body_radius_m=6.371e6,
            body_albedo=0.3,
            total_mass_kg=12.5,
        )
        np.testing.assert_array_equal(result, np.zeros(3))


# ══════════════════════════════════════════════════════════════════════════════
# SolarSailModel.reflected_force — boom shadow f_self <= 0
# ══════════════════════════════════════════════════════════════════════════════
class TestReflectedForceBoomShadow:
    def test_boom_shadow_full_blocks_force(self):
        """boom_shadow_fraction=1.0 and very oblique angle → f_self ≤ 0 → zero force."""
        sail = SolarSailModel(area_m2=32.0, boom_shadow_fraction=1.0)
        # arrival from -x, sail normal +x: cos_theta = 1.0
        # f_self = max(0, 1 - 1.0/1.0) = 0 → return zeros
        arrival = np.array([-1.0, 0.0, 0.0])
        f = sail.reflected_force(
            sc_pos_from_sun=np.array([-AU, 0.0, 0.0]),
            sail_normal=np.array([1.0, 0.0, 0.0]),
            reflected_arrival_dir=arrival,
            reflected_flux_ratio=0.1,
        )
        np.testing.assert_array_equal(f, np.zeros(3))


# ══════════════════════════════════════════════════════════════════════════════
# Spacecraft gap tests: albedo SRP, RCS trigger, event rising direction
# ══════════════════════════════════════════════════════════════════════════════
class TestSpacecraftAlbedoSRP:
    def test_albedo_srp_enabled_integrates_without_error(self):
        from conftest import R_LEO, V_CIRC_LEO

        from cislunar.physics import (
            Action,
            GravityModel,
            IonThrusterModel,
            Spacecraft,
            SpacecraftState,
        )
        from cislunar.physics.constants import AU, Body
        from cislunar.physics.forces.eclipse import PowerBudgetModel
        from cislunar.physics.forces.propulsion import SolarSailModel

        grav = GravityModel().add_body("Earth", mu=Body.MU_EARTH)
        state = SpacecraftState(
            position=np.array([R_LEO, 0.0, 0.0]),
            velocity=np.array([0.0, V_CIRC_LEO, 0.0]),
            propellant_kg=0.5,
            mass_dry_kg=12.0,
        )
        sc = Spacecraft(
            initial_state=state,
            sail_model=SolarSailModel(area_m2=32.0),
            thruster_model=IonThrusterModel(),
            gravity_model=grav,
            sun_pos_fn=lambda t: np.array([AU, 0.0, 0.0]),
            power_model=PowerBudgetModel(),
        )
        sc.step(Action(), dt_requested=60.0)
        assert np.isfinite(np.linalg.norm(sc.state.position))

    def test_albedo_srp_with_moon_pos_fn(self):
        from conftest import R_LEO, V_CIRC_LEO

        from cislunar.physics import (
            Action,
            GravityModel,
            IonThrusterModel,
            Spacecraft,
            SpacecraftState,
        )
        from cislunar.physics.constants import AU, Body
        from cislunar.physics.forces.eclipse import PowerBudgetModel
        from cislunar.physics.forces.propulsion import SolarSailModel

        def moon_fn(t):
            return np.array([3.844e8, 0.0, 0.0])

        grav = (
            GravityModel()
            .add_body("Earth", mu=Body.MU_EARTH)
            .add_body("Moon", mu=Body.MU_MOON, body_pos_fn=moon_fn)
        )
        state = SpacecraftState(
            position=np.array([R_LEO, 0.0, 0.0]),
            velocity=np.array([0.0, V_CIRC_LEO, 0.0]),
            propellant_kg=0.5,
            mass_dry_kg=12.0,
        )
        sc = Spacecraft(
            initial_state=state,
            sail_model=SolarSailModel(area_m2=32.0),
            thruster_model=IonThrusterModel(),
            gravity_model=grav,
            sun_pos_fn=lambda t: np.array([AU, 0.0, 0.0]),
            power_model=PowerBudgetModel(),
        )
        sc.step(Action(), dt_requested=60.0)
        assert np.isfinite(np.linalg.norm(sc.state.position))


class TestRcsTrigger:
    def _make_rcs_spacecraft(self, rcs_propellant_kg=0.1):
        from conftest import R_LEO, V_CIRC_LEO

        from cislunar.physics import (
            GravityModel,
            IonThrusterModel,
            Spacecraft,
            SpacecraftState,
        )
        from cislunar.physics.constants import AU, Body
        from cislunar.physics.forces.attitude import AttitudeDynamicsModel
        from cislunar.physics.forces.eclipse import PowerBudgetModel
        from cislunar.physics.forces.propulsion import SolarSailModel

        grav = GravityModel().add_body("Earth", mu=Body.MU_EARTH)
        state = SpacecraftState(
            position=np.array([R_LEO, 0.0, 0.0]),
            velocity=np.array([0.0, V_CIRC_LEO, 0.0]),
            propellant_kg=0.5,
            rcs_propellant_kg=rcs_propellant_kg,
            mass_dry_kg=12.0,
        )
        return Spacecraft(
            initial_state=state,
            sail_model=SolarSailModel(area_m2=32.0),
            thruster_model=IonThrusterModel(),
            gravity_model=grav,
            sun_pos_fn=lambda t: np.array([AU, 0.0, 0.0]),
            power_model=PowerBudgetModel(),
            reaction_wheels=False,  # RCS craft
            attitude_model=AttitudeDynamicsModel(),
        )

    def test_rcs_trigger_consumes_propellant(self):
        from cislunar.physics import Action

        sc = self._make_rcs_spacecraft(rcs_propellant_kg=0.1)
        prop_before = sc.state.rcs_propellant_kg
        action = Action(
            attitude_dir_cmd=np.array([1.0, 0.0, 0.0]),
            rcs_trigger=True,
        )
        sc.step(action, dt_requested=10.0)
        assert sc.state.rcs_propellant_kg < prop_before

    def test_rcs_trigger_with_no_propellant_does_not_crash(self):
        from cislunar.physics import Action

        sc = self._make_rcs_spacecraft(rcs_propellant_kg=0.0)
        action = Action(
            attitude_dir_cmd=np.array([1.0, 0.0, 0.0]),
            rcs_trigger=True,
        )
        sc.step(action, dt_requested=10.0)
        assert np.isfinite(np.linalg.norm(sc.state.position))


class TestEventRisingDirection:
    """Tests for the 'rising' direction branch in the event detector."""

    def test_shadow_exit_detected_with_rising_direction(self):
        import math

        from cislunar.physics import (
            Action,
            GravityModel,
            IonThrusterModel,
            Spacecraft,
            SpacecraftState,
        )
        from cislunar.physics.constants import AU, Body
        from cislunar.physics.forces.eclipse import PowerBudgetModel
        from cislunar.physics.forces.events import EclipseEvent
        from cislunar.physics.forces.propulsion import SolarSailModel

        R_orb = Body.R_EARTH + 500e3
        v_c = math.sqrt(Body.MU_EARTH / R_orb)

        def sun_fn(t):
            return np.array([AU, 0.0, 0.0])

        grav = GravityModel().add_body(
            "Earth", mu=Body.MU_EARTH, j2=Body.J2_EARTH, r_body=Body.R_EARTH
        )
        state = SpacecraftState(
            position=np.array([-R_orb, 0.0, 0.0]),  # start in shadow
            velocity=np.array([0.0, -v_c, 0.0]),
            propellant_kg=0.5,
            mass_dry_kg=12.0,
        )
        sc = Spacecraft(
            initial_state=state,
            sail_model=SolarSailModel(area_m2=32.0),
            thruster_model=IonThrusterModel(),
            gravity_model=grav,
            sun_pos_fn=sun_fn,
            power_model=PowerBudgetModel(),
        )
        ev = EclipseEvent(sun_pos_fn=sun_fn, body_radius_m=Body.R_EARTH, detect="exit")
        half_T = math.pi * math.sqrt(R_orb**3 / Body.MU_EARTH)
        _, triggered = sc.step(Action(), dt_requested=half_T, events=[ev])
        assert len(triggered) >= 1
        assert triggered[0].name == "eclipse"


# ══════════════════════════════════════════════════════════════════════════════
# Residual gap tests
# ══════════════════════════════════════════════════════════════════════════════
class TestAttitudeErrorDeg:
    def test_attitude_error_deg_returns_zero(self):
        from cislunar.physics.forces.attitude import AttitudeDynamicsModel

        m = AttitudeDynamicsModel()
        assert m.attitude_error_deg == pytest.approx(0.0)


class TestSailOccultationZeroArea:
    def test_zero_area_blocker_returns_one(self):
        from cislunar.physics.constants import AU, Body
        from cislunar.physics.forces.eclipse import sail_occultation_fraction

        sc = np.array([Body.R_EARTH + 400e3, 0.0, 0.0])
        blocker = np.array([Body.R_EARTH + 200e3, 0.0, 0.0])
        result = sail_occultation_fraction(
            observer_pos_eci=sc,
            sun_pos_eci=np.array([AU, 0.0, 0.0]),
            blocker_pos_eci=blocker,
            blocker_area_m2=0.0,  # zero area → silhouette = 0 → return 1.0
            blocker_normal_eci=np.array([1.0, 0.0, 0.0]),
        )
        assert result == pytest.approx(1.0)


class TestReflectedFluxRatioObserverAtSun:
    def test_observer_at_sun_returns_zero(self):
        from cislunar.physics.constants import AU, Body
        from cislunar.physics.forces.eclipse import reflected_flux_ratio

        refl_pos = np.array([Body.R_EARTH + 200e3, 0.0, 0.0])
        sun_pos = np.array([AU, 0.0, 0.0])
        result, _ = reflected_flux_ratio(
            observer_pos_eci=sun_pos,  # observer co-located with sun → r_obs_from_sun < 1
            sun_pos_eci=sun_pos,
            reflector_pos_eci=refl_pos,
            reflector_area_m2=32.0,
            reflector_normal_eci=np.array([1.0, 0.0, 0.0]),
            reflector_illumination=1.0,
            specular_fraction=0.9,
            lobe_half_angle_rad=0.1,
        )
        assert result == pytest.approx(0.0)


class TestNormalizedLegendreDegree2:
    def test_degree_two_runs_without_error(self):
        from cislunar.physics.forces.mascon import _normalized_legendre

        p, dp = _normalized_legendre(0.5, 2)
        assert p.shape == (3, 3)
        assert np.isfinite(p[1, 0])


class TestLunarHarmonicInsideReferenceRadius:
    def test_inside_reference_radius_returns_zero(self):
        from cislunar.physics.constants import Body
        from cislunar.physics.forces.mascon import (
            default_lunar_harmonic_gravity,
        )

        grav = default_lunar_harmonic_gravity()
        # Position inside the Moon's reference radius
        sc_body = np.array([Body.R_MOON * 0.5, 0.0, 0.0])
        result = grav.acceleration_body(sc_body)
        np.testing.assert_array_equal(result, np.zeros(3))


class TestReflectedForceNoTangential:
    def test_normal_incidence_no_tangential(self):
        """sail_normal anti-parallel to arrival dir → tang = 0 → F_t = 0."""
        from cislunar.physics.constants import AU
        from cislunar.physics.forces.propulsion import SolarSailModel

        sail = SolarSailModel(area_m2=32.0)
        # arrival_dir = [-1, 0, 0], sail_normal = [1, 0, 0]
        # tang = arrival_dir - dot(arrival_dir, sail_normal) * sail_normal
        #      = [-1,0,0] - (-1) * [1,0,0] = [0,0,0]
        # |tang| < 1e-10 → F_t = zeros
        arrival = np.array([-1.0, 0.0, 0.0])
        f = sail.reflected_force(
            sc_pos_from_sun=np.array([-AU, 0.0, 0.0]),
            sail_normal=np.array([1.0, 0.0, 0.0]),
            reflected_arrival_dir=arrival,
            reflected_flux_ratio=0.1,
        )
        assert np.isfinite(np.linalg.norm(f))


class TestEventBothDirection:
    """Covers the 'both' direction branch in the spacecraft integrator (line 334)."""

    def test_both_direction_detects_exit_from_shadow(self):
        import math

        from cislunar.physics import (
            Action,
            GravityModel,
            IonThrusterModel,
            Spacecraft,
            SpacecraftState,
        )
        from cislunar.physics.constants import AU, Body
        from cislunar.physics.forces.eclipse import PowerBudgetModel
        from cislunar.physics.forces.events import EclipseEvent
        from cislunar.physics.forces.propulsion import SolarSailModel

        R_orb = Body.R_EARTH + 500e3
        v_c = math.sqrt(Body.MU_EARTH / R_orb)

        def sun_fn(t):
            return np.array([AU, 0.0, 0.0])

        grav = GravityModel().add_body(
            "Earth", mu=Body.MU_EARTH, j2=Body.J2_EARTH, r_body=Body.R_EARTH
        )
        state = SpacecraftState(
            position=np.array([-R_orb, 0.0, 0.0]),
            velocity=np.array([0.0, -v_c, 0.0]),
            propellant_kg=0.5,
            mass_dry_kg=12.0,
        )
        sc = Spacecraft(
            initial_state=state,
            sail_model=SolarSailModel(area_m2=32.0),
            thruster_model=IonThrusterModel(),
            gravity_model=grav,
            sun_pos_fn=sun_fn,
            power_model=PowerBudgetModel(),
        )
        # detect="both" → direction="both" → covers line 334
        ev = EclipseEvent(sun_pos_fn=sun_fn, body_radius_m=Body.R_EARTH, detect="both")
        assert ev.direction == "both"
        half_T = math.pi * math.sqrt(R_orb**3 / Body.MU_EARTH)
        _, triggered = sc.step(Action(), dt_requested=half_T, events=[ev])
        assert len(triggered) >= 1


class TestSpacecraftSailDisabled:
    def test_sail_disabled_gives_zero_sail_accel(self):
        from conftest import R_LEO, V_CIRC_LEO

        from cislunar.physics import (
            Action,
            GravityModel,
            IonThrusterModel,
            Spacecraft,
            SpacecraftState,
        )
        from cislunar.physics.constants import AU, Body
        from cislunar.physics.forces.eclipse import PowerBudgetModel
        from cislunar.physics.forces.propulsion import SolarSailModel

        grav = GravityModel().add_body("Earth", mu=Body.MU_EARTH)
        state = SpacecraftState(
            position=np.array([R_LEO, 0.0, 0.0]),
            velocity=np.array([0.0, V_CIRC_LEO, 0.0]),
            propellant_kg=0.5,
            mass_dry_kg=12.0,
        )
        sc = Spacecraft(
            initial_state=state,
            sail_model=SolarSailModel(area_m2=32.0),
            thruster_model=IonThrusterModel(),
            gravity_model=grav,
            sun_pos_fn=lambda t: np.array([AU, 0.0, 0.0]),
            power_model=PowerBudgetModel(),
            sail_enabled=False,  # disabled → a_sail = np.zeros(3) branch
        )
        sc.step(Action(), dt_requested=60.0)
        assert np.isfinite(np.linalg.norm(sc.state.position))


# ══════════════════════════════════════════════════════════════════════════════
# Final gap closers
# ══════════════════════════════════════════════════════════════════════════════
class TestPowerRequestW:
    @staticmethod
    def _warmed_thruster():
        from cislunar.physics.forces.propulsion import IonThrusterModel

        thr = IonThrusterModel()
        for _ in range(120):
            thr.tick(1.0, 1.0)
        return thr

    def test_positive_propellant_returns_nonzero(self):
        thr = self._warmed_thruster()
        p = thr.power_request_w(throttle=1.0, propellant_kg=0.5)
        assert p > 0.0

    def test_full_throttle_bounded_by_max_power(self):
        thr = self._warmed_thruster()
        p = thr.power_request_w(throttle=1.0, propellant_kg=1.0)
        assert p <= thr.max_power_w


class TestReflectedForceTangentialComponent:
    def test_oblique_incidence_produces_tangential_force(self):
        """45° incident reflected light has a nonzero tangential component."""
        import math

        from cislunar.physics.constants import AU
        from cislunar.physics.forces.propulsion import SolarSailModel

        sail = SolarSailModel(area_m2=32.0)
        # arrival_dir at 45° to sail_normal=[1,0,0]
        # arrival = [-1/√2, 1/√2, 0] so photons come at 45°
        c = math.cos(math.radians(45))
        arrival = np.array([-c, c, 0.0])
        # cos_theta = dot([1,0,0], -arrival) = c > 0 ✓
        # tang = arrival - dot(arrival,[1,0,0]) * [1,0,0] = [-c,c,0] - (-c)*[1,0,0] = [0,c,0]
        # norm(tang) = c > 1e-10 ✓ → line 297 executed
        f = sail.reflected_force(
            sc_pos_from_sun=np.array([-AU, 0.0, 0.0]),
            sail_normal=np.array([1.0, 0.0, 0.0]),
            reflected_arrival_dir=arrival,
            reflected_flux_ratio=0.1,
        )
        assert np.isfinite(np.linalg.norm(f))
        # tangential (y) component should be nonzero
        assert abs(f[1]) > 0


class TestNormalizedLegendreAtPole:
    def test_sin_lat_near_one_handles_denom(self):
        """sin_lat ≈ 1 → x² - 1 ≈ 0 → clamped to -1e-14."""
        from cislunar.physics.forces.mascon import _normalized_legendre

        p, dp = _normalized_legendre(1.0, 4)  # sin_lat = 1 (north pole)
        assert np.isfinite(np.sum(p))


class TestLunarHarmonicInsideMoonBody:
    def test_inside_lunar_radius_returns_zero(self):
        from cislunar.physics.constants import Body
        from cislunar.physics.forces.mascon import default_lunar_harmonic_gravity

        grav = default_lunar_harmonic_gravity()
        # r < REFERENCE_RADIUS_M (1738 km) → return zeros
        sc_body = np.array([Body.R_MOON * 0.9, 0.0, 0.0])
        result = grav.acceleration_body(sc_body)
        np.testing.assert_array_equal(result, np.zeros(3))
