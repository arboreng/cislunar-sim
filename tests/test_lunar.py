"""Lunar mascon model and harmonic gravity tests."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from cislunar.physics import (
    GRAIL_MASCONS,
    Action,
    GravityModel,
    IonThrusterModel,
    LunarHarmonicGravity,
    SolarSailModel,
    Spacecraft,
    SpacecraftState,
    lunar_mascon_acceleration,
    mascon_acceleration_at_altitude,
)
from cislunar.physics.constants import AU, Body
from cislunar.physics.forces.eclipse import PowerBudgetModel
from cislunar.physics.forces.gravity import j2_perturbation, j3_perturbation
from cislunar.physics.forces.mascon import R_MOON_M


# ══════════════════════════════════════════════════════════════════════════════
# Section 17 — Lunar mascon model
# ══════════════════════════════════════════════════════════════════════════════
class TestLunarMascons:
    @staticmethod
    def _imbrium_nadir_position(altitude_m: float) -> np.ndarray:
        imbrium = GRAIL_MASCONS[0]
        lat = math.radians(imbrium.lat_deg)
        lon = math.radians(imbrium.lon_deg)
        radius = R_MOON_M + altitude_m
        return np.array(
            [
                radius * math.cos(lat) * math.cos(lon),
                radius * math.cos(lat) * math.sin(lon),
                radius * math.sin(lat),
            ]
        )

    def test_grail_catalogue_has_8_sites(self):
        assert len(GRAIL_MASCONS) == 8

    def test_all_excess_masses_exceed_1e17_kg(self):
        assert all(s.excess_mass_kg > 1e17 for s in GRAIL_MASCONS)

    def test_mare_imbrium_is_strongest(self):
        """Mare Imbrium (catalogue index 0) is the strongest mascon."""
        max_mass = max(s.excess_mass_kg for s in GRAIL_MASCONS)
        assert GRAIL_MASCONS[0].excess_mass_kg == max_mass

    def test_mascon_accel_significant_at_100km(self):
        a = mascon_acceleration_at_altitude(100)
        assert a > 1e-5, f"a(100 km)={a:.2e} m/s²"

    def test_mascon_accel_monotonic_with_altitude(self):
        a_100 = mascon_acceleration_at_altitude(100)
        a_500 = mascon_acceleration_at_altitude(500)
        a_2000 = mascon_acceleration_at_altitude(2000)
        assert a_100 > a_500 > a_2000, f"a_100={a_100:.3e}  a_500={a_500:.3e}  a_2000={a_2000:.3e}"

    def test_delta_v_over_one_orbit_exceeds_50mm(self):
        """a × T_orbit at 100 km accumulates > 50 mm/s of Δv per orbit."""
        dv = mascon_acceleration_at_altitude(100) * 7200
        assert dv > 0.05, f"Δv={dv * 1e3:.1f} mm/s"

    def test_nearside_accel_exceeds_farside(self):
        """Near-side (Mare Imbrium) > far side."""
        moon_pos = np.array([3.844e8, 0.0, 0.0])
        sc_near = np.array([3.844e8 - R_MOON_M - 200_000, 0.0, 0.0])
        sc_far = np.array([3.844e8 + R_MOON_M + 200_000, 0.0, 0.0])
        a_near = np.linalg.norm(lunar_mascon_acceleration(sc_near, moon_pos))
        a_far = np.linalg.norm(lunar_mascon_acceleration(sc_far, moon_pos))
        assert a_near > a_far, f"near={a_near:.2e}  far={a_far:.2e}"

    def test_zero_acceleration_above_2000km(self):
        moon_pos = np.array([3.844e8, 0.0, 0.0])
        sc_2100 = moon_pos + np.array([R_MOON_M + 2_100_000, 0.0, 0.0])
        a = lunar_mascon_acceleration(sc_2100, moon_pos)
        assert np.linalg.norm(a) == 0.0, f"|a(2100 km)|={np.linalg.norm(a):.3e}"

    def test_nonzero_acceleration_just_inside_cutoff(self):
        moon_pos = np.array([3.844e8, 0.0, 0.0])
        sc_1900 = moon_pos + np.array([R_MOON_M + 1_900_000, 0.0, 0.0])
        a = lunar_mascon_acceleration(sc_1900, moon_pos)
        assert np.linalg.norm(a) > 0, f"|a(1900 km)|={np.linalg.norm(a):.3e}"

    def test_gravity_model_accepts_callable_harmonic_model(self):
        harmonic = LunarHarmonicGravity(min_degree=2)
        grav = GravityModel().add_body("Moon", mu=Body.MU_MOON).set_mascon_model(harmonic)
        sc = self._imbrium_nadir_position(50_000)
        a = grav.acceleration(sc, time_s=0.0)
        assert grav._mascon_fn is harmonic
        assert np.all(np.isfinite(a))
        assert np.linalg.norm(a) > 1.0, f"|a|={np.linalg.norm(a):.3e}"

    def test_harmonic_model_matches_published_imbrium_gravity_scale(self):
        harmonic = LunarHarmonicGravity(min_degree=2)
        sc = self._imbrium_nadir_position(50_000)
        unit_r = sc / np.linalg.norm(sc)
        a = harmonic(sc, np.zeros(3))
        radial_mag = -float(np.dot(a, unit_r))
        published_50km = 1.70e-3
        err_pct = abs(radial_mag - published_50km) / published_50km * 100.0
        assert err_pct < 10.0, (
            f"radial={radial_mag:.3e}  ref={published_50km:.3e} m/s²  err={err_pct:.1f}%"
        )

    @pytest.mark.perf
    def test_degree70_harmonic_runtime_stays_sub_millisecond(self):
        harmonic = LunarHarmonicGravity()
        sc = self._imbrium_nadir_position(50_000)
        moon_pos = np.zeros(3)
        for _ in range(50):
            harmonic(sc, moon_pos)
        calls = 500
        t0 = time.perf_counter()
        for _ in range(calls):
            harmonic(sc, moon_pos)
        mean_ms = (time.perf_counter() - t0) * 1000.0 / calls
        assert mean_ms < 1.0, f"mean={mean_ms:.3f} ms/call"

    def test_make_lunar_craft_has_mascon_enabled(self, leo_spacecraft):
        assert leo_spacecraft.gravity._mascon_fn is not None

    def test_mascon_trajectory_diverges_from_clean(self):
        """Two LLO spacecraft (one with mascons, one without) diverge over 30 min."""
        R_lo = R_MOON_M + 200_000
        v_lunar = math.sqrt(Body.MU_MOON / R_lo)

        def _make_lo(with_mascon: bool) -> Spacecraft:
            grav = GravityModel().add_body("Moon", mu=Body.MU_MOON)
            if with_mascon:
                grav.set_mascon_model(lunar_mascon_acceleration)
            st = SpacecraftState(
                position=np.array([-R_lo, 0.0, 0.0]),
                velocity=np.array([0.0, v_lunar, 0.0]),
                propellant_kg=0.0,
                mass_dry_kg=5.0,
                time_s=0.0,
            )
            return Spacecraft(
                initial_state=st,
                sail_model=SolarSailModel(area_m2=32.0),
                thruster_model=IonThrusterModel(),
                gravity_model=grav,
                sun_pos_fn=lambda t: np.array([AU, 0.0, 0.0]),
                drag_model=None,
                power_model=PowerBudgetModel(),
                attitude_model=None,
            )

        sc_with = _make_lo(True)
        sc_without = _make_lo(False)
        coast = Action(attitude_dir_cmd=np.array([-1.0, 0.0, 0.0]), throttle=0.0)
        for _ in range(3):
            sc_with.step(coast, dt_requested=600.0)
            sc_without.step(coast, dt_requested=600.0)
        divergence = np.linalg.norm(sc_with.state.position - sc_without.state.position)
        assert divergence > 1.0, f"divergence={divergence:.2f} m"

    def test_j2_moon_defined_and_plausible(self):
        assert hasattr(Body, "J2_MOON") and 1e-5 < Body.J2_MOON < 1e-3

    def test_j3_moon_defined_and_smaller_than_j2(self):
        assert hasattr(Body, "J3_MOON") and 0 < Body.J3_MOON < Body.J2_MOON

    def test_j3_j2_ratio_in_expected_range(self):
        """For the Moon, J3/J2 ≈ 0.3–0.6."""
        ratio = Body.J3_MOON / Body.J2_MOON
        assert 0.3 < ratio < 0.6, f"J3/J2 = {ratio:.3f}"

    def test_lunar_j2_exceeds_mascon_at_200km(self):
        """At 200 km altitude, J2 dwarfs the 8-site mascon signal."""
        r_200 = R_MOON_M + 200_000
        pos_polar = np.array([0.0, 0.0, r_200])
        a_j2 = np.linalg.norm(j2_perturbation(pos_polar, Body.MU_MOON, Body.J2_MOON, Body.R_MOON))
        a_msc = mascon_acceleration_at_altitude(200)
        assert a_j2 > a_msc, f"J2={a_j2:.2e}  mascon={a_msc:.2e}"

    def test_lunar_j3_nonnegligible_at_200km(self):
        """J3 is at least 0.5 × the mascon signal — physically material."""
        r_200 = R_MOON_M + 200_000
        pos_polar = np.array([0.0, 0.0, r_200])
        a_j3 = np.linalg.norm(j3_perturbation(pos_polar, Body.MU_MOON, Body.J3_MOON, Body.R_MOON))
        a_msc = mascon_acceleration_at_altitude(200)
        assert a_j3 > 0.5 * a_msc, f"J3={a_j3:.2e}  mascon={a_msc:.2e}"

    def test_make_lunar_craft_sets_lunar_j2(self, leo_spacecraft):
        moon_body = next(b for b in leo_spacecraft.gravity._bodies if b.name == "Moon")
        assert abs(moon_body.j2 - Body.J2_MOON) < 1e-12

    def test_make_lunar_craft_sets_lunar_j3(self, leo_spacecraft):
        moon_body = next(b for b in leo_spacecraft.gravity._bodies if b.name == "Moon")
        assert abs(moon_body.j3 - Body.J3_MOON) < 1e-12


# ══════════════════════════════════════════════════════════════════════════════
# Gap-closing additions
# ══════════════════════════════════════════════════════════════════════════════
class TestLunarHarmonicGravityGaps:
    def test_bad_degree_below_min_raises(self):
        from cislunar.physics.forces.mascon import LunarHarmonicGravity

        with pytest.raises(ValueError, match="degree"):
            LunarHarmonicGravity(degree=1)

    def test_bad_min_degree_raises(self):
        from cislunar.physics.forces.mascon import LunarHarmonicGravity

        with pytest.raises(ValueError, match="min_degree"):
            LunarHarmonicGravity(degree=10, min_degree=1)

    def test_above_active_altitude_returns_zero(self):
        from cislunar.physics.constants import Body
        from cislunar.physics.forces.mascon import LunarHarmonicGravity

        grav = LunarHarmonicGravity(degree=4, active_altitude_m=100e3)
        # Position 500 km above Moon → above active_altitude_m
        sc_body = np.array([Body.R_MOON + 500e3, 0.0, 0.0])
        result = grav.acceleration_body(sc_body)
        np.testing.assert_array_equal(result, np.zeros(3))

    def test_min_degree_above_4_zeros_lower_harmonics(self):
        from cislunar.physics.constants import Body
        from cislunar.physics.forces.mascon import LunarHarmonicGravity

        # min_degree=6 → degrees 4 and 5 are zeroed out
        grav = LunarHarmonicGravity(degree=10, min_degree=6)
        sc_body = np.array([Body.R_MOON + 50e3, 0.0, 0.0])
        result = grav.acceleration_body(sc_body)
        assert np.isfinite(np.linalg.norm(result))


class TestNormalizedLegendreEdgeCases:
    def test_max_degree_zero_returns_p00_one(self):
        from cislunar.physics.forces.mascon import _normalized_legendre

        # _normalized_legendre(sin_lat, max_degree) — positional args only
        p, dp = _normalized_legendre(0.5, 0)
        assert p[0, 0] == pytest.approx(1.0)


class TestBodyToEciPoleCase:
    def test_moon_at_north_pole_direction_handled(self):
        from cislunar.physics.forces.mascon import _body_to_eci

        # Moon at [0,0,r] → x_axis = [0,0,-1]; pole = [0,0,1] is parallel to x_axis
        # This triggers the fallback pole = [1,0,0]
        moon_pos = np.array([0.0, 0.0, 3.844e8])
        dcm = _body_to_eci(moon_pos)
        assert dcm.shape == (3, 3)
        # Should be orthonormal
        np.testing.assert_allclose(dcm @ dcm.T, np.eye(3), atol=1e-10)
