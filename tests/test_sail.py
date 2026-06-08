"""Solar radiation pressure tests: sail force, illumination gating, shadowing, albedo."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cislunar.physics import (
    AttitudeConfig,
    AttitudeDynamicsModel,
    SolarSailModel,
    body_shadow_illumination,
    reflected_flux_ratio,
    sail_occultation_fraction,
)
from cislunar.physics.constants import AU, Body
from cislunar.physics.constants import Sail as SailConst
from cislunar.physics.forces.propulsion import solar_flux_at_position


# ══════════════════════════════════════════════════════════════════════════════
# Section 2 — Solar flux
# ══════════════════════════════════════════════════════════════════════════════
class TestSolarFlux:
    def test_solar_constant_at_1au(self):
        """Solar flux at 1 AU ≈ 1361 W/m²."""
        flux = solar_flux_at_position(np.array([AU, 0, 0]))
        assert abs(flux - 1361) < 5, f"flux={flux:.1f} W/m²"

    def test_inverse_square_with_distance(self):
        """Flux at 2 AU = ¼ flux at 1 AU."""
        f1 = solar_flux_at_position(np.array([AU, 0, 0]))
        f2 = solar_flux_at_position(np.array([2 * AU, 0, 0]))
        assert abs(f2 / f1 - 0.25) < 1e-9, f"f2/f1={f2 / f1}"


# ══════════════════════════════════════════════════════════════════════════════
# Section 3 — Solar sail
# ══════════════════════════════════════════════════════════════════════════════
class TestSolarSail:
    @pytest.fixture
    def sail(self):
        return SolarSailModel(area_m2=32.0)

    def test_head_on_force_dominates_x(self, sail):
        """Head-on incidence: force component along Sun-line dominates."""
        pos = np.array([AU, 0.0, 0.0])
        n_toward_sun = np.array([-1.0, 0.0, 0.0])
        a = sail.acceleration(pos, n_toward_sun, 12.0)
        assert abs(a[0]) > abs(a[1]) and abs(a[0]) > abs(a[2]), f"a={a}"

    def test_head_on_force_points_away_from_sun(self, sail):
        """Radiation pressure pushes spacecraft outward (+x when Sun at -x)."""
        pos = np.array([AU, 0.0, 0.0])
        n_toward_sun = np.array([-1.0, 0.0, 0.0])
        a = sail.acceleration(pos, n_toward_sun, 12.0)
        assert a[0] > 0, f"a_x={a[0]}"

    def test_back_facing_sail_zero_force(self, sail):
        """A sail whose normal points away from the Sun receives no force."""
        pos = np.array([AU, 0.0, 0.0])
        n_away = np.array([1.0, 0.0, 0.0])
        a = sail.acceleration(pos, n_away, 12.0)
        assert np.linalg.norm(a) < 1e-30, f"|a|={np.linalg.norm(a)}"

    def test_characteristic_acceleration_in_lightsail_range(self, sail):
        """LightSail-2-class sail has nonzero characteristic acceleration."""
        a_c = sail.characteristic_acceleration(12.0)
        assert a_c > 0.01, f"a_c={a_c:.4f} mm/s²"


# ══════════════════════════════════════════════════════════════════════════════
# Section 3b — SRP illumination gating (Earth/Moon eclipse blocks sail force)
# ══════════════════════════════════════════════════════════════════════════════
class TestSRPIlluminationGating:
    """
    Closes the fidelity gap where the flat-plate SRP model applied full sail
    force even when the spacecraft was in Earth's umbra.  ``SolarSailModel``
    now accepts an ``illumination`` kwarg in [0, 1], and ``Spacecraft``
    derives it from ``body_shadow_illumination(pos, sun_pos, moon_pos)`` so
    that the sail force and the power model see the same physical truth.
    """

    @pytest.fixture
    def sail(self):
        return SolarSailModel(area_m2=32.0, boom_shadow_fraction=0.0)

    def test_srp_zero_in_earth_umbra(self, sail):
        """Inside Earth's umbra the sail must produce zero force."""
        pos = np.array([AU, 0.0, 0.0])
        n = np.array([-1.0, 0.0, 0.0])
        F0 = sail.force(pos, n, 12.0, illumination=1.0)
        F_umbra = sail.force(pos, n, 12.0, illumination=0.0)
        assert np.linalg.norm(F0) > 0.0
        assert np.linalg.norm(F_umbra) == 0.0, f"|F|={np.linalg.norm(F_umbra)}"

    def test_srp_scales_linearly_through_penumbra(self, sail):
        """Penumbra illumination fractions scale force linearly."""
        pos = np.array([AU, 0.0, 0.0])
        n = np.array([-1.0, 0.0, 0.0])
        F_full = np.linalg.norm(sail.force(pos, n, 12.0, illumination=1.0))
        for frac in (0.1, 0.25, 0.5, 0.75):
            F = np.linalg.norm(sail.force(pos, n, 12.0, illumination=frac))
            expected = F_full * frac
            rel_err = abs(F - expected) / expected
            assert rel_err < 1e-9, f"frac={frac} F={F:.3e} expected={expected:.3e}"

    def test_srp_zero_in_moon_umbra_via_helper(self):
        """
        body_shadow_illumination → 0 when spacecraft is in lunar umbra;
        the sail should produce zero force when that value is plumbed through.
        """
        sail = SolarSailModel(area_m2=32.0, boom_shadow_fraction=0.0)

        sun_pos = np.array([AU, 0.0, 0.0])
        moon_pos = np.array([3.84e8, 0.0, 0.0])
        sc_pos = moon_pos - np.array([1.0e5, 0.0, 0.0])

        illum = body_shadow_illumination(sc_pos, sun_pos, moon_pos)
        assert illum == 0.0, f"expected umbra, got illum={illum}"

        F = sail.force(sc_pos - sun_pos, np.array([-1.0, 0.0, 0.0]), 12.0, illumination=illum)
        assert np.linalg.norm(F) == 0.0

    def test_srp_sunlit_unchanged(self, sail):
        """In full sunlight (illumination=1.0), force matches the default call."""
        pos = np.array([AU, 0.0, 0.0])
        n = np.array([-1.0, 0.0, 0.0])
        F_default = sail.force(pos, n, 12.0)
        F_explicit = sail.force(pos, n, 12.0, illumination=1.0)
        np.testing.assert_allclose(F_default, F_explicit, rtol=0, atol=0)

    def test_srp_illumination_default_backcompat(self, sail):
        """Calling force() without the illumination kwarg must still work."""
        pos = np.array([AU, 0.0, 0.0])
        n = np.array([-1.0, 0.0, 0.0])
        F = sail.force(pos, n, 12.0)
        assert np.linalg.norm(F) > 0.0

    def test_body_shadow_helper_composition(self):
        """
        body_shadow_illumination multiplies Earth and Moon fractions.  At a
        point well outside both shadow cones the result is exactly 1.0.
        """
        sun_pos = np.array([AU, 0.0, 0.0])
        sc_pos = np.array([+7.0e6, 0.0, 0.0])
        moon_pos = np.array([0.0, 3.84e8, 0.0])
        assert body_shadow_illumination(sc_pos, sun_pos, moon_pos) == 1.0
        assert body_shadow_illumination(sc_pos, sun_pos, None) == 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Section 3c — Sail structural self-shadowing (boom shadow on membrane)
# ══════════════════════════════════════════════════════════════════════════════
class TestSailSelfShadowing:
    def test_self_shadow_zero_at_normal_incidence(self):
        """At cos θ = 1 the boom shadow scales only by boom_shadow_fraction."""
        boom_frac = 0.02
        sail = SolarSailModel(area_m2=32.0, boom_shadow_fraction=boom_frac)
        sail_no_boom = SolarSailModel(area_m2=32.0, boom_shadow_fraction=0.0)

        pos = np.array([AU, 0.0, 0.0])
        n = np.array([-1.0, 0.0, 0.0])
        F = np.linalg.norm(sail.force(pos, n, 12.0))
        F_ref = np.linalg.norm(sail_no_boom.force(pos, n, 12.0))

        expected_ratio = 1.0 - boom_frac
        assert abs(F / F_ref - expected_ratio) < 1e-9, (
            f"ratio={F / F_ref:.6f} expected={expected_ratio:.6f}"
        )

    def test_self_shadow_grows_with_obliquity(self):
        boom_frac = 0.05
        sail = SolarSailModel(area_m2=32.0, boom_shadow_fraction=boom_frac)
        sail_ref = SolarSailModel(area_m2=32.0, boom_shadow_fraction=0.0)

        pos = np.array([AU, 0.0, 0.0])
        ratios = []
        for theta_deg in (0.0, 30.0, 60.0, 75.0):
            th = math.radians(theta_deg)
            n = np.array([-math.cos(th), math.sin(th), 0.0])
            F = np.linalg.norm(sail.force(pos, n, 12.0))
            F_ref_val = np.linalg.norm(sail_ref.force(pos, n, 12.0))
            ratios.append(F / F_ref_val)

        for a, b in zip(ratios, ratios[1:]):
            assert b < a, f"ratios={ratios} — not monotone"

        assert abs(ratios[0] - (1.0 - boom_frac)) < 1e-9

    def test_self_shadow_clamps_to_zero(self):
        sail = SolarSailModel(area_m2=32.0, boom_shadow_fraction=0.05)
        th = math.acos(0.01)
        pos = np.array([AU, 0.0, 0.0])
        n = np.array([-math.cos(th), math.sin(th), 0.0])
        F = sail.force(pos, n, 12.0)
        assert np.linalg.norm(F) == 0.0, f"|F|={np.linalg.norm(F)}"

    def test_self_shadow_disabled_when_unset(self):
        """boom_shadow_fraction=0.0 reproduces pre-change behaviour exactly."""
        sail = SolarSailModel(area_m2=32.0, boom_shadow_fraction=0.0)
        pos = np.array([AU, 0.0, 0.0])
        for theta_deg in (0.0, 30.0, 60.0, 80.0):
            th = math.radians(theta_deg)
            n = np.array([-math.cos(th), math.sin(th), 0.0])
            F = sail.force(pos, n, 12.0)
            assert np.linalg.norm(F) > 0.0, f"θ={theta_deg}: unexpected zero force"


# ══════════════════════════════════════════════════════════════════════════════
# Section 3d — Inter-spacecraft sail shadowing
# ══════════════════════════════════════════════════════════════════════════════
class TestInterSpacecraftShadowing:
    SUN_POS = np.array([AU, 0.0, 0.0])
    N_TOWARD_SUN = np.array([+1.0, 0.0, 0.0])
    SAIL_AREA = 32.0

    def test_sail_occultation_full_umbra_in_line(self):
        """Blocker 100 m sunward on-axis → observer fully in umbra (illum=0)."""
        observer = np.zeros(3)
        blocker = np.array([100.0, 0.0, 0.0])
        f = sail_occultation_fraction(
            observer,
            self.SUN_POS,
            blocker,
            self.SAIL_AREA,
            self.N_TOWARD_SUN,
        )
        assert f == 0.0, f"expected umbra, got f={f}"

    def test_sail_occultation_zero_at_long_range(self):
        observer = np.zeros(3)
        blocker = np.array([10_000.0, 1_000.0, 0.0])
        f = sail_occultation_fraction(
            observer,
            self.SUN_POS,
            blocker,
            self.SAIL_AREA,
            self.N_TOWARD_SUN,
        )
        assert f == 1.0, f"expected full sunlight, got f={f}"

    def test_sail_occultation_zero_when_blocker_edge_on(self):
        """Blocker normal perpendicular to Sun line → zero silhouette, no shadow."""
        observer = np.zeros(3)
        blocker = np.array([100.0, 0.0, 0.0])
        edge_on_normal = np.array([0.0, 1.0, 0.0])
        f = sail_occultation_fraction(
            observer,
            self.SUN_POS,
            blocker,
            self.SAIL_AREA,
            edge_on_normal,
        )
        assert f == 1.0, f"expected no shadow, got f={f}"

    def test_sail_occultation_backface_blocker_no_shadow(self):
        """Blocker normal pointing AWAY from Sun → back-face, no silhouette."""
        observer = np.zeros(3)
        blocker = np.array([100.0, 0.0, 0.0])
        back_face_normal = np.array([-1.0, 0.0, 0.0])
        f = sail_occultation_fraction(
            observer,
            self.SUN_POS,
            blocker,
            self.SAIL_AREA,
            back_face_normal,
        )
        assert f == 1.0, f"expected no shadow, got f={f}"

    def test_sail_occultation_penumbra_is_continuous(self):
        """Sweeping lateral offset across the penumbra boundary is monotonic."""
        observer = np.zeros(3)
        results = []
        for lat_m in (0.0, 2.0, 5.0, 8.0, 11.0, 15.0, 30.0):
            blocker = np.array([2000.0, lat_m, 0.0])
            f = sail_occultation_fraction(
                observer,
                self.SUN_POS,
                blocker,
                self.SAIL_AREA,
                self.N_TOWARD_SUN,
            )
            results.append(f)
        for a, b in zip(results, results[1:]):
            assert b >= a - 1e-12, f"non-monotonic: {results}"
        assert results[0] == 0.0
        assert results[-1] == 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Section 3e — Planetary albedo SRP (Lambertian-disc model)
# ══════════════════════════════════════════════════════════════════════════════
class TestAlbedoSRP:
    SUN_POS = np.array([AU, 0.0, 0.0])
    R_LEO = Body.R_EARTH + 400e3
    SC_POS = np.array([R_LEO, 0.0, 0.0])

    @pytest.fixture
    def sail(self):
        return SolarSailModel(area_m2=32.0, boom_shadow_fraction=0.0)

    def test_albedo_force_is_positive_on_sunlit_side(self, sail):
        """Spacecraft on sunlit side gets a positive (+x) albedo push."""
        sc_from_sun = self.SC_POS - self.SUN_POS
        a = sail.albedo_acceleration(
            self.SC_POS,
            np.zeros(3),
            self.SUN_POS,
            sc_from_sun,
            sail_normal=np.array([-1.0, 0.0, 0.0]),
            body_radius_m=Body.R_EARTH,
            body_albedo=Body.ALBEDO_EARTH,
            total_mass_kg=12.5,
        )
        assert a[0] > 0.0, f"expected positive x-acceleration, got {a}"

    def test_albedo_zero_on_night_side(self, sail):
        """Spacecraft on the night side (phase angle > 90°) gets zero albedo."""
        night_pos = np.array([-self.R_LEO, 0.0, 0.0])
        sc_from_sun = night_pos - self.SUN_POS
        a = sail.albedo_acceleration(
            night_pos,
            np.zeros(3),
            self.SUN_POS,
            sc_from_sun,
            sail_normal=np.array([+1.0, 0.0, 0.0]),
            body_radius_m=Body.R_EARTH,
            body_albedo=Body.ALBEDO_EARTH,
            total_mass_kg=12.5,
        )
        assert np.allclose(a, 0.0), f"expected zero on night side, got {a}"

    def test_albedo_at_leo_is_significant_fraction_of_direct(self, sail):
        mass = 12.5
        sc_from_sun = self.SC_POS - self.SUN_POS
        a_direct = sail.acceleration(sc_from_sun, np.array([+1.0, 0.0, 0.0]), mass)
        a_albedo = sail.albedo_acceleration(
            self.SC_POS,
            np.zeros(3),
            self.SUN_POS,
            sc_from_sun,
            sail_normal=np.array([-1.0, 0.0, 0.0]),
            body_radius_m=Body.R_EARTH,
            body_albedo=Body.ALBEDO_EARTH,
            total_mass_kg=mass,
        )
        ratio_leo = np.linalg.norm(a_albedo) / np.linalg.norm(a_direct)
        assert 0.10 < ratio_leo < 0.40, (
            f"LEO albedo/direct ratio {ratio_leo:.3f} outside expected 10–40%"
        )

        r_geo = Body.R_EARTH + 36_000e3
        sc_geo = np.array([r_geo, 0.0, 0.0])
        sc_geo_sun = sc_geo - self.SUN_POS
        a_direct_geo = sail.acceleration(sc_geo_sun, np.array([+1.0, 0.0, 0.0]), mass)
        a_albedo_geo = sail.albedo_acceleration(
            sc_geo,
            np.zeros(3),
            self.SUN_POS,
            sc_geo_sun,
            sail_normal=np.array([-1.0, 0.0, 0.0]),
            body_radius_m=Body.R_EARTH,
            body_albedo=Body.ALBEDO_EARTH,
            total_mass_kg=mass,
        )
        ratio_geo = np.linalg.norm(a_albedo_geo) / np.linalg.norm(a_direct_geo)
        assert ratio_geo < 0.01, f"GEO albedo/direct ratio {ratio_geo:.4f} should be < 1%"

    def test_albedo_falls_off_with_altitude_ordering(self, sail):
        mass = 12.5
        altitudes = [Body.R_EARTH + alt for alt in [400e3, 2000e3, 36000e3]]
        accels = []
        for r in altitudes:
            sc_pos = np.array([r, 0.0, 0.0])
            sc_from_sun = sc_pos - self.SUN_POS
            a = sail.albedo_acceleration(
                sc_pos,
                np.zeros(3),
                self.SUN_POS,
                sc_from_sun,
                sail_normal=np.array([-1.0, 0.0, 0.0]),
                body_radius_m=Body.R_EARTH,
                body_albedo=Body.ALBEDO_EARTH,
                total_mass_kg=mass,
            )
            accels.append(float(np.linalg.norm(a)))
        assert accels[0] > accels[1] > accels[2], f"albedo should decrease with altitude: {accels}"

    def test_albedo_zero_inside_body_radius(self, sail):
        """Guard: spacecraft inside Earth radius returns zero (no physics)."""
        sc_pos = np.array([Body.R_EARTH * 0.5, 0.0, 0.0])
        sc_from_sun = sc_pos - self.SUN_POS
        a = sail.albedo_acceleration(
            sc_pos,
            np.zeros(3),
            self.SUN_POS,
            sc_from_sun,
            sail_normal=np.array([-1.0, 0.0, 0.0]),
            body_radius_m=Body.R_EARTH,
            body_albedo=Body.ALBEDO_EARTH,
            total_mass_kg=12.5,
        )
        assert np.allclose(a, 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# Section 20 — SRP torque coupling via centre-of-pressure offset
# ══════════════════════════════════════════════════════════════════════════════
class TestSRPTorqueCoupling:
    F_SRP_ECI = np.array([1e-3, 0.0, 0.0])

    @pytest.fixture
    def cfg_with_cp(self):
        return AttitudeConfig(
            enable_disturbances=False,
            gravity_gradient=False,
            cp_offset_body=np.array([0.0, 0.0, 0.05]),
        )

    @pytest.fixture
    def cfg_no_cp(self):
        return AttitudeConfig(
            enable_disturbances=False,
            gravity_gradient=False,
            cp_offset_body=np.zeros(3),
        )

    def test_cp_offset_loads_wheels(self, cfg_with_cp):
        """Non-zero CoP offset → SRP torque spins up the reaction wheels."""
        att = AttitudeDynamicsModel(config=cfg_with_cp)
        h_before = att.h_wheels.copy()
        att.step(q_cmd=np.array([1.0, 0.0, 0.0, 0.0]), dt=5.0, srp_force_eci=self.F_SRP_ECI)
        dh = np.linalg.norm(att.h_wheels - h_before)
        assert dh > 0, f"Δ|h_wheels|={dh:.2e} N·m·s"

    def test_zero_cp_offset_no_wheel_load(self, cfg_no_cp):
        """No CoP offset → no SRP-induced torque on the wheels."""
        att = AttitudeDynamicsModel(config=cfg_no_cp)
        h_before = att.h_wheels.copy()
        att.step(q_cmd=np.array([1.0, 0.0, 0.0, 0.0]), dt=5.0, srp_force_eci=self.F_SRP_ECI)
        dh = np.linalg.norm(att.h_wheels - h_before)
        assert dh < 1e-20, f"Δ|h_wheels|={dh:.2e}"

    def test_srp_torque_direction_matches_r_cross_f(self, cfg_with_cp):
        """τ = r_cp × F_srp. With r_cp=[0,0,0.05] and F_srp=[1e-3,0,0], body-y dominates."""
        att = AttitudeDynamicsModel(config=cfg_with_cp)
        att.step(q_cmd=np.array([1.0, 0.0, 0.0, 0.0]), dt=1.0, srp_force_eci=self.F_SRP_ECI)
        h_eci = att.A_rw @ att.h_wheels
        assert abs(h_eci[1]) > abs(h_eci[0]) and abs(h_eci[1]) > abs(h_eci[2]), (
            f"h_eci={h_eci} — expected y to dominate"
        )

    def test_none_srp_force_is_backward_compatible(self, cfg_with_cp):
        """srp_force_eci=None must not raise and must leave wheels untouched."""
        att = AttitudeDynamicsModel(config=cfg_with_cp)
        h_before = att.h_wheels.copy()
        att.step(q_cmd=np.array([1.0, 0.0, 0.0, 0.0]), dt=5.0, srp_force_eci=None)
        dh = np.linalg.norm(att.h_wheels - h_before)
        assert dh < 1e-15, f"Δ|h_wheels|={dh:.2e}"


# ══════════════════════════════════════════════════════════════════════════════
# Section 21 — Reflected SRP boost (close-range inter-sail specular coupling)
# ══════════════════════════════════════════════════════════════════════════════
class TestReflectedSRPBoost:
    SUN_POS = np.array([AU, 0.0, 0.0])

    @staticmethod
    def _reflector_normal_for_tilt(tilt_deg: float) -> np.ndarray:
        half = math.radians(tilt_deg) / 2.0
        return np.array([math.cos(half), math.sin(half), 0.0])

    @staticmethod
    def _beam_axis(n_r: np.ndarray, sun_pos: np.ndarray, refl_pos: np.ndarray) -> np.ndarray:
        s_hat = (sun_pos - refl_pos) / np.linalg.norm(sun_pos - refl_pos)
        cos_i = float(np.dot(n_r, s_hat))
        d_refl = -s_hat + 2.0 * cos_i * n_r
        return d_refl / np.linalg.norm(d_refl)

    def test_on_axis_peak_matches_closed_form(self):
        refl = np.array([0.0, 0.0, 0.0])
        n_r = self._reflector_normal_for_tilt(30.0)
        d_refl = self._beam_axis(n_r, self.SUN_POS, refl)

        d = 100.0
        obs = refl + d * d_refl

        ratio, u_hat = reflected_flux_ratio(
            obs,
            self.SUN_POS,
            refl,
            32.0,
            n_r,
            1.0,
            specular_fraction=SailConst.SPECULAR_FRACTION,
            lobe_half_angle_rad=SailConst.SPECULAR_LOBE_HALF_ANGLE_RAD,
        )

        rho_s = SailConst.REFLECTIVITY * SailConst.SPECULAR_FRACTION
        cos_i = math.cos(math.radians(15.0))
        sigma = SailConst.SPECULAR_LOBE_HALF_ANGLE_RAD
        expected = (rho_s * 32.0 * cos_i * 1.0) / (math.pi * sigma * sigma * d * d)

        assert abs(ratio - expected) / expected < 0.05, (
            f"ratio={ratio:.5f}  expected={expected:.5f}"
        )
        assert abs(float(np.dot(u_hat, d_refl)) - 1.0) < 1e-9

    def test_three_sigma_offset_attenuates_to_tail(self):
        """Observer at 3σ off the beam axis receives ≤ 1.5% of on-axis peak."""
        refl = np.array([0.0, 0.0, 0.0])
        n_r = self._reflector_normal_for_tilt(30.0)
        d_refl = self._beam_axis(n_r, self.SUN_POS, refl)

        d = 100.0
        obs_on = refl + d * d_refl
        ratio_on, _ = reflected_flux_ratio(
            obs_on,
            self.SUN_POS,
            refl,
            32.0,
            n_r,
            1.0,
            specular_fraction=SailConst.SPECULAR_FRACTION,
            lobe_half_angle_rad=SailConst.SPECULAR_LOBE_HALF_ANGLE_RAD,
        )

        sigma = SailConst.SPECULAR_LOBE_HALF_ANGLE_RAD
        three_sigma = 3.0 * sigma
        c, s = math.cos(three_sigma), math.sin(three_sigma)
        d_off = np.array([c * d_refl[0] - s * d_refl[1], s * d_refl[0] + c * d_refl[1], d_refl[2]])
        obs_off = refl + d * d_off

        ratio_off, _ = reflected_flux_ratio(
            obs_off,
            self.SUN_POS,
            refl,
            32.0,
            n_r,
            1.0,
            specular_fraction=SailConst.SPECULAR_FRACTION,
            lobe_half_angle_rad=SailConst.SPECULAR_LOBE_HALF_ANGLE_RAD,
        )

        expected_attenuation = math.exp(-4.5)
        assert ratio_off / ratio_on < 1.5 * expected_attenuation, (
            f"ratio_off/ratio_on = {ratio_off / ratio_on:.6f} "
            f"(expected ≈ {expected_attenuation:.6f})"
        )

    def test_inverse_square_distance_scaling(self):
        """Doubling observer distance cuts the ratio by ≈ 4×."""
        refl = np.array([0.0, 0.0, 0.0])
        n_r = self._reflector_normal_for_tilt(30.0)
        d_refl = self._beam_axis(n_r, self.SUN_POS, refl)

        obs_close = refl + 100.0 * d_refl
        obs_far = refl + 200.0 * d_refl

        ratio_close, _ = reflected_flux_ratio(
            obs_close,
            self.SUN_POS,
            refl,
            32.0,
            n_r,
            1.0,
            specular_fraction=SailConst.SPECULAR_FRACTION,
            lobe_half_angle_rad=SailConst.SPECULAR_LOBE_HALF_ANGLE_RAD,
        )
        ratio_far, _ = reflected_flux_ratio(
            obs_far,
            self.SUN_POS,
            refl,
            32.0,
            n_r,
            1.0,
            specular_fraction=SailConst.SPECULAR_FRACTION,
            lobe_half_angle_rad=SailConst.SPECULAR_LOBE_HALF_ANGLE_RAD,
        )

        ratio_of_ratios = ratio_close / ratio_far
        assert abs(ratio_of_ratios - 4.0) / 4.0 < 1e-6, (
            f"ratio_close/ratio_far = {ratio_of_ratios:.6f} (expected 4.0)"
        )

    def test_reflector_in_shadow_produces_zero_boost(self):
        refl = np.array([0.0, 0.0, 0.0])
        n_r = self._reflector_normal_for_tilt(30.0)
        d_refl = self._beam_axis(n_r, self.SUN_POS, refl)
        obs = refl + 100.0 * d_refl

        ratio, u_hat = reflected_flux_ratio(
            obs,
            self.SUN_POS,
            refl,
            32.0,
            n_r,
            reflector_illumination=0.0,
            specular_fraction=SailConst.SPECULAR_FRACTION,
            lobe_half_angle_rad=SailConst.SPECULAR_LOBE_HALF_ANGLE_RAD,
        )
        assert ratio == 0.0
        assert np.allclose(u_hat, np.zeros(3))

    def test_reflector_back_face_produces_zero_boost(self):
        refl = np.array([0.0, 0.0, 0.0])
        n_r = np.array([-1.0, 0.0, 0.0])
        obs = refl + 100.0 * np.array([-1.0, 0.3, 0.0])
        obs = obs / np.linalg.norm(obs) * 100.0

        ratio, _ = reflected_flux_ratio(
            obs,
            self.SUN_POS,
            refl,
            32.0,
            n_r,
            1.0,
            specular_fraction=SailConst.SPECULAR_FRACTION,
            lobe_half_angle_rad=SailConst.SPECULAR_LOBE_HALF_ANGLE_RAD,
        )
        assert ratio == 0.0

    def test_observer_back_face_produces_zero_force(self):
        sail = SolarSailModel(area_m2=32.0, boom_shadow_fraction=0.0)
        sc_from_sun = np.array([-AU, 0.0, 0.0])
        arrival_dir = np.array([+1.0, 0.0, 0.0])

        sail_normal_back = np.array([+1.0, 0.0, 0.0])
        F_back = sail.reflected_force(
            sc_from_sun,
            sail_normal_back,
            reflected_flux_ratio=0.1,
            reflected_arrival_dir=arrival_dir,
        )
        assert np.allclose(F_back, np.zeros(3))

        sail_normal_front = np.array([-1.0, 0.0, 0.0])
        F_front = sail.reflected_force(
            sc_from_sun,
            sail_normal_front,
            reflected_flux_ratio=0.1,
            reflected_arrival_dir=arrival_dir,
        )
        assert np.linalg.norm(F_front) > 0.0

    def test_sun_normal_trailing_geometry_zero_boost(self):
        refl = np.array([0.0, 0.0, 0.0])
        n_r = np.array([+1.0, 0.0, 0.0])
        obs = refl + np.array([-10.0, 0.0, 0.0])

        ratio, _ = reflected_flux_ratio(
            obs,
            self.SUN_POS,
            refl,
            32.0,
            n_r,
            1.0,
            specular_fraction=SailConst.SPECULAR_FRACTION,
            lobe_half_angle_rad=SailConst.SPECULAR_LOBE_HALF_ANGLE_RAD,
        )
        assert ratio == 0.0
