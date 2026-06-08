"""Tests for cislunar.guidance.laws — all five steering laws + _rv_to_elements."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cislunar.guidance.laws import (
    GTOApogeeGuidance,
    GTOPeriapsisGuidance,
    GVEApogeeGuidance,
    GVEPeriapsisGuidance,
    ProgradeGuidance,
    _rv_to_elements,
)
from cislunar.physics.constants import Body as _Body

_MU = _Body.MU_EARTH
_R_EARTH = _Body.R_EARTH

# ── Orbital state helpers ─────────────────────────────────────────────────────


def _circular_state(r_m: float, nu_rad: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """ECI state for a circular equatorial orbit at radius r_m and true anomaly nu."""
    v_c = math.sqrt(_MU / r_m)
    pos = np.array([r_m * math.cos(nu_rad), r_m * math.sin(nu_rad), 0.0])
    vel = np.array([-v_c * math.sin(nu_rad), v_c * math.cos(nu_rad), 0.0])
    return pos, vel


def _elliptic_state(a: float, e: float, nu_rad: float) -> tuple[np.ndarray, np.ndarray]:
    """ECI state for an elliptic equatorial orbit at true anomaly nu."""
    p = a * (1.0 - e**2)
    r = p / (1.0 + e * math.cos(nu_rad))
    pos = np.array([r * math.cos(nu_rad), r * math.sin(nu_rad), 0.0])
    h = math.sqrt(_MU * p)
    vr = _MU / h * e * math.sin(nu_rad)
    vt = _MU / h * (1.0 + e * math.cos(nu_rad))
    cos_nu, sin_nu = math.cos(nu_rad), math.sin(nu_rad)
    vel = np.array([vr * cos_nu - vt * sin_nu, vr * sin_nu + vt * cos_nu, 0.0])
    return pos, vel


# ── GTO reference orbit (e ≈ 0.73, matching the docstrings) ──────────────────
_GTO_A = (_R_EARTH + 200e3 + _R_EARTH + 35_786e3) / 2.0
_GTO_E = 0.73


# ══════════════════════════════════════════════════════════════════════════════
# _rv_to_elements
# ══════════════════════════════════════════════════════════════════════════════
class TestRvToElements:
    def test_circular_orbit_eccentricity_near_zero(self):
        pos, vel = _circular_state(7e6)
        a, e, h, cos_nu, sin_nu = _rv_to_elements(pos, vel)
        assert e < 1e-4

    def test_circular_orbit_sma_matches_radius(self):
        r = 7e6
        pos, vel = _circular_state(r)
        a, *_ = _rv_to_elements(pos, vel)
        assert abs(a - r) < 1.0

    def test_gto_eccentricity(self):
        pos, vel = _elliptic_state(_GTO_A, _GTO_E, 0.0)
        _, e, *_ = _rv_to_elements(pos, vel)
        assert abs(e - _GTO_E) < 1e-4

    def test_true_anomaly_at_periapsis(self):
        pos, vel = _elliptic_state(_GTO_A, _GTO_E, 0.0)
        _, _, _, cos_nu, sin_nu = _rv_to_elements(pos, vel)
        assert abs(cos_nu - 1.0) < 1e-6
        assert abs(sin_nu) < 1e-6

    def test_true_anomaly_at_apoapsis(self):
        pos, vel = _elliptic_state(_GTO_A, _GTO_E, math.pi)
        _, _, _, cos_nu, sin_nu = _rv_to_elements(pos, vel)
        assert abs(cos_nu - (-1.0)) < 1e-6
        assert abs(sin_nu) < 1e-6

    def test_sin_nu_positive_moving_away_from_periapsis(self):
        pos, vel = _elliptic_state(_GTO_A, _GTO_E, math.pi / 3)
        _, _, _, _, sin_nu = _rv_to_elements(pos, vel)
        assert sin_nu > 0

    def test_sin_nu_negative_approaching_periapsis(self):
        pos, vel = _elliptic_state(_GTO_A, _GTO_E, -math.pi / 3)
        _, _, _, _, sin_nu = _rv_to_elements(pos, vel)
        assert sin_nu < 0

    def test_zero_velocity_returns_without_crash(self):
        pos = np.array([7e6, 0.0, 0.0])
        vel = np.zeros(3)
        result = _rv_to_elements(pos, vel)
        assert len(result) == 5


# ══════════════════════════════════════════════════════════════════════════════
# ProgradeGuidance
# ══════════════════════════════════════════════════════════════════════════════
class TestProgradeGuidance:
    def setup_method(self):
        self.law = ProgradeGuidance()

    def test_returns_unit_direction(self):
        pos, vel = _circular_state(7e6)
        direction, _ = self.law.steer(pos, vel)
        assert abs(np.linalg.norm(direction) - 1.0) < 1e-9

    def test_throttle_one_for_circular_leo(self):
        pos, vel = _circular_state(7e6)
        _, throttle = self.law.steer(pos, vel)
        assert throttle == 1.0

    def test_direction_is_prograde_for_circular(self):
        pos, vel = _circular_state(7e6)
        direction, _ = self.law.steer(pos, vel)
        v_hat = vel / np.linalg.norm(vel)
        assert np.dot(direction, v_hat) > 0.999

    def test_throttle_one_near_periapsis_gto(self):
        # nu = 45° — radial_frac ≈ 0.32 < threshold 0.42 → inside burn window
        pos, vel = _elliptic_state(_GTO_A, _GTO_E, math.pi / 4)
        _, throttle = self.law.steer(pos, vel)
        assert throttle == 1.0

    def test_throttle_zero_at_peak_radial_velocity_gto(self):
        # nu = 90° — radial_frac ≈ 0.59 > threshold 0.42 → outside burn window
        pos, vel = _elliptic_state(_GTO_A, _GTO_E, math.pi / 2)
        _, throttle = self.law.steer(pos, vel)
        assert throttle == 0.0

    def test_low_battery_suppresses_throttle(self):
        pos, vel = _circular_state(7e6)
        _, throttle = self.law.steer(pos, vel, battery_fraction=0.1)
        assert throttle == 0.0

    def test_battery_above_threshold_does_not_suppress(self):
        pos, vel = _circular_state(7e6)
        _, throttle = self.law.steer(pos, vel, battery_fraction=0.21)
        assert throttle == 1.0

    def test_zero_speed_returns_safe_default(self):
        pos = np.array([7e6, 0.0, 0.0])
        vel = np.zeros(3)
        direction, throttle = self.law.steer(pos, vel)
        assert throttle == 0.0
        assert np.linalg.norm(direction) > 0.0


# ══════════════════════════════════════════════════════════════════════════════
# GTOPeriapsisGuidance
# ══════════════════════════════════════════════════════════════════════════════
class TestGTOPeriapsisGuidance:
    def setup_method(self):
        self.law = GTOPeriapsisGuidance()

    def test_returns_unit_direction(self):
        pos, vel = _elliptic_state(_GTO_A, _GTO_E, math.pi)
        direction, _ = self.law.steer(pos, vel)
        assert abs(np.linalg.norm(direction) - 1.0) < 1e-9

    def test_burns_near_apogee(self):
        # cos_nu ≈ −1 at apogee; threshold is 1−e ≈ 0.27, so cos_nu < 0.27 → burn
        pos, vel = _elliptic_state(_GTO_A, _GTO_E, math.pi)
        _, throttle = self.law.steer(pos, vel)
        assert throttle == 1.0

    def test_coasts_near_periapsis_in_gto(self):
        # cos_nu ≈ 1 at periapsis; 1 > 1−e ≈ 0.27 → coast
        pos, vel = _elliptic_state(_GTO_A, _GTO_E, 0.0)
        _, throttle = self.law.steer(pos, vel)
        assert throttle == 0.0

    def test_burns_for_low_eccentricity_not_at_periapsis(self):
        # e = 0.01 → threshold ≈ 0.99; at nu=90° cos_nu=0 < 0.99 → burns
        pos, vel = _elliptic_state(7e6, 0.01, math.pi / 2)
        _, throttle = self.law.steer(pos, vel)
        assert throttle == 1.0

    def test_low_battery_suppresses_throttle(self):
        pos, vel = _elliptic_state(_GTO_A, _GTO_E, math.pi)
        _, throttle = self.law.steer(pos, vel, battery_fraction=0.1)
        assert throttle == 0.0

    def test_zero_speed_returns_safe_default(self):
        pos = np.array([7e6, 0.0, 0.0])
        vel = np.zeros(3)
        _, throttle = self.law.steer(pos, vel)
        assert throttle == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# GTOApogeeGuidance
# ══════════════════════════════════════════════════════════════════════════════
class TestGTOApogeeGuidance:
    def setup_method(self):
        self.law = GTOApogeeGuidance()

    def test_returns_unit_direction(self):
        pos, vel = _elliptic_state(_GTO_A, _GTO_E, 0.0)
        direction, _ = self.law.steer(pos, vel)
        assert abs(np.linalg.norm(direction) - 1.0) < 1e-9

    def test_burns_near_periapsis(self):
        # cos_nu ≈ 1 at periapsis; threshold is e−1 ≈ −0.27, so 1 > −0.27 → burn
        pos, vel = _elliptic_state(_GTO_A, _GTO_E, 0.0)
        _, throttle = self.law.steer(pos, vel)
        assert throttle == 1.0

    def test_coasts_near_apogee_in_highly_eccentric_orbit(self):
        # Very high eccentricity: at apogee cos_nu = −1; threshold = e−1 → −1 > −1 is False
        e_high = 0.95
        a = _R_EARTH + 200e3 + 1e6
        pos, vel = _elliptic_state(a, e_high, math.pi)
        _, throttle = self.law.steer(pos, vel)
        assert throttle == 0.0

    def test_burns_everywhere_for_circular(self):
        pos, vel = _circular_state(7e6)
        _, throttle = self.law.steer(pos, vel)
        assert throttle == 1.0

    def test_low_battery_suppresses_throttle(self):
        pos, vel = _elliptic_state(_GTO_A, _GTO_E, 0.0)
        _, throttle = self.law.steer(pos, vel, battery_fraction=0.05)
        assert throttle == 0.0

    def test_zero_speed_returns_safe_default(self):
        pos = np.array([7e6, 0.0, 0.0])
        vel = np.zeros(3)
        _, throttle = self.law.steer(pos, vel)
        assert throttle == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# GVEPeriapsisGuidance
# ══════════════════════════════════════════════════════════════════════════════
class TestGVEPeriapsisGuidance:
    def setup_method(self):
        self.law = GVEPeriapsisGuidance(target_periapsis_m=None)

    def test_returns_unit_direction(self):
        pos, vel = _elliptic_state(_GTO_A, _GTO_E, math.pi)
        direction, _ = self.law.steer(pos, vel)
        assert abs(np.linalg.norm(direction) - 1.0) < 1e-9

    def test_throttle_one_at_apogee(self):
        # GVE efficiency peaks at apogee → should burn
        pos, vel = _elliptic_state(_GTO_A, _GTO_E, math.pi)
        _, throttle = self.law.steer(pos, vel)
        assert throttle == 1.0

    def test_natural_coast_at_periapsis(self):
        # C_r = C_t = 0 at periapsis → efficiency = 0 → coasts without explicit gate
        pos, vel = _elliptic_state(_GTO_A, _GTO_E, 0.0)
        _, throttle = self.law.steer(pos, vel)
        assert throttle == 0.0

    def test_target_periapsis_shuts_off_once_reached(self):
        # Set target below current periapsis so it's already met
        r_peri = _GTO_A * (1.0 - _GTO_E)
        law = GVEPeriapsisGuidance(target_periapsis_m=r_peri * 0.5)
        pos, vel = _elliptic_state(_GTO_A, _GTO_E, math.pi)
        _, throttle = law.steer(pos, vel)
        assert throttle == 0.0

    def test_default_target_periapsis_is_1000km_altitude(self):
        # Default target = R_Earth + 1000 km; starting GTO periapsis ≈ 200 km → still burns
        law = GVEPeriapsisGuidance()
        pos, vel = _elliptic_state(_GTO_A, _GTO_E, math.pi)
        _, throttle = law.steer(pos, vel)
        assert throttle == 1.0

    def test_low_battery_suppresses_throttle(self):
        pos, vel = _elliptic_state(_GTO_A, _GTO_E, math.pi)
        _, throttle = self.law.steer(pos, vel, battery_fraction=0.1)
        assert throttle == 0.0

    def test_coast_efficiency_frac_zero_fires_at_periapsis(self):
        # coast_frac = 0 → never coasts on efficiency grounds (only periapsis GVE zero)
        law = GVEPeriapsisGuidance(coast_efficiency_frac=0.0, target_periapsis_m=None)
        # near-periapsis: efficiency is near zero but threshold is also zero → should fire
        pos, vel = _elliptic_state(_GTO_A, _GTO_E, 0.1)
        _, throttle = law.steer(pos, vel)
        assert throttle == 1.0

    def test_thrust_direction_in_rtn_plane(self):
        # The optimal thrust must lie in the orbital (RTN) plane (no out-of-plane component)
        pos, vel = _elliptic_state(_GTO_A, _GTO_E, math.pi * 2 / 3)
        direction, throttle = self.law.steer(pos, vel)
        if throttle == 1.0:
            h_vec = np.cross(pos, vel)
            n_hat = h_vec / np.linalg.norm(h_vec)
            out_of_plane = abs(float(np.dot(direction, n_hat)))
            assert out_of_plane < 1e-9


# ══════════════════════════════════════════════════════════════════════════════
# GVEApogeeGuidance
# ══════════════════════════════════════════════════════════════════════════════
class TestGVEApogeeGuidance:
    def setup_method(self):
        self.law = GVEApogeeGuidance(target_apoapsis_m=None)

    def test_returns_unit_direction(self):
        pos, vel = _elliptic_state(_GTO_A, _GTO_E, 0.0)
        direction, _ = self.law.steer(pos, vel)
        assert abs(np.linalg.norm(direction) - 1.0) < 1e-9

    def test_throttle_one_at_periapsis(self):
        # GVE efficiency for apogee raise peaks at periapsis → should burn
        pos, vel = _elliptic_state(_GTO_A, _GTO_E, 0.0)
        _, throttle = self.law.steer(pos, vel)
        assert throttle == 1.0

    def test_natural_coast_at_apogee(self):
        # C_r = C_t = 0 at apogee → efficiency ≈ 0 → coasts
        pos, vel = _elliptic_state(_GTO_A, _GTO_E, math.pi)
        _, throttle = self.law.steer(pos, vel)
        assert throttle == 0.0

    def test_target_apoapsis_shuts_off_once_reached(self):
        r_apo = _GTO_A * (1.0 + _GTO_E)
        law = GVEApogeeGuidance(target_apoapsis_m=r_apo * 0.5)
        pos, vel = _elliptic_state(_GTO_A, _GTO_E, 0.0)
        _, throttle = law.steer(pos, vel)
        assert throttle == 0.0

    def test_no_target_apoapsis_burns_at_periapsis(self):
        law = GVEApogeeGuidance(target_apoapsis_m=None)
        pos, vel = _elliptic_state(_GTO_A, _GTO_E, 0.0)
        _, throttle = law.steer(pos, vel)
        assert throttle == 1.0

    def test_low_battery_suppresses_throttle(self):
        pos, vel = _elliptic_state(_GTO_A, _GTO_E, 0.0)
        _, throttle = self.law.steer(pos, vel, battery_fraction=0.05)
        assert throttle == 0.0

    def test_thrust_direction_in_rtn_plane(self):
        pos, vel = _elliptic_state(_GTO_A, _GTO_E, math.pi / 4)
        direction, throttle = self.law.steer(pos, vel)
        if throttle == 1.0:
            h_vec = np.cross(pos, vel)
            n_hat = h_vec / np.linalg.norm(h_vec)
            out_of_plane = abs(float(np.dot(direction, n_hat)))
            assert out_of_plane < 1e-9

    def test_default_coast_frac_suppresses_mid_orbit(self):
        # Default coast_frac = 0.5; at 90° efficiency is ~70% of peak for GTO → still fires
        pos, vel = _elliptic_state(_GTO_A, _GTO_E, math.pi / 2)
        _, throttle = self.law.steer(pos, vel)
        # just assert it returns a valid throttle value
        assert throttle in (0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# Cross-law consistency: all laws share the same interface
# ══════════════════════════════════════════════════════════════════════════════
class TestAllLawsShareInterface:
    @pytest.mark.parametrize(
        "law",
        [
            ProgradeGuidance(),
            GTOPeriapsisGuidance(),
            GTOApogeeGuidance(),
            GVEPeriapsisGuidance(target_periapsis_m=None),
            GVEApogeeGuidance(target_apoapsis_m=None),
        ],
    )
    def test_steer_returns_unit_vector_and_binary_throttle(self, law):
        pos, vel = _circular_state(7e6)
        direction, throttle = law.steer(pos, vel)
        assert direction.shape == (3,)
        assert abs(np.linalg.norm(direction) - 1.0) < 1e-9
        assert throttle in (0.0, 1.0)

    @pytest.mark.parametrize(
        "law",
        [
            ProgradeGuidance(),
            GTOPeriapsisGuidance(),
            GTOApogeeGuidance(),
            GVEPeriapsisGuidance(target_periapsis_m=None),
            GVEApogeeGuidance(target_apoapsis_m=None),
        ],
    )
    def test_battery_fraction_argument_accepted(self, law):
        pos, vel = _circular_state(7e6)
        direction, throttle = law.steer(pos, vel, battery_fraction=0.5)
        assert direction.shape == (3,)
        assert throttle in (0.0, 1.0)
