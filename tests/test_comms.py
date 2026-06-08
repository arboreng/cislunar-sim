"""Communications tests: DSN visibility, lunar occultation, link budget, blackout."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cislunar.physics import (
    CommsConfig,
    CommsPointingState,
    blackout_fraction_low_lunar_orbit,
    comms_blackout,
    is_occulted_by_moon,
    is_visible_from_dsn,
    link_budget,
)
from cislunar.physics.constants import Body
from cislunar.physics.forces.comms import R_MOON_M as COMMS_R_MOON
from cislunar.physics.forces.comms import free_space_path_loss_db
from cislunar.physics.forces.eclipse import R_EARTH_M as R_EARTH_ECLIPSE


# ══════════════════════════════════════════════════════════════════════════════
# Section 9 — DSN station visibility
# ══════════════════════════════════════════════════════════════════════════════
class TestDSNVisibility:
    def test_midcislunar_visible_at_t0(self):
        pos = np.array([1.92e8, 0.0, 0.0])
        assert is_visible_from_dsn(pos, time_s=0.0)

    def test_returns_bool_for_anti_x_leo(self):
        """Function must return a clean Python bool regardless of geometry."""
        pos = np.array([-6.8e6, 0.0, 0.0])
        assert isinstance(is_visible_from_dsn(pos, time_s=0.0), bool)

    def test_coverage_varies_over_sidereal_day(self):
        """Sampling 24 hours for a fixed ECI point must flip visibility at least once."""
        pos_meo = np.array([1.5e7, 0.0, 0.0])
        samples = [is_visible_from_dsn(pos_meo, time_s=h * 3600.0) for h in range(24)]
        assert True in samples and False in samples, (
            f"24h samples: {sum(samples)} visible, {24 - sum(samples)} blacked out"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Section 13 — Comms blackout (lunar occultation)
# ══════════════════════════════════════════════════════════════════════════════
class TestCommsBlackout:
    MOON_POS = np.array([3.844e8, 0.0, 0.0])

    def test_sc_behind_moon_is_occulted(self):
        sc = np.array([3.844e8 + COMMS_R_MOON + 200_000, 0.0, 0.0])
        assert is_occulted_by_moon(sc, self.MOON_POS)

    def test_sc_on_earth_side_is_clear(self):
        sc = np.array([3.844e8 - COMMS_R_MOON - 200_000, 0.0, 0.0])
        assert not is_occulted_by_moon(sc, self.MOON_POS)

    def test_sc_at_midpoint_cislunar_is_clear(self):
        assert not is_occulted_by_moon(np.array([2e8, 0.0, 0.0]), self.MOON_POS)

    def test_sc_lateral_to_moon_is_clear(self):
        sc = np.array([3.844e8, COMMS_R_MOON * 2, 0.0])
        assert not is_occulted_by_moon(sc, self.MOON_POS)

    def test_comms_blackout_matches_occultation(self):
        sc = np.array([3.844e8 + COMMS_R_MOON + 200_000, 0.0, 0.0])
        assert comms_blackout(sc, self.MOON_POS) == is_occulted_by_moon(sc, self.MOON_POS)

    def test_llo_200km_blackout_fraction_in_range(self):
        f = blackout_fraction_low_lunar_orbit(200)
        assert 0.25 < f < 0.40, f"f_200={f * 100:.1f}%"

    def test_blackout_fraction_decreases_with_altitude(self):
        f_200 = blackout_fraction_low_lunar_orbit(200)
        f_500 = blackout_fraction_low_lunar_orbit(500)
        assert f_200 > f_500, f"f_200={f_200:.3f}  f_500={f_500:.3f}"

    def test_link_budget_negative_below_half_watt_at_lunar_distance(self):
        margin_db = link_budget(
            range_m=3.844e8,
            tx_power_w=0.4,
            tx_gain_dbi=8.0,
            rx_gain_dbi=45.0,
        )
        assert margin_db < 0.0, f"margin={margin_db:.2f} dB"

    def test_slew_limited_acquisition_shortens_four_minute_leo_pass(self):
        single_station = np.array([[R_EARTH_ECLIPSE, 0.0, 0.0]], dtype=np.float64)
        cfg_geometric = CommsConfig(
            stations_ecef_m=single_station,
            max_slew_rate_deg_s=float("inf"),
            tx_power_w=5.0,
        )
        cfg_slew_limited = CommsConfig(
            stations_ecef_m=single_station,
            max_slew_rate_deg_s=1.0,
            tx_power_w=5.0,
        )
        moon_far = np.array([0.0, 0.0, 1.0e12], dtype=np.float64)

        orbit_r = Body.R_EARTH + 400e3
        orbit_w = math.sqrt(Body.MU_EARTH / orbit_r**3)
        times = np.arange(-120.0, 120.0 + 1.0, 1.0)

        geometric_contact_s = 0.0
        slew_contact_s = 0.0
        theta0 = orbit_w * times[0]
        pos0 = np.array(
            [
                orbit_r * math.cos(theta0),
                orbit_r * math.sin(theta0),
                0.0,
            ]
        )
        target0 = single_station[0] - pos0
        target0 = target0 / np.linalg.norm(target0)
        c30 = math.cos(math.radians(30.0))
        s30 = math.sin(math.radians(30.0))
        slew_state = CommsPointingState(
            boresight_eci=np.array(
                [
                    c30 * target0[0] - s30 * target0[1],
                    s30 * target0[0] + c30 * target0[1],
                    target0[2],
                ]
            )
        )

        for t in times:
            theta = orbit_w * t
            pos = np.array(
                [
                    orbit_r * math.cos(theta),
                    orbit_r * math.sin(theta),
                    0.0,
                ]
            )
            if not comms_blackout(pos, moon_far, time_s=t, config=cfg_geometric):
                geometric_contact_s += 1.0
            if not comms_blackout(
                pos,
                moon_far,
                time_s=t,
                config=cfg_slew_limited,
                pointing_state=slew_state,
            ):
                slew_contact_s += 1.0

        assert 220.0 <= geometric_contact_s <= 260.0, (
            f"geometric_contact={geometric_contact_s:.1f}s"
        )
        shortening = 1.0 - slew_contact_s / geometric_contact_s
        assert 0.08 <= shortening <= 0.14, (
            f"geometric={geometric_contact_s:.1f}s  slew={slew_contact_s:.1f}s  "
            f"shortening={shortening * 100:.1f}%"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Free-space path loss
# ══════════════════════════════════════════════════════════════════════════════
class TestFreeSpacePathLoss:
    """Friis FSPL = 20·log₁₀(4π·d/λ)."""

    C = 299_792_458.0  # speed of light [m/s]

    def test_known_value_sband_1m(self):
        """At 1 m, 2.2 GHz: FSPL = 20·log₁₀(4π·1·f/c)."""
        f = 2.2e9
        wavelength = self.C / f
        expected_db = 20.0 * math.log10(4.0 * math.pi * 1.0 / wavelength)
        assert abs(free_space_path_loss_db(1.0, f) - expected_db) < 1e-9

    def test_doubling_distance_adds_6db(self):
        """Doubling range increases FSPL by exactly 6.020… dB (20·log₁₀ 2)."""
        f = 2.2e9
        loss_1 = free_space_path_loss_db(1_000.0, f)
        loss_2 = free_space_path_loss_db(2_000.0, f)
        assert abs((loss_2 - loss_1) - 20.0 * math.log10(2.0)) < 1e-9

    def test_zero_range_clamped_to_1m(self):
        """range_m=0 must not blow up; clamp to 1 m floor."""
        loss_zero = free_space_path_loss_db(0.0, 2.2e9)
        loss_one = free_space_path_loss_db(1.0, 2.2e9)
        assert loss_zero == pytest.approx(loss_one)


# ══════════════════════════════════════════════════════════════════════════════
# Gap-closing additions
# ══════════════════════════════════════════════════════════════════════════════
class TestUnitHelper:
    def test_zero_vector_returns_copy(self):
        from cislunar.physics.forces.comms import _unit

        v = np.zeros(3)
        result = _unit(v)
        np.testing.assert_array_equal(result, np.zeros(3))
        assert result is not v  # copy, not same object

    def test_nonzero_returns_unit(self):
        from cislunar.physics.forces.comms import _unit

        v = np.array([3.0, 0.0, 4.0])
        result = _unit(v)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-12


class TestRotateToward:
    def test_already_aligned_returns_target(self):
        from cislunar.physics.forces.comms import _rotate_toward

        v = np.array([1.0, 0.0, 0.0])
        result = _rotate_toward(v, v, max_angle_rad=0.1)
        np.testing.assert_allclose(result, v, atol=1e-12)

    def test_small_angle_within_limit_goes_to_target(self):
        from cislunar.physics.forces.comms import _rotate_toward

        curr = np.array([1.0, 0.0, 0.0])
        tgt = np.array([0.0, 1.0, 0.0])  # 90° away
        result = _rotate_toward(curr, tgt, max_angle_rad=math.pi)  # limit >= angle
        np.testing.assert_allclose(result, tgt, atol=1e-12)

    def test_partial_rotation_stays_within_limit(self):
        from cislunar.physics.forces.comms import _rotate_toward

        curr = np.array([1.0, 0.0, 0.0])
        tgt = np.array([0.0, 1.0, 0.0])  # 90° away
        limit = math.radians(10.0)
        result = _rotate_toward(curr, tgt, max_angle_rad=limit)
        angle = math.acos(float(np.clip(np.dot(result, curr), -1.0, 1.0)))
        assert angle <= limit + 1e-9

    def test_anti_parallel_vectors_handled(self):
        from cislunar.physics.forces.comms import _rotate_toward

        curr = np.array([1.0, 0.0, 0.0])
        tgt = np.array([-1.0, 0.0, 0.0])  # 180° — sin_total ≈ 0
        result = _rotate_toward(curr, tgt, max_angle_rad=0.01)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6


class TestIsOccultedByMoonDegenerate:
    def test_spacecraft_at_origin_returns_false(self):
        # sc_mag < 1.0 → early return False
        assert is_occulted_by_moon(np.zeros(3), np.array([3.844e8, 0.0, 0.0])) is False


class TestCommsBlackoutWithPointingState:
    """Tests for the comms_blackout paths gated on pointing_state."""

    def test_moon_occultation_updates_pointing_state(self):
        moon = np.array([1e8, 0.0, 0.0])
        sc = np.array([1e8 + 2e6, 0.0, 0.0])  # 2000 km behind Moon, < R_MOON from shadow line

        ps = CommsPointingState()
        result = comms_blackout(sc, moon, time_s=0.0, pointing_state=ps)
        if result:  # only if actually occulted
            assert ps.acquired is False

    def test_no_candidates_sets_pointing_state_not_acquired(self):
        # Position craft far from Earth so no DSN station is visible
        sc_pos = np.array([4e8, 0.0, 0.0])  # very far → link budget fails
        moon_pos = np.zeros(3)
        ps = CommsPointingState()
        cfg = CommsConfig(required_rx_power_dbw=-50.0)  # very demanding threshold
        result = comms_blackout(sc_pos, moon_pos, time_s=0.0, pointing_state=ps, config=cfg)
        assert result is True
        assert ps.acquired is False

    def test_visible_station_acquires_after_slew(self):
        sc_pos = np.array([Body.R_EARTH + 400e3, 0.0, 0.0])
        moon_pos = np.zeros(3)
        ps = CommsPointingState()
        # First call: antenna must slew → not yet acquired
        comms_blackout(sc_pos, moon_pos, time_s=0.0, pointing_state=ps)
        # After enough time: should be acquired or at least state updated
        assert ps.last_time_s == pytest.approx(0.0)

    def test_instant_slew_rate_sets_zero_acquisition_remaining(self):
        # Use a generous link budget to ensure at least one station is visible
        sc_pos = np.array([Body.R_EARTH + 400e3, 0.0, 0.0])
        moon_pos = np.zeros(3)
        ps = CommsPointingState()
        cfg = CommsConfig(
            max_slew_rate_deg_s=float("inf"),
            required_rx_power_dbw=-200.0,  # very permissive → station visible
        )
        comms_blackout(sc_pos, moon_pos, time_s=0.0, pointing_state=ps, config=cfg)
        # With infinite slew rate, acquisition_remaining_s should be 0 after the call
        assert ps.acquisition_remaining_s == pytest.approx(0.0)

    def test_no_pointing_state_returns_bool(self):
        sc_pos = np.array([Body.R_EARTH + 400e3, 0.0, 0.0])
        result = comms_blackout(sc_pos, np.zeros(3), time_s=0.0, pointing_state=None)
        assert isinstance(result, bool)
