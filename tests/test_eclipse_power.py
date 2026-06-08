"""Eclipse detection, shadow fraction, power model, and attitude-power coupling tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cislunar.physics import (
    R_SUN_M,
    Action,
    GravityModel,
    IonThrusterModel,
    PowerBudgetModel,
    SolarSailModel,
    Spacecraft,
    SpacecraftState,
    eclipse_fraction_circular_orbit,
    is_in_eclipse,
    shadow_fraction,
)
from cislunar.physics.constants import AU, Body
from cislunar.physics.forces.eclipse import R_EARTH_M as R_EARTH_ECLIPSE
from cislunar.physics.forces.eclipse import SolarArrayConfig


# ══════════════════════════════════════════════════════════════════════════════
# Section 8 — Eclipse detection & power model
# ══════════════════════════════════════════════════════════════════════════════
class TestEclipseAndPower:
    """Boolean eclipse detection, fractional shadow geometry, and battery dynamics."""

    @pytest.fixture(scope="class")
    def geometry(self):
        """Sun at +1 AU on +x, two test points on either side of Earth."""
        return dict(
            sun=np.array([AU, 0.0, 0.0]),
            behind_earth=np.array([-7e6, 0.0, 0.0]),
            sun_side=np.array([7e6, 0.0, 0.0]),
            perpendicular=np.array([0.0, 8e6, 0.0]),
        )

    def test_eclipse_detected_behind_earth(self, geometry):
        assert is_in_eclipse(geometry["behind_earth"], geometry["sun"])

    def test_no_eclipse_on_sun_side(self, geometry):
        assert not is_in_eclipse(geometry["sun_side"], geometry["sun"])

    def test_no_eclipse_perpendicular_to_sun_line(self, geometry):
        assert not is_in_eclipse(geometry["perpendicular"], geometry["sun"])

    def test_eclipse_fraction_at_leo(self):
        f = eclipse_fraction_circular_orbit(400e3)
        assert 0.33 < f < 0.42, f"f_LEO={f * 100:.1f}% (expect 33–42%)"

    def test_eclipse_fraction_decreases_with_altitude(self):
        f_leo = eclipse_fraction_circular_orbit(400e3)
        f_geo = eclipse_fraction_circular_orbit(35_786e3)
        assert f_leo > f_geo, f"f_LEO={f_leo:.3f}  f_GEO={f_geo:.3f}"

    def test_power_available_in_sunlight(self):
        power = PowerBudgetModel()
        pos_sun = np.array([6.771e6, 0.0, 0.0])
        sun_pos = np.array([AU, 0.0, 0.0])
        helio = pos_sun - sun_pos
        p = power.step(pos_sun, sun_pos, helio, dt_s=60.0)
        assert p > 0, f"P={p:.2f} W"

    def test_in_eclipse_flag_false_in_sunlight(self):
        power = PowerBudgetModel()
        pos_sun = np.array([6.771e6, 0.0, 0.0])
        sun_pos = np.array([AU, 0.0, 0.0])
        power.step(pos_sun, sun_pos, pos_sun - sun_pos, dt_s=60.0)
        assert not power.in_eclipse

    def test_in_eclipse_flag_true_on_shadow_side(self):
        power = PowerBudgetModel()
        pos_shadow = np.array([-6.771e6, 0.0, 0.0])
        sun_pos = np.array([AU, 0.0, 0.0])
        power.step(pos_shadow, sun_pos, pos_shadow - sun_pos, dt_s=60.0)
        assert power.in_eclipse

    def test_battery_drains_during_sustained_eclipse(self):
        pm = PowerBudgetModel()
        pos_shadow = np.array([-6.771e6, 0.0, 0.0])
        sun_pos = np.array([AU, 0.0, 0.0])
        helio = pos_shadow - sun_pos
        for _ in range(200):
            pm.step(pos_shadow, sun_pos, helio, dt_s=60.0)
        assert pm.battery_pct < 80.0, f"batt={pm.battery_pct:.1f}%"

    def test_battery_recharges_in_sunlight_after_eclipse(self):
        pm = PowerBudgetModel()
        pos_shadow = np.array([-6.771e6, 0.0, 0.0])
        pos_sun = np.array([6.771e6, 0.0, 0.0])
        sun_pos = np.array([AU, 0.0, 0.0])
        for _ in range(200):
            pm.step(pos_shadow, sun_pos, pos_shadow - sun_pos, dt_s=60.0)
        soc_after_eclipse = pm.battery_pct
        for _ in range(100):
            pm.step(pos_sun, sun_pos, pos_sun - sun_pos, dt_s=60.0)
        assert pm.battery_pct > soc_after_eclipse, (
            f"{soc_after_eclipse:.1f}% → {pm.battery_pct:.1f}% (expected increase)"
        )

    class TestShadowFraction:
        """Fractional illumination at umbra boundary, penumbra midpoint, full sun."""

        def test_full_sunlight_returns_one(self):
            pos_sun = np.array([6.771e6, 0.0, 0.0])
            sun_pos = np.array([AU, 0.0, 0.0])
            f = shadow_fraction(pos_sun, sun_pos)
            assert f == 1.0, f"f={f}"

        def test_deep_umbra_returns_zero(self):
            pos = np.array([-7.5e6, 0.0, 0.0])
            sun_pos = np.array([AU, 0.0, 0.0])
            f = shadow_fraction(pos, sun_pos)
            assert f == 0.0, f"f={f:.4f}"

        def test_umbra_agrees_with_is_in_eclipse(self):
            """``shadow_fraction == 0`` iff ``is_in_eclipse`` is True."""
            pos = np.array([-7.5e6, 0.0, 0.0])
            sun_pos = np.array([AU, 0.0, 0.0])
            f = shadow_fraction(pos, sun_pos)
            assert is_in_eclipse(pos, sun_pos) == (f == 0.0)

        def test_penumbra_strictly_between_0_and_1(self):
            sun_pos = np.array([AU, 0.0, 0.0])
            d = R_EARTH_ECLIPSE + 400e3
            r_u = R_EARTH_ECLIPSE - d * (R_SUN_M - R_EARTH_ECLIPSE) / AU
            r_p = R_EARTH_ECLIPSE + d * R_SUN_M / AU
            pos_penumbra = np.array([-d, (r_u + r_p) / 2.0, 0.0])
            f = shadow_fraction(pos_penumbra, sun_pos)
            assert 0.0 < f < 1.0, f"f={f:.4f}"

        def test_penumbra_midpoint_near_half(self):
            sun_pos = np.array([AU, 0.0, 0.0])
            d = R_EARTH_ECLIPSE + 400e3
            r_u = R_EARTH_ECLIPSE - d * (R_SUN_M - R_EARTH_ECLIPSE) / AU
            r_p = R_EARTH_ECLIPSE + d * R_SUN_M / AU
            pos_penumbra = np.array([-d, (r_u + r_p) / 2.0, 0.0])
            f = shadow_fraction(pos_penumbra, sun_pos)
            assert abs(f - 0.5) < 0.1, f"f={f:.4f} (expect ~0.5 ± 0.1)"


# ══════════════════════════════════════════════════════════════════════════════
# Section 8b — Power model state wiring
# ══════════════════════════════════════════════════════════════════════════════
class TestPowerModelStateWiring:
    def test_power_model_receives_post_step_position(self):
        R = Body.R_EARTH + 400e3
        pos = np.array([R, 0.0, 0.0])
        vel = np.array([0.0, math.sqrt(Body.MU_EARTH / R), 0.0])
        state = SpacecraftState(
            position=pos,
            velocity=vel,
            sail_normal=np.array([-1.0, 0.0, 0.0]),
            propellant_kg=0.5,
            mass_dry_kg=12.0,
        )

        power = PowerBudgetModel()
        captured_pos = []
        original_step = power.step

        def recording_step(**kwargs):
            captured_pos.append(kwargs["spacecraft_pos"].copy())
            return original_step(**kwargs)

        power.step = recording_step

        sc = Spacecraft(
            initial_state=state,
            sail_model=SolarSailModel(area_m2=32.0),
            thruster_model=IonThrusterModel(),
            gravity_model=GravityModel().add_body(
                "Earth",
                mu=Body.MU_EARTH,
                j2=Body.J2_EARTH,
                r_body=Body.R_EARTH,
            ),
            sun_pos_fn=lambda t: np.array([AU, 0.0, 0.0]),
            power_model=power,
        )
        sc.step(Action(), dt_requested=60.0)

        assert captured_pos, "power.step() was never called"
        lag = float(np.linalg.norm(captured_pos[-1] - sc.state.position))
        assert lag < 1.0, f"lag={lag:.3f} m"


# ══════════════════════════════════════════════════════════════════════════════
# Section 10 — Attitude-power coupling
# ══════════════════════════════════════════════════════════════════════════════
class TestAttitudePowerCoupling:
    """Sail orientation modulates solar-array yield and thruster power budget."""

    @pytest.fixture
    def geometry(self):
        pos = np.array([6.771e6, 0.0, 0.0])
        sun = np.array([AU, 0.0, 0.0])
        helio = pos - sun
        return dict(pos=pos, sun=sun, helio=helio)

    def test_sail_toward_sun_beats_perpendicular(self, geometry):
        pm_toward = PowerBudgetModel()
        pm_perp = PowerBudgetModel()
        pm_toward.step(
            geometry["pos"],
            geometry["sun"],
            geometry["helio"],
            dt_s=60.0,
            sail_normal_eci=np.array([1.0, 0.0, 0.0]),
        )
        pm_perp.step(
            geometry["pos"],
            geometry["sun"],
            geometry["helio"],
            dt_s=60.0,
            sail_normal_eci=np.array([0.0, 1.0, 0.0]),
        )
        assert pm_toward.P_solar_w > pm_perp.P_solar_w and pm_toward.P_solar_w > 0, (
            f"toward={pm_toward.P_solar_w:.2f}W  perp={pm_perp.P_solar_w:.2f}W"
        )

    def test_sail_perpendicular_gives_zero_solar(self, geometry):
        pm = PowerBudgetModel()
        pm.step(
            geometry["pos"],
            geometry["sun"],
            geometry["helio"],
            dt_s=60.0,
            sail_normal_eci=np.array([0.0, 1.0, 0.0]),
        )
        assert abs(pm.P_solar_w) < 0.01, f"P_solar={pm.P_solar_w:.4f} W"

    def test_sail_facing_away_gives_zero_solar(self, geometry):
        pm = PowerBudgetModel()
        pm.step(
            geometry["pos"],
            geometry["sun"],
            geometry["helio"],
            dt_s=60.0,
            sail_normal_eci=np.array([-1.0, 0.0, 0.0]),
        )
        assert abs(pm.P_solar_w) < 0.01, f"P_solar={pm.P_solar_w:.4f} W"

    def test_thruster_power_respects_c_rate_cap(self, geometry):
        """P_thruster cannot exceed solar surplus + peak battery C-rate."""
        pm = PowerBudgetModel()
        p_thruster = pm.step(
            geometry["pos"],
            geometry["sun"],
            geometry["helio"],
            dt_s=60.0,
            sail_normal_eci=np.array([1.0, 0.0, 0.0]),
        )
        expected_max = pm.P_solar_w - pm.P_hk + pm.c_rate_peak_w
        assert p_thruster <= expected_max + 0.01, (
            f"P_thruster={p_thruster:.2f}W > cap {expected_max:.2f}W"
        )

    def test_thruster_power_below_legacy_burst(self, geometry):
        """Regression guard: P_thruster must not balloon to the old ~997 W value."""
        pm = PowerBudgetModel()
        p = pm.step(
            geometry["pos"],
            geometry["sun"],
            geometry["helio"],
            dt_s=60.0,
            sail_normal_eci=np.array([1.0, 0.0, 0.0]),
        )
        assert p < 100.0, f"P_thruster={p:.1f}W — regressed to legacy burst"

    def test_requested_eclipse_burn_debits_battery_for_thruster_draw(self):
        """An eclipse burn must drain battery at housekeeping plus propulsion load."""
        pm = PowerBudgetModel()
        pos_shadow = np.array([-6.771e6, 0.0, 0.0])
        sun_pos = np.array([AU, 0.0, 0.0])
        helio = pos_shadow - sun_pos
        batt_before = pm.batt_energy_j

        p_thr = pm.step(
            pos_shadow,
            sun_pos,
            helio,
            dt_s=60.0,
            propulsion_power_request_w=pm.c_rate_peak_w,
        )

        expected_thruster_w = pm.c_rate_peak_w - pm.P_hk
        expected_drop_j = (pm.P_hk + expected_thruster_w) * 60.0 / pm.eta_batt
        actual_drop_j = batt_before - pm.batt_energy_j

        assert p_thr == pytest.approx(expected_thruster_w, rel=1e-9)
        assert actual_drop_j == pytest.approx(expected_drop_j, rel=1e-9), (
            f"battery drop {actual_drop_j:.2f} J != expected {expected_drop_j:.2f} J"
        )

    def test_better_attitude_gives_more_thruster_power(self, geometry):
        pm_good = PowerBudgetModel()
        pm_bad = PowerBudgetModel()
        p_good = pm_good.step(
            geometry["pos"],
            geometry["sun"],
            geometry["helio"],
            dt_s=60.0,
            sail_normal_eci=np.array([1.0, 0.0, 0.0]),
        )
        p_bad = pm_bad.step(
            geometry["pos"],
            geometry["sun"],
            geometry["helio"],
            dt_s=60.0,
            sail_normal_eci=np.array([-1.0, 0.0, 0.0]),
        )
        assert p_good > p_bad, f"p_toward={p_good:.1f}W  p_away={p_bad:.1f}W"

    def test_panel_offset_reduces_peak_power_by_cos_30(self, geometry):
        q_30deg_y = np.array(
            [
                math.cos(math.radians(15.0)),
                0.0,
                math.sin(math.radians(15.0)),
                0.0,
            ]
        )
        pm_zero = PowerBudgetModel()
        pm_off = PowerBudgetModel(array_cfg=SolarArrayConfig(panel_body_offset_q=q_30deg_y))

        pm_zero.step(
            geometry["pos"],
            geometry["sun"],
            geometry["helio"],
            dt_s=60.0,
            sail_normal_eci=np.array([1.0, 0.0, 0.0]),
        )
        pm_off.step(
            geometry["pos"],
            geometry["sun"],
            geometry["helio"],
            dt_s=60.0,
            sail_normal_eci=np.array([1.0, 0.0, 0.0]),
        )

        ratio = pm_off.P_solar_w / pm_zero.P_solar_w
        assert 0.85 < ratio < 0.88, f"ratio={ratio:.3f} (expected ~0.866)"


# ══════════════════════════════════════════════════════════════════════════════
# Gap-closing additions
# ══════════════════════════════════════════════════════════════════════════════
class TestShadowFractionDegenerateSun:
    def test_sun_at_origin_returns_one(self):
        """r_sun < 1 → degenerate, assume sunlit → return 1.0."""
        sc_pos = np.array([Body.R_EARTH + 400e3, 0.0, 0.0])
        frac = shadow_fraction(sc_pos, sun_pos=np.zeros(3))
        assert frac == pytest.approx(1.0)


class TestSailOccultationFractionGuards:
    def test_degenerate_sun_at_blocker_returns_one(self):
        from cislunar.physics.forces.eclipse import sail_occultation_fraction

        sc_pos = np.array([Body.R_EARTH + 400e3, 0.0, 0.0])
        blocker_pos = np.array([Body.R_EARTH + 200e3, 0.0, 0.0])
        result = sail_occultation_fraction(
            observer_pos_eci=sc_pos,
            sun_pos_eci=blocker_pos,  # sun co-located with blocker → r_sun < 1
            blocker_pos_eci=blocker_pos,
            blocker_area_m2=10.0,
            blocker_normal_eci=np.array([1.0, 0.0, 0.0]),
        )
        assert result == pytest.approx(1.0)

    def test_back_face_blocker_returns_one(self):
        from cislunar.physics.forces.eclipse import sail_occultation_fraction

        sc_pos = np.array([Body.R_EARTH + 400e3, 0.0, 0.0])
        blocker_pos = np.array([Body.R_EARTH + 200e3, 0.0, 0.0])
        sun_pos = np.array([AU, 0.0, 0.0])
        # normal points away from sun → cos_theta_blocker ≤ 0
        back_normal = np.array([-1.0, 0.0, 0.0])
        result = sail_occultation_fraction(
            observer_pos_eci=sc_pos,
            sun_pos_eci=sun_pos,
            blocker_pos_eci=blocker_pos,
            blocker_area_m2=10.0,
            blocker_normal_eci=back_normal,
        )
        assert result == pytest.approx(1.0)


class TestReflectedFluxRatioGuards:
    _LOBE = 0.1  # radians

    def test_reflector_in_shadow_returns_zero(self):
        from cislunar.physics.forces.eclipse import reflected_flux_ratio

        sc = np.array([Body.R_EARTH + 400e3, 0.0, 0.0])
        refl_pos = np.array([Body.R_EARTH + 200e3, 0.0, 0.0])
        result, direction = reflected_flux_ratio(
            observer_pos_eci=sc,
            sun_pos_eci=np.array([AU, 0.0, 0.0]),
            reflector_pos_eci=refl_pos,
            reflector_area_m2=32.0,
            reflector_normal_eci=np.array([1.0, 0.0, 0.0]),
            reflector_illumination=0.0,  # in shadow → early return
            specular_fraction=0.9,
            lobe_half_angle_rad=self._LOBE,
        )
        assert result == pytest.approx(0.0)
        np.testing.assert_array_equal(direction, np.zeros(3))

    def test_degenerate_sun_at_reflector_returns_zero(self):
        from cislunar.physics.forces.eclipse import reflected_flux_ratio

        sc = np.array([Body.R_EARTH + 400e3, 0.0, 0.0])
        refl_pos = np.array([Body.R_EARTH + 200e3, 0.0, 0.0])
        result, _ = reflected_flux_ratio(
            observer_pos_eci=sc,
            sun_pos_eci=refl_pos,  # sun co-located with reflector → r_sun < 1
            reflector_pos_eci=refl_pos,
            reflector_area_m2=32.0,
            reflector_normal_eci=np.array([1.0, 0.0, 0.0]),
            reflector_illumination=1.0,
            specular_fraction=0.9,
            lobe_half_angle_rad=self._LOBE,
        )
        assert result == pytest.approx(0.0)

    def test_back_face_reflector_returns_zero(self):
        from cislunar.physics.forces.eclipse import reflected_flux_ratio

        sc = np.array([Body.R_EARTH + 400e3, 0.0, 0.0])
        refl_pos = np.array([Body.R_EARTH + 200e3, 0.0, 0.0])
        result, _ = reflected_flux_ratio(
            observer_pos_eci=sc,
            sun_pos_eci=np.array([AU, 0.0, 0.0]),
            reflector_pos_eci=refl_pos,
            reflector_area_m2=32.0,
            reflector_normal_eci=np.array([-1.0, 0.0, 0.0]),  # back face toward sun
            reflector_illumination=1.0,
            specular_fraction=0.9,
            lobe_half_angle_rad=self._LOBE,
        )
        assert result == pytest.approx(0.0)

    def test_co_located_observer_reflector_returns_zero(self):
        from cislunar.physics.forces.eclipse import reflected_flux_ratio

        pos = np.array([Body.R_EARTH + 400e3, 0.0, 0.0])
        result, _ = reflected_flux_ratio(
            observer_pos_eci=pos,
            sun_pos_eci=np.array([AU, 0.0, 0.0]),
            reflector_pos_eci=pos,  # same position → d < 1
            reflector_area_m2=32.0,
            reflector_normal_eci=np.array([1.0, 0.0, 0.0]),
            reflector_illumination=1.0,
            specular_fraction=0.9,
            lobe_half_angle_rad=self._LOBE,
        )
        assert result == pytest.approx(0.0)

    def test_beam_pointing_away_returns_zero(self):
        from cislunar.physics.forces.eclipse import reflected_flux_ratio

        # Place observer far off-axis so the reflected beam does not reach it
        sc = np.array([Body.R_EARTH + 400e3, 1e9, 0.0])  # 1 Gm off-axis
        refl_pos = np.array([Body.R_EARTH + 200e3, 0.0, 0.0])
        result, _ = reflected_flux_ratio(
            observer_pos_eci=sc,
            sun_pos_eci=np.array([AU, 0.0, 0.0]),
            reflector_pos_eci=refl_pos,
            reflector_area_m2=32.0,
            reflector_normal_eci=np.array([1.0, 0.0, 0.0]),
            reflector_illumination=1.0,
            specular_fraction=0.9,
            lobe_half_angle_rad=self._LOBE,
        )
        assert result == pytest.approx(0.0)


class TestPowerBudgetModelGaps:
    def test_battery_pct_at_full_soc(self):
        pm = PowerBudgetModel()
        assert pm.battery_pct == pytest.approx(100.0)

    def test_battery_pct_at_half_soc(self):
        pm = PowerBudgetModel()
        pm.batt_soc = 0.5
        assert pm.battery_pct == pytest.approx(50.0)

    def test_status_line_contains_eclipse_word(self):
        pm = PowerBudgetModel()
        pm.in_eclipse = True
        s = pm.status_line()
        assert "ECLIPSE" in s

    def test_status_line_sun_when_not_in_eclipse(self):
        pm = PowerBudgetModel()
        s = pm.status_line()
        assert "SUN" in s

    def test_has_previous_position_false_initially(self):
        pm = PowerBudgetModel()
        assert pm.has_previous_position is False

    def test_has_previous_position_true_after_step(self):
        pm = PowerBudgetModel()
        pm._prev_spacecraft_pos = np.array([1.0, 0.0, 0.0])
        assert pm.has_previous_position is True

    def test_reset_clears_state(self):
        pm = PowerBudgetModel()
        pm.batt_soc = 0.3
        pm.in_eclipse = True
        pm._prev_spacecraft_pos = np.array([1.0, 0.0, 0.0])
        pm.reset(soc=0.8)
        assert pm.batt_soc == pytest.approx(0.8)
        assert pm.in_eclipse is False
        assert pm._prev_spacecraft_pos is None

    def test_reset_default_soc_is_full(self):
        pm = PowerBudgetModel()
        pm.batt_soc = 0.0
        pm.reset()
        assert pm.batt_soc == pytest.approx(1.0)
