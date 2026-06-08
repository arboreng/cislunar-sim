"""Gravity force model tests: point-mass, J2, J3, third-body, conservation laws."""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import ONE_DAY, R_LEO, V_CIRC_LEO, _integrate_for

from cislunar.physics import (
    Action,
    GravityModel,
    IonThrusterModel,
    SolarSailModel,
    Spacecraft,
    SpacecraftState,
)
from cislunar.physics.constants import AU, Body
from cislunar.physics.forces.eclipse import PowerBudgetModel
from cislunar.physics.forces.gravity import (
    j2_perturbation,
    j3_perturbation,
    point_mass_gravity,
    third_body_perturbation,
)


# ══════════════════════════════════════════════════════════════════════════════
# Section 1 — Gravity
# ══════════════════════════════════════════════════════════════════════════════
class TestGravity:
    """Point-mass inverse-square law and J2 perturbation magnitudes."""

    def test_inverse_square_scaling(self):
        """Doubling distance gives a factor-4 weaker acceleration."""
        r1 = np.array([7e6, 0, 0])
        r2 = np.array([14e6, 0, 0])
        a1 = point_mass_gravity(r1, Body.MU_EARTH)
        a2 = point_mass_gravity(r2, Body.MU_EARTH)
        ratio = np.linalg.norm(a1) / np.linalg.norm(a2)
        assert abs(ratio - 4.0) < 1e-9, f"ratio={ratio}"

    def test_leo_gravity_magnitude(self):
        """At 400 km altitude, |g| ≈ 8.7 m/s²."""
        leo = np.array([Body.R_EARTH + 400e3, 0, 0])
        a = point_mass_gravity(leo, Body.MU_EARTH)
        assert abs(np.linalg.norm(a) - 8.7) < 0.1, f"|a|={np.linalg.norm(a):.3f}"

    def test_j2_much_smaller_than_point_mass_at_leo(self):
        """J2 at LEO is < 1 % of point-mass attraction."""
        leo = np.array([Body.R_EARTH + 400e3, 0, 0])
        a_pm = point_mass_gravity(leo, Body.MU_EARTH)
        a_j2 = j2_perturbation(leo, Body.MU_EARTH, Body.J2_EARTH, Body.R_EARTH)
        ratio = np.linalg.norm(a_j2) / np.linalg.norm(a_pm)
        assert ratio < 1e-2, f"J2/pm ratio={ratio:.2e}"

    def test_j3_transverse_zero_in_equatorial_plane(self):
        """At z=0 the transverse (x,y) j3 components vanish (fac_xy ∝ z)."""
        r = Body.R_MOON + 200e3
        pos = np.array([r, 0.0, 0.0])
        acc = j3_perturbation(pos, Body.MU_MOON, Body.J3_MOON, Body.R_MOON)
        assert abs(acc[0]) < 1e-30 and abs(acc[1]) < 1e-30, (
            f"transverse components non-zero: acc={acc}"
        )

    def test_j3_transverse_antisymmetric_north_south(self):
        """j3 transverse (x,y) components are odd in z; axial (z) is even in z."""
        r = Body.R_MOON + 200e3
        z = 1e5
        acc_north = j3_perturbation(np.array([r, 0.0, z]), Body.MU_MOON, Body.J3_MOON, Body.R_MOON)
        acc_south = j3_perturbation(np.array([r, 0.0, -z]), Body.MU_MOON, Body.J3_MOON, Body.R_MOON)
        # Transverse components: odd in z → flip sign
        assert abs(acc_north[0] + acc_south[0]) < 1e-20, "x not antisymmetric"
        # Axial component: even in z → same value
        np.testing.assert_allclose(acc_north[2], acc_south[2], rtol=1e-12)

    def test_moon_j3_fraction_of_j2(self):
        """At 200 km lunar altitude, |J3| / |J2| is in published range 0.1–0.6."""
        r = Body.R_MOON + 200e3
        pos = np.array([r, 0.0, r * 0.5])
        a_j2 = np.linalg.norm(j2_perturbation(pos, Body.MU_MOON, Body.J2_MOON, Body.R_MOON))
        a_j3 = np.linalg.norm(j3_perturbation(pos, Body.MU_MOON, Body.J3_MOON, Body.R_MOON))
        ratio = a_j3 / a_j2
        assert 0.1 < ratio < 0.6, f"J3/J2 ratio={ratio:.3f} (expect 0.1–0.6)"


# ══════════════════════════════════════════════════════════════════════════════
# Section 6 — Sun as third-body perturber
# ══════════════════════════════════════════════════════════════════════════════
def _make_earth_moon_sun_spacecraft(include_sun: bool):
    """Build a minimal Earth-Moon(-Sun) spacecraft for the Sun 3rd-body divergence test.

    Uses analytic circular ephemerides so the comparison is exactly controlled.
    """
    MOON_ORBIT_RADIUS = 3.844e8
    MOON_PERIOD = 27.3217 * ONE_DAY
    EARTH_ORBIT_PERIOD = 365.25 * ONE_DAY

    def moon_pos(t):
        th = 2 * math.pi * t / MOON_PERIOD
        return np.array([MOON_ORBIT_RADIUS * math.cos(th), MOON_ORBIT_RADIUS * math.sin(th), 0.0])

    def sun_pos(t):
        th = 2 * math.pi * t / EARTH_ORBIT_PERIOD
        return np.array([AU * math.cos(th), AU * math.sin(th), 0.0])

    grav = (
        GravityModel()
        .add_body("Earth", mu=Body.MU_EARTH, j2=Body.J2_EARTH, r_body=Body.R_EARTH)
        .add_body("Moon", mu=Body.MU_MOON, body_pos_fn=moon_pos)
    )
    if include_sun:
        grav.add_body("Sun", mu=Body.MU_SUN, body_pos_fn=sun_pos)

    state = SpacecraftState(
        position=np.array([R_LEO, 0.0, 0.0]),
        velocity=np.array([0.0, V_CIRC_LEO, 0.0]),
        propellant_kg=0.5,
        mass_dry_kg=12.0,
    )
    return Spacecraft(
        initial_state=state,
        sail_model=SolarSailModel(area_m2=32.0),
        thruster_model=IonThrusterModel(),
        gravity_model=grav,
        sun_pos_fn=sun_pos,
        power_model=PowerBudgetModel(),
    )


class TestSunThirdBodyPerturber:
    @pytest.fixture(scope="class")
    def lunar_sun_geometry(self):
        """Spacecraft at lunar distance, Sun at +1 AU on +x axis."""
        return dict(
            lunar_pos=np.array([3.844e8, 0.0, 0.0]),
            sun_pos=np.array([AU, 0.0, 0.0]),
        )

    def test_sun_3rd_body_nonzero_at_lunar_distance(self, lunar_sun_geometry):
        a = third_body_perturbation(
            lunar_sun_geometry["lunar_pos"],
            lunar_sun_geometry["sun_pos"],
            Body.MU_SUN,
        )
        mag = np.linalg.norm(a)
        assert mag > 1e-6, f"|a_sun|={mag:.2e} m/s²"

    def test_sun_3rd_body_within_tidal_estimate(self, lunar_sun_geometry):
        """Magnitude matches the analytical tidal estimate ~2–5 × 10⁻⁵ m/s²."""
        a = third_body_perturbation(
            lunar_sun_geometry["lunar_pos"],
            lunar_sun_geometry["sun_pos"],
            Body.MU_SUN,
        )
        mag = np.linalg.norm(a)
        assert 2e-5 < mag < 5e-5, f"|a_sun|={mag:.2e} m/s²"

    def test_sun_accel_small_vs_earth_at_lunar_distance(self, lunar_sun_geometry):
        """At lunar distance, Sun 3rd-body pull is a ~1 % perturbation on Earth."""
        lunar_pos = lunar_sun_geometry["lunar_pos"]
        sun_pos = lunar_sun_geometry["sun_pos"]
        a_sun = np.linalg.norm(third_body_perturbation(lunar_pos, sun_pos, Body.MU_SUN))
        a_earth = Body.MU_EARTH / np.linalg.norm(lunar_pos) ** 2
        ratio = a_sun / a_earth
        assert ratio < 0.02, f"Sun/Earth ratio at lunar distance = {ratio:.4f}"

    def test_indirect_term_nearly_cancels_direct(self, lunar_sun_geometry):
        """
        At lunar distance the direct and indirect Sun terms nearly cancel —
        the net perturbation is << the direct Sun pull on the spacecraft.
        """
        lunar_pos = lunar_sun_geometry["lunar_pos"]
        sun_pos = lunar_sun_geometry["sun_pos"]
        a_net = np.linalg.norm(third_body_perturbation(lunar_pos, sun_pos, Body.MU_SUN))
        a_direct = Body.MU_SUN / np.linalg.norm(lunar_pos - sun_pos) ** 2
        assert a_net / a_direct < 0.01, (
            f"net/direct = {a_net / a_direct:.4f} — indirect term not cancelling"
        )

    def test_make_lunar_craft_includes_earth_moon_sun(self, leo_spacecraft):
        body_names = {b.name for b in leo_spacecraft.gravity._bodies}
        assert body_names == {"Earth", "Moon", "Sun"}, f"bodies={body_names}"

    def test_make_lunar_craft_moon_trajectory_is_not_legacy_circular(self, leo_spacecraft):
        moon_body = next(b for b in leo_spacecraft.gravity._bodies if b.name == "Moon")
        moon_pos_fn = moon_body.body_pos_fn

        t0 = 0.0
        t1 = 3.5 * ONE_DAY
        moon_t0 = moon_pos_fn(t0)
        moon_t1 = moon_pos_fn(t1)

        theta = 2.0 * math.pi * t1 / (27.3217 * ONE_DAY)
        rot_z = np.array(
            [
                [math.cos(theta), -math.sin(theta), 0.0],
                [math.sin(theta), math.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        circular_prediction = rot_z @ moon_t0

        deviation_m = float(np.linalg.norm(moon_t1 - circular_prediction))
        assert deviation_m > 1e6, f"deviation_km={deviation_m / 1e3:.1f}"

    def test_3day_trajectory_diverges_with_sun(self):
        """Integrating 3 days with vs. without Sun third-body gives ≥ 50 m offset."""
        sc_with = _make_earth_moon_sun_spacecraft(include_sun=True)
        sc_without = _make_earth_moon_sun_spacecraft(include_sun=False)
        coast = Action(attitude_dir_cmd=np.array([-1.0, 0.0, 0.0]), throttle=0.0)
        sc_with.step(coast, dt_requested=3 * ONE_DAY)
        sc_without.step(coast, dt_requested=3 * ONE_DAY)
        divergence_m = np.linalg.norm(sc_with.state.position - sc_without.state.position)
        assert divergence_m > 50, f"divergence={divergence_m / 1e3:.3f} km"


# ══════════════════════════════════════════════════════════════════════════════
# Section 18 — Two-body conservation (specific energy and angular momentum)
# ══════════════════════════════════════════════════════════════════════════════


def _orbital_energy(pos: np.ndarray, vel: np.ndarray, mu: float) -> float:
    return 0.5 * np.dot(vel, vel) - mu / np.linalg.norm(pos)


def _angular_momentum_mag(pos: np.ndarray, vel: np.ndarray) -> float:
    return float(np.linalg.norm(np.cross(pos, vel)))


class TestTwoBodyConservation:
    @pytest.fixture(scope="class")
    def orbital_period_s(self) -> float:
        return 2 * math.pi * math.sqrt(R_LEO**3 / Body.MU_EARTH)

    def test_specific_energy_conserved(self, two_body_sc, orbital_period_s):
        """ΔE / |E| < 0.001 % over one orbital period."""
        s0 = two_body_sc.state.copy()
        E0 = _orbital_energy(s0.position, s0.velocity, Body.MU_EARTH)
        _integrate_for(two_body_sc, orbital_period_s, step_s=60.0)
        s1 = two_body_sc.state
        E1 = _orbital_energy(s1.position, s1.velocity, Body.MU_EARTH)
        dE_pct = abs(E1 - E0) / abs(E0) * 100.0
        assert dE_pct < 0.001, f"ΔE={dE_pct:.5f}%  E0={E0:.6e}  E1={E1:.6e}"

    def test_angular_momentum_conserved(self, two_body_sc, orbital_period_s):
        """Δ|h| / |h| < 0.001 % over one orbital period."""
        s0 = two_body_sc.state.copy()
        h0 = _angular_momentum_mag(s0.position, s0.velocity)
        _integrate_for(two_body_sc, orbital_period_s, step_s=60.0)
        s1 = two_body_sc.state
        h1 = _angular_momentum_mag(s1.position, s1.velocity)
        dh_pct = abs(h1 - h0) / h0 * 100.0
        assert dh_pct < 0.001, f"Δh={dh_pct:.5f}%  h0={h0:.6e}  h1={h1:.6e}"


# ══════════════════════════════════════════════════════════════════════════════
# Section 19 — J2 nodal precession (Brouwer)
# ══════════════════════════════════════════════════════════════════════════════


def _raan_from_state(pos: np.ndarray, vel: np.ndarray) -> float:
    """Right ascension of ascending node [rad] from ECI state."""
    h = np.cross(pos, vel)
    N = np.array([-h[1], h[0], 0.0])
    if np.linalg.norm(N) < 1e-10:
        return 0.0
    return float(math.atan2(N[1], N[0]))


class TestJ2NodalPrecession:
    INC_DEG = 51.6  # ISS-like inclination
    INTEGRATE_S = ONE_DAY

    @pytest.fixture
    def j2_spacecraft(self):
        inc = math.radians(self.INC_DEG)
        grav = GravityModel().add_body(
            "Earth", mu=Body.MU_EARTH, j2=Body.J2_EARTH, r_body=Body.R_EARTH
        )
        v_inc = np.array([0.0, V_CIRC_LEO * math.cos(inc), V_CIRC_LEO * math.sin(inc)])
        state = SpacecraftState(
            position=np.array([R_LEO, 0.0, 0.0]),
            velocity=v_inc,
            propellant_kg=0.0,
            mass_dry_kg=12.0,
        )
        return Spacecraft(
            initial_state=state,
            sail_model=SolarSailModel(area_m2=0.0),
            thruster_model=IonThrusterModel(),
            gravity_model=grav,
            sun_pos_fn=lambda t: np.array([AU, 0.0, 0.0]),
            power_model=PowerBudgetModel(),
        )

    @pytest.fixture(scope="class")
    def brouwer_theory(self) -> float:
        inc = math.radians(self.INC_DEG)
        n = math.sqrt(Body.MU_EARTH / R_LEO**3)
        return -1.5 * n * Body.J2_EARTH * (Body.R_EARTH / R_LEO) ** 2 * math.cos(inc)

    @staticmethod
    def _measure_drift(sc: Spacecraft, duration_s: float) -> float:
        """Integrate *sc* and return the unwrapped RAAN drift in rad/s."""
        raan0 = _raan_from_state(sc.state.position, sc.state.velocity)
        _integrate_for(sc, duration_s, step_s=60.0)
        raan1 = _raan_from_state(sc.state.position, sc.state.velocity)
        d_raan = raan1 - raan0
        if d_raan > math.pi:
            d_raan -= 2 * math.pi
        if d_raan < -math.pi:
            d_raan += 2 * math.pi
        return d_raan / duration_s

    def test_raan_drift_is_westward(self, j2_spacecraft):
        """J2 precession is westward (dΩ/dt < 0) for prograde orbits at i < 90°."""
        drift = self._measure_drift(j2_spacecraft, self.INTEGRATE_S)
        assert drift < 0, f"dΩ/dt = {math.degrees(drift * 86400):+.3f}°/day"

    def test_raan_drift_within_1pct_of_brouwer(self, j2_spacecraft, brouwer_theory):
        drift = self._measure_drift(j2_spacecraft, self.INTEGRATE_S)
        err_pct = abs(drift - brouwer_theory) / abs(brouwer_theory) * 100.0
        assert err_pct < 1.0, (
            f"theory={math.degrees(brouwer_theory * 86400):+.3f}°/day  "
            f"meas={math.degrees(drift * 86400):+.3f}°/day  err={err_pct:.2f}%"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Section — Point-mass degenerate position guard
# ══════════════════════════════════════════════════════════════════════════════
class TestPointMassGravityEdgeCases:
    def test_near_zero_position_raises(self):
        with pytest.raises(ValueError, match="Degenerate"):
            point_mass_gravity(np.array([0.1, 0.0, 0.0]), Body.MU_EARTH)

    def test_exactly_zero_position_raises(self):
        with pytest.raises(ValueError):
            point_mass_gravity(np.zeros(3), Body.MU_EARTH)


# ══════════════════════════════════════════════════════════════════════════════
# Section — GravityModel with fixed_pos body
# ══════════════════════════════════════════════════════════════════════════════
class TestGravityModelFixedPos:
    def test_fixed_pos_body_produces_nonzero_acceleration(self):
        """A body added with a fixed static position should perturb the spacecraft."""
        grav = (
            GravityModel()
            .add_body("Earth", mu=Body.MU_EARTH)
            .add_body("Moon", mu=Body.MU_MOON, fixed_pos=np.array([3.844e8, 0.0, 0.0]))
        )
        pos = np.array([R_LEO, 0.0, 0.0])
        a = grav.acceleration(pos, 0.0)
        assert np.linalg.norm(a) > 0

    def test_fixed_pos_mascon_moon_path(self):
        """Fixed Moon position should be found by the mascon moon-search loop."""
        from cislunar.physics.forces.mascon import default_lunar_harmonic_gravity

        grav = (
            GravityModel()
            .add_body("Earth", mu=Body.MU_EARTH)
            .add_body("Moon", mu=Body.MU_MOON, fixed_pos=np.array([3.844e8, 0.0, 0.0]))
        )
        grav.set_mascon_model(default_lunar_harmonic_gravity())
        pos = np.array([R_LEO, 0.0, 0.0])
        a = grav.acceleration(pos, 0.0)
        assert np.isfinite(np.linalg.norm(a))
