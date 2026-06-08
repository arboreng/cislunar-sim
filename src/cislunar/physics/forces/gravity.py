"""
Gravitational force models.

Implements:
  - Newtonian point-mass gravity (single body)
  - J2 oblateness perturbation
  - J3 pear-shape perturbation (important for Moon: J3/J2 ≈ 0.42)
  - N-body third-body perturbations (Moon + Sun) via third_body_perturbation
  - Composable GravityModel for cislunar scenarios
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class _GravityBody:
    name: str
    mu: float
    j2: float = 0.0
    j3: float = 0.0
    r_body: float = 0.0
    body_pos_fn: Callable[[float], NDArray[np.float64]] | None = None
    fixed_pos: NDArray[np.float64] | None = field(default=None, repr=False)


def point_mass_gravity(
    position: NDArray[np.float64],
    mu: float,
) -> NDArray[np.float64]:
    """
    Acceleration due to a point-mass gravitational body.

    Args:
        position: spacecraft position relative to the attracting body [m]
        mu:       gravitational parameter GM of the body [m³ s⁻²]

    Returns:
        acceleration vector [m s⁻²]
    """
    r = np.linalg.norm(position)
    if r < 1.0:
        raise ValueError(f"Degenerate position vector: r={r:.3e} m")
    return -mu / r**3 * position


def j2_perturbation(
    position: NDArray[np.float64],
    mu: float,
    j2: float,
    r_body: float,
) -> NDArray[np.float64]:
    """
    J2 oblateness perturbation acceleration.

    Significant for low lunar / low Earth orbits; negligible beyond ~10 body radii.
    Uses the standard analytic expression in inertial Cartesian coordinates.

    Args:
        position: spacecraft position relative to body centre [m]
        mu:       gravitational parameter of the body [m³ s⁻²]
        j2:       dimensionless J2 coefficient
        r_body:   equatorial radius of the body [m]

    Returns:
        acceleration vector [m s⁻²]
    """
    x, y, z = position
    r = np.linalg.norm(position)
    r2 = r * r
    r5 = r2 * r2 * r

    factor = (3.0 / 2.0) * j2 * mu * r_body**2 / r5
    z_r2 = z * z / r2

    ax = factor * x * (5.0 * z_r2 - 1.0)
    ay = factor * y * (5.0 * z_r2 - 1.0)
    az = factor * z * (5.0 * z_r2 - 3.0)

    return np.array([ax, ay, az])


def j3_perturbation(
    position: NDArray[np.float64],
    mu: float,
    j3: float,
    r_body: float,
) -> NDArray[np.float64]:
    """
    J3 zonal harmonic perturbation acceleration (north-south asymmetry).

    J3 is the "pear-shape" term.  For the Moon J3/J2 ≈ 0.42, so J3 is
    non-negligible at low lunar altitudes: at 200 km it is ~1/6 of J2.
    For Earth J3/J2 ≈ 0.08 — small enough to omit for cislunar transfers.

    Derived from the gravitational potential
        U_J3 = (μ/r) J3 (R/r)³ P₃(z/r),   P₃(s) = (5s³ − 3s)/2
    and taking the gradient analytically.

    Args:
        position: spacecraft position relative to body centre [m]
        mu:       gravitational parameter of the body [m³ s⁻²]
        j3:       dimensionless J3 coefficient
        r_body:   equatorial radius of the body [m]

    Returns:
        acceleration vector [m s⁻²]
    """
    x, y, z = position
    r = np.linalg.norm(position)
    r2 = r * r
    r5 = r2 * r2 * r
    r7 = r5 * r2
    z_r2 = (z * z) / r2  # (z/r)²

    # Transverse factor shared by ax and ay
    # ax = x * fac_xy,  ay = y * fac_xy
    fac_xy = (5.0 / 2.0) * j3 * mu * r_body**3 * z * (3.0 - 7.0 * z_r2) / r7

    # Axial component
    fac_z = (1.0 / 2.0) * j3 * mu * r_body**3 * (-3.0 + 30.0 * z_r2 - 35.0 * z_r2**2) / r5

    return np.array([x * fac_xy, y * fac_xy, fac_z])


def third_body_perturbation(
    spacecraft_pos: NDArray[np.float64],
    third_body_pos: NDArray[np.float64],
    mu_third: float,
) -> NDArray[np.float64]:
    """
    Perturbation from a third body (e.g. Sun when integrating in Earth-centred frame,
    or Earth when integrating near the Moon).

    Uses the indirect-term formulation to avoid numerical cancellation
    when the spacecraft is far from the third body.

    Args:
        spacecraft_pos:  spacecraft position relative to primary body [m]
        third_body_pos:  third body position relative to primary body [m]
        mu_third:        gravitational parameter of the third body [m³ s⁻²]

    Returns:
        acceleration vector [m s⁻²]
    """
    # Vector from third body to spacecraft
    r_sc_tb = spacecraft_pos - third_body_pos
    d = np.linalg.norm(r_sc_tb)

    # Vector from origin (primary) to third body
    r_tb = np.linalg.norm(third_body_pos)

    # Indirect + direct terms
    a_direct = -mu_third / d**3 * r_sc_tb
    a_indirect = -mu_third / r_tb**3 * third_body_pos  # indirect term cancels

    return a_direct + a_indirect


class GravityModel:
    """
    Composable gravity model for a cislunar simulation.

    Build one at setup time and call .acceleration() inside the
    integrator's derivative function.

    Example (Earth-Moon system):
        grav = GravityModel()
        grav.add_body("Earth", mu=Body.MU_EARTH, j2=Body.J2_EARTH,
                       r_body=Body.R_EARTH)
        grav.add_body("Moon",  mu=Body.MU_MOON,  body_pos_fn=moon_position_fn)
    """

    def __init__(self):
        self._bodies: list[_GravityBody] = []
        self._mascon_fn: (
            Callable[
                [NDArray[np.float64], NDArray[np.float64]],
                NDArray[np.float64],
            ]
            | None
        ) = None

    def add_body(
        self,
        name: str,
        mu: float,
        j2: float = 0.0,
        j3: float = 0.0,
        r_body: float = 0.0,
        body_pos_fn: Callable[[float], NDArray[np.float64]] | None = None,
        fixed_pos: NDArray[np.float64] | None = None,
    ) -> GravityModel:
        """
        Register a gravitational body.

        For the primary attractor, omit body_pos_fn and fixed_pos.
        For perturbing bodies, supply body_pos_fn (preferred) or fixed_pos.

        Args:
            name:        human-readable label (used for Moon mascon lookup)
            mu:          gravitational parameter GM [m³ s⁻²]
            j2:          J2 oblateness coefficient (0 = spherical)
            j3:          J3 pear-shape coefficient (0 = omit; significant for Moon)
            r_body:      equatorial radius [m] — required when j2 or j3 ≠ 0
            body_pos_fn: callable(t) → ECI position [m]; for moving perturbers
            fixed_pos:   constant ECI position [m]; for static scenarios
        """
        self._bodies.append(
            _GravityBody(
                name=name,
                mu=mu,
                j2=j2,
                j3=j3,
                r_body=r_body,
                body_pos_fn=body_pos_fn,
                fixed_pos=fixed_pos,
            )
        )
        return self  # fluent interface

    def set_mascon_model(
        self,
        mascon_fn: Callable[
            [NDArray[np.float64], NDArray[np.float64]],
            NDArray[np.float64],
        ],
    ) -> GravityModel:
        """
        Enable lunar mascon perturbations.

        Args:
            mascon_fn: callable(spacecraft_pos_eci, moon_pos_eci) → NDArray[3]
                       Returns mascon acceleration [m/s²] in ECI frame.
                       Use lunar_mascon_acceleration from cislunar.physics.forces.mascon.
        """
        self._mascon_fn = mascon_fn
        return self

    def acceleration(
        self,
        spacecraft_pos: NDArray[np.float64],
        time_s: float,
    ) -> NDArray[np.float64]:
        """
        Total gravitational acceleration at the current position/time.

        Args:
            spacecraft_pos: spacecraft position relative to the primary body [m]
            time_s:         simulation time (used for moving-body lookups)

        Returns:
            total acceleration vector [m s⁻²]
        """
        total = np.zeros(3)

        for body in self._bodies:
            if body.body_pos_fn is not None:
                body_pos = body.body_pos_fn(time_s)
                # Non-primary perturber: use full third-body formula which
                # includes the indirect term — accounts for the primary body
                # also being accelerated by the perturber. Without this the
                # Moon and Sun perturbations are missing ~10-30% of their
                # effect over multi-day trajectories.
                total += third_body_perturbation(spacecraft_pos, body_pos, body.mu)
                rel_pos = (
                    spacecraft_pos - body_pos
                )  # spacecraft pos relative to body; needed for J2/J3
            elif body.fixed_pos is not None:
                body_pos = body.fixed_pos
                total += third_body_perturbation(spacecraft_pos, body_pos, body.mu)
                rel_pos = spacecraft_pos - body_pos
            else:
                rel_pos = spacecraft_pos  # primary attractor
                total += point_mass_gravity(rel_pos, body.mu)

            if body.j2 != 0.0 and body.r_body != 0.0:
                total += j2_perturbation(rel_pos, body.mu, body.j2, body.r_body)
            if body.j3 != 0.0 and body.r_body != 0.0:
                total += j3_perturbation(rel_pos, body.mu, body.j3, body.r_body)

        # ── Lunar mascon / harmonic perturbation ─────────────────────────────
        # Default model: point mass + J2 + J3 + 8 representative mascons.
        # Adequate for broad cislunar simulation (trans-lunar trajectories,
        # rough perilune passes, orbital gate geometry).
        #
        # For low lunar orbit (< ~200 km) use LunarHarmonicGravity (degree-70
        # GRGM truncation) via make_lunar_craft(use_grgm_harmonic=True) or by
        # calling set_mascon_model(default_lunar_harmonic_gravity()) directly.
        # ─────────────────────────────────────────────────────────────────────
        if self._mascon_fn is not None:
            # Find Moon position for this timestep.
            # Case 1: Moon is a perturbing body (ECI frame) → use body_pos_fn.
            # Case 2: Moon is the primary body (Moon-centred frame) → moon at origin.
            moon_pos = np.zeros(3)  # default: Moon-centred frame
            for body in self._bodies:
                if body.name == "Moon":
                    if body.body_pos_fn is not None:
                        moon_pos = body.body_pos_fn(time_s)
                    elif body.fixed_pos is not None:
                        moon_pos = body.fixed_pos
                    break  # found Moon; keep moon_pos as-is for primary case
            total += self._mascon_fn(spacecraft_pos, moon_pos)

        return total
