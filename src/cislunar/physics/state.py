"""
State representation for a cislunar spacecraft.

Everything in the physics engine flows through SpacecraftState.
Using a dataclass keeps it readable; NumPy views keep it fast.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MASTER FRAME: Earth-Centred Inertial (ECI)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Origin    : Earth's centre of mass
Axes      : non-rotating, aligned to the mean equinox/equatorial
            plane at J2000.0 (GCRF approximation)
Units     : metres (position), metres per second (velocity)

All position and velocity fields in SpacecraftState, all ephemeris
outputs (Moon, Sun), and all force vectors are expressed in this
frame.

Solar radiation pressure needs the spacecraft's position relative
to the Sun.  This is computed on-the-fly in Spacecraft._derivatives
as ``sc_pos_from_sun = pos_eci - sun_pos_eci`` and passed to the
sail model.  It is an ECI-frame vector; it is NOT a heliocentric
coordinate in the solar-system barycentric sense.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class SpacecraftState:
    """
    Full state of one spacecraft at a single point in time.

    All vector fields are in the Earth-Centred Inertial (ECI) frame
    (see module docstring for the full frame contract).
    """

    # ── Dynamics ─────────────────────────────────────────────────────────────
    position: NDArray[np.float64]  # [x, y, z]  metres
    velocity: NDArray[np.float64]  # [vx, vy, vz]  m s⁻¹

    # ── Attitude (sail pointing) ──────────────────────────────────────────────
    # Unit vector normal to the sail face, in ECI.
    # Agents control this via angular rate commands to the attitude model.
    sail_normal: NDArray[np.float64] = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))

    # ── Resources ────────────────────────────────────────────────────────────
    propellant_kg: float = 0.5  # Remaining ion thruster propellant [kg]
    rcs_propellant_kg: float = 0.0  # Remaining RCS propellant (cold gas) [kg]
    power_w: float = 30.0  # Available electrical power        [W]

    # ── Simulation bookkeeping ────────────────────────────────────────────────
    time_s: float = 0.0  # Mission elapsed time           [s]
    mass_dry_kg: float = 12.0  # Dry mass (bus + structure)     [kg]

    # ── Derived (computed, not set directly) ──────────────────────────────────
    # Populated by the integrator after each step.
    acceleration_last: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        # Normalise the sail normal in case caller passed a non-unit vector.
        n = np.linalg.norm(self.sail_normal)
        if n > 0:
            self.sail_normal = self.sail_normal / n

    @property
    def total_mass_kg(self) -> float:
        return self.mass_dry_kg + self.propellant_kg + self.rcs_propellant_kg

    @property
    def speed_ms(self) -> float:
        return float(np.linalg.norm(self.velocity))

    @property
    def altitude_from_body_m(self, body_radius_m: float = 6.371e6) -> float:
        """Convenience: radial distance minus a body radius."""
        return float(np.linalg.norm(self.position)) - body_radius_m

    def as_vector(self) -> NDArray[np.float64]:
        """
        Flatten to a single 1-D array for the numerical integrator.
        Layout: [px, py, pz, vx, vy, vz, nx, ny, nz]
        Resources (propellant, power) are tracked separately since they
        evolve via discrete events, not ODE integration.
        """
        return np.concatenate([self.position, self.velocity, self.sail_normal])

    @classmethod
    def from_vector(
        cls,
        vec: NDArray[np.float64],
        propellant_kg: float,
        power_w: float,
        time_s: float,
        mass_dry_kg: float,
        rcs_propellant_kg: float = 0.0,
    ) -> SpacecraftState:
        """Reconstruct from a flattened integrator vector."""
        return cls(
            position=vec[0:3].copy(),
            velocity=vec[3:6].copy(),
            sail_normal=vec[6:9].copy(),
            propellant_kg=propellant_kg,
            rcs_propellant_kg=rcs_propellant_kg,
            power_w=power_w,
            time_s=time_s,
            mass_dry_kg=mass_dry_kg,
        )

    def copy(self) -> SpacecraftState:
        import copy

        return copy.deepcopy(self)


@dataclass
class Checkpoint:
    """A waypoint the spacecraft must pass through."""

    position: NDArray[np.float64]  # Centre of the checkpoint gate
    radius_m: float  # Half-width of the gate
    label: str = ""

    def is_passed(self, spacecraft_pos: NDArray[np.float64]) -> bool:
        return bool(np.linalg.norm(spacecraft_pos - self.position) <= self.radius_m)
