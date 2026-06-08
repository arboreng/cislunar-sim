"""
Hall-thruster plume impingement on a neighbouring spacecraft's sail.

Close-range formation-flying or proximity operations can expose one
spacecraft to another's exhaust plume. Hall thrusters at 1 mN–5 mN produce
momentum flux that, at 100–300 m separation, is large enough to perturb a
32 m² sail on a multi-minute timescale.

Physical model — cosine-lobe plume
───────────────────────────────────
The angular momentum-flux distribution of a Hall-effect thruster plume is
well-represented by a cosine-power lobe centred on the thrust axis:

    dF/dΩ = ((n+1) / 2π) · F_thrust · cos^n(θ)        (θ ≤ π/2)

where
    F_thrust  = total thrust magnitude produced by the source [N]
    θ         = angle between plume axis and source→target line
    n         = ``PLUME_DIVERGENCE_EXPONENT`` (≈ 3 for Hall thrusters)

The ((n+1)/2π) normalization ensures ∫dF/dΩ over the forward hemisphere
equals F_thrust exactly — momentum conservation with the source.

At distance r, the momentum flux per unit perpendicular area is
    J_perp = (dF/dΩ) / r²

For a flat target of area A with front-face normal n̂_t, the captured
momentum rate (= force on target) is

    F_target = J_perp · A · max(0, -n̂_t · r̂) · r̂
             = ((n+1)/2π) · F_thrust · A · cos^n(θ) · cos(α) / r²  ·  r̂

where α is the angle between the target's front-face normal and the
incoming plume direction (-r̂).

The force vector points FROM the source TOWARD the target — the plume
pushes the target away.  Full momentum absorption is assumed: specular
reflection would introduce a factor of 2 on the normal component and an
angle-dependent tangential component, but sail reflectivity at ion-beam
energies (100–300 eV) is very low — the plume particles embed in the
membrane rather than bouncing.  Absorption is the physically accurate
limit for this energy range.

Gates
─────
Two independent geometric gates produce bit-exact zero force:

  1. θ > π/2  — target is behind or beside the thruster (outside the
     forward hemisphere of the plume).  Implemented as ``cos(θ) ≤ 0``.

  2. n̂_t · (-r̂) ≤ 0  — target's back face is toward the source.  The
     sail's front face is what intercepts plume; a sail presenting its
     back face intercepts no plume.  Same convention as ``SolarSailModel``
     (normal points toward the illuminated face).

Either gate also triggers on ``thrust_mag_n ≤ 0`` (cold / idle source).

Frame note
──────────
All vectors are in ECI.  Only differences (target − source) are used,
so the choice of ECI origin is irrelevant.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ..constants import Thruster as ThrConst


def plume_impingement_force(
    source_pos: NDArray[np.float64],
    plume_axis: NDArray[np.float64],
    thrust_mag_n: float,
    target_pos: NDArray[np.float64],
    target_normal: NDArray[np.float64],
    target_area: float,
    *,
    divergence_exponent: float = ThrConst.PLUME_DIVERGENCE_EXPONENT,
) -> NDArray[np.float64]:
    """
    Force on a target sail from a source thruster's exhaust plume.

    Args:
        source_pos:    position of the source spacecraft [m, ECI]
        plume_axis:    unit vector along the direction the plume TRAVELS
                       (= -thrust_direction of the source).  When the
                       source thrusts in +x, its plume exits in -x;
                       plume_axis = -thrust_dir.
        thrust_mag_n:  magnitude of the source's current thrust force [N].
                       Zero / negative ⇒ bit-exact zero output.
        target_pos:    position of the target spacecraft [m, ECI]
        target_normal: unit normal of the target sail's front face [ECI].
                       Front-face convention matches ``SolarSailModel``:
                       normal points toward the illuminated / impingement
                       face of the sail.
        target_area:   target's sail area [m²]
        divergence_exponent: cosine-lobe exponent n.  Default from
                             ``ThrConst.PLUME_DIVERGENCE_EXPONENT``.

    Returns:
        Force vector on the target in ECI [N].  Direction is from source
        toward target (the plume pushes the target away from the source).
        Bit-exact zero vector on any of:
          • thrust_mag_n ≤ 0
          • source and target co-located (numerical safety, r < 1 m)
          • target in rear hemisphere of plume (cos θ ≤ 0)
          • target presenting back face to the source (cos α ≤ 0)
    """
    # Gate 0: cold / idle thruster.
    if thrust_mag_n <= 0.0:
        return np.zeros(3)

    # Source → target separation.
    r_vec = target_pos - source_pos
    r = float(np.linalg.norm(r_vec))
    if r < 1.0:
        return np.zeros(3)  # numerical safety — effectively co-located
    r_hat = r_vec / r

    # Gate 1: target behind thruster (outside forward hemisphere).
    # plume_axis is a unit vector along the plume propagation direction.
    cos_theta = float(np.dot(plume_axis, r_hat))
    if cos_theta <= 0.0:
        return np.zeros(3)

    # Gate 2: target back face toward source.
    # target_normal points toward the illuminated face; plume photons
    # arrive travelling along +r_hat, so the front face is struck iff
    # target_normal · (-r_hat) > 0.
    cos_alpha = float(np.dot(target_normal, -r_hat))
    if cos_alpha <= 0.0:
        return np.zeros(3)

    # Cosine-lobe magnitude.
    n = divergence_exponent
    coeff = (n + 1.0) / (2.0 * math.pi)
    mag = coeff * thrust_mag_n * target_area * (cos_theta**n) * cos_alpha / (r * r)

    return mag * r_hat
