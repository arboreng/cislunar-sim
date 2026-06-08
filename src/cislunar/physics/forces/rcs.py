"""
Cold gas reaction control system model.

Nozzle layout (body frame, z = spacecraft long / forward axis)
──────────────────────────────────────────────────────────────
Cluster A — nose (4 nozzles, cruciform, at z = +NOSE_ARM_M from CoM)
  Each nozzle fires laterally (perpendicular to z).
  Controls pitch (±y torque) and yaw (±x torque).

    Nozzle A0: fires +x  →  torque +y (pitch up)      force +x
    Nozzle A1: fires -x  →  torque -y (pitch down)    force -x
    Nozzle A2: fires -y  →  torque +x (yaw right)     force -y
    Nozzle A3: fires +y  →  torque -x (yaw left)      force +y

Cluster B — mid-body (4 nozzles, tangential, at ±BODY_ARM_M from CoM axis)
  Fire tangentially around the roll axis.
  Controls roll (±z torque); symmetric pairs cancel net translational force.

    B0+B2: fire in +roll direction  →  torque +z
    B1+B3: fire in -roll direction  →  torque -z

Continuous model
────────────────
Rather than discrete nozzle selection, the model treats nozzle firing as
continuous: the commanded torque direction d = rcs_dir_cmd / |rcs_dir_cmd|
is decomposed into (pitch, yaw, roll) body-frame components.  Each component
activates nozzle capacity proportionally.

  τ_x (yaw)   = d_x × F_per_nozzle × NOSE_ARM_M          (1 nose nozzle)
  τ_y (pitch) = d_y × F_per_nozzle × NOSE_ARM_M          (1 nose nozzle)
  τ_z (roll)  = d_z × 2 × F_per_nozzle × BODY_ARM_M      (1 pair = 2 nozzles)

Net translational force from nose cluster:
  F_x = +d_y × F_per_nozzle   (pitch nozzle fires laterally in ±x)
  F_y = -d_x × F_per_nozzle   (yaw nozzle fires laterally in ∓y)
  F_z = 0                      (no axial component from attitude nozzles)
Mid-body roll nozzles cancel in translation (symmetric pair).

Propellant consumption is proportional to the total impulse of all
active nozzles (continuous equivalent):
  F_active = (|d_x| + |d_y|) × F_per_nozzle + |d_z| × 2 × F_per_nozzle
  ṁ = F_active / (Isp × g₀)

References
──────────
  Sutton, G.P., Biblarz, O. (2017). Rocket Propulsion Elements (9th ed.)
  Wertz, J.R. (ed.) (1999). Space Mission Engineering: The New SMAD. §10.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..constants import RCS as RCSConst


@dataclass
class RCSConfig:
    """Physical parameters for the cold gas RCS."""

    f_per_nozzle_n: float = RCSConst.F_PER_NOZZLE_N  # N per nozzle
    isp: float = RCSConst.ISP  # s
    nose_arm_m: float = RCSConst.NOSE_ARM_M  # m from CoM to nose cluster
    body_arm_m: float = RCSConst.BODY_ARM_M  # m from CoM to mid-body nozzles


class ColdGasRCSModel:
    """
    8-nozzle cold gas reaction control system.

    Two clusters control three attitude axes independently:
      - Nose cluster (4 nozzles, cruciform): pitch + yaw
      - Mid-body cluster (4 nozzles, tangential pairs): roll

    Args:
        config: RCSConfig with nozzle geometry and propellant parameters.
                Defaults to RCSConst values from cislunar.physics.constants.
    """

    G0 = 9.80665  # Standard gravity [m s⁻²]

    def __init__(self, config: RCSConfig | None = None):
        cfg = config or RCSConfig()
        self.f_per_nozzle = cfg.f_per_nozzle_n
        self.isp = cfg.isp
        self.nose_arm = cfg.nose_arm_m
        self.body_arm = cfg.body_arm_m

        # Pre-compute max torque per axis for diagnostics
        self.max_torque_pitch_yaw_nm = self.f_per_nozzle * self.nose_arm
        self.max_torque_roll_nm = 2.0 * self.f_per_nozzle * self.body_arm

    def fire(
        self,
        rcs_dir_cmd: NDArray,
        rcs_trigger: bool,
        dt_s: float,
        rcs_propellant_kg: float,
    ) -> tuple[NDArray, NDArray, float]:
        """
        Fire the RCS and return torque, force, and propellant consumed.

        Args:
            rcs_dir_cmd:       Desired torque direction in body frame (need not
                               be unit length; zero vector produces no output).
            rcs_trigger:       True when the caller commands an RCS burn.
            dt_s:              Timestep duration [s].
            rcs_propellant_kg: Remaining R-134a propellant [kg].

        Returns:
            torque_nm:         Body-frame torque vector [N·m].
            force_n:           Body-frame translational force vector [N].
            prop_consumed_kg:  Propellant mass consumed this step [kg].

        Notes:
            Zero output is returned when ``rcs_trigger`` is False or
            ``rcs_propellant_kg <= 0``.  A UserWarning is issued on the
            first call with trigger=True and empty propellant.
        """
        _zero = (np.zeros(3), np.zeros(3), 0.0)

        if not rcs_trigger:
            return _zero

        if rcs_propellant_kg <= 0.0:
            warnings.warn(
                "RCS fired with empty propellant tank — no thrust produced.",
                UserWarning,
                stacklevel=2,
            )
            return _zero

        # Normalise command direction
        mag = float(np.linalg.norm(rcs_dir_cmd))
        if mag < 1e-12:
            return _zero
        d = np.asarray(rcs_dir_cmd, dtype=float) / mag

        # ── Torque (body frame) ───────────────────────────────────────────────
        # Pitch (tau_y) and yaw (tau_x): 1 nose nozzle each, scaled by d component.
        # Roll  (tau_z): 1 pair of mid-body nozzles (2 nozzles), scaled by d[2].
        tau_x = d[0] * self.f_per_nozzle * self.nose_arm  # yaw
        tau_y = d[1] * self.f_per_nozzle * self.nose_arm  # pitch
        tau_z = d[2] * 2.0 * self.f_per_nozzle * self.body_arm  # roll
        torque_nm = np.array([tau_x, tau_y, tau_z])

        # ── Net translational force (body frame) ─────────────────────────────
        # Nose nozzles fire laterally:
        #   Pitch nozzle (±x thrust) → force along ±x
        #   Yaw   nozzle (∓y thrust) → force along ∓y
        # Mid-body roll nozzles fire as symmetric pairs → net force ≈ 0.
        force_n = np.array(
            [
                d[1] * self.f_per_nozzle,  # pitch nozzle lateral component → +x
                -d[0] * self.f_per_nozzle,  # yaw nozzle lateral component  → -y
                0.0,  # no axial component
            ]
        )

        # ── Propellant consumption ────────────────────────────────────────────
        # Total effective thrust from all active nozzles (continuous model).
        f_active = (abs(d[0]) + abs(d[1])) * self.f_per_nozzle + abs(d[2]) * 2.0 * self.f_per_nozzle
        mdot = f_active / (self.isp * self.G0)
        prop_requested = mdot * dt_s

        if prop_requested <= rcs_propellant_kg:
            prop_consumed = prop_requested
        else:
            # Tank nearly empty — scale output proportionally
            scale = rcs_propellant_kg / (prop_requested + 1e-30)
            torque_nm *= scale
            force_n *= scale
            prop_consumed = rcs_propellant_kg

        return torque_nm, force_n, prop_consumed
