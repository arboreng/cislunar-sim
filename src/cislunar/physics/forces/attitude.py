"""
Rigid-body attitude dynamics with reaction wheel control.

The simple fixed-rate sail-slew fallback is useful for lightweight studies,
but higher-fidelity simulations benefit from explicit rotational dynamics:

  - Rotational inertia: large slews take minutes, not seconds
  - Reaction wheel saturation: max angular momentum storage ~0.015 N·m·s
    per wheel; pushing past it stops control authority entirely
  - Gyroscopic coupling: angular momentum from one axis bleeds into others
  - Gravity-gradient torque: tidal forces try to lock the long axis to nadir

This model captures those effects so mission analyses can account for slew
timing, wheel momentum buildup, and desaturation costs.

State representation
─────────────────────
Attitude is stored as a quaternion q = [qw, qx, qy, qz] (scalar-first).
Quaternions avoid gimbal lock and are numerically stable over long
integrations. The attitude matrix R = R(q) converts body-frame vectors
to inertial frame.

Angular velocity ω = [ωx, ωy, ωz] is in the body frame [rad/s].

Reaction wheels
────────────────
Four reaction wheels in a pyramid configuration (standard for CubeSats):
three aligned with body axes + one skewed 54.7° for redundancy. Each wheel
stores angular momentum h_i [N·m·s].

Total wheel momentum: H_rw = A_rw @ h_w  (A_rw is the distribution matrix)

Control law: PD attitude control
  τ_cmd = -Kp × e_att - Kd × ω

  where e_att is the attitude error quaternion's vector part.

The commanded torque is distributed to wheels via the pseudoinverse of A_rw.
Wheel angular velocity is saturated at ω_max; when saturated the wheel
cannot accept more momentum (torque authority is lost in that direction).

References
───────────
  Markley, F.L., Crassidis, J.L. (2014). Fundamentals of Spacecraft
  Attitude Determination and Control. Springer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

# ── Quaternion utilities ──────────────────────────────────────────────────────


def quat_multiply(p: NDArray, q: NDArray) -> NDArray:
    """Hamilton product of two quaternions [pw,px,py,pz] × [qw,qx,qy,qz]."""
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return np.array(
        [
            pw * qw - px * qx - py * qy - pz * qz,
            pw * qx + px * qw + py * qz - pz * qy,
            pw * qy - px * qz + py * qw + pz * qx,
            pw * qz + px * qy - py * qx + pz * qw,
        ]
    )


def quat_normalize(q: NDArray) -> NDArray:
    """Return a unit quaternion, or identity if the input norm is tiny."""
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


def quat_conjugate(q: NDArray) -> NDArray:
    """Quaternion conjugate for a scalar-first quaternion."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_dcm(q: NDArray) -> NDArray:
    """Direction cosine matrix from quaternion (body → inertial)."""
    qw, qx, qy, qz = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)],
        ]
    )


def attitude_error_quat(q_current: NDArray, q_target: NDArray) -> NDArray:
    """Error quaternion: q_err = q_target^-1 ⊗ q_current."""
    return quat_multiply(quat_conjugate(q_target), q_current)


def body_to_inertial(v_body: NDArray, q: NDArray) -> NDArray:
    """Rotate a body-frame vector to inertial frame."""
    return quat_to_dcm(q) @ v_body


def inertial_to_body(v_inertial: NDArray, q: NDArray) -> NDArray:
    """Rotate an inertial-frame vector to body frame."""
    return quat_to_dcm(q).T @ v_inertial


def sail_normal_from_attitude(q: NDArray, sail_body_axis: NDArray) -> NDArray:
    """Sail normal in inertial frame given attitude quaternion."""
    return body_to_inertial(sail_body_axis, q)


# ── Inertia tensor ────────────────────────────────────────────────────────────


def cubesat_inertia_12u(mass_kg: float = 12.0) -> NDArray:
    """
    Approximate inertia tensor for a 12U CubeSat.

    Geometry: 226.3 mm × 226.3 mm × 340.5 mm rectangular prism.
    Assumes uniform mass distribution (real distribution will differ
    when sail boom is deployed — update with CAD mass properties).
    """
    a = 0.2263  # x dimension [m]
    b = 0.2263  # y dimension [m]
    c = 0.3405  # z dimension (long axis) [m]  → sail boom along z

    Ixx = mass_kg / 12.0 * (b**2 + c**2)
    Iyy = mass_kg / 12.0 * (a**2 + c**2)
    Izz = mass_kg / 12.0 * (a**2 + b**2)

    return np.diag([Ixx, Iyy, Izz])


# ── Reaction wheel geometry ───────────────────────────────────────────────────


def pyramid_wheel_matrix() -> NDArray:
    """
    Distribution matrix A_rw [3×4] for a 4-wheel pyramid configuration.

    Wheel layout:
      Wheel 0: aligned with +x
      Wheel 1: aligned with +y
      Wheel 2: aligned with +z
      Wheel 3: skewed at 54.7° from z (redundant)

    H_total = A_rw @ h_wheels  →  total angular momentum vector [N·m·s]
    """
    beta = math.radians(54.7)  # pyramid half-angle
    return np.array(
        [
            [1.0, 0.0, 0.0, math.sin(beta)],
            [0.0, 1.0, 0.0, math.sin(beta)],
            [0.0, 0.0, 1.0, math.cos(beta)],
        ]
    )


# ── Attitude dynamics model ───────────────────────────────────────────────────


@dataclass
class AttitudeConfig:
    """Physical parameters for the attitude control system."""

    # Reaction wheel specs (per wheel)
    wheel_max_momentum_nms: float = 0.015  # [N·m·s]  saturation limit
    wheel_max_torque_nm: float = 0.001  # [N·m]    peak torque per wheel
    wheel_friction_coeff: float = 0.0001  # viscous friction [N·m·s]

    # PD control gains
    # Gains tuned for ~10-minute 90° slew (LightSail 2 performance target)
    # ωn = sqrt(kp/Izz) = sqrt(1e-5/0.10) = 0.01 rad/s → period ≈ 630 s
    # Critical damping: kd = 2*sqrt(kp*Izz) = 2*sqrt(1e-5*0.10) ≈ 6.3e-4
    kp: float = 1e-5  # proportional gain [N·m/rad]
    kd: float = 6e-4  # derivative gain [N·m·s/rad]

    # Gravity-gradient coefficient (enabled near bodies)
    gravity_gradient: bool = True

    # Environmental disturbance torques [N·m]
    # Magnetic torque from eddy currents (dominant at LEO)
    magnetic_torque_nm: float = 1e-5  # 10 μN·m — typical CubeSat worst case
    # Adds realistic ~4-hour saturation cycle, matching real CubeSat operations.
    # Set to 0 to disable for faster simulations.
    enable_disturbances: bool = True

    # Sail normal direction in body frame (which face the sail is on)
    sail_body_axis: NDArray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))

    # Centre-of-pressure offset from centre-of-mass in body frame [m].
    # For a boom-deployed sail the CoP is at the sail geometric centre,
    # offset from the CubeSat bus (CoM) along the deployment axis.
    # Default 5 cm along body-z is representative of a 32 m² sail on a 12U bus.
    # This offset means SRP force produces a net attitude torque that the
    # reaction wheels must counteract, which couples attitude control to
    # propulsion and power-budget studies.
    # Set to np.zeros(3) to disable SRP torque coupling.
    cp_offset_body: NDArray = field(default_factory=lambda: np.array([0.0, 0.0, 0.05]))


class AttitudeDynamicsModel:
    """
    Full rigid-body attitude dynamics for a CubeSat with reaction wheels.

    State:
        q  [4]  — attitude quaternion  (body → inertial)
        ω  [3]  — body angular velocity  [rad/s]
        h  [4]  — reaction wheel angular momenta  [N·m·s]

    Control input:
        q_cmd  — target attitude quaternion

    Args:
        inertia:    3×3 inertia tensor in body frame [kg·m²]
        config:     AttitudeConfig with wheel and control parameters
    """

    def __init__(
        self,
        inertia: NDArray | None = None,
        config: AttitudeConfig | None = None,
    ):
        self.I = inertia if inertia is not None else cubesat_inertia_12u()
        self.I_inv = np.linalg.inv(self.I)
        self.cfg = config or AttitudeConfig()
        self.A_rw = pyramid_wheel_matrix()  # [3×4]
        self.A_rw_pinv = np.linalg.pinv(self.A_rw)  # pseudoinverse [4×3]

        # State — initialise to identity attitude
        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # unit quaternion
        self.omega = np.zeros(3)  # body angular velocity [rad/s]
        self.h_wheels = np.zeros(4)  # wheel momenta [N·m·s]
        self._q_cmd = np.array([1.0, 0.0, 0.0, 0.0])  # last commanded quaternion
        # Disturbance direction (set here and updated each step)
        self._disturbance_direction = np.array([1.0, 0.0, 0.0])
        self._dist_phase = 0.0

    # ── Public interface ──────────────────────────────────────────────────────

    def step(
        self,
        q_cmd: NDArray,
        dt: float,
        position: NDArray | None = None,
        mu: float = 3.986e14,
        srp_force_eci: NDArray | None = None,
        craft_type: str = "wheel",
    ) -> NDArray:
        """
        Advance attitude dynamics by dt seconds toward q_cmd.

        Uses RK4 integration of Euler's equations with PD control.

        Args:
            q_cmd:         target attitude quaternion [qw, qx, qy, qz]
            dt:            timestep [s]
            position:      spacecraft position for gravity-gradient [m]
            mu:            primary body gravitational parameter [m³/s²]
            srp_force_eci: total SRP force vector in ECI [N].  When provided,
                           the CoP offset (cfg.cp_offset_body) generates a
                           body-frame torque τ = r_cp × F_srp that loads the
                           reaction wheels during sustained illumination.
                           Pass None to disable (backward-compatible default).
            craft_type:    ``"wheel"`` (default) — reaction wheel path;
                           ``"rcs"`` — cold gas path: wheel ODE terms are
                           zeroed and ``h_wheels`` is held at zero.
                           Quaternion + omega propagation is identical in
                           both branches; only the torque source changes.

        Returns:
            body +Z axis in inertial frame [3] (unit vector); for wheel
            craft this is the sail normal, for RCS craft the attitude axis.
        """
        q_cmd = quat_normalize(q_cmd)
        self._q_cmd = q_cmd

        # Control torque from PD law
        tau_ctrl = self._pd_torque(q_cmd)

        # Gravity-gradient torque
        tau_gg = np.zeros(3)
        if self.cfg.gravity_gradient and position is not None:
            tau_gg = self._gravity_gradient_torque(position, mu)

        # Environmental disturbance torques (magnetic eddy currents, etc.)
        tau_dist = np.zeros(3)
        if self.cfg.enable_disturbances and self.cfg.magnetic_torque_nm > 0:
            tau_dist = self.cfg.magnetic_torque_nm * self._disturbance_direction

        # SRP torque from CoP offset ──────────────────────────────────────────
        # τ_srp (body) = r_cp (body) × F_srp (body)
        # F_srp is supplied in ECI; rotate to body frame before cross-product.
        tau_srp = np.zeros(3)
        if srp_force_eci is not None and np.any(self.cfg.cp_offset_body != 0.0):
            F_srp_body = inertial_to_body(srp_force_eci, self.q)
            tau_srp = np.cross(self.cfg.cp_offset_body, F_srp_body)

        # Total external torque (control + gravity-gradient + disturbances + SRP)
        tau_ext = tau_ctrl + tau_gg + tau_dist + tau_srp

        # RK4 integration
        y0 = np.concatenate([self.q, self.omega, self.h_wheels])

        # Renormalize quaternion before each RK4 stage to prevent overflow
        def safe_deriv(y, tau):
            y = y.copy()
            y[0:4] = quat_normalize(y[0:4])
            return self._derivatives(y, tau, craft_type)

        k1 = safe_deriv(y0, tau_ext)
        k2 = safe_deriv(y0 + 0.5 * dt * k1, tau_ext)
        k3 = safe_deriv(y0 + 0.5 * dt * k2, tau_ext)
        k4 = safe_deriv(y0 + dt * k3, tau_ext)
        y1 = y0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Unpack and normalise quaternion
        self.q = quat_normalize(y1[0:4])
        self.omega = y1[4:7]
        self._update_disturbance(dt)
        if craft_type == "rcs":
            # No reaction wheels on RCS craft — hold at zero to prevent
            # any numerical drift from leaking into the wheel ODE.
            self.h_wheels = np.zeros(4)
        else:
            # Hard backstop: the smooth taper in _pd_torque is the primary
            # saturation mechanism. This clip only catches the small overshoot
            # that RK4 discretisation can produce at large dt values.
            self.h_wheels = np.clip(
                y1[7:11],
                -self.cfg.wheel_max_momentum_nms,
                self.cfg.wheel_max_momentum_nms,
            )

        return self.sail_normal_inertial

    @property
    def sail_normal_inertial(self) -> NDArray:
        """Sail face normal vector in inertial frame."""
        return body_to_inertial(self.cfg.sail_body_axis, self.q)

    @property
    def wheel_saturation_fraction(self) -> float:
        """Max wheel momentum as fraction of saturation limit (0–1)."""
        return float(np.max(np.abs(self.h_wheels))) / self.cfg.wheel_max_momentum_nms

    @property
    def attitude_error_deg(self) -> float:
        """Angle between current attitude and last commanded target [degrees]."""
        q_err = attitude_error_quat(self.q, self._q_cmd)
        # rotation angle = 2 * arccos(|scalar part|), clamped for numerical safety
        return float(np.degrees(2.0 * np.arccos(np.clip(abs(q_err[0]), 0.0, 1.0))))

    def reset(self, q_init: NDArray | None = None) -> None:
        """Reset to initial attitude with zero angular velocity."""
        self.q = quat_normalize(q_init) if q_init is not None else np.array([1.0, 0.0, 0.0, 0.0])
        self.omega = np.zeros(3)
        self.h_wheels = np.zeros(4)
        # Disturbance direction rotates slowly to ensure secular accumulation
        self._disturbance_direction = np.array([1.0, 0.0, 0.0])
        self._dist_phase = 0.0

    def desaturate(self, thruster_torque_available_nm: float = 1e-3) -> float:
        """
        One desaturation step using thruster torques.

        Applies an impulsive counter-torque proportional to accumulated wheel
        momentum. Call this over multiple steps to fully desaturate. Each
        call costs a small amount of Δv.

        Returns:
            delta_v_cost_ms [m/s] — approximate Δv cost of this burn.
            A typical desaturation step costs ~4 mm/s.
        """
        H_total = self.A_rw @ self.h_wheels
        h_mag = np.linalg.norm(H_total)
        if h_mag < 1e-10:
            return 0.0

        # Desaturation torque opposes net wheel momentum direction
        tau_desat = -H_total / h_mag * min(h_mag / 5.0, thruster_torque_available_nm)

        # Wheel momentum change to shrink total angular momentum H toward zero.
        # We want: H_new = A_rw @ (h + dh) = H + tau_desat  (H shrinks)
        # Since tau_desat = -H/|H| * factor (points opposite to H):
        # H_new = H - factor*H/|H|  → |H_new| = |H| - factor  ✓
        # So: dh = A_rw_pinv @ tau_desat  (same sign, not negative)
        dh = self.A_rw_pinv @ tau_desat
        max_change = thruster_torque_available_nm * 1.0
        dh = np.clip(dh, -max_change, max_change)
        self.h_wheels = np.clip(
            self.h_wheels + dh,
            -self.cfg.wheel_max_momentum_nms,
            self.cfg.wheel_max_momentum_nms,
        )

        lever_arm_m = 0.1
        mass_kg = 12.0
        F_n = np.linalg.norm(tau_desat) / lever_arm_m
        return float((F_n / mass_kg) * 1.0)  # Δv for 1-second burn

    def sail_normal_to_quaternion(self, sail_normal_inertial: NDArray) -> NDArray:
        """
        Compute the attitude quaternion that points the sail toward the
        given inertial direction.

        Minimal-rotation solution — rotates the body sail axis to align
        with the target normal using the shortest arc.
        """
        sail_body = self.cfg.sail_body_axis
        target = sail_normal_inertial / (np.linalg.norm(sail_normal_inertial) + 1e-12)

        cross = np.cross(sail_body, target)
        cross_mag = np.linalg.norm(cross)
        dot = float(np.dot(sail_body, target))

        if cross_mag < 1e-8:
            if dot > 0:
                return np.array([1.0, 0.0, 0.0, 0.0])  # already aligned
            else:
                # 180° rotation — pick the most orthogonal basis vector
                # to avoid numerical issues
                candidates = [
                    np.array([0.0, 0.0, 1.0]),
                    np.array([0.0, 1.0, 0.0]),
                    np.array([1.0, 0.0, 0.0]),
                ]
                perp = min(candidates, key=lambda v: abs(float(np.dot(sail_body, v))))
                perp = np.cross(sail_body, perp)
                perp = perp / np.linalg.norm(perp)
                # Half-angle = π/2, so q = [cos(π/2), sin(π/2)*axis] = [0, axis]
                return np.array([0.0, perp[0], perp[1], perp[2]])

        angle = math.acos(np.clip(dot, -1.0, 1.0))
        axis = cross / np.linalg.norm(cross)
        return np.array([math.cos(angle / 2), *(math.sin(angle / 2) * axis)])

    # ── Internals ─────────────────────────────────────────────────────────────

    def _pd_torque(self, q_cmd: NDArray) -> NDArray:
        """PD control law — computes wheel torque commands."""
        q_err = attitude_error_quat(self.q, q_cmd)

        # Attitude error: vector part of error quaternion (sign ensures short-arc)
        sign = 1.0 if q_err[0] >= 0 else -1.0
        e_att = sign * q_err[1:4]

        # Desired torque
        tau_desired = -self.cfg.kp * e_att - self.cfg.kd * self.omega

        # Distribute to wheels via pseudoinverse.
        # h_dot = -(A_rw_pinv @ tau_desired) so that:
        #   spacecraft reaction = -A_rw @ h_dot = +tau_desired (correct sign)
        dh_dt = -(self.A_rw_pinv @ tau_desired)

        # Saturate at max wheel torque
        dh_dt = np.clip(dh_dt, -self.cfg.wheel_max_torque_nm, self.cfg.wheel_max_torque_nm)

        # Smooth capacity taper: linearly reduce authority as each wheel
        # approaches its momentum limit. Authority is full below taper_start
        # and reaches zero exactly at h_max, eliminating the discontinuity
        # that the old binary 95%-cliff produced.
        h_max = self.cfg.wheel_max_momentum_nms
        taper_start = 0.85 * h_max  # full authority below 85% saturation
        taper_zone = h_max - taper_start
        for i in range(4):
            if dh_dt[i] * self.h_wheels[i] > 0:  # only when pushing toward limit
                headroom = h_max - abs(self.h_wheels[i])
                alpha = float(np.clip(headroom / taper_zone, 0.0, 1.0))
                dh_dt[i] *= alpha

        # Reaction torque on spacecraft = -A_rw @ h_dot = +tau_desired
        return -self.A_rw @ dh_dt

    def _gravity_gradient_torque(
        self,
        position: NDArray,
        mu: float,
    ) -> NDArray:
        """
        Gravity-gradient torque in body frame.

        τ_gg = (3μ/r⁵) × r × (I × r)

        where r is the nadir vector (position) rotated to body frame.
        Significant at LEO; negligible beyond ~10,000 km.
        """
        r_mag = float(np.linalg.norm(position))
        r_body = inertial_to_body(position, self.q) / r_mag  # unit nadir in body

        tau = (3.0 * mu / r_mag**3) * np.cross(r_body, self.I @ r_body)
        return tau

    def _derivatives(self, y: NDArray, tau_ext: NDArray, craft_type: str = "wheel") -> NDArray:
        """
        dy/dt for the full attitude state vector.

        Layout: y = [q0, q1, q2, q3, ωx, ωy, ωz, h0, h1, h2, h3]

        When craft_type == "rcs" the wheel ODE (dh/dt) is zeroed and
        h_w is treated as zero for the angular-momentum coupling term,
        so Euler's equations simplify to the pure rigid-body form.
        """
        q = quat_normalize(y[0:4])
        omega = y[4:7]
        h_w = y[7:11] if craft_type == "wheel" else np.zeros(4)

        # ── Quaternion kinematics: dq/dt = 0.5 × q ⊗ [0, ω] ─────────────
        omega_quat = np.array([0.0, *omega])
        dq_dt = 0.5 * quat_multiply(q, omega_quat)

        # ── Euler's equations: I·dω/dt = τ_ext - ω × (I·ω + H_rw) ──────
        H_rw = self.A_rw @ h_w  # zero for RCS craft
        L_body = self.I @ omega + H_rw
        domega_dt = self.I_inv @ (tau_ext - np.cross(omega, L_body))

        # ── Wheel dynamics: dh/dt = τ_wheel - friction ───────────────────
        if craft_type == "rcs":
            dh_dt = np.zeros(4)  # no reaction wheels
        else:
            tau_wheel = -self.A_rw_pinv @ tau_ext  # torque reaction on wheels
            dh_dt = tau_wheel - self.cfg.wheel_friction_coeff * h_w

        return np.concatenate([dq_dt, domega_dt, dh_dt])

    def _update_disturbance(self, dt: float) -> None:
        """Slowly rotate the disturbance direction vector."""
        # Rotates at 1 rev per ~100 minutes (orbital period)
        self._dist_phase = (self._dist_phase + dt * 2 * math.pi / 6000.0) % (2 * math.pi)
        self._disturbance_direction = np.array(
            [
                math.cos(self._dist_phase),
                math.sin(self._dist_phase),
                0.1 * math.sin(self._dist_phase * 0.7),
            ]
        )
