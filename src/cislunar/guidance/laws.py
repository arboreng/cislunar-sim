"""
Orbital steering laws for cislunar spacecraft.

Four laws are provided, in increasing sophistication:

ProgradeGuidance — burns prograde, narrows to periapsis arc as eccentricity
    rises (Oberth timing).  Good baseline for LEO raising.

GTOPeriapsisGuidance — raises periapsis by burning near apogee (the long slow
    arc).  Skips the fast draggy periapsis dash.  Designed for GTO (e ≈ 0.73)
    where a 60 s thruster warmup delay makes periapsis burns impractical.

GTOApogeeGuidance — raises apogee by burning near periapsis.  Complement of
    GTOPeriapsisGuidance; used once periapsis is safely raised.

GVEPeriapsisGuidance — GVE-optimal direction for periapsis raise.  Computes
    the thrust direction that maximises dr_p/dt at each instant.  Naturally
    coasts at periapsis (C_r = C_t = 0 there); no explicit gate required.

GVEApogeeGuidance — GVE-optimal direction for apogee raise.  Symmetric
    complement of GVEPeriapsisGuidance; naturally coasts at apogee.

All laws share a single interface::

    thrust_dir, throttle = law.steer(pos_m, vel_ms)

where ``pos_m`` and ``vel_ms`` are ECI state vectors in SI units (metres, m/s)
and the return values are a unit thrust direction in ECI and a throttle in
{0.0, 1.0}.  An optional ``battery_fraction`` argument (0–1) suppresses
thrusting below 0.2 to protect the power bus.
"""

from __future__ import annotations

import numpy as np

from ..physics.constants import Body as _Body

_LOW_BATTERY = 0.2


def _rv_to_elements(
    pos_m: np.ndarray, vel_ms: np.ndarray
) -> tuple[float, float, float, float, float]:
    """
    Compute two-body osculating elements from ECI state vectors.

    Returns (a, e, h, cos_nu, sin_nu):
      a       semi-major axis, metres
      e       eccentricity
      h       specific angular momentum magnitude, m²/s
      cos_nu  cosine of true anomaly
      sin_nu  sine of true anomaly (positive = moving away from periapsis)
    """
    r = float(np.linalg.norm(pos_m))
    v = float(np.linalg.norm(vel_ms))

    h_vec = np.cross(pos_m, vel_ms)
    h = float(np.linalg.norm(h_vec))

    # Vis-viva: 1/a = 2/r - v²/μ
    vis_viva_denom = 2.0 / r - v**2 / _Body.MU_EARTH
    a = 1.0 / vis_viva_denom if abs(vis_viva_denom) > 1e-20 else 1e12

    # Eccentricity vector (points toward periapsis)
    e_vec = np.cross(vel_ms, h_vec) / _Body.MU_EARTH - pos_m / r
    e = float(np.linalg.norm(e_vec))

    if e > 1e-9:
        e_hat = e_vec / e
        cos_nu = float(np.clip(np.dot(e_hat, pos_m / r), -1.0, 1.0))
        # r·v > 0 when moving away from periapsis → sin_nu > 0
        sin_nu = float(np.sign(np.dot(pos_m, vel_ms)) * np.sqrt(max(0.0, 1.0 - cos_nu**2)))
    else:
        cos_nu, sin_nu = 1.0, 0.0

    return a, e, h, cos_nu, sin_nu


class ProgradeGuidance:
    """
    Prograde-burn steering with Oberth timing.

    Burns prograde at all times for circular orbits.  As eccentricity rises,
    narrows the burn window to the ascending arc near periapsis where Oberth
    efficiency is highest.

    Burn gate: r̂·v̂ < 1 − 0.8·e
      e = 0  → cutoff 1.0 → always burns.
      e = 0.73 (GTO) → cutoff 0.42 → ascending arc only.
    """

    def steer(
        self,
        pos_m: np.ndarray,
        vel_ms: np.ndarray,
        battery_fraction: float = 1.0,
    ) -> tuple[np.ndarray, float]:
        """Return ``(thrust_direction, throttle)`` for the given ECI state."""
        pos = np.asarray(pos_m, dtype=np.float64)
        vel = np.asarray(vel_ms, dtype=np.float64)
        speed = float(np.linalg.norm(vel))

        if speed < 1e-9:
            return np.array([1.0, 0.0, 0.0]), 0.0

        prograde = vel / speed
        r = float(np.linalg.norm(pos))
        in_burn_window = True

        if r > 1e-9:
            _, e, _, _, _ = _rv_to_elements(pos, vel)
            radial_frac = float(np.dot(pos / r, prograde))
            in_burn_window = radial_frac < (1.0 - 0.8 * e)

        throttle = 1.0 if (in_burn_window and battery_fraction > _LOW_BATTERY) else 0.0
        return prograde, throttle


class GTOPeriapsisGuidance:
    """
    GTO periapsis-raise steering (apogee-burn strategy).

    Burns prograde near apogee, coasting through the fast low-altitude
    periapsis passage.  Raises periapsis each orbit.

    Burn gate: cos(ν) < 1 − e
      e = 0 (circular)  → burn everywhere.
      e = 0.73 (GTO)    → fire for ν > 74° (skip periapsis dash).
      e → 1             → fire only near apogee.
    """

    def steer(
        self,
        pos_m: np.ndarray,
        vel_ms: np.ndarray,
        battery_fraction: float = 1.0,
    ) -> tuple[np.ndarray, float]:
        """Return ``(thrust_direction, throttle)`` for the given ECI state."""
        pos = np.asarray(pos_m, dtype=np.float64)
        vel = np.asarray(vel_ms, dtype=np.float64)
        speed = float(np.linalg.norm(vel))

        if speed < 1e-9:
            return np.array([1.0, 0.0, 0.0]), 0.0

        prograde = vel / speed
        _, e, _, cos_nu, _ = _rv_to_elements(pos, vel)
        in_burn_window = cos_nu < (1.0 - e)

        throttle = 1.0 if (in_burn_window and battery_fraction > _LOW_BATTERY) else 0.0
        return prograde, throttle


class GTOApogeeGuidance:
    """
    GTO apogee-raise steering (periapsis-burn strategy).

    Burns prograde through the periapsis half of each orbit, raising the
    apoapsis incrementally.  Complement of GTOPeriapsisGuidance.

    Burn gate: cos(ν) > e − 1
      e = 0 (circular)  → burn everywhere.
      e = 0.73 (GTO)    → fire for ν < 106° (broad periapsis window).
      e → 1             → fire when cos(ν) > 0.
    """

    def steer(
        self,
        pos_m: np.ndarray,
        vel_ms: np.ndarray,
        battery_fraction: float = 1.0,
    ) -> tuple[np.ndarray, float]:
        """Return ``(thrust_direction, throttle)`` for the given ECI state."""
        pos = np.asarray(pos_m, dtype=np.float64)
        vel = np.asarray(vel_ms, dtype=np.float64)
        speed = float(np.linalg.norm(vel))

        if speed < 1e-9:
            return np.array([1.0, 0.0, 0.0]), 0.0

        prograde = vel / speed
        _, e, _, cos_nu, _ = _rv_to_elements(pos, vel)
        in_burn_window = cos_nu > (e - 1.0)

        throttle = 1.0 if (in_burn_window and battery_fraction > _LOW_BATTERY) else 0.0
        return prograde, throttle


def _gve_steer(
    pos_m: np.ndarray,
    vel_ms: np.ndarray,
    battery_fraction: float,
    sign: float,  # -1 → periapsis raise;  +1 → apogee raise
    coast_efficiency_frac: float,
    target_m: float | None,
) -> tuple[np.ndarray, float]:
    """
    GVE-optimal single-element orbit raise.

    With sign = -1 the objective is dr_periapsis/dt = (1-e)·da/dt - a·de/dt.
    With sign = +1 the objective is dr_apoapsis/dt  = (1+e)·da/dt + a·de/dt.
    In both cases C_r = C_t = 0 at the opposite apse, so coasting there is natural.
    """
    pos = np.asarray(pos_m, dtype=np.float64)
    vel = np.asarray(vel_ms, dtype=np.float64)
    r = float(np.linalg.norm(pos))
    v = float(np.linalg.norm(vel))

    thrust_eci = vel / (v + 1e-12)
    in_burn_window = False
    a, e_raw, h_orb, cos_nu, sin_nu = _rv_to_elements(pos, vel)

    if r > 1e3 and v > 1e-3:
        h_vec = np.cross(pos, vel)
        h_mag = float(np.linalg.norm(h_vec))

        if h_mag > 1e-3:
            R_hat = pos / r
            N_hat = h_vec / h_mag
            T_hat = np.cross(N_hat, R_hat)

            e = float(np.clip(e_raw, 0.0, 0.9999))
            p_orb = a * (1.0 - e**2)
            r_cur = p_orb / (1.0 + e * cos_nu + 1e-30)

            # GVE partials for da/dt and de/dt (Battin §10.6)
            da_fr = (2.0 * a**2 / h_orb) * e * sin_nu
            da_ft = (2.0 * a**2 / h_orb) * (p_orb / r_cur)
            de_fr = p_orb * sin_nu / h_orb
            de_ft = ((p_orb + r_cur) * cos_nu + r_cur * e) / h_orb

            # dr/dt = (1 + sign·e)·da/dt + sign·a·de/dt
            C_r = (1.0 + sign * e) * da_fr + sign * a * de_fr
            C_t = (1.0 + sign * e) * da_ft + sign * a * de_ft

            efficiency = float(np.sqrt(C_r**2 + C_t**2))
            peak = 4.0 * a**2 * (1.0 + sign * e) / h_orb

            if efficiency > coast_efficiency_frac * peak:
                in_burn_window = True
                thrust_eci = (C_r * R_hat + C_t * T_hat) / efficiency

    if target_m is not None:
        e = float(np.clip(e_raw, 0.0, 0.9999))
        if a * (1.0 + sign * e) >= target_m:
            in_burn_window = False

    throttle = 1.0 if (in_burn_window and battery_fraction > _LOW_BATTERY) else 0.0
    return thrust_eci, throttle


class GVEPeriapsisGuidance:
    """
    GVE-optimal steering for GTO periapsis raise.

    At each step the Gauss Variational Equations (GVE) give dr_p/dt as a
    linear function of thrust components in the RTN frame::

        dr_p/dt = C_r · f_r + C_t · f_t

    The optimal unit-thrust direction is (C_r·R̂ + C_t·T̂) / ‖(C_r, C_t)‖.

    Key result: C_r = C_t = 0 exactly at periapsis, so the law naturally coasts
    there — no explicit gate required.  Peak efficiency occurs at apogee:
    4a²(1−e)/h.

    **Naming note:** despite the class name, this is *not* the Lyapunov Q-law of
    Petropoulos (2004), which minimises a weighted sum of element errors over
    all five orbital elements simultaneously.  This law is the simpler
    single-element GVE-optimal direction for periapsis raise; it is
    well-suited for GTO and low-thrust LEO raising but does not generalise to
    multi-element targeting.

    Reference: Battin §10.6; Schaub & Junkins eq 9.78.

    Parameters
    ----------
    coast_efficiency_frac:
        Coast when instantaneous efficiency < this fraction of the apogee peak.
        Default 0.05 captures most of the apogee arc while skipping the
        lowest-return region near periapsis.
    target_periapsis_m:
        Coast once the osculating periapsis radius reaches this value, preserving
        propellant for subsequent mission phases.
        Default: R_Earth + 1000 km.  Set to None to raise until halted.
    """

    def __init__(
        self,
        coast_efficiency_frac: float = 0.05,
        target_periapsis_m: float | None = _Body.R_EARTH + 1_000e3,
    ) -> None:
        self._coast_frac = coast_efficiency_frac
        self._target_peri_m = target_periapsis_m

    def steer(
        self,
        pos_m: np.ndarray,
        vel_ms: np.ndarray,
        battery_fraction: float = 1.0,
    ) -> tuple[np.ndarray, float]:
        """Return ``(thrust_direction, throttle)`` for the given ECI state."""
        return _gve_steer(
            pos_m, vel_ms, battery_fraction, -1.0, self._coast_frac, self._target_peri_m
        )


class GVEApogeeGuidance:
    """
    GVE-optimal steering for GTO apogee raise.

    Symmetric complement of GVEPeriapsisGuidance.  See that class for a note
    on the naming convention (this is single-element GVE-optimal steering, not
    the Petropoulos Q-law).

    Uses GVE to compute the thrust direction that maximally increases dr_apo/dt::

        r_apo = a(1+e)  →  dr_apo/dt = (1+e)·da/dt + a·de/dt
        C_r = (1+e)·da_fr + a·de_fr
        C_t = (1+e)·da_ft + a·de_ft

    C_r = C_t = 0 at apogee — natural coast.  Peak efficiency at periapsis:
    4a²(1+e)/h.

    Parameters
    ----------
    coast_efficiency_frac:
        Coast when instantaneous efficiency < this fraction of the periapsis
        peak.  Default 0.5 (higher than GVEPeriapsisGuidance because the
        apogee-raise efficiency curve falls slowly — at ν=90° it is ~70% of
        peak for GTO, so 0.05 would fire almost the full orbit).
    target_apoapsis_m:
        Coast once the osculating apoapsis radius reaches this value.
        Set to None to raise until halted externally.
    """

    def __init__(
        self,
        coast_efficiency_frac: float = 0.5,
        target_apoapsis_m: float | None = None,
    ) -> None:
        self._coast_frac = coast_efficiency_frac
        self._target_apo_m = target_apoapsis_m

    def steer(
        self,
        pos_m: np.ndarray,
        vel_ms: np.ndarray,
        battery_fraction: float = 1.0,
    ) -> tuple[np.ndarray, float]:
        """Return ``(thrust_direction, throttle)`` for the given ECI state."""
        return _gve_steer(
            pos_m, vel_ms, battery_fraction, +1.0, self._coast_frac, self._target_apo_m
        )
