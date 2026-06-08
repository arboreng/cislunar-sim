"""
Eclipse detection and solar power budget model.

Without this model, power would be treated as a constant. In reality:

  - At 400 km LEO: 35 min in shadow per 92-min orbit (38% blackout)
  - During eclipse: solar array output = 0 W → thruster cannot fire
  - After eclipse: battery charges from solar array, thruster can fire again
  - In deep cislunar: eclipse rarer but comms blackouts still apply

Power-aware simulations can therefore time burns around illuminated orbital
arcs, quantify recovery after eclipse, and study the coupling between sail
orientation and electrical generation.

Eclipse geometry — conical umbra and penumbra
──────────────────────────────────────────────
The spacecraft is in eclipse when the Earth's shadow cone blocks the Sun.
We model the full dual-cone geometry (umbra + penumbra):

  Umbra cone    — at axial distance d behind Earth:
                  r_u(d) = R_body − d × (R_sun − R_body) / D_sun

  Penumbra cone — at axial distance d behind Earth:
                  r_p(d) = R_body + d × R_sun / D_sun

  shadow_fraction() returns:
    0.0  — full umbra (total eclipse)
    1.0  — full sunlight
   (0,1) — penumbra (partial illumination)

At a 60-second propagation cadence the penumbra crossing takes ~8 seconds,
which is below the step size of many baseline simulations. The
model is nonetheless physically correct and ready for higher-resolution use.

  APPROXIMATION: Earth is treated as a sphere (no oblateness).
  Error: < 20 km at the shadow boundary → < 2.6 s timing error.
  APPROXIMATION: Moon's shadow uses the same geometry.
  APPROXIMATION: Sun–Earth distance is read from the supplied sun_pos vector;
                 no annual parallax correction is applied separately.

Power model
────────────
    P_solar = shadow_fraction × η × G_sc × A_array × cos(θ_array)  [W]

    η         = array efficiency (0.28 typical for triple-junction GaAs)
    G_sc      = solar constant at spacecraft heliocentric distance [W/m²]
    A_array   = total solar array area [m²]
    cos(θ)    = projection of array normal onto sun direction

During full eclipse shadow_fraction=0 → P_solar=0.
During penumbra P_solar scales continuously with illumination fraction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..constants import Sail as SailConst
from .attitude import quat_to_dcm

# ── Constants ─────────────────────────────────────────────────────────────────

R_EARTH_M = 6.371e6  # Earth mean radius [m]
R_MOON_M = 1.7374e6  # Moon mean radius [m]  (GRAIL-refined)
R_SUN_M = 6.957e8  # Solar radius [m]
AU_M = 1.496e11  # 1 AU in metres
SOLAR_CONST = 1361.0  # Solar irradiance at 1 AU [W m⁻²]
SPEED_OF_LIGHT = 2.998e8  # [m s⁻¹]


# ── Eclipse detection ─────────────────────────────────────────────────────────


def shadow_fraction(
    spacecraft_pos: NDArray[np.float64],
    sun_pos: NDArray[np.float64],
    body_radius_m: float = R_EARTH_M,
    sun_radius_m: float = R_SUN_M,
) -> float:
    """
    Solar illumination fraction at the spacecraft position.

    Implements the conical dual-shadow geometry (umbra + penumbra) using
    the finite angular size of the Sun.

    Args:
        spacecraft_pos: spacecraft position in body-centred ECI [m]
        sun_pos:        Sun position in body-centred ECI [m]
        body_radius_m:  radius of the occulting body [m]
        sun_radius_m:   Solar radius [m]

    Returns:
        float in [0, 1]:
          0.0 → full umbra (zero illumination)
          1.0 → full sunlight
         (0,1) → penumbra (partial illumination, linear interpolation)
    """
    r_sun = float(np.linalg.norm(sun_pos))
    if r_sun < 1.0:
        return 1.0  # degenerate: assume sunlit

    sun_dir = sun_pos / r_sun

    # Axial distance behind the body (positive = anti-Sun side)
    sc_along_sun = float(np.dot(spacecraft_pos, sun_dir))
    if sc_along_sun >= 0:
        return 1.0  # spacecraft on Sun side → always sunlit

    d = -sc_along_sun  # distance behind body centre along anti-Sun axis

    # Perpendicular distance from the body–Sun axis
    sc_perp_vec = spacecraft_pos - sc_along_sun * sun_dir
    perp_dist = float(np.linalg.norm(sc_perp_vec))

    # Conical umbra and penumbra radii at this axial distance
    # Umbra cone:    narrows away from body;  r_u = R_body − d·(R_sun−R_body)/D_sun
    # Penumbra cone: widens away from body;   r_p = R_body + d·R_sun/D_sun
    r_umbra = body_radius_m - d * (sun_radius_m - body_radius_m) / r_sun
    r_penumbra = body_radius_m + d * sun_radius_m / r_sun

    if r_umbra > 0 and perp_dist < r_umbra:
        return 0.0  # full umbra

    if perp_dist >= r_penumbra:
        return 1.0  # full sunlight

    # Penumbra: linear interpolation between umbra boundary and penumbra edge
    r_inner = max(0.0, r_umbra)
    return float((perp_dist - r_inner) / (r_penumbra - r_inner))


def is_in_eclipse(
    spacecraft_pos: NDArray[np.float64],
    sun_pos: NDArray[np.float64],
    body_radius_m: float = R_EARTH_M,
) -> bool:
    """
    True if the spacecraft is in full umbra (zero illumination).

    Backward-compatible wrapper around shadow_fraction().  Use
    shadow_fraction() directly when partial penumbra illumination matters.

    Args:
        spacecraft_pos: spacecraft position in body-centred ECI [m]
        sun_pos:        Sun position in body-centred ECI [m]
        body_radius_m:  radius of the occulting body [m]

    Returns:
        True if shadow_fraction == 0 (full umbra)
    """
    return shadow_fraction(spacecraft_pos, sun_pos, body_radius_m) == 0.0


def bisect_eclipse_boundary(
    pos_start: NDArray[np.float64],
    pos_end: NDArray[np.float64],
    sun_pos: NDArray[np.float64],
    body_radius_m: float = R_EARTH_M,
    n_iter: int = 8,
) -> float:
    """
    Average shadow fraction over an integration step that crosses an
    eclipse boundary.

    The standard integrator evaluates ``shadow_fraction`` at a single
    point (the end of the accepted sub-step).  When the spacecraft
    crosses the umbra boundary mid-step the integrator smears the
    transition: half a 60-s step can span ~3-4 km of shadow-boundary
    geometry, introducing a ~30-s timing error in the power-off event.

    This function uses bisection to pin the crossing to within
    ``(pos_end - pos_start) / 2**n_iter`` in position space — at
    n_iter=8 that is < 10 m for typical LEO steps — and returns a
    time-weighted average illumination for the step.

    Args:
        pos_start:      spacecraft ECI position at step start [m]
        pos_end:        spacecraft ECI position at step end   [m]
        sun_pos:        Sun ECI position [m]
        body_radius_m:  occulting body radius [m]  (Earth default)
        n_iter:         bisection iterations (8 → ~256× step resolution)

    Returns:
        float in [0, 1]: average illumination fraction for the step.

    Notes:
        * Called only when there is a prior stored position, so the very
          first step of an episode uses the plain ``shadow_fraction`` value.
        * The linear position interpolation is exact for straight-line
          motion; curvature over a 60-s LEO sub-step is < 100 m, giving
          a < 0.1 s error on the boundary time — acceptable for this model.
        * Moon shadow uses the same geometry; pass ``body_radius_m=R_MOON_M``
          when checking lunar occultation.
    """
    f_start = shadow_fraction(pos_start, sun_pos, body_radius_m)
    f_end = shadow_fraction(pos_end, sun_pos, body_radius_m)

    in_shadow_start = f_start == 0.0
    in_shadow_end = f_end == 0.0

    # No boundary crossing — midpoint average is accurate enough.
    if in_shadow_start == in_shadow_end:
        return 0.5 * (f_start + f_end)

    # Binary search for the crossing fraction α ∈ [0, 1] along the step.
    lo, hi = 0.0, 1.0
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        pos_mid = pos_start + mid * (pos_end - pos_start)
        f_mid = shadow_fraction(pos_mid, sun_pos, body_radius_m)
        if (f_mid == 0.0) == in_shadow_start:
            lo = mid  # still in same state as start
        else:
            hi = mid  # crossed to the other state

    alpha = 0.5 * (lo + hi)  # fraction of step spent in the starting state

    # Weighted average: alpha in starting illumination, (1-alpha) in ending.
    return alpha * f_start + (1.0 - alpha) * f_end


def eclipse_fraction_circular_orbit(altitude_m: float) -> float:
    """
    Fraction of a circular orbit spent in Earth's umbra shadow.

    Analytical formula for equatorial orbit (worst case is slightly
    higher for inclined orbits — this is a conservative lower bound).

    APPROXIMATION: uses the umbra half-angle (cylindrical approximation
    for the orbit-crossing fraction).  Error < 0.5% at LEO.
    Penumbra adds ~0.4% to this fraction — negligible for energy budgets.

    Args:
        altitude_m: orbit altitude above Earth's surface [m]

    Returns:
        fraction in [0, 1] of orbit in umbra shadow
    """
    r = R_EARTH_M + altitude_m
    half_angle = math.asin(R_EARTH_M / r)  # half-angle of shadow cone
    return half_angle / math.pi  # fraction of full circle


def body_shadow_illumination(
    spacecraft_pos_eci: NDArray[np.float64],
    sun_pos_eci: NDArray[np.float64],
    moon_pos_eci: NDArray[np.float64] | None = None,
) -> float:
    """
    Solar illumination fraction seen by the spacecraft, accounting for
    Earth AND (optionally) Moon occultation of the Sun.

    This is the quantity the solar-sail force and the solar-array power
    model should both consume: during full umbra the sail should produce
    zero SRP force (not full force, as the naive flat-plate model
    implies), and the solar array should produce zero power.

    Composition is multiplicative — the two bodies occulting the Sun
    simultaneously is vanishingly rare but the product
    ``f_earth × f_moon`` is the correct general form and costs nothing.

    Frame note
    ──────────
    All inputs are in Earth-centred ECI [m].  This function translates
    into Moon-centred coordinates internally when computing the lunar
    shadow, then returns the combined illumination fraction.

    Args:
        spacecraft_pos_eci: spacecraft position in ECI [m]
        sun_pos_eci:        Sun position in ECI [m]
                            (Earth→Sun vector in Earth-centred ECI)
        moon_pos_eci:       Moon position in ECI [m], or None to skip the
                            lunar-shadow check (Earth-only eclipse).

    Returns:
        float in [0, 1]:
          1.0 → full sunlight (no occultation)
          0.0 → full umbra of at least one body
         (0,1) → penumbra of one or both bodies
    """
    f_earth = shadow_fraction(spacecraft_pos_eci, sun_pos_eci, R_EARTH_M)

    # Short-circuit if already in Earth's umbra — no need to check Moon.
    if f_earth == 0.0 or moon_pos_eci is None:
        return f_earth

    # Translate into Moon-centred frame for the lunar occultation check.
    sc_from_moon = spacecraft_pos_eci - moon_pos_eci
    sun_from_moon = sun_pos_eci - moon_pos_eci
    f_moon = shadow_fraction(sc_from_moon, sun_from_moon, R_MOON_M)

    return f_earth * f_moon


def sail_occultation_fraction(
    observer_pos_eci: NDArray[np.float64],
    sun_pos_eci: NDArray[np.float64],
    blocker_pos_eci: NDArray[np.float64],
    blocker_area_m2: float,
    blocker_normal_eci: NDArray[np.float64],
) -> float:
    """
    Solar illumination fraction at the observer due to ONE other sail
    occulting the Sun.

    Geometry
    ────────
    The blocker's silhouette perpendicular to the Sun direction is a disc
    of area ``A_blocker × max(cos θ_blocker, 0)`` where θ_blocker is the
    angle between the blocker's sail normal and the direction toward the
    Sun.  The effective blocker "body" is therefore a disc of radius

        r_blocker = sqrt(A_blocker × cos θ_blocker / π)

    which is treated exactly like Earth or the Moon in ``shadow_fraction``:
    the umbra/penumbra cones are formed against the finite Sun disc, and
    the observer's illumination is computed from its perpendicular distance
    to the Sun–blocker axis.

    Composition across multiple blockers is multiplicative:
    ``Π_j sail_occultation_fraction(..., blocker_j)`` — the same rule the
    Earth × Moon composition uses.

    Range sanity: for a 32 m² sail at cos θ = 1 the geometric umbra
    extends only to ~600 m separation, penumbra out to ~1 km.  Beyond that
    the return value saturates to 1.0 and callers pay only the dot-product
    early-exits.

    Frame note
    ──────────
    Inputs are ECI (any origin — only differences are used).

    Args:
        observer_pos_eci:    position of the observer [m]
        sun_pos_eci:         position of the Sun [m]
        blocker_pos_eci:     position of the blocking spacecraft [m]
        blocker_area_m2:     blocker's sail area [m²]
        blocker_normal_eci:  blocker's sail normal (unit) in ECI

    Returns:
        float in [0, 1]:
          1.0 → no occultation (observer not behind blocker along Sun line,
                or blocker presents zero silhouette)
          0.0 → full umbra of the blocker's sail
         (0,1) → penumbra
    """
    # Blocker silhouette perpendicular to Sun direction.  Edge-on blocker
    # (|n·(-sun_dir)| = 0) casts no shadow; back-face blocker (dot > 0)
    # also contributes nothing because its illuminated face is pointed
    # away from the observer.
    sun_dir = sun_pos_eci - blocker_pos_eci
    r_sun_from_blocker = float(np.linalg.norm(sun_dir))
    if r_sun_from_blocker < 1.0:
        return 1.0
    sun_dir_unit = sun_dir / r_sun_from_blocker

    cos_theta_blocker = float(np.dot(blocker_normal_eci, sun_dir_unit))
    if cos_theta_blocker <= 0.0:
        return 1.0  # blocker back-face or edge-on → no silhouette

    silhouette_area = blocker_area_m2 * cos_theta_blocker
    if silhouette_area <= 0.0:
        return 1.0
    r_blocker = math.sqrt(silhouette_area / math.pi)

    # Reuse the standard shadow_fraction geometry with the blocker as the
    # occulting body.  shadow_fraction expects body-centred coordinates.
    sc_from_blocker = observer_pos_eci - blocker_pos_eci
    sun_from_blocker = sun_pos_eci - blocker_pos_eci
    return shadow_fraction(sc_from_blocker, sun_from_blocker, r_blocker)


# ── Reflected-light boost (close-range inter-sail SRP) ────────────────────────


def reflected_flux_ratio(
    observer_pos_eci: NDArray[np.float64],
    sun_pos_eci: NDArray[np.float64],
    reflector_pos_eci: NDArray[np.float64],
    reflector_area_m2: float,
    reflector_normal_eci: NDArray[np.float64],
    reflector_illumination: float,
    *,
    specular_fraction: float,
    lobe_half_angle_rad: float,
) -> tuple[float, NDArray[np.float64]]:
    """
    Ratio of specularly-reflected flux at the observer to the direct solar
    flux at the observer's heliocentric distance, plus the unit direction
    the reflected light travels (from reflector toward observer).

    Counterpart to ``sail_occultation_fraction``.  Where that function
    *removes* direct solar flux when a sail lies between observer and Sun,
    this function *adds* a secondary beam contribution when a neighbouring
    sail reflects sunlight toward the observer.

    Physical model
    ──────────────
    The reflector is treated as a finite-lobe specular mirror.  For a
    perfectly flat sail the reflected beam would be a collimated delta
    function, but real sails have membrane billow, wrinkles, and boom-
    induced warp that spread the reflection into a narrow cone with
    Gaussian angular profile of 1-σ half-width ``lobe_half_angle_rad``.

    The peak reflected flux at an on-axis observer at distance d is::

        F_peak = ρ_s · F_sun_at_reflector · A_r · cos_i · illum_r
                 / (π · σ² · d²)

    where ρ_s is the specular reflectance (REFLECTIVITY · specular_fraction),
    cos_i is the reflector's solar-incidence cosine, illum_r is the
    reflector's own direct-solar illumination fraction (eclipse gating —
    a reflector in Earth's shadow cannot reflect), and π·σ² is the small-
    angle approximation to the Gaussian lobe solid angle.

    Off-axis, the flux is attenuated by exp(-δ²/(2σ²)) where δ is the
    angular offset from the specular beam axis.

    The returned *ratio* is F_peak_at_observer / F_sun_at_observer — a
    dimensionless multiplier the caller applies to the direct solar flux
    in the sail-force computation.  Keeping it dimensionless avoids an
    extra flux argument in the force method and composes cleanly with the
    observer's own position-dependent solar flux.

    Early-exit conditions (return (0.0, zero-vector)):
      • reflector self-illumination is zero (reflector in eclipse)
      • reflector's illuminated face is not Sun-facing (back-face or edge-on)
      • observer and reflector coincide (numerical safety)
      • reflected beam points away from observer (cos_align ≤ 0)
      • observer's Gaussian falloff δ/σ is large (> 5σ — below noise floor)

    Frame note
    ──────────
    Inputs are ECI (any origin — only differences are used).  The returned
    direction vector is ECI.

    Args:
        observer_pos_eci:       position of observer [m]
        sun_pos_eci:            position of Sun [m]
        reflector_pos_eci:      position of reflector [m]
        reflector_area_m2:      reflector's sail area [m²]
        reflector_normal_eci:   reflector's sail normal (unit, ECI), pointing
                                toward its illuminated face
        reflector_illumination: reflector's own direct-solar illumination
                                factor in [0, 1] from body-shadow gating
        specular_fraction:      fraction of total reflectance that is specular
                                (ρ_s = REFLECTIVITY · specular_fraction)
        lobe_half_angle_rad:    1-σ Gaussian half-width of the specular lobe

    Returns:
        (flux_ratio, arrival_direction_unit)
          flux_ratio: F_reflected_at_observer / F_sun_at_observer  (>= 0)
          arrival_direction_unit: unit vector from reflector toward observer,
                                  i.e. the direction the reflected photons
                                  travel (ECI).  Caller uses this as the
                                  effective "sun direction" for the
                                  sail-force computation on the observer.
                                  Zero vector on any early-exit.
    """
    # Reflector in full shadow → no reflection.
    if reflector_illumination <= 0.0:
        return 0.0, np.zeros(3)

    # Sun direction at the reflector (unit vector pointing TOWARD Sun from
    # the reflector).  Edge cases: co-location, degenerate vectors.
    sun_from_refl = sun_pos_eci - reflector_pos_eci
    r_sun = float(np.linalg.norm(sun_from_refl))
    if r_sun < 1.0:
        return 0.0, np.zeros(3)
    s_hat = sun_from_refl / r_sun

    # Reflector's solar-incidence cosine.  Normal points toward illuminated
    # face; for a Sun-facing reflector, n_r · s_hat > 0.
    cos_i = float(np.dot(reflector_normal_eci, s_hat))
    if cos_i <= 0.0:
        return 0.0, np.zeros(3)  # back-face or edge-on — no specular reflection

    # Observer line-of-sight from reflector.
    obs_from_refl = observer_pos_eci - reflector_pos_eci
    d = float(np.linalg.norm(obs_from_refl))
    if d < 1.0:
        return 0.0, np.zeros(3)  # numerical safety — co-located sails
    u_hat = obs_from_refl / d

    # Specular-beam axis: reflection of the incoming photon direction (-s_hat)
    # about the reflector normal.  Closed form: d_refl = -s_hat + 2·cos_i·n_r.
    d_refl = -s_hat + 2.0 * cos_i * reflector_normal_eci

    # Alignment of the observer's line-of-sight with the beam axis.
    cos_align = float(np.dot(u_hat, d_refl))
    if cos_align <= 0.0:
        return 0.0, np.zeros(3)  # reflected beam points away from observer

    # Gaussian angular falloff.  For small δ, use δ² ≈ 2·(1 − cos_align)
    # to avoid an acos() call (and the numerical noise at cos_align ≈ 1).
    # This approximation is < 0.3% error out to δ = 15° (3σ for σ=5°).
    delta_sq = max(0.0, 2.0 * (1.0 - cos_align))
    sigma_sq = lobe_half_angle_rad * lobe_half_angle_rad
    # Cut off at 5σ: exp(-12.5) ≈ 3.7e-6 — contribution is numerically zero.
    if delta_sq > 25.0 * sigma_sq:
        return 0.0, np.zeros(3)
    g_lobe = math.exp(-0.5 * delta_sq / sigma_sq)

    # Specular reflectance.
    rho_s = SailConst.REFLECTIVITY * specular_fraction

    # Flux ratio at observer vs the direct solar flux at its own position.
    # Derivation: the reflector sees flux F_r = F_sun(d_refl) = SOLAR_FLUX_1AU·(AU/r_sun)².
    # Reflected power along specular axis: P_s = ρ_s · F_r · A_r · cos_i · illum_r.
    # Gaussian lobe solid angle: Ω ≈ π·σ²  (small-angle).
    # Peak intensity: I_peak = P_s / Ω.
    # Flux at observer on-axis: F_obs = I_peak / d²  (inverse-square from reflector).
    # Direct solar flux at observer's location:
    #     F_sun_at_obs = SOLAR_FLUX_1AU · (AU / |observer - sun|)².
    # Ratio = F_obs / F_sun_at_obs.  The two SOLAR_FLUX_1AU·(AU)² factors
    # cancel, leaving (|observer - sun| / r_sun)² in the numerator.
    r_obs_from_sun = float(np.linalg.norm(observer_pos_eci - sun_pos_eci))
    if r_obs_from_sun < 1.0:
        return 0.0, np.zeros(3)
    helio_ratio_sq = (r_obs_from_sun / r_sun) ** 2

    ratio = (
        rho_s
        * reflector_area_m2
        * cos_i
        * reflector_illumination
        * g_lobe
        * helio_ratio_sq
        / (math.pi * sigma_sq * d * d)
    )

    return ratio, u_hat


# ── Solar array power ─────────────────────────────────────────────────────────


@dataclass
class SolarArrayConfig:
    """
    Configuration for the spacecraft solar array.

    Defaults match a typical 12U CubeSat with body-mounted GaAs cells
    augmented by two deployable panels.

    Attitude coupling
    ──────────────────
    On a sailcraft the solar array is body-mounted.  When the sail points
    toward the Sun (optimal for SRP), the side panels face perpendicular to
    it.  ``cos_theta`` must therefore be derived from the *actual* sail
    attitude; the PowerBudgetModel.step() caller supplies this via the
    ``sail_normal_eci`` argument.  ``cos_theta = 1`` (full Sun-tracking) is
    only used as a fallback when attitude is unavailable.

    Battery C-rate
    ───────────────
    ``c_rate_peak`` is the peak discharge rate in multiples of the 1-hour
    (1C) capacity.  A standard LiPo 12U pack is rated at 2C continuous,
    giving 40 W peak for a 20 Wh battery.  This caps the instantaneous
    power the battery can contribute to the thruster, preventing the
    previous behaviour where the entire stored energy was treated as
    instantaneously available.
    """

    area_m2: float = 0.12  # Total array area [m²]
    # 12U body: ~0.04 m², 2 panels: ~0.08 m²
    efficiency: float = 0.28  # Triple-junction GaAs efficiency
    degradation_rate: float = 0.005  # Annual efficiency loss (radiation)
    c_rate_peak: float = 2.0  # Peak discharge rate [1/h]  (2C = 40 W for 20 Wh)
    panel_body_offset_q: NDArray[np.float64] = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    decouple_attitude: bool = False  # When True, always use cos(θ)=1 (gimballed
    # assumption). Use for scenarios where sail
    # attitude has no power consequence (e.g.
    # pure orbit-raise with negligible SRP).


def _rotate_vector_by_quaternion(
    q: NDArray[np.float64],
    vec: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Rotate *vec* by scalar-first quaternion *q*."""
    return quat_to_dcm(np.asarray(q, dtype=np.float64)) @ np.asarray(vec, dtype=np.float64)


# ── Power budget model ────────────────────────────────────────────────────────


class PowerBudgetModel:
    """
    Spacecraft electrical power budget with eclipse-aware solar generation.

    Tracks battery state-of-charge (SoC) and computes available power
    for the thruster at each timestep.

    Power architecture
    ───────────────────
    P_solar  = flux × area × η × cos(θ_array)
    P_net    = P_solar − P_housekeeping
    Battery charges/discharges at P_net (with round-trip efficiency).
    P_thruster = P_solar_surplus + P_battery_burst

    P_battery_burst is capped by the battery C-rate: a 2C 20 Wh pack can
    deliver at most 40 W continuously.  This prevents the battery from
    acting as an instantaneous infinite-power source.

    Attitude coupling
    ───────────────────
    Pass ``sail_normal_eci`` to step() so that cos(θ_array) is computed
    from the real sail attitude.  On this sailcraft the solar panels are
    body-mounted: when the sail faces the Sun, the panels face away.
    cos(θ) = dot(sail_normal_eci, toward_sun_eci) — same angle the sail
    force model uses. Orientations that align well for SRP also maximise
    power, while off-Sun attitudes pay a power penalty.

    PHASE 1 FIDELITY LIMIT — power geometry
    ────────────────────────────────────────
    The model assumes that the solar array normal and the sail normal share
    a fixed, close alignment: cos(θ_array) is derived directly from the
    sail-to-Sun angle.  Real bus/panel geometry may be less forgiving:

      - Deployable panels on a separate boom can point independently of
        the sail — decoupling the SRP-optimal attitude from the power-
        optimal attitude.
      - Fixed body-mounted arrays (common on 6U/12U CubeSats) may have
        a significant constant offset angle between the array normal and
        the sail boom axis.
      - Multi-face body arrays (e.g., six-face CubeSat) aggregate the
        per-face cos(θ) contributions, which can maintain non-zero power
        generation across a wider range of orientations than modelled here.

    Consequence: the current model slightly overestimates eclipse-exit
    power recovery when the spacecraft is held in an SRP-favourable attitude
    that is not simultaneously panel-favourable.

    Upgrade path: add a ``panel_body_offset_q`` quaternion to
    SolarArrayConfig describing the fixed rotation between the sail normal
    axis and the panel normal axis.  step() then evaluates
    cos(θ_array) = dot(R(q) @ sail_normal, toward_sun) instead of the
    raw sail-normal dot product.  No environment interface changes needed.

    Args:
        array_cfg:           Solar array configuration
        battery_capacity_wh: Battery energy capacity [Wh]
        battery_soc_init:    Initial battery state of charge [0–1]
        P_housekeeping:      Always-on power draw (avionics, comms, ADCS) [W]
        charge_efficiency:   Battery charge/discharge round-trip efficiency
    """

    def __init__(
        self,
        array_cfg: SolarArrayConfig | None = None,
        battery_capacity_wh: float = 20.0,  # ~typical 12U
        battery_soc_init: float = 1.0,
        P_housekeeping: float = 8.0,  # [W] — avionics + comms + ADCS
        charge_efficiency: float = 0.92,
    ):
        self.array = array_cfg or SolarArrayConfig()
        self.batt_cap = battery_capacity_wh * 3600.0  # convert to Joules
        self.batt_soc = battery_soc_init
        self.P_hk = P_housekeeping
        self.eta_batt = charge_efficiency

        # C-rate cap: maximum instantaneous battery discharge power [W].
        # c_rate_peak=2.0 and 20 Wh → 40 W peak.  Prevents the battery from
        # appearing as a ~960 W instantaneous source when batt_avail_j/dt_s
        # would otherwise dominate.
        self.c_rate_peak_w = self.array.c_rate_peak * battery_capacity_wh

        # Diagnostics (updated each call to step())
        self.in_eclipse = False
        self.P_solar_w = 0.0
        self.P_available_w = 0.0
        self.P_thruster_w = 0.0
        self.batt_energy_j = battery_soc_init * self.batt_cap

        # Previous spacecraft position for eclipse-boundary bisection.
        # None on the very first step → falls back to point evaluation.
        self._prev_spacecraft_pos: NDArray[np.float64] | None = None

    # ── Main update ───────────────────────────────────────────────────────────

    def step(
        self,
        spacecraft_pos: NDArray[np.float64],
        sun_pos: NDArray[np.float64],
        helio_pos: NDArray[np.float64],
        dt_s: float,
        mission_age_days: float = 0.0,
        sail_normal_eci: NDArray[np.float64] | None = None,
        moon_pos: NDArray[np.float64] | None = None,
        propulsion_power_request_w: float | None = None,
    ) -> float:
        """
        Advance the power model by dt_s seconds.

        Args:
            spacecraft_pos:  spacecraft position in ECI [m]
            sun_pos:         Sun position in ECI [m]
            helio_pos:       spacecraft position relative to Sun in ECI [m]
                             (= spacecraft_pos - sun_pos).  Used for
                             inverse-square solar flux and toward-Sun direction.
            dt_s:            Timestep [s]
            mission_age_days: For degradation modelling
            sail_normal_eci: Sail face unit normal in ECI.  When provided,
                             array cos(θ) = dot(sail_normal_eci, toward_sun)
                             so that attitude and power are coupled correctly.
                             When None, falls back to cos(θ) = 1 (gimballed
                             assumption — optimistic but backward-compatible).
            moon_pos:        Moon position in ECI [m], or None to model only
                             Earth's shadow.  When provided, the illumination
                             fraction is the product of Earth and Moon shadow
                             fractions — same quantity the sail force consumes.
            propulsion_power_request_w:
                             Requested propulsion bus power draw [W] for this
                             interval. When omitted, the method acts as a
                             pure availability query and does not debit any
                             propulsion draw from the battery.

        Returns:
            Thruster bus power [W]. With ``propulsion_power_request_w=None``,
            this is the maximum propulsion power available this step. When a
            request is supplied, the return value is the power actually
            deliverable after solar generation, housekeeping, battery reserve,
            and C-rate limits are applied.
        """
        # ── Eclipse / penumbra ────────────────────────────────────────────
        # in_eclipse: current-state flag; always evaluated at the endpoint
        # position so downstream consumers and the power model know where the spacecraft is.
        # When moon_pos is supplied, lunar occultation also counts as eclipse.
        illum_point = body_shadow_illumination(spacecraft_pos, sun_pos, moon_pos)
        self.in_eclipse = illum_point == 0.0

        # illum: time-averaged illumination over the step, used for power.
        # Bisection refinement is used when a previous position is available
        # so that boundary crossings within a sub-step contribute the correct
        # fraction of illuminated time rather than the step-end point value.
        # Only Earth's shadow is bisected here — the Moon boundary-crossing
        # rate is far lower and endpoint sampling is adequate at the 60-s
        # env-step cadence.  The Moon-shadow contribution is folded into the
        # endpoint value via the multiplicative ratio below.
        if self._prev_spacecraft_pos is not None:
            illum_earth_avg = bisect_eclipse_boundary(
                self._prev_spacecraft_pos, spacecraft_pos, sun_pos
            )
            if moon_pos is not None:
                # Scale the time-averaged Earth illumination by the current
                # Earth→combined ratio to fold Moon occultation into the
                # average without bisecting on the Moon boundary.
                f_earth_endpoint = shadow_fraction(spacecraft_pos, sun_pos)
                if f_earth_endpoint > 0.0:
                    illum = illum_earth_avg * (illum_point / f_earth_endpoint)
                else:
                    illum = illum_point  # in umbra → no averaging benefit
            else:
                illum = illum_earth_avg
        else:
            illum = illum_point
        self._prev_spacecraft_pos = spacecraft_pos.copy()

        # ── Solar generation ──────────────────────────────────────────────
        if illum == 0.0:
            self.P_solar_w = 0.0
        else:
            # Solar flux at current distance from Sun (inverse-square)
            r_helio = float(np.linalg.norm(helio_pos))
            flux = SOLAR_CONST * (AU_M / r_helio) ** 2

            # Radiation degradation
            years = mission_age_days / 365.25
            eta_deg = self.array.efficiency * (1.0 - self.array.degradation_rate) ** years

            # cos(θ): angle between array normal and toward-Sun direction.
            # When sail_normal_eci is available, rotate it by the fixed panel
            # offset to get the true array normal before evaluating cos(theta).
            if self.array.decouple_attitude or sail_normal_eci is None:
                # Ideal Sun-tracking (cos θ = 1): gimballed panels or unit tests.
                cos_theta = 1.0
            else:
                toward_sun = -helio_pos / (r_helio + 1e-30)  # unit vec toward Sun
                array_normal_eci = _rotate_vector_by_quaternion(
                    self.array.panel_body_offset_q,
                    sail_normal_eci,
                )
                cos_theta = float(np.dot(array_normal_eci, toward_sun))
                cos_theta = max(0.0, cos_theta)  # back-face → zero

            self.P_solar_w = eta_deg * flux * self.array.area_m2 * cos_theta * illum

        # ── Thruster available power ───────────────────────────────────────
        # Housekeeping is served first. Any remaining solar power can drive
        # propulsion directly; only the residual propulsion draw should come
        # from the battery. This avoids the previous "free energy" path where
        # propulsion could use the battery burst budget without actually
        # debiting stored energy.
        solar_surplus_w = max(0.0, self.P_solar_w - self.P_hk)
        hk_deficit_w = max(0.0, self.P_hk - self.P_solar_w)

        min_batt_j = 0.20 * self.batt_cap  # 20% depth-of-discharge limit
        batt_avail_j = max(0.0, self.batt_energy_j - min_batt_j)
        batt_output_cap_w = min(
            batt_avail_j / max(dt_s, 1.0),  # energy-limited rate
            self.c_rate_peak_w,  # C-rate cap (40 W at 2C/20Wh)
        )
        batt_for_thruster_w = max(0.0, batt_output_cap_w - hk_deficit_w)
        max_thruster_power_w = solar_surplus_w + batt_for_thruster_w

        self.P_available_w = max_thruster_power_w

        if propulsion_power_request_w is None:
            delivered_thruster_w = 0.0
            self.P_thruster_w = max_thruster_power_w
        else:
            requested_thruster_w = max(0.0, float(propulsion_power_request_w))
            delivered_thruster_w = min(requested_thruster_w, max_thruster_power_w)
            self.P_thruster_w = delivered_thruster_w

        # ── Battery update ────────────────────────────────────────────────
        # Battery output first covers any solar shortfall in housekeeping,
        # then any propulsion draw not met by solar surplus. Unused solar
        # surplus charges the battery.
        batt_to_thruster_w = max(0.0, delivered_thruster_w - solar_surplus_w)
        batt_output_w = hk_deficit_w + batt_to_thruster_w
        solar_to_charge_w = max(0.0, solar_surplus_w - delivered_thruster_w)

        if batt_output_w > 0.0:
            discharge_j = batt_output_w * dt_s / self.eta_batt
            self.batt_energy_j = max(0.0, self.batt_energy_j - discharge_j)

        if solar_to_charge_w > 0.0:
            charge_j = solar_to_charge_w * dt_s * self.eta_batt
            self.batt_energy_j = min(self.batt_cap, self.batt_energy_j + charge_j)

        self.batt_soc = self.batt_energy_j / self.batt_cap

        return self.P_thruster_w

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @property
    def battery_pct(self) -> float:
        return self.batt_soc * 100.0

    def status_line(self) -> str:
        eclipse_str = "ECLIPSE" if self.in_eclipse else "SUN    "
        return (
            f"{eclipse_str} | "
            f"P_solar={self.P_solar_w:6.1f}W  "
            f"P_thr={self.P_thruster_w:6.1f}W  "
            f"batt={self.battery_pct:5.1f}%"
        )

    @property
    def has_previous_position(self) -> bool:
        return self._prev_spacecraft_pos is not None

    def reset(self, soc: float = 1.0) -> None:
        self.batt_soc = soc
        self.batt_energy_j = soc * self.batt_cap
        self.in_eclipse = False
        self._prev_spacecraft_pos = None
