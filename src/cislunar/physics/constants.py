"""
Physical and astronomical constants for cislunar-sim.
All values in SI units unless noted.
"""

import math

# ── Fundamental ──────────────────────────────────────────────────────────────
G = 6.674_30e-11  # Gravitational constant         [m³ kg⁻¹ s⁻²]
C = 2.997_924_58e8  # Speed of light                 [m s⁻¹]
AU = 1.495_978_707e11  # Astronomical unit              [m]
SOLAR_FLUX_1AU = 1_361.0  # Solar irradiance at 1 AU       [W m⁻²]


# ── Solar System Bodies ──────────────────────────────────────────────────────
class Body:
    """Gravitational parameters (μ = GM) and radii for common bodies."""

    # μ = GM values [m³ s⁻²]  — more precise than G*M separately
    MU_SUN = 1.327_124_400_18e20
    MU_EARTH = 3.986_004_418e14
    MU_MOON = 4.902_800_118e12
    MU_MARS = 4.282_837_00e13

    # Mean radii [m]
    R_SUN = 6.957e8
    R_EARTH = 6.371e6
    R_MOON = 1.737_400e6  # GRAIL-refined mean radius (Lemoine et al. 2014)
    R_MARS = 3.390e6

    # Approximate J2 oblateness terms (dimensionless) for perturbations
    J2_EARTH = 1.082_630e-3

    # Lunar gravity field zonal harmonics — GRAIL primary mission
    # Source: Lemoine et al. (2014), GRAIL SGM150J, degrees 2 and 3.
    # J2_MOON dominates all other perturbations at LLO (8× larger than the
    # 8-mascon signal at 200 km altitude).  J3 adds the north-south asymmetry
    # (pear-shape) that shifts perilune altitude over multi-orbit timescales.
    # A full spherical-harmonics model (LP150Q / SGM150J, degree/order 150)
    # is available for higher-fidelity studies; for many cislunar flybys,
    # J2+J3+mascons captures the dominant structure.
    J2_MOON = 2.027_0e-4  # oblateness (dominant zonal term)
    J3_MOON = 8.473_0e-5  # pear-shape (north-south asymmetry)

    J2_MARS = 1.955_45e-3

    # Bond albedo (fraction of incident sunlight diffusely reflected).
    # Used by the Lambertian-disc albedo SRP model.
    # Earth: Stephens et al. (2015) CERES composite, 0.29–0.31 range.
    # Moon: Kieffer & Stone (2005) photometric model, mean 0.12.
    ALBEDO_EARTH = 0.30
    ALBEDO_MOON = 0.12


# ── Sail / Propulsion ────────────────────────────────────────────────────────
class Sail:
    """Default sail material properties (Mylar-based, similar to LightSail 2)."""

    REFLECTIVITY = 0.88  # ρ — specular + diffuse reflectance
    ABSORPTIVITY = 0.06  # α — fraction of photons absorbed
    EMISSIVITY_FRONT = 0.05  # ε_f — thermal emission, front face
    EMISSIVITY_BACK = 0.55  # ε_b — thermal emission, back face
    AREA_DENSITY = 0.007  # kg m⁻² — sail film areal density

    # Dimensionless ratio of boom/structural shadow area to total membrane area.
    # For a 4-boom TRAC-boom square sail at the 30-50 m² LightSail 2 scale this
    # is ~2%.  At oblique incidence the structural shadow on the membrane grows
    # as 1/cos θ, so this parameter sets only the normal-incidence value.
    # Override to 0.0 for booms-free sails (IKAROS-style spinning heliogyro).
    BOOM_SHADOW_FRACTION = 0.02

    # ── Reflected-light boost (close-range inter-sail SRP) ──────────────────
    # Fraction of REFLECTIVITY that is specular (mirror-like) rather than
    # diffuse (Lambertian).  Aluminised Mylar is overwhelmingly specular;
    # ~0.94 of the 0.88 total reflectance is specular, giving ρ_s ≈ 0.83.
    # Only the specular fraction contributes a directional beam that couples
    # significantly onto a nearby observer sail — the diffuse hemispherical
    # component falls off as 1/(π·d²) and is ~10⁻⁴ of direct solar at 100 m,
    # below any detectable threshold.
    SPECULAR_FRACTION = 0.94

    # Angular half-width (1-σ) of the specular lobe.  Real solar sails are
    # not perfect mirrors — membrane billow, wrinkles, and boom-induced
    # warping spread the reflected beam into a narrow cone.  σ = 5° is a
    # middle-ground value bracketing the LightSail 2 (~7°, TRAC-boom
    # billow) and IKAROS (~2°, spin-stabilised) ranges.  Smaller σ makes
    # the on-axis boost larger but the angular window rarer; missions must
    # actively maneuver to catch the beam.  At σ = 5°, 32 m² reflector,
    # 100 m separation, on-axis incidence, the peak reflected flux ratio
    # vs direct solar is ~11% — "small but detectable" and only when the
    # reflector is tilted away from the Sun (pure sun-normal attitudes
    # reflect straight back toward the Sun, not toward trailing craft).
    SPECULAR_LOBE_HALF_ANGLE_RAD = math.radians(5.0)


class Thruster:
    """Hall-effect ion thruster parameters (iodine-fed)."""

    ISP = 2_300.0  # Specific impulse              [s]
    MAX_THRUST = 0.015  # Peak thrust force             [N]
    MAX_POWER_W = 260.0  # Wall power at full throttle   [W]
    MIN_THROTTLE = 0.05  # Minimum stable throttle fraction
    POWER_PER_NEWTON = 16_000.0  # Watts per Newton at full thrust [W N⁻¹]
    WARMUP_DELAY_S = 60.0  # Hall-thruster ignition delay  [s]
    # Physical range: 30–120 s; 60 s is mid-range.
    # Rapid on/off pulsing delivers far less total impulse
    # than a sustained burn because each re-ignition incurs
    # a full warmup cycle.

    # Plume divergence exponent for the cosine-lobe impingement model.
    # The angular distribution of exhaust momentum flux follows
    #     dF/dΩ = ((n+1)/2π) · F_thrust · cos^n(θ)
    # where θ is the angle between the plume axis and the line to the target.
    # Higher n → narrower beam.  For Hall-effect thrusters, measured plume
    # half-angles (90% of mass flow) are typically 30–45°, which corresponds
    # to n ≈ 2.5–4.  n = 3.0 is a mid-range value that matches published
    # XR-5 / BHT-600 plume characterizations.
    PLUME_DIVERGENCE_EXPONENT = 3.0


class Spacecraft:
    """Physical properties of a 12U CubeSat bus."""

    DRY_MASS_KG = 8.8  # Dry mass for the reference spacecraft bus [kg]
    INITIAL_PROPELLANT_KG = 3.0  # Reference iodine propellant load [kg]
    SOLAR_PANEL_AREA_M2 = 0.720  # Body + deployable wings [m²]
    # Panel area drives PowerBudgetModel; see eclipse.SolarArrayConfig.


class RCS:
    """Cold gas reaction control system.

    Two clusters of 4 nozzles each:
      Nose cluster  — ±pitch and ±yaw authority (4 nozzles, cruciform)
      Mid-body cluster — ±roll authority (4 nozzles around roll axis)

    Propellant: R-134a (tetrafluoroethane), stored as saturated liquid.
    """

    F_PER_NOZZLE_N = 0.020  # Thrust per nozzle          [N]
    N_NOZZLES = 8  # Total nozzles (4 nose + 4 mid-body)
    ISP = 65.0  # Specific impulse           [s]
    INITIAL_PROPELLANT_KG = 0.15  # R-134a propellant load     [kg]
    NOSE_ARM_M = 0.25  # Moment arm — nose cluster  [m from CoM]
    BODY_ARM_M = 0.12  # Moment arm — mid-body      [m from CoM]
