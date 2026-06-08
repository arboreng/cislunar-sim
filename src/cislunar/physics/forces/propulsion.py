"""
Propulsion force models — solar sail and ion/Hall thruster.

Solar sail model is a flat-plate optical force model validated
against IKAROS and LightSail 2 telemetry coefficients.

Frame note
──────────
The master simulation frame is Earth-Centred Inertial (ECI).
Solar radiation pressure requires the spacecraft's distance from the
Sun and the Sun-to-spacecraft direction.  Both are obtained from
``sc_pos_from_sun = pos_eci - sun_pos_eci``, an ECI-frame vector that
points from the Sun to the spacecraft.  Functions in this module
accept that vector under the name ``sc_pos_from_sun``; it is NOT a
heliocentric coordinate in the barycentric sense.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..constants import AU, SOLAR_FLUX_1AU, C
from ..constants import Sail as SailConst
from ..constants import Thruster as ThrConst

# ── Utility ───────────────────────────────────────────────────────────────────


def solar_flux_at_position(sc_pos_from_sun: NDArray[np.float64]) -> float:
    """
    Solar flux [W m⁻²] at the spacecraft's current distance from the Sun.

    Args:
        sc_pos_from_sun: spacecraft position relative to the Sun, expressed
                         in ECI coordinates [m].  Derived as
                         ``pos_eci - sun_pos_eci`` in Spacecraft._derivatives.
    """
    r_m = float(np.linalg.norm(sc_pos_from_sun))
    return SOLAR_FLUX_1AU * (AU / r_m) ** 2


def solar_direction(
    sc_pos_from_sun: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Unit vector pointing FROM the Sun TO the spacecraft (away from Sun),
    in ECI coordinates.

    Args:
        sc_pos_from_sun: spacecraft position relative to the Sun, expressed
                         in ECI coordinates [m].
    """
    r = np.linalg.norm(sc_pos_from_sun)
    return sc_pos_from_sun / (r + 1e-30)


# ── Solar Sail ────────────────────────────────────────────────────────────────


class SolarSailModel:
    """
    Flat-plate optical solar sail force model.

    Accounts for radiation pressure, specular reflection, diffuse reflection,
    and thermal emission asymmetry — the standard four-term optical model used
    in IKAROS and LightSail 2 analysis.

    Reference: McInnes, C.R. (1999) "Solar Sailing: Technology, Dynamics
               and Mission Applications", Springer.

    Args:
        area_m2:              Sail surface area [m²]
        reflectivity:         Specular reflectance coefficient ρ  (0–1)
        absorptivity:         Absorptance coefficient α  (0–1)
        emissivity_f:         Front face emissivity ε_f
        emissivity_b:         Back face emissivity ε_b
        boom_shadow_fraction: Ratio of structural (boom) shadow area to
                              membrane area at normal incidence.  At oblique
                              angles the structural shadow on the membrane
                              grows as 1/cos θ.  Default 0.02 ≈ 4-boom
                              TRAC sails; set to 0.0 for spinning heliogyros
                              (IKAROS) and other rigid-boom-free designs.
    """

    def __init__(
        self,
        area_m2: float,
        reflectivity: float = SailConst.REFLECTIVITY,
        absorptivity: float = SailConst.ABSORPTIVITY,
        emissivity_f: float = SailConst.EMISSIVITY_FRONT,
        emissivity_b: float = SailConst.EMISSIVITY_BACK,
        boom_shadow_fraction: float = SailConst.BOOM_SHADOW_FRACTION,
    ):
        self.area_m2 = area_m2
        self.rho = reflectivity
        self.alpha = absorptivity
        self.eps_f = emissivity_f
        self.eps_b = emissivity_b
        self.boom_shadow_fraction = boom_shadow_fraction

    def force(
        self,
        sc_pos_from_sun: NDArray[np.float64],
        sail_normal: NDArray[np.float64],
        total_mass_kg: float,
        *,
        illumination: float = 1.0,
    ) -> NDArray[np.float64]:
        """
        Radiation pressure force on the sail.

        Args:
            sc_pos_from_sun: spacecraft position relative to the Sun, expressed
                             in ECI coordinates [m].  Derived as
                             ``pos_eci - sun_pos_eci`` in Spacecraft._derivatives.
            sail_normal:     unit normal to the sail face, in ECI
            total_mass_kg:   current total spacecraft mass [kg]
            illumination:    solar flux attenuation factor in [0, 1] from
                             external occultation (Earth/Moon eclipse,
                             inter-spacecraft shadowing).  1.0 = full sunlight,
                             0.0 = full umbra.  Composes multiplicatively with
                             the internal self-shadow factor.

        Returns:
            force vector [N]  (not acceleration — divide by mass externally)
        """
        # Early exit for full umbra — avoids all downstream work and
        # guarantees a bit-exact zero for tests.
        if illumination <= 0.0:
            return np.zeros(3)

        # Sun-to-spacecraft direction (in ECI)
        sun_dir = solar_direction(sc_pos_from_sun)

        # Cosine of incidence angle θ between sail normal and sun direction
        # Clamp to [0, 1] — sail only produces force when Sun-facing
        cos_theta = float(np.dot(sail_normal, -sun_dir))  # -sun_dir = toward Sun
        cos_theta = max(0.0, cos_theta)  # back-face produces no sail force

        # Self-shadow factor: booms/structural elements in the sail plane
        # cast a shadow on the membrane whose projected area grows as
        # 1/cos θ with obliquity.  f_self ∈ [0, 1]; at θ=0 equals
        # 1 - boom_shadow_fraction; goes to 0 at grazing incidence where
        # the structural shadow sweeps the entire membrane.
        if self.boom_shadow_fraction > 0.0 and cos_theta > 1e-6:
            f_self = max(0.0, 1.0 - self.boom_shadow_fraction / cos_theta)
        else:
            f_self = 1.0

        # Effective illumination combines all attenuation sources.
        illum_eff = illumination * f_self
        if illum_eff <= 0.0:
            return np.zeros(3)

        # Solar radiation pressure [Pa] at current distance from Sun,
        # attenuated by the effective illumination.
        flux = solar_flux_at_position(sc_pos_from_sun) * illum_eff
        P = flux / C  # Radiation pressure [N m⁻²]

        # ── Four-term optical force model (McInnes 1999, §2.3) ───────────────
        # Convention: sail_normal points toward the illuminated (sun-facing) side.
        # Net force is along -sail_normal (away from Sun) for normal incidence.
        #
        # Term 1 — absorbed photon momentum (along sun_dir, normal + tangential)
        # Term 2 — specularly reflected photons (along -sail_normal)
        # Term 3 — diffusely re-emitted photons (Lambertian, along -sail_normal)
        # Term 4 — asymmetric thermal emission (along -sail_normal)
        #           F_thermal = P·A·cos θ · α·(ε_f − ε_b)/(ε_f + ε_b)
        #           Factor α comes from thermal-equilibrium: T⁴ ∝ α·flux.
        #           Omitting α overestimates this term by 1/α ≈ 16× for Mylar.

        coeff_n = (
            (1.0 + self.rho) * cos_theta
            + (2.0 / 3.0) * self.alpha * cos_theta
            + self.alpha * (self.eps_f - self.eps_b) / (self.eps_f + self.eps_b)
        )

        # Normal force: away from Sun (i.e. in the -sail_normal direction
        # when sail_normal points toward Sun)
        F_n = P * self.area_m2 * cos_theta * coeff_n * (-sail_normal)

        # Tangential force: absorbed photons at oblique incidence transfer
        # momentum along the sail plane with magnitude cos θ · sin θ.
        # `tang` is the projection of sun_dir onto the sail plane; its
        # magnitude equals sin θ, so multiplying by alpha·tang gives the
        # correct cos θ · sin θ · alpha scaling without a separate sin θ lookup.
        tang = sun_dir - np.dot(sun_dir, sail_normal) * sail_normal
        if np.linalg.norm(tang) > 1e-10:
            F_t = P * self.area_m2 * cos_theta * self.alpha * tang
        else:
            F_t = np.zeros(3)

        return F_n + F_t

    def acceleration(
        self,
        sc_pos_from_sun: NDArray[np.float64],
        sail_normal: NDArray[np.float64],
        total_mass_kg: float,
        *,
        illumination: float = 1.0,
    ) -> NDArray[np.float64]:
        """Convenience: force / mass.  See ``force()`` for illumination semantics."""
        return (
            self.force(
                sc_pos_from_sun,
                sail_normal,
                total_mass_kg,
                illumination=illumination,
            )
            / total_mass_kg
        )

    def reflected_force(
        self,
        sc_pos_from_sun: NDArray[np.float64],
        sail_normal: NDArray[np.float64],
        reflected_flux_ratio: float,
        reflected_arrival_dir: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Force on this sail from a specularly-reflected beam arriving from a
        neighbouring sail.

        Geometrically identical to ``force()`` — the same four-term flat-plate
        optical model — but the photon source is a directional beam from
        another sail instead of the Sun.  The caller supplies:

          • sc_pos_from_sun: observer's own position relative to the Sun.
            Used to compute the observer's direct solar flux, which is
            multiplied by ``reflected_flux_ratio`` to get the reflected
            flux at the observer.
          • reflected_flux_ratio: dimensionless ratio F_reflected / F_sun
            at the observer (from ``reflected_flux_ratio`` in eclipse.py).
          • reflected_arrival_dir: unit vector pointing FROM the reflector
            TOWARD the observer (ECI).  This is the direction the reflected
            photons are travelling — analogous to ``sun_dir`` in force().

        The observer's own body-shadow eclipse state is NOT applied here —
        if the observer is in Earth's shadow, the direct solar flux at that
        point is the *unobscured* 1-AU-scaled value, and the reflected
        flux is a fraction of *that*.  Physically: a reflector outside the
        shadow can still illuminate an observer inside it.  In practice
        both are usually sunlit together or eclipsed together (separation
        < 1 km vs shadow boundary > 6000 km) so this edge case is rare.

        Self-shadow (boom f_self) and thermal-emission/absorption terms are
        retained because the underlying optics are identical — the reflected
        beam behaves exactly like a weak directional Sun.

        Args:
            sc_pos_from_sun:       observer's position relative to the Sun [m]
            sail_normal:           observer's sail normal (unit, ECI)
            reflected_flux_ratio:  F_reflected / F_sun at observer  (>= 0)
            reflected_arrival_dir: unit direction the reflected photons
                                   travel (ECI, FROM reflector TO observer)

        Returns:
            force vector [N]  (add to baseline SRP force; do NOT replace)
        """
        if reflected_flux_ratio <= 0.0:
            return np.zeros(3)

        # Cosine of incidence between reflected beam and observer's sail normal.
        # The reflected beam plays the role of "photons from the Sun" — for the
        # observer to feel force on its illuminated face, its normal must point
        # back toward the reflector, i.e. opposite to the arrival direction.
        cos_theta = float(np.dot(sail_normal, -reflected_arrival_dir))
        if cos_theta <= 0.0:
            return np.zeros(3)

        # Self-shadow (booms): same obliquity growth as direct-Sun case.
        if self.boom_shadow_fraction > 0.0 and cos_theta > 1e-6:
            f_self = max(0.0, 1.0 - self.boom_shadow_fraction / cos_theta)
        else:
            f_self = 1.0
        if f_self <= 0.0:
            return np.zeros(3)

        # Reflected flux at observer = direct solar flux × ratio × self-shadow.
        flux = solar_flux_at_position(sc_pos_from_sun) * reflected_flux_ratio * f_self
        P = flux / C  # radiation pressure [N m⁻²]

        # Four-term optical model — identical coefficients to force(), but
        # the incoming ray is reflected_arrival_dir (photons from reflector).
        coeff_n = (
            (1.0 + self.rho) * cos_theta
            + (2.0 / 3.0) * self.alpha * cos_theta
            + self.alpha * (self.eps_f - self.eps_b) / (self.eps_f + self.eps_b)
        )
        F_n = P * self.area_m2 * cos_theta * coeff_n * (-sail_normal)

        # Tangential component: same cos θ · sin θ · alpha scaling as force().
        tang = reflected_arrival_dir - np.dot(reflected_arrival_dir, sail_normal) * sail_normal
        if np.linalg.norm(tang) > 1e-10:
            F_t = P * self.area_m2 * cos_theta * self.alpha * tang
        else:
            F_t = np.zeros(3)

        return F_n + F_t

    def reflected_acceleration(
        self,
        sc_pos_from_sun: NDArray[np.float64],
        sail_normal: NDArray[np.float64],
        reflected_flux_ratio: float,
        reflected_arrival_dir: NDArray[np.float64],
        total_mass_kg: float,
    ) -> NDArray[np.float64]:
        """Convenience: reflected_force / mass."""
        return (
            self.reflected_force(
                sc_pos_from_sun,
                sail_normal,
                reflected_flux_ratio,
                reflected_arrival_dir,
            )
            / total_mass_kg
        )

    def albedo_acceleration(
        self,
        sc_pos_eci: NDArray[np.float64],
        body_pos_eci: NDArray[np.float64],
        sun_pos_eci: NDArray[np.float64],
        sc_pos_from_sun: NDArray[np.float64],
        sail_normal: NDArray[np.float64],
        body_radius_m: float,
        body_albedo: float,
        total_mass_kg: float,
    ) -> NDArray[np.float64]:
        """
        Acceleration from Lambertian-disc planetary albedo SRP.

        Models the body as a flat Lambertian disc facing the spacecraft.
        Albedo flux at the spacecraft:

            F_albedo = a × F_sun_at_body × (R_body / r)² × max(0, cos φ)

        where a = Bond albedo, r = body–spacecraft distance, and φ = phase
        angle at the body between the Sun direction and the spacecraft direction.

        The reflected photons travel from the body toward the spacecraft and
        are handled by the same four-term flat-plate sail model used for
        direct SRP and inter-sail reflections.

        Typical magnitudes (32 m² sail):
          Earth at LEO (400 km):  ~1–3% of direct SRP
          Moon at LLO  (100 km):  ~0.5–1% of direct SRP
          Beyond GEO:             < 0.1% — negligible

        Args:
            sc_pos_eci:      spacecraft position in ECI [m]
            body_pos_eci:    body (Earth/Moon) centre in ECI [m]
                             (Earth = [0,0,0] in the standard ECI frame)
            sun_pos_eci:     Sun centre in ECI [m]
            sc_pos_from_sun: sc_pos_eci − sun_pos_eci (pre-computed by caller)
            sail_normal:     sail face unit normal (ECI)
            body_radius_m:   mean equatorial radius of the body [m]
            body_albedo:     Bond albedo (dimensionless, 0–1)
            total_mass_kg:   current spacecraft mass [kg]

        Returns:
            acceleration vector [m s⁻²]
        """
        body_to_sc = sc_pos_eci - body_pos_eci
        r = float(np.linalg.norm(body_to_sc))
        if r < body_radius_m:
            return np.zeros(3)

        # Unit direction from body toward spacecraft (albedo photon travel direction)
        albedo_dir = body_to_sc / r

        # Phase angle at body: angle between toward-Sun and toward-spacecraft
        body_from_sun = body_pos_eci - sun_pos_eci
        r_body_sun = float(np.linalg.norm(body_from_sun))
        toward_sun_at_body = -body_from_sun / (r_body_sun + 1e-30)
        cos_phase = float(np.dot(toward_sun_at_body, albedo_dir))
        if cos_phase <= 0.0:
            return np.zeros(3)  # spacecraft on night side — no albedo

        # Solar flux at the body (not at the spacecraft)
        F_sun_at_body = SOLAR_FLUX_1AU * (AU / (r_body_sun + 1e-30)) ** 2

        # Albedo flux at spacecraft: Lambertian disc formula
        F_albedo = body_albedo * F_sun_at_body * (body_radius_m / r) ** 2 * cos_phase

        # Express as a ratio of solar flux at the spacecraft for reuse of
        # reflected_force(), which internally scales by solar_flux_at_position()
        F_sun_at_sc = solar_flux_at_position(sc_pos_from_sun)
        if F_sun_at_sc <= 0.0:
            return np.zeros(3)
        ratio = F_albedo / F_sun_at_sc

        return self.reflected_acceleration(
            sc_pos_from_sun,
            sail_normal,
            ratio,
            albedo_dir,
            total_mass_kg,
        )

    def characteristic_acceleration(self, total_mass_kg: float) -> float:
        """
        a_c: acceleration at 1 AU with sail normal to Sun [mm s⁻²].
        Standard figure-of-merit for sail designs.
        """
        # sc_pos_from_sun = [AU, 0, 0]: spacecraft 1 AU directly "above" Sun
        sc_pos_from_sun_1au = np.array([AU, 0.0, 0.0])
        normal_toward_sun = np.array([-1.0, 0.0, 0.0])
        a = self.acceleration(sc_pos_from_sun_1au, normal_toward_sun, total_mass_kg)
        return float(np.linalg.norm(a)) * 1e3  # convert to mm s⁻²


# ── Ion / Hall Thruster ───────────────────────────────────────────────────────


class IonThrusterModel:
    """
    Simplified Hall-effect thruster model for maneuvering.

    Tsiolkovsky rocket equation governs propellant consumption.
    Power-limited: actual thrust is capped by the lower of two ceilings:

      1. ``max_power_w`` — hardware limit on thruster wall-plug power draw.
         Binds when the bus has surplus power but the thruster electronics cap
         out (for example, at the default 260 W rating).
      2. ``available_power_w`` — bus power the power model can deliver this step.
         Binds during eclipse or high-housekeeping periods.

    Hall thrusters require a plasma ignition and warm-up phase before reaching
    stable operating conditions.  ``warmup_time_s`` (default 60 s, physical range
    30–120 s) controls how long the effective throttle takes to ramp from zero to
    the commanded level.  Rapid on/off pulsing therefore delivers far less total
    impulse than a sustained burn.

    Args:
        isp:              Specific impulse [s]
        max_thrust_n:     Peak thrust at full power [N]
        max_power_w:      Hardware wall-plug power ceiling [W]
        power_per_newton: Power required per Newton [W N⁻¹]
        min_throttle:     Minimum stable throttle (fraction 0–1)
        warmup_time_s:    Ignition-to-steady-state delay [s]
    """

    G0 = 9.80665  # Standard gravity [m s⁻²]
    _SHUTDOWN_S = 5.0  # Plasma extinction time; re-ignition needs a full warmup cycle

    def __init__(
        self,
        isp: float = ThrConst.ISP,
        max_thrust_n: float = ThrConst.MAX_THRUST,
        max_power_w: float = ThrConst.MAX_POWER_W,
        power_per_newton: float = ThrConst.POWER_PER_NEWTON,
        min_throttle: float = ThrConst.MIN_THROTTLE,
        warmup_time_s: float = ThrConst.WARMUP_DELAY_S,
    ):
        self.isp = isp
        self.max_thrust_n = max_thrust_n
        self.max_power_w = max_power_w
        self.power_per_newton = power_per_newton
        self.min_throttle = min_throttle
        self.warmup_time_s = max(warmup_time_s, 1.0)  # guard against zero

        # Warmup state: 0.0 = cold (just ignited), 1.0 = full steady-state.
        # Effective throttle = commanded_throttle × _warmup_frac.
        # Updated by tick() once per accepted integrator sub-step.
        self._warmup_frac: float = 0.0

    def tick(self, commanded_throttle: float, dt_s: float) -> None:
        """
        Advance thruster warmup state by dt_s seconds.

        Call once per **accepted** integrator sub-step, before any call to
        ``thrust()`` or ``acceleration_and_mdot()`` for that same step.
        Call only on accepted steps — not on rejected RK45 trial steps.

        Ramp-up:   _warmup_frac increases at 1/warmup_time_s per second.
        Ramp-down: plasma extinguishes quickly (~5 s) when throttle is cut;
                   re-ignition then requires a full warmup cycle again.
        """
        if commanded_throttle >= self.min_throttle:
            self._warmup_frac = min(1.0, self._warmup_frac + dt_s / self.warmup_time_s)
        else:
            self._warmup_frac = max(0.0, self._warmup_frac - dt_s / self._SHUTDOWN_S)

    def thrust(
        self,
        throttle: float,
        available_power_w: float,
        propellant_kg: float,
    ) -> tuple[float, float]:
        """
        Compute the thrust magnitude and propellant flow rate.

        Effective throttle = commanded_throttle × _warmup_frac.
        A cold thruster (_warmup_frac = 0) produces zero thrust even at
        full commanded throttle.  Call tick() each accepted step to advance
        the warmup state.

        Args:
            throttle:          Requested throttle fraction [0–1]
            available_power_w: Current bus power available for propulsion [W]
            propellant_kg:     Remaining propellant mass [kg]

        Returns:
            (thrust_magnitude_N, mdot_kg_s) — scalar magnitudes.
            Direction is applied externally by the caller.
        """
        effective_throttle = throttle * self._warmup_frac

        if propellant_kg <= 0.0 or effective_throttle < self.min_throttle:
            return 0.0, 0.0

        effective_throttle = min(1.0, effective_throttle)

        # Power-limited thrust ceiling: lower of hardware cap and bus supply.
        usable_power_w = min(available_power_w, self.max_power_w)
        power_limited_thrust = usable_power_w / self.power_per_newton
        requested_thrust = effective_throttle * self.max_thrust_n
        actual_thrust_n = min(requested_thrust, power_limited_thrust)

        # Propellant mass flow rate from Tsiolkovsky
        mdot = actual_thrust_n / (self.isp * self.G0)

        return actual_thrust_n, mdot

    def power_request_w(
        self,
        throttle: float,
        propellant_kg: float,
    ) -> float:
        """
        Bus power the thruster would like to draw this step before power limits.

        Uses the current warmup fraction, throttle floor, and propellant state,
        but does not apply any spacecraft power-budget cap. The caller can feed
        the result into the power model to determine the actually deliverable
        propulsion power for the same interval.
        """
        effective_throttle = throttle * self._warmup_frac
        if propellant_kg <= 0.0 or effective_throttle < self.min_throttle:
            return 0.0

        effective_throttle = min(1.0, effective_throttle)
        requested_thrust_n = effective_throttle * self.max_thrust_n
        return min(requested_thrust_n * self.power_per_newton, self.max_power_w)

    def acceleration_and_mdot(
        self,
        thrust_dir: NDArray[np.float64],
        throttle: float,
        available_power_w: float,
        propellant_kg: float,
        total_mass_kg: float,
    ) -> tuple[NDArray[np.float64], float]:
        """
        Acceleration vector and mass flow rate for one timestep.

        Args:
            thrust_dir:    Unit vector giving thrust direction
            throttle:      Throttle fraction [0–1]
            available_power_w, propellant_kg, total_mass_kg: as named

        Returns:
            (acceleration [m s⁻²], mdot [kg s⁻¹])
        """
        F, mdot = self.thrust(throttle, available_power_w, propellant_kg)
        accel = (F / total_mass_kg) * thrust_dir
        return accel, mdot

    def delta_v_remaining(self, propellant_kg: float, dry_mass_kg: float) -> float:
        """
        Maximum Δv remaining via Tsiolkovsky [m s⁻¹].
        Useful for mission-level propellant budgeting.
        """
        if propellant_kg <= 0.0:
            return 0.0
        return self.isp * self.G0 * np.log((dry_mass_kg + propellant_kg) / dry_mass_kg)
