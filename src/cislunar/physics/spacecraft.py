"""
Numerical integrator and spacecraft simulation loop.

Uses adaptive-step RK4(5) (Dormand-Prince) for accuracy control:
 - Tight tolerance near bodies (periapsis maneuvers, close approaches)
 - Loose tolerance in interplanetary cruise (100–1000x speedup)

The ``Spacecraft`` class wraps state plus physics models and exposes a
``step(action, dt)`` interface for propagation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MASTER FRAME: Earth-Centred Inertial (ECI)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
All state vectors (position, velocity, sail_normal), ephemeris
outputs, and force vectors are in ECI throughout.

The one derived quantity that looks "non-ECI" is the Sun-relative
vector used for solar radiation pressure:

    sc_pos_from_sun = pos_eci - sun_pos_eci

This is the spacecraft's position measured from the Sun, expressed
in ECI coordinates.  It gives the correct Sun-to-spacecraft distance
and direction for the sail force model.  It is NOT a heliocentric
coordinate in the solar-system barycentric sense — Earth's absolute
position in the solar system is never tracked by this code.

sun_pos_fn(t) returns the Sun's position in ECI [m]. In the astropy-backed
ephemeris used for cislunar scenarios, the Sun moves along a smooth path
around the ECI origin (Earth's centre).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from . import constants
from .constants import Body
from .forces.attitude import AttitudeDynamicsModel, body_to_inertial, inertial_to_body
from .forces.drag import AtmosphericDragModel
from .forces.eclipse import PowerBudgetModel, body_shadow_illumination
from .forces.gravity import GravityModel
from .forces.propulsion import IonThrusterModel, SolarSailModel
from .forces.rcs import ColdGasRCSModel
from .state import SpacecraftState

# ── Dormand-Prince RK45 coefficients ─────────────────────────────────────────

_DP_A = np.array(
    [
        [0, 0, 0, 0, 0, 0],
        [1 / 5, 0, 0, 0, 0, 0],
        [3 / 40, 9 / 40, 0, 0, 0, 0],
        [44 / 45, -56 / 15, 32 / 9, 0, 0, 0],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0],
    ]
)
_DP_B = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84])
_DP_BE = np.array([5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])


@dataclass
class IntegratorConfig:
    """Tolerances and step-size bounds for the adaptive integrator."""

    atol: float = 1.0  # Absolute tolerance [m or m/s]
    rtol: float = 1e-6  # Relative tolerance
    dt_min_s: float = 0.1  # Minimum allowed step [s]
    dt_max_s: float = 3_600.0  # Maximum allowed step [s]  (1 hour)
    dt_init_s: float = 10.0  # Initial step guess [s]


@dataclass
class Action:
    """
    Control action for one simulation step.

    attitude_dir_cmd: Desired attitude direction unit vector (inertial frame).
                      Sail craft: desired sail normal.
                      RCS craft:  desired torque direction (body +Z target).
                      The attitude controller slews toward this at the craft's
                      max attitude rate; attitude changes are never instant.
    thrust_dir:       Unit vector for ion thruster pointing.
    throttle:         Ion throttle fraction [0–1].  0 = off.
    rcs_trigger:      True → fire RCS (attitude manoeuvre for RCS craft;
                      wheel desaturation burn for wheel craft).
    """

    attitude_dir_cmd: NDArray[np.float64] = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    thrust_dir: NDArray[np.float64] = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    throttle: float = 0.0
    rcs_trigger: bool = False


@dataclass
class Event:
    """
    Zero-crossing event for the adaptive integrator.

    The integrator evaluates fn(y, t) at the start and end of every accepted
    RK45 substep.  When a sign change is detected it bisects within the
    substep interval to locate the crossing to within tol_s seconds, then
    records an EventResult and — if terminal=True — stops integration there.

    Args:
        name:      Human-readable label (used in EventResult and logs).
        fn:        Signed scalar function of (state_vector, time_s).
                   Negative before the event, positive after (or vice-versa
                   depending on direction).  Zero = event occurred.
        direction: "falling" = negative-going zero (fn goes + → −),
                   "rising"  = positive-going zero (fn goes − → +),
                   "both"    = detect either crossing direction.
        terminal:  If True, integration halts at the crossing time and
                   step() returns the event state as the final state.
        tol_s:     Bisection convergence tolerance in seconds.
    """

    name: str
    fn: Callable[[NDArray, float], float]
    direction: Literal["rising", "falling", "both"] = "both"
    terminal: bool = False
    tol_s: float = 1.0  # 1-second default is adequate for all event detections


@dataclass
class EventResult:
    """Outcome of a zero-crossing detection for one Event."""

    name: str
    t_event: float  # simulation time of the crossing [s]
    y_event: NDArray  # integrated state at the crossing
    terminal: bool  # copied from Event.terminal


def _bisect_event(
    fn: Callable[[NDArray, float], float],
    y0: NDArray,
    t0: float,
    y1: NDArray,
    t1: float,
    tol_s: float,
) -> tuple[NDArray, float]:
    """
    Bisect the interval [t0, t1] to find the zero of fn to within tol_s.

    Uses linear interpolation of the state vector (not RK integration) for
    speed.  This is valid for the short substep intervals produced by the
    RK45 adaptive stepper — errors from linear interpolation over a 1–10 s
    accepted step are well below the 1 s timing tolerance for all event detections.

    Returns (y_event, t_event).
    """
    fa = fn(y0, t0)
    for _ in range(50):  # 50 bisections → tolerance ~(t1-t0)/2^50, far below 1 s
        if t1 - t0 < tol_s:
            break
        t_mid = 0.5 * (t0 + t1)
        alpha = (t_mid - t0) / (t1 - t0 + 1e-300)
        y_mid = y0 + alpha * (y1 - y0)
        fm = fn(y_mid, t_mid)
        if fa * fm <= 0.0:  # zero is in left half
            t1, y1 = t_mid, y_mid
        else:  # zero is in right half
            t0, y0, fa = t_mid, y_mid, fm
    t_event = 0.5 * (t0 + t1)
    alpha = (t_event - t0) / (t1 - t0 + 1e-300)
    y_event = y0 + alpha * (y1 - y0)
    return y_event, t_event


class Spacecraft:
    """
    One spacecraft: physics models + state + integrator.

    Designed to be instantiated once per craft per simulation.

    Args:
        initial_state:      Starting conditions
        sail_model:         Configured SolarSailModel
        thruster_model:     Configured IonThrusterModel
        gravity_model:      Configured GravityModel for this scenario
        max_attitude_rate:  Max sail slew rate [rad s⁻¹] — physical limit
        integrator_cfg:     Tolerances for the adaptive stepper
    """

    def __init__(
        self,
        initial_state: SpacecraftState,
        sail_model: SolarSailModel | None,
        thruster_model: IonThrusterModel,
        gravity_model: GravityModel,
        max_attitude_rate: float = 0.005,
        integrator_cfg: IntegratorConfig | None = None,
        sun_pos_fn: Callable[[float], NDArray[np.float64]] | None = None,
        drag_model: AtmosphericDragModel | None = None,
        power_model: PowerBudgetModel | None = None,  # required; None raises immediately
        attitude_model: AttitudeDynamicsModel | None = None,
        sail_enabled: bool = True,
        reaction_wheels: bool = True,
    ):
        self.state = initial_state.copy()
        self.sail = sail_model
        self.thruster = thruster_model
        self.gravity = gravity_model
        if power_model is None:
            raise ValueError(
                "power_model is required. Pass PowerBudgetModel() or use make_lunar_craft()."
            )
        self.drag = drag_model
        self.power = power_model
        self.attitude = attitude_model
        self.max_att_rate = max_attitude_rate
        self.cfg = integrator_cfg or IntegratorConfig()

        # Craft-type flags — determined by ScenarioConfig at env level and
        # forwarded here.  Controls which physics paths are active.
        self._sail_enabled = sail_enabled and (sail_model is not None)
        self._reaction_wheels = reaction_wheels
        self._craft_type = "wheel" if reaction_wheels else "rcs"

        # Cold gas RCS model — instantiated when craft has no reaction wheels.
        self._rcs: ColdGasRCSModel | None = ColdGasRCSModel() if not reaction_wheels else None

        if sun_pos_fn is None:
            raise ValueError(
                "sun_pos_fn is required. Use build_cislunar_ephemeris() to get a "
                "DE430-backed callable, or pass make_lunar_craft() which handles this automatically."
            )
        self._sun_pos_fn = sun_pos_fn

        # Moon position function: auto-detected from the gravity model's body
        # list when the Moon is included as a perturber.  When the gravity
        # model is Earth-only the Moon's occultation contribution is zero
        # (Moon cannot plausibly eclipse the Sun from an Earth-centred two-
        # body trajectory), so leaving this as None is correct.  Same access
        # pattern as the comms-blackout utilities use for lunar visibility checks.
        self._moon_pos_fn: Callable[[float], NDArray[np.float64]] | None = None
        for _body in self.gravity._bodies:
            if _body.name == "Moon" and _body.body_pos_fn is not None:
                self._moon_pos_fn = _body.body_pos_fn
                break

        self._dt = self.cfg.dt_init_s
        self._action = Action()
        self.history: list[SpacecraftState] = []

    # ── Public interface ──────────────────────────────────────────────────────

    def step(
        self,
        action: Action,
        dt_requested: float,
        events: list[Event] | None = None,
    ) -> tuple[SpacecraftState, list[EventResult]]:
        """
        Advance the spacecraft by dt_requested seconds under the given action.

        The integrator may take multiple sub-steps internally.
        Propellant and attitude are updated as side effects.

        Args:
            action:       Control input for this step.
            dt_requested: Requested wall-clock advance [s].
            events:       Optional list of Event objects.  After each accepted
                          RK45 substep the integrator evaluates every event
                          function at (y_prev, t) and (y_new, t+dt).  A sign
                          change triggers bisection to locate the crossing time
                          to within event.tol_s seconds.  If event.terminal is
                          True, integration halts at the crossing and the event
                          state is used as the final state for that step.

        Returns:
            (new_state, triggered_events) — triggered_events is an empty list
            when no events parameter is passed or no crossings occurred.
        """
        self._action = action
        t_step_start = self.state.time_s
        t_end = self.state.time_s + dt_requested
        t = self.state.time_s
        y = self.state.as_vector()

        triggered: list[EventResult] = []
        _events = events or []
        # Pre-evaluate all event functions at the start of this env step
        _ev_vals: list[float] = [ev.fn(y, t) for ev in _events]

        while t < t_end:
            dt = min(self._dt, t_end - t)
            y_new, dt_next, accepted = self._rk45_step(y, t, dt)

            if accepted:
                _y_before = y
                _t_before = t
                _t_after = t + dt

                # ── Event detection ───────────────────────────────────────
                # Detect any crossings before mutating side-effectful state
                # (battery, propellant, warmup). A terminal event trims the
                # accepted substep to the true crossing time, and only that
                # elapsed interval is allowed to update spacecraft resources.
                stop_integration = False
                y_effective = y_new
                t_effective = _t_after
                earliest_terminal: EventResult | None = None

                for k, ev in enumerate(_events):
                    f_new = ev.fn(y_new, _t_after)
                    f_old = _ev_vals[k]
                    crossed = False
                    if ev.direction == "both":
                        crossed = f_old * f_new < 0.0
                    elif ev.direction == "rising":
                        crossed = f_old < 0.0 <= f_new
                    elif ev.direction == "falling":
                        crossed = f_old >= 0.0 > f_new

                    if crossed:
                        y_ev, t_ev = _bisect_event(
                            ev.fn,
                            _y_before,
                            _t_before,
                            y_new,
                            _t_after,
                            ev.tol_s,
                        )
                        result = EventResult(
                            name=ev.name,
                            t_event=t_ev,
                            y_event=y_ev,
                            terminal=ev.terminal,
                        )
                        triggered.append(result)
                        if ev.terminal and (
                            earliest_terminal is None or t_ev < earliest_terminal.t_event
                        ):
                            earliest_terminal = result
                    _ev_vals[k] = f_new

                if earliest_terminal is not None:
                    stop_integration = True
                    y_effective = earliest_terminal.y_event
                    t_effective = earliest_terminal.t_event

                dt_effective = max(0.0, t_effective - _t_before)

                # ── Power model update (gates thruster) ───────────────────
                # Use the true end-of-interval state (accepted endpoint or
                # terminal-event crossing) so the power model sees the correct
                # geometry and only advances over the elapsed portion.
                sun_pos = self._sun_pos_fn(t_effective)
                moon_pos = self._moon_pos_fn(t_effective) if self._moon_pos_fn is not None else None
                pos_new = y_effective[0:3]
                sc_pos_from_sun = pos_new - sun_pos
                sail_norm_new = y_effective[6:9]
                p_req = self.thruster.power_request_w(
                    action.throttle,
                    self.state.propellant_kg,
                )
                p_thr = self.power.step(
                    spacecraft_pos=pos_new,
                    sun_pos=sun_pos,
                    helio_pos=sc_pos_from_sun,
                    dt_s=dt_effective,
                    sail_normal_eci=sail_norm_new,
                    moon_pos=moon_pos,
                    propulsion_power_request_w=p_req,
                )
                self.state.power_w = p_thr

                # ── Propellant accounting ─────────────────────────────────
                # Must use the same pre-tick thruster state that the accepted
                # RK45 substep integrated. Advancing warmup before this call
                # would charge propellant against a later throttle state than
                # the one that produced the impulse in _derivatives().
                _, mdot = self.thruster.thrust(
                    action.throttle,
                    self.state.power_w,
                    self.state.propellant_kg,
                )
                self.state.propellant_kg = max(0.0, self.state.propellant_kg - mdot * dt_effective)

                # ── Thruster warmup tick ───────────────────────────────────
                # Advance warmup only after accounting for the interval that
                # just completed. Rejected RK45 trial steps must not advance
                # the warmup clock at all.
                self.thruster.tick(action.throttle, dt_effective)
                y = y_effective
                t = t_effective
                self.state.time_s = t

                if stop_integration:
                    break

            self._dt = np.clip(dt_next, self.cfg.dt_min_s, self.cfg.dt_max_s)

        # Reconstruct state from integrated vector
        new_state = SpacecraftState.from_vector(
            y,
            propellant_kg=self.state.propellant_kg,
            rcs_propellant_kg=self.state.rcs_propellant_kg,
            power_w=self.state.power_w,
            time_s=t,
            mass_dry_kg=self.state.mass_dry_kg,
        )
        # ── Attitude: decoupled fixed-rate control loop (5 s steps) ─────────
        # Attitude dynamics are slow (90° slew ≈ 5-10 min) and must NOT run
        # at the orbital RK45 adaptive step cadence (can be <1 s at LEO).
        # We advance the attitude controller at a fixed 5 s rate AFTER the
        # orbit integration completes, using the final position as anchor.
        att_elapsed_total = t - t_step_start
        if self.attitude is not None and att_elapsed_total > 0.0:
            att_dt = 5.0
            n_att = max(1, int(round(att_elapsed_total / att_dt)))
            dt_att = att_elapsed_total / n_att

            cmd_norm = action.attitude_dir_cmd
            cmd_norm = cmd_norm / (np.linalg.norm(cmd_norm) + 1e-30)
            q_cmd = self.attitude.sail_normal_to_quaternion(cmd_norm)
            sail_out = self.attitude.sail_normal_inertial  # fallback

            # SRP torque coupling: compute sail force at the post-integration
            # state and forward it to the attitude model so the CoP offset
            # loads the reaction wheels realistically.  Only for sail craft.
            _srp_force: NDArray | None = None
            if self._sail_enabled and self.sail is not None:
                _sun_pos_att = self._sun_pos_fn(new_state.time_s)
                _moon_pos_att = (
                    self._moon_pos_fn(new_state.time_s) if self._moon_pos_fn is not None else None
                )
                _sc_from_sun = new_state.position - _sun_pos_att
                _illum_att = body_shadow_illumination(
                    new_state.position,
                    _sun_pos_att,
                    _moon_pos_att,
                )
                _srp_force = self.sail.force(
                    _sc_from_sun,
                    new_state.sail_normal,
                    new_state.total_mass_kg,
                    illumination=_illum_att,
                )

            for _ in range(n_att):
                out = self.attitude.step(
                    q_cmd,
                    dt=dt_att,
                    position=new_state.position,
                    srp_force_eci=_srp_force,
                    craft_type=self._craft_type,
                )
                if np.all(np.isfinite(out)):
                    sail_out = out
                else:
                    self.attitude.reset()  # NaN guard: restart at current q
                    break

            new_state.sail_normal = sail_out

            # ── RCS trigger ───────────────────────────────────────────────
            if action.rcs_trigger:
                if (
                    self._craft_type == "rcs"
                    and self._rcs is not None
                    and new_state.rcs_propellant_kg > 0.0
                ):
                    # Cold gas RCS: apply translational force kick and consume propellant.
                    # Torque is already handled by attitude.step() (craft_type="rcs").
                    rcs_dir_body = inertial_to_body(action.attitude_dir_cmd, self.attitude.q)
                    _, force_body, rcs_prop_used = self._rcs.fire(
                        rcs_dir_body,
                        True,
                        att_elapsed_total,
                        new_state.rcs_propellant_kg,
                    )
                    # Rotate force to ECI and apply as velocity kick (operator splitting).
                    force_eci = body_to_inertial(force_body, self.attitude.q)
                    if new_state.total_mass_kg > 0.0:
                        new_state.velocity = (
                            new_state.velocity
                            + force_eci / new_state.total_mass_kg * att_elapsed_total
                        )
                    new_state.rcs_propellant_kg = max(
                        0.0, new_state.rcs_propellant_kg - rcs_prop_used
                    )
                else:
                    # Wheel craft: desaturation burn using ion thruster.
                    dv_cost = self.attitude.desaturate()
                    if dv_cost > 0 and new_state.propellant_kg > 0:
                        isp_g0 = self.thruster.G0 * self.thruster.isp
                        dm = new_state.total_mass_kg * (1.0 - math.exp(-dv_cost / isp_g0))
                        new_state.propellant_kg = max(0.0, new_state.propellant_kg - dm)

        self.state = new_state
        return new_state, triggered

    # ── Derivative function (the ODE right-hand side) ─────────────────────────

    def _derivatives(
        self,
        y: NDArray[np.float64],
        t: float,
    ) -> NDArray[np.float64]:
        """
        dy/dt for the state vector [pos, vel, sail_normal].

        Layout of y: [px, py, pz, vx, vy, vz, nx, ny, nz]
        Layout of dy: [vx, vy, vz, ax, ay, az, dnx, dny, dnz]
        """
        pos = y[0:3]
        vel = y[3:6]
        norm = y[6:9] / (np.linalg.norm(y[6:9]) + 1e-30)  # normalise

        action = self._action
        mass = self.state.total_mass_kg

        # ── Accelerations ─────────────────────────────────────────────────
        a_grav = self.gravity.acceleration(pos, t)

        # Sun and (optionally) Moon positions at this stage time t.
        # Both are analytic functions of t, so RK45 stages stay independent.
        sun_pos_t = self._sun_pos_fn(t)
        moon_pos_t = self._moon_pos_fn(t) if self._moon_pos_fn is not None else None

        # Sail illumination: Earth umbra/penumbra × Moon umbra/penumbra.
        # During full eclipse illum == 0 and the sail produces zero SRP.
        illum = body_shadow_illumination(pos, sun_pos_t, moon_pos_t)

        # Solar sail acceleration — gated on sail_enabled.
        # Spacecraft without an active sail contribute no SRP acceleration.
        if self._sail_enabled and self.sail is not None:
            a_sail = self.sail.acceleration(
                pos - sun_pos_t,  # sc_pos_from_sun: ECI vector from Sun to spacecraft
                norm,
                mass,
                illumination=illum,
            )
        else:
            a_sail = np.zeros(3)

        thrust_dir_norm = action.thrust_dir / (np.linalg.norm(action.thrust_dir) + 1e-30)
        a_thrust, _ = self.thruster.acceleration_and_mdot(
            thrust_dir_norm,
            action.throttle,
            self.state.power_w,
            self.state.propellant_kg,
            mass,
        )

        # ── Planetary albedo SRP (Lambertian-disc model) ─────────────────────
        if self._sail_enabled and self.sail is not None:
            sc_from_sun = pos - sun_pos_t
            earth_pos = np.zeros(3)  # Earth at ECI origin
            a_sail = a_sail + self.sail.albedo_acceleration(
                pos,
                earth_pos,
                sun_pos_t,
                sc_from_sun,
                norm,
                constants.Body.R_EARTH,
                constants.Body.ALBEDO_EARTH,
                mass,
            )
            if moon_pos_t is not None:
                a_sail = a_sail + self.sail.albedo_acceleration(
                    pos,
                    moon_pos_t,
                    sun_pos_t,
                    sc_from_sun,
                    norm,
                    constants.Body.R_MOON,
                    constants.Body.ALBEDO_MOON,
                    mass,
                )

        accel = a_grav + a_sail + a_thrust

        # ── Atmospheric drag (zero above ~1200 km or if model not provided) ──
        if self.drag is not None:
            a_drag = self.drag.acceleration(pos, vel, norm, mass, t)
            accel = accel + a_drag

        # ── Sail-normal rate ──────────────────────────────────────────────
        if self.attitude is not None:
            # Attitude is managed on its own fixed-rate timeline (5 s steps),
            # fully decoupled from the orbital RK45 integrator.
            #
            # _derivatives() must be a pure function of (y, t) so that the
            # RK45 stage evaluations are independent and can be safely rejected
            # or re-evaluated.  Calling self.attitude.step() here would mutate
            # attitude state (quaternion, omega, wheel momenta) on every stage
            # call — 6-7 times per accepted step — corrupting the controller.
            #
            # Instead: freeze the sail normal at the value it held at the start
            # of the environment step.  The post-integration loop in step()
            # advances the attitude model once on its own 5 s grid and then
            # writes the result into new_state.sail_normal.
            dn_dt = np.zeros(3)
        else:
            # First-order slew fallback (no rigid-body attitude model)
            cmd_normal = action.attitude_dir_cmd
            cmd_normal = cmd_normal / (np.linalg.norm(cmd_normal) + 1e-30)
            error = cmd_normal - norm
            error_mag = np.linalg.norm(error)
            if error_mag > 1e-9:
                dn_dt = self.max_att_rate * error / error_mag
            else:
                dn_dt = np.zeros(3)

        return np.concatenate([vel, accel, dn_dt])

    # ── Adaptive RK45 step (Dormand-Prince) ──────────────────────────────────

    def _rk45_step(
        self,
        y: NDArray[np.float64],
        t: float,
        dt: float,
    ) -> tuple[NDArray[np.float64], float, bool]:
        """
        One Dormand-Prince RK45 step with error estimate.

        Returns (y_new, dt_next, accepted).
        """
        k = np.zeros((7, len(y)))
        k[0] = self._derivatives(y, t)

        for i in range(1, 6):
            yi = y + dt * (_DP_A[i, :i] @ k[:i])
            k[i] = self._derivatives(yi, t + dt * float(_DP_A[i, :i].sum()))

        y4 = y + dt * (_DP_B @ k[:6])
        k[6] = self._derivatives(y4, t + dt)

        y5 = y + dt * (_DP_BE @ k)

        # Error estimate: difference between 4th and 5th order solutions
        scale = self.cfg.atol + self.cfg.rtol * np.maximum(np.abs(y), np.abs(y5))
        err = np.linalg.norm((y5 - y4) / scale) / np.sqrt(len(y))

        # Step-size control (PI controller, standard formula)
        if err == 0.0:
            factor = 5.0
        else:
            factor = 0.9 * (1.0 / err) ** 0.2

        factor = np.clip(factor, 0.1, 5.0)
        dt_next = dt * factor

        accepted = err <= 1.0
        y_out = y5 if accepted else y  # use 5th order if accepted

        return y_out, dt_next, accepted


# ── Convenience factory functions ─────────────────────────────────────────────


def make_lunar_craft(
    position_m: Sequence[float] | NDArray[np.float64],
    velocity_ms: Sequence[float] | NDArray[np.float64],
    sail_area_m2: float = 32.0,  # 32 m² ≈ LightSail 2 scale
    propellant_kg: float = 0.5,
    start_date: str = "2025-01-01",  # ISO date for ephemeris epoch
    use_live_drag_indices: bool = False,
    use_grgm_harmonic: bool = False,
) -> Spacecraft:
    """
    Pre-configured cislunar spacecraft.

    Gravity: Earth (J2) + Moon (real ephemeris) + Sun (real ephemeris).
    Uses astropy built-in DE430 ephemeris via build_cislunar_ephemeris().
    astropy is a required dependency for cislunar scenarios.
    """
    from .forces.ephemeris import build_cislunar_ephemeris
    from .forces.mascon import default_lunar_harmonic_gravity, lunar_mascon_acceleration

    # ── Body ephemerides (JPL-quality via astropy built-in DE430) ────────────
    # start_date sets the absolute epoch; simulation time is offset from it.
    moon_pos, sun_pos = build_cislunar_ephemeris(
        start_time=start_date,
        duration_days=60.0,
    )

    # ── Gravity model ─────────────────────────────────────────────────────────
    lunar_perturbation = (
        default_lunar_harmonic_gravity() if use_grgm_harmonic else lunar_mascon_acceleration
    )

    grav = (
        GravityModel()
        .add_body("Earth", mu=Body.MU_EARTH, j2=Body.J2_EARTH, r_body=Body.R_EARTH)
        .add_body(
            "Moon",
            mu=Body.MU_MOON,
            j2=Body.J2_MOON,
            j3=Body.J3_MOON,
            r_body=Body.R_MOON,
            body_pos_fn=moon_pos,
        )
        .add_body("Sun", mu=Body.MU_SUN, body_pos_fn=sun_pos)
        .set_mascon_model(lunar_perturbation)
    )

    state = SpacecraftState(
        position=np.array(position_m, dtype=float),
        velocity=np.array(velocity_ms, dtype=float),
        propellant_kg=propellant_kg,
        mass_dry_kg=12.0,
    )

    from .forces.eclipse import SolarArrayConfig

    return Spacecraft(
        initial_state=state,
        sail_model=SolarSailModel(area_m2=sail_area_m2),
        thruster_model=IonThrusterModel(),
        gravity_model=grav,
        sun_pos_fn=sun_pos,
        drag_model=AtmosphericDragModel(
            body_area_m2=0.077,
            sail_area_m2=sail_area_m2,
            use_live_indices=use_live_drag_indices,
        ),
        power_model=PowerBudgetModel(
            array_cfg=SolarArrayConfig(area_m2=constants.Spacecraft.SOLAR_PANEL_AREA_M2)
        ),
        attitude_model=AttitudeDynamicsModel(),
    )
