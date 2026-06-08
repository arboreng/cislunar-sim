"""
Telemetry replay and residual analysis utilities.

Ingest real mission telemetry (such as LightSail 2), run the physics engine
forward from the same initial conditions, and measure divergence between the
simulated and observed trajectories.

Data format:
    Telemetry is a time-series of spacecraft state dictionaries:
    [
        {
            "time_s":      float,          mission elapsed seconds
            "position_m":  [x, y, z],      Earth-centred inertial [m]
            "velocity_ms": [vx, vy, vz],   [m/s]
            "sail_area_m2": float,          effective area (may vary)
        },
        ...
    ]

For LightSail 2 the real data can be fetched from:
    - Planetary Society: planetary.org/lightsail (TLE + attitude data)
    - NASA Horizons: ssd.jpl.nasa.gov/horizons (for ephemeris)

This module also ships with a synthetic data generator so you can develop and
test the replay pipeline before obtaining real telemetry files.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from cislunar.physics import (
    Action,
    GravityModel,
    IonThrusterModel,
    PowerBudgetModel,
    SolarSailModel,
    Spacecraft,
    SpacecraftState,
)
from cislunar.physics.constants import AU, Body


def _semi_major_axis(pos_m: NDArray[np.float64], vel_ms: NDArray[np.float64], mu: float) -> float:
    """Return osculating semi-major axis from a Cartesian state."""
    radius_m = np.linalg.norm(pos_m)
    speed_sq = float(np.dot(vel_ms, vel_ms))
    return float(radius_m / (2.0 - radius_m * speed_sq / mu))


def _eccentricity(pos_m: NDArray[np.float64], vel_ms: NDArray[np.float64], mu: float) -> float:
    """Return osculating eccentricity magnitude from a Cartesian state."""
    radius_m = np.linalg.norm(pos_m)
    h_vec = np.cross(pos_m, vel_ms)
    e_vec = np.cross(vel_ms, h_vec) / mu - pos_m / radius_m
    return float(np.linalg.norm(e_vec))


# ── Telemetry data structures ─────────────────────────────────────────────────


@dataclass
class TelemetryPoint:
    """One observation from real mission telemetry."""

    time_s: float
    position_m: NDArray[np.float64]
    velocity_ms: NDArray[np.float64]
    sail_area_m2: float = 32.0
    # Optional observed quantities (used for residual analysis)
    altitude_m: float | None = None
    solar_flux_wm2: float | None = None


@dataclass
class TelemetryDataset:
    """
    A complete mission telemetry dataset.

    Attributes:
        name:        Mission name (e.g. "LightSail 2")
        points:      Time-ordered list of telemetry observations
        sail_area:   Nominal sail area [m²]
        dry_mass_kg: Spacecraft dry mass [kg]
        propellant_kg: Propellant at mission start [kg]
        notes:       Free-form metadata
    """

    name: str
    points: list[TelemetryPoint]
    sail_area_m2: float = 32.0
    dry_mass_kg: float = 5.0
    propellant_kg: float = 0.0  # LightSail 2 has no thruster
    notes: str = ""

    @property
    def duration_s(self) -> float:
        if not self.points:
            return 0.0
        return self.points[-1].time_s - self.points[0].time_s

    @property
    def initial_state(self) -> SpacecraftState:
        p = self.points[0]
        return SpacecraftState(
            position=p.position_m.copy(),
            velocity=p.velocity_ms.copy(),
            sail_normal=np.array([-1.0, 0.0, 0.0]),  # toward Sun initially
            propellant_kg=self.propellant_kg,
            mass_dry_kg=self.dry_mass_kg,
            time_s=p.time_s,
        )

    @classmethod
    def load_json(cls, path: str) -> TelemetryDataset:
        """Load a telemetry dataset from JSON."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        points = [
            TelemetryPoint(
                time_s=pt["time_s"],
                position_m=np.array(pt["position_m"], dtype=np.float64),
                velocity_ms=np.array(pt["velocity_ms"], dtype=np.float64),
                sail_area_m2=pt.get("sail_area_m2", data.get("sail_area_m2", 32.0)),
                altitude_m=pt.get("altitude_m"),
                solar_flux_wm2=pt.get("solar_flux_wm2"),
            )
            for pt in data["points"]
        ]
        return cls(
            name=data.get("name", "Unknown"),
            points=points,
            sail_area_m2=data.get("sail_area_m2", 32.0),
            dry_mass_kg=data.get("dry_mass_kg", 5.0),
            propellant_kg=data.get("propellant_kg", 0.0),
            notes=data.get("notes", ""),
        )

    def save_json(self, path: str) -> None:
        """Write the telemetry dataset to JSON."""
        points_payload = []
        for pt in self.points:
            point_data = {
                "time_s": pt.time_s,
                "position_m": pt.position_m.tolist(),
                "velocity_ms": pt.velocity_ms.tolist(),
                "sail_area_m2": pt.sail_area_m2,
            }
            if pt.altitude_m is not None:
                point_data["altitude_m"] = pt.altitude_m
            if pt.solar_flux_wm2 is not None:
                point_data["solar_flux_wm2"] = pt.solar_flux_wm2
            points_payload.append(point_data)

        data = {
            "name": self.name,
            "sail_area_m2": self.sail_area_m2,
            "dry_mass_kg": self.dry_mass_kg,
            "propellant_kg": self.propellant_kg,
            "notes": self.notes,
            "points": points_payload,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


# ── Synthetic data generator ──────────────────────────────────────────────────


def generate_lightsail2_synthetic(
    duration_days: float = 30.0,
    sample_interval_s: float = 3600.0,  # 1 sample per hour
    seed: int = 42,
) -> TelemetryDataset:
    """
    Generate synthetic telemetry mimicking LightSail 2's trajectory.

    LightSail 2 (launched 2019-06-25):
        - Initial orbit: ~720 km altitude, 24° inclination
        - Sail area: 32 m²
        - Mass: ~5 kg
        - No chemical propulsion

    Uses our physics engine to generate the "ground truth" — in real
    use you replace this with actual downloaded telemetry.
    """
    R_orbit = Body.R_EARTH + 720e3  # 720 km altitude
    v_circ = math.sqrt(Body.MU_EARTH / R_orbit)

    rng = np.random.default_rng(seed)

    # Build gravity model matching LightSail 2 regime
    gravity = GravityModel().add_body(
        "Earth", mu=Body.MU_EARTH, j2=Body.J2_EARTH, r_body=Body.R_EARTH
    )

    state = SpacecraftState(
        position=np.array([R_orbit, 0.0, 0.0]),
        velocity=np.array(
            [0.0, v_circ * math.cos(math.radians(24)), v_circ * math.sin(math.radians(24))]
        ),
        sail_normal=np.array([-1.0, 0.0, 0.0]),
        propellant_kg=0.0,
        mass_dry_kg=5.0,
    )

    sc = Spacecraft(
        initial_state=state,
        sail_model=SolarSailModel(area_m2=32.0),
        thruster_model=IonThrusterModel(),
        gravity_model=gravity,
        sun_pos_fn=lambda t: np.array([AU, 0.0, 0.0]),
        power_model=PowerBudgetModel(),
    )

    # Attitude: LightSail 2 actively orients sail toward Sun
    action = Action(
        attitude_dir_cmd=np.array([-1.0, 0.0, 0.0]),
        throttle=0.0,
    )

    points: list[TelemetryPoint] = []
    duration_s = duration_days * 86400.0
    t = 0.0

    # Record initial state
    points.append(
        TelemetryPoint(
            time_s=t,
            position_m=sc.state.position.copy(),
            velocity_ms=sc.state.velocity.copy(),
        )
    )

    while t < duration_s:
        dt = min(sample_interval_s, duration_s - t)
        sc.step(action, dt_requested=dt)
        t += dt

        # Add small measurement noise (±100 m position, ±0.1 m/s velocity)
        noise_pos = rng.normal(0, 100.0, 3)
        noise_vel = rng.normal(0, 0.1, 3)

        points.append(
            TelemetryPoint(
                time_s=t,
                position_m=sc.state.position.copy() + noise_pos,
                velocity_ms=sc.state.velocity.copy() + noise_vel,
            )
        )

    return TelemetryDataset(
        name="LightSail 2 (synthetic)",
        points=points,
        sail_area_m2=32.0,
        dry_mass_kg=5.0,
        propellant_kg=0.0,
        notes="Synthetic dataset generated by physics engine. "
        "Replace with real planetary.org/lightsail TLE data.",
    )


# ── Replay engine ─────────────────────────────────────────────────────────────


@dataclass
class ReplayMetrics:
    """Divergence statistics between simulation and telemetry."""

    mission_name: str
    duration_s: float
    n_points: int

    # Position errors
    pos_error_mean_m: float = 0.0
    pos_error_max_m: float = 0.0
    pos_error_rms_m: float = 0.0
    pos_error_pct: float = 0.0  # as % of orbital radius

    # Velocity errors
    vel_error_mean_ms: float = 0.0
    vel_error_rms_ms: float = 0.0

    # RTN (radial-transverse-normal) error decomposition
    # Along-track (transverse) error is dominated by drag phase drift.
    # Radial and cross-track errors reflect force model accuracy.
    radial_error_mean_m: float = 0.0  # altitude accuracy
    along_track_error_mean_m: float = 0.0  # phase/drag accuracy
    cross_track_error_mean_m: float = 0.0  # inclination accuracy

    # Orbit element errors
    sma_error_pct: float = 0.0  # semi-major axis
    eccentricity_err: float = 0.0

    # Per-point residuals (for plotting)
    times_s: list[float] = field(default_factory=list)
    pos_errors_m: list[float] = field(default_factory=list)
    vel_errors_ms: list[float] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Replay: {self.mission_name}",
            f"  Duration:        {self.duration_s / 86400:.1f} days  "
            f"({self.n_points} telemetry points)",
            f"  Position error:  mean={self.pos_error_mean_m / 1e3:.3f} km  "
            f"max={self.pos_error_max_m / 1e3:.3f} km  "
            f"rms={self.pos_error_rms_m / 1e3:.3f} km  "
            f"({self.pos_error_pct:.3f}% of orbit radius)",
            f"  RTN breakdown:   R={self.radial_error_mean_m / 1e3:.3f} km  "
            f"T={self.along_track_error_mean_m / 1e3:.3f} km  "
            f"N={self.cross_track_error_mean_m / 1e3:.3f} km",
            f"  Velocity error:  mean={self.vel_error_mean_ms:.3f} m/s  "
            f"rms={self.vel_error_rms_ms:.3f} m/s",
            f"  SMA error:       {self.sma_error_pct:.4f}%",
        ]
        # Primary fidelity check: radial (altitude) error < 2% of orbit radius.
        # Along-track phase drift is driven by drag uncertainty, not force model error.
        # Radial and cross-track errors reflect the accuracy of gravity + sail forces.
        # Check radial vs orbit radius directly
        orbit_radius_m = self.pos_error_mean_m / max(self.pos_error_pct / 100, 1e-10)
        radial_pct_of_r = self.radial_error_mean_m / max(orbit_radius_m, 1e6) * 100
        grade_pos = "✓" if self.pos_error_pct < 2.0 else "✗"
        grade_sma = "✓" if self.sma_error_pct < 0.1 else "✗"
        grade_radial = "✓" if radial_pct_of_r < 0.1 else "✗"
        lines.append(
            f"  Fidelity grade:  position {grade_pos} ({self.pos_error_pct:.3f}%)  "
            f"SMA {grade_sma} ({self.sma_error_pct:.4f}%)  "
            f"radial {grade_radial} ({radial_pct_of_r:.4f}%)"
        )
        return "\n".join(lines)


def replay(
    dataset: TelemetryDataset,
    attitude_fn=None,  # callable(time_s, state) -> sail_normal; default is fixed toward Sun
) -> ReplayMetrics:
    """
    Run the physics engine from dataset's initial conditions and compare
    against all subsequent telemetry points.

    Args:
        dataset:      TelemetryDataset with real or synthetic observations
        attitude_fn:  Optional function returning sail_normal at each step.
                      If None, sail is held toward Sun (simplest assumption).

    Returns:
        ReplayMetrics with full divergence analysis.
    """
    if len(dataset.points) < 2:
        raise ValueError("Need at least 2 telemetry points for replay")

    # Build spacecraft matching the mission
    gravity = GravityModel().add_body(
        "Earth", mu=Body.MU_EARTH, j2=Body.J2_EARTH, r_body=Body.R_EARTH
    )

    sc = Spacecraft(
        initial_state=dataset.initial_state,
        sail_model=SolarSailModel(area_m2=dataset.sail_area_m2),
        thruster_model=IonThrusterModel(),
        gravity_model=gravity,
        sun_pos_fn=lambda t: np.array([AU, 0.0, 0.0]),
        power_model=PowerBudgetModel(),
    )

    pos_errors: list[float] = []
    vel_errors: list[float] = []
    times: list[float] = []
    orbital_radii: list[float] = []
    sim_positions: list[NDArray[np.float64]] = []
    observed_positions: list[NDArray[np.float64]] = []
    observed_velocities: list[NDArray[np.float64]] = []

    t_prev = dataset.points[0].time_s

    for pt in dataset.points[1:]:
        dt = pt.time_s - t_prev
        if dt <= 0:
            continue

        # Determine sail attitude for this step
        if attitude_fn is not None:
            sail_cmd = attitude_fn(sc.state.time_s, sc.state)
        else:
            sail_cmd = np.array([-1.0, 0.0, 0.0])  # fixed toward Sun

        action = Action(attitude_dir_cmd=sail_cmd, throttle=0.0)
        sc.step(action, dt_requested=dt)

        pos_err = float(np.linalg.norm(sc.state.position - pt.position_m))
        vel_err = float(np.linalg.norm(sc.state.velocity - pt.velocity_ms))

        pos_errors.append(pos_err)
        vel_errors.append(vel_err)
        times.append(pt.time_s)
        orbital_radii.append(float(np.linalg.norm(pt.position_m)))
        sim_positions.append(sc.state.position.copy())
        observed_positions.append(pt.position_m.copy())
        observed_velocities.append(pt.velocity_ms.copy())

        t_prev = pt.time_s

    if not pos_errors:
        raise ValueError("Replay produced no comparable telemetry samples")

    pos_arr = np.array(pos_errors)
    vel_arr = np.array(vel_errors)
    r_arr = np.array(orbital_radii)
    sim_pos_arr = np.array(sim_positions)
    obs_pos_arr = np.array(observed_positions)
    obs_vel_arr = np.array(observed_velocities)

    sma_obs = _semi_major_axis(
        dataset.points[-1].position_m,
        dataset.points[-1].velocity_ms,
        Body.MU_EARTH,
    )
    sma_sim = _semi_major_axis(sc.state.position, sc.state.velocity, Body.MU_EARTH)
    sma_err_pct = abs(sma_sim - sma_obs) / sma_obs * 100.0
    ecc_obs = _eccentricity(
        dataset.points[-1].position_m,
        dataset.points[-1].velocity_ms,
        Body.MU_EARTH,
    )
    ecc_sim = _eccentricity(sc.state.position, sc.state.velocity, Body.MU_EARTH)

    # ── RTN error decomposition ──────────────────────────────────────────────
    radial_errs = []
    along_errs = []
    cross_errs = []

    for sim_pos, obs_pos, obs_vel in zip(
        sim_pos_arr,
        obs_pos_arr,
        obs_vel_arr,
        strict=False,
    ):
        r_hat = obs_pos / (np.linalg.norm(obs_pos) + 1e-30)
        n_hat = np.cross(obs_pos, obs_vel)
        n_mag = np.linalg.norm(n_hat)
        if n_mag > 1e-10:
            n_hat /= n_mag
        else:
            n_hat = np.array([0.0, 0.0, 1.0])
        t_hat = np.cross(n_hat, r_hat)
        t_mag = np.linalg.norm(t_hat)
        if t_mag > 1e-10:
            t_hat /= t_mag
        else:
            t_hat = np.array([0.0, 1.0, 0.0])

        delta_pos = sim_pos - obs_pos
        radial_errs.append(abs(float(np.dot(delta_pos, r_hat))))
        along_errs.append(abs(float(np.dot(delta_pos, t_hat))))
        cross_errs.append(abs(float(np.dot(delta_pos, n_hat))))

    return ReplayMetrics(
        mission_name=dataset.name,
        duration_s=dataset.duration_s,
        n_points=len(dataset.points) - 1,
        pos_error_mean_m=float(pos_arr.mean()),
        pos_error_max_m=float(pos_arr.max()),
        pos_error_rms_m=float(np.sqrt((pos_arr**2).mean())),
        pos_error_pct=float((pos_arr / r_arr).mean() * 100.0),
        vel_error_mean_ms=float(vel_arr.mean()),
        vel_error_rms_ms=float(np.sqrt((vel_arr**2).mean())),
        sma_error_pct=sma_err_pct,
        eccentricity_err=abs(ecc_sim - ecc_obs),
        radial_error_mean_m=float(np.mean(radial_errs)) if radial_errs else 0.0,
        along_track_error_mean_m=float(np.mean(along_errs)) if along_errs else 0.0,
        cross_track_error_mean_m=float(np.mean(cross_errs)) if cross_errs else 0.0,
        times_s=times,
        pos_errors_m=pos_errors,
        vel_errors_ms=vel_errors,
    )
