"""
Trajectory export utility for CCSDS Orbit Ephemeris Message (OEM) files.

Generate a cislunar-sim trajectory and write it as a CCSDS OEM v2 file that
can be imported directly into GMAT, Orekit, or STK for external validation.

Usage
-----
    # Generate a 3-day cislunar trajectory at 60-second steps
    python -m cislunar.validation.trajectory_export --days 3 --step 60 --out trajectory.oem

    # Generate at 10-second steps for a finer reference
    python -m cislunar.validation.trajectory_export --days 3 --step 10 --out trajectory_ref.oem

    # Print the OEM to stdout
    python -m cislunar.validation.trajectory_export --days 1 --step 60

CCSDS OEM format reference
---------------------------
CCSDS 502.0-B-3 (2023), "Orbit Data Messages".
The OEM encodes time-tagged Cartesian state vectors (position + velocity)
in a specified reference frame and time system.

Conventions used here
----------------------
  TIME_SYSTEM = TDB         (matches astropy ephemeris scale)
  REF_FRAME   = EME2000     (Earth-centred inertial, J2000 orientation)
                            Equivalent to ICRF for this application.
  CENTER      = EARTH
  Units: km and km/s (CCSDS standard)

GMAT import procedure
---------------------
1. Mission > Add New Ephemeris Propagator
2. Format: CCSDS_OEM_V2  File: trajectory.oem
3. Add a propagate command with the EphemerisPropagator pointed at that file
4. Compare final state vector with your reference propagation

Orekit import procedure
-----------------------
    from org.orekit.files.ccsds.ndm.odm.oem import OEMParser
    parser = OEMParser().withMissionReferenceDate(epoch)
    file   = parser.parseMessage(DataSource(Path("trajectory.oem")))
    # file.getSatellites()[id].getSegments() gives time-tagged states

External comparison protocol
-----------------------------
See validation/EXTERNAL_COMPARISON_PROTOCOL.md for the complete procedure
including force model configuration, tolerance criteria, and how to interpret
disagreements.
"""

from __future__ import annotations

import argparse
import math
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any, NamedTuple, cast

import numpy as np

# ── State record ──────────────────────────────────────────────────────────────


class StateVector(NamedTuple):
    """Time-tagged Cartesian state sample used for OEM export."""

    t_s: float  # simulation elapsed seconds from epoch
    pos_m: np.ndarray  # ECI position [m]   (shape 3)
    vel_ms: np.ndarray  # ECI velocity [m/s] (shape 3)


# ── Trajectory propagator ─────────────────────────────────────────────────────


def propagate(
    start_date: str = "2025-01-01",
    duration_days: float = 3.0,
    step_s: float = 60.0,
    altitude_km: float = 400.0,
    inclination_deg: float = 0.0,
    record_every: int = 1,  # record every Nth step
    verbose: bool = True,
) -> list[StateVector]:
    """
    Propagate a spacecraft from a circular LEO initial condition.

    Args:
        start_date:      ISO date string; sets the ephemeris epoch.
        duration_days:   Total propagation time [days].
        step_s:          Environment step size [s].
        altitude_km:     Initial circular orbit altitude [km].
        inclination_deg: Initial orbit inclination [deg].
        record_every:    Record state every N steps (default 1 = every step).
        verbose:         Print progress.

    Returns:
        List of StateVector namedtuples.
    """
    from cislunar.physics.constants import Body
    from cislunar.physics.forces.ephemeris import build_cislunar_ephemeris
    from cislunar.physics.forces.gravity import GravityModel
    from cislunar.physics.forces.propulsion import IonThrusterModel, SolarSailModel
    from cislunar.physics.spacecraft import Action, Spacecraft
    from cislunar.physics.state import SpacecraftState

    MU = Body.MU_EARTH
    R_E = Body.R_EARTH
    a = R_E + altitude_km * 1e3
    v_c = math.sqrt(MU / a)
    inc = math.radians(inclination_deg)

    if verbose:
        print(f"  Building ephemeris cache ({start_date}, {duration_days:.0f} days)...", flush=True)

    moon_fn, sun_fn = build_cislunar_ephemeris(
        start_time=start_date,
        duration_days=duration_days * 1.1 + 1.0,
        verbose=False,
    )
    grav = (
        GravityModel()
        .add_body("Earth", mu=MU, j2=Body.J2_EARTH, r_body=R_E)
        .add_body("Moon", mu=Body.MU_MOON, body_pos_fn=moon_fn)
        .add_body("Sun", mu=Body.MU_SUN, body_pos_fn=sun_fn)
    )
    sc = Spacecraft(
        initial_state=SpacecraftState(
            position=np.array([a, 0.0, 0.0]),
            velocity=np.array([0.0, v_c * math.cos(inc), v_c * math.sin(inc)]),
            sail_normal=np.array([1.0, 0.0, 0.0]),
            propellant_kg=0.5,
            mass_dry_kg=12.0,
        ),
        sail_model=SolarSailModel(area_m2=32.0),
        thruster_model=IonThrusterModel(),
        gravity_model=grav,
        sun_pos_fn=sun_fn,
    )

    total_s = duration_days * 86400.0
    n_steps = int(total_s / step_s)
    coast = Action()
    records: list[StateVector] = []

    # Record initial state
    records.append(
        StateVector(
            t_s=0.0,
            pos_m=sc.state.position.copy(),
            vel_ms=sc.state.velocity.copy(),
        )
    )

    if verbose:
        print(
            f"  Propagating {n_steps} steps × {step_s:.0f} s = {duration_days:.1f} days...",
            flush=True,
        )

    for step_i in range(1, n_steps + 1):
        sc.step(coast, dt_requested=step_s)
        if step_i % record_every == 0:
            records.append(
                StateVector(
                    t_s=sc.state.time_s,
                    pos_m=sc.state.position.copy(),
                    vel_ms=sc.state.velocity.copy(),
                )
            )
        if verbose and step_i % max(1, n_steps // 10) == 0:
            pct = step_i / n_steps * 100
            r = np.linalg.norm(sc.state.position) / 1e6
            print(f"    {pct:5.1f}%  r={r:.3f} Mm", flush=True)

    if verbose:
        print(f"  Done. {len(records)} state vectors recorded.")

    return records


# ── CCSDS OEM writer ──────────────────────────────────────────────────────────


@lru_cache(maxsize=16)
def _tdb_epoch_base(start_date: str):
    """Parse and cache the OEM epoch as an astropy TDB time."""
    from astropy.time import Time

    return Time(start_date, scale="tdb")


def _tdb_epoch_to_isot(start_date: str, offset_s: float) -> str:
    """
    Format an epoch as CCSDS ISO 8601 string: YYYY-MM-DDTHH:MM:SS.ffffff

    Args:
        start_date: ISO date of simulation epoch (TDB scale).
        offset_s:   Elapsed seconds from the epoch.
    """
    import astropy.units as u

    epoch_tdb = cast(Any, (_tdb_epoch_base(start_date) + float(offset_s) * u.s).tdb)
    epoch_tdb.precision = 6
    return str(epoch_tdb.isot)


def write_oem(
    records: list[StateVector],
    start_date: str,
    path: str | None,
    object_id: str = "CISLUNAR-SIM-1",
    object_name: str = "cislunar-sim reference spacecraft",
    step_s: float = 60.0,
) -> str:
    """
    Write a CCSDS OEM v2 file from a list of StateVector records.

    Args:
        records:     State history from propagate().
        start_date:  ISO epoch date (TDB).
        path:        Output file path.  None = return string only.
        object_id:   CCSDS OBJECT_ID field.
        object_name: CCSDS OBJECT_NAME field.
        step_s:      Nominal step size (written to USEABLE_STOP_TIME comment).

    Returns:
        The OEM content as a string.
    """
    created = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

    # OEM header
    lines = [
        "CCSDS_OEM_VERS = 2.0",
        f"CREATION_DATE  = {created}",
        "ORIGINATOR     = cislunar-sim",
        "",
        "META_START",
        f"  OBJECT_NAME  = {object_name}",
        f"  OBJECT_ID    = {object_id}",
        "  CENTER_NAME  = EARTH",
        "  REF_FRAME    = EME2000",
        "  TIME_SYSTEM  = TDB",
        f"  START_TIME   = {_tdb_epoch_to_isot(start_date, records[0].t_s)}",
        f"  STOP_TIME    = {_tdb_epoch_to_isot(start_date, records[-1].t_s)}",
        f"  COMMENT      Nominal step {step_s:.0f} s; "
        "Earth+Moon+Sun gravity; J2 Earth; astropy DE430 ephemeris",
        "  COMMENT      Generated by cislunar.validation.trajectory_export",
        "  COMMENT      See validation/EXTERNAL_COMPARISON_PROTOCOL.md "
        "for GMAT/Orekit import instructions",
        "META_STOP",
        "",
    ]

    # Data block — units: km and km/s (CCSDS standard)
    for sv in records:
        epoch = _tdb_epoch_to_isot(start_date, sv.t_s)
        px, py, pz = sv.pos_m / 1e3  # m → km
        vx, vy, vz = sv.vel_ms / 1e3  # m/s → km/s
        lines.append(
            f"{epoch}  {px:18.9f}  {py:18.9f}  {pz:18.9f}  {vx:15.12f}  {vy:15.12f}  {vz:15.12f}"
        )

    content = "\n".join(lines) + "\n"

    if path is not None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    return content


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    """Run the command-line OEM export utility."""
    parser = argparse.ArgumentParser(
        description="Export a cislunar-sim trajectory to CCSDS OEM format for "
        "external validation against GMAT, Orekit, or STK.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--days", type=float, default=3.0, help="Propagation duration [days] (default 3)"
    )
    parser.add_argument(
        "--step", type=float, default=60.0, help="Propagation step size [s] (default 60)"
    )
    parser.add_argument(
        "--alt", type=float, default=400.0, help="Initial altitude [km] (default 400)"
    )
    parser.add_argument(
        "--inc", type=float, default=0.0, help="Initial inclination [deg] (default 0)"
    )
    parser.add_argument(
        "--epoch", type=str, default="2025-01-01", help="Start date ISO string (default 2025-01-01)"
    )
    parser.add_argument(
        "--record", type=int, default=1, help="Record every Nth step (default 1 = every step)"
    )
    parser.add_argument(
        "--out", type=str, default=None, help="Output OEM file path (omit to print to stdout)"
    )
    parser.add_argument("--object-id", default="CISLUNAR-SIM-1")
    parser.add_argument("--object-name", default="cislunar-sim reference spacecraft")
    args = parser.parse_args()

    records = propagate(
        start_date=args.epoch,
        duration_days=args.days,
        step_s=args.step,
        altitude_km=args.alt,
        inclination_deg=args.inc,
        record_every=args.record,
        verbose=args.out is not None,  # quiet when printing to stdout
    )

    oem = write_oem(
        records=records,
        start_date=args.epoch,
        path=args.out,
        object_id=args.object_id,
        object_name=args.object_name,
        step_s=args.step,
    )

    if args.out is None:
        print(oem)
    else:
        n_points = len(records)
        duration = records[-1].t_s / 86400.0
        final_r = np.linalg.norm(records[-1].pos_m) / 1e6
        print(f"  Written: {args.out}")
        print(f"  Points:  {n_points}  Duration: {duration:.2f} days  Final r: {final_r:.3f} Mm")


if __name__ == "__main__":
    main()
