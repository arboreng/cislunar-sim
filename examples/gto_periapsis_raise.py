"""
GTO periapsis raising with GVE guidance.

Raises a spacecraft from a geostationary transfer orbit (185 km × 35 786 km)
toward a 1 000 km periapsis using an ion thruster guided by
GVEPeriapsisGuidance.  Runs in roughly 25–30 seconds on a laptop.

Intended effect: periapsis altitude rises from 185 km to ≥ 1 000 km within
7 days.  The spacecraft starts at apogee so the GVE guidance burns near apogee
each orbit — the efficient arc for periapsis raising.

Run::

    python examples/gto_periapsis_raise.py
"""

import numpy as np

from cislunar.guidance import GVEPeriapsisGuidance
from cislunar.physics import Action, make_lunar_craft

# ── Orbital constants ─────────────────────────────────────────────────────────

R_E = 6.371e6  # Earth radius [m]
MU = 3.986004418e14  # Earth GM [m³/s²]

_TARGET_PERIAPSIS_M = R_E + 1_000e3  # 1 000 km periapsis target


def run(days: float = 7.0) -> tuple[float, float]:
    """
    Simulate GTO periapsis raising.

    Returns
    -------
    periapsis_alt_km : float
        Final periapsis altitude above Earth's surface [km].
    propellant_used_kg : float
        Propellant consumed during the manoeuvre [kg].
    """
    # GTO: 185 km periapsis, 35 786 km apogee — start from apogee
    r_p = R_E + 185e3
    r_a = R_E + 35_786e3
    v_a = np.sqrt(MU * 2 * r_p / (r_a * (r_p + r_a)))  # vis-viva at apogee

    sc = make_lunar_craft(
        position_m=[r_a, 0.0, 0.0],
        velocity_ms=[0.0, v_a, 0.0],
        sail_area_m2=0.0,  # ion-thruster-only; no sail
        propellant_kg=3.0,
        start_date="2025-01-01",
    )

    guidance = GVEPeriapsisGuidance(target_periapsis_m=_TARGET_PERIAPSIS_M)

    DT = 300.0  # 5-minute control steps
    N = int(86400 * days / DT)

    prop_initial = sc.state.propellant_kg
    for _ in range(N):
        thrust_dir, throttle = guidance.steer(sc.state.position, sc.state.velocity)
        sc.step(
            Action(attitude_dir_cmd=thrust_dir, thrust_dir=thrust_dir, throttle=throttle),
            dt_requested=DT,
        )

    r = np.linalg.norm(sc.state.position)
    v = np.linalg.norm(sc.state.velocity)
    a = 1.0 / (2.0 / r - v**2 / MU)
    e_vec = (
        np.cross(sc.state.velocity, np.cross(sc.state.position, sc.state.velocity)) / MU
        - sc.state.position / r
    )
    e = np.linalg.norm(e_vec)
    r_peri = a * (1.0 - e)

    periapsis_alt_km = (r_peri - R_E) / 1e3
    propellant_used_kg = prop_initial - sc.state.propellant_kg
    return periapsis_alt_km, propellant_used_kg


if __name__ == "__main__":
    DAYS = 7
    print(f"Simulating {DAYS}-day GTO periapsis raise …")
    peri_km, prop_kg = run(days=DAYS)
    print(f"  Periapsis altitude : {peri_km:8.1f} km  (target ≥ 1 000 km)")
    print(f"  Propellant used    : {prop_kg:.3f} kg")
