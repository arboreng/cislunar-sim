"""
Atmospheric drag orbital decay at 400 km LEO.

Demonstrates the force model's drag implementation: a 12U CubeSat with a
32 m² solar sail deployed in the edge-on (minimum-drag) attitude loses SMA
over time due to NRLMSISE-00 atmospheric drag.

Intended effect: semi-major axis decreases measurably after 5 days at 400 km.
The example coasts (throttle=0) with attitude held edge-on to the velocity
vector, so only the 0.077 m² CubeSat bus area contributes drag.  ΔSMA after
5 days is on the order of −0.1 km — small but strictly negative.  The
spacecraft bus area alone cannot produce a multi-kilometre decay over 5 days at
400 km; a 32 m² sail in ram orientation would show a much larger decay rate.

Run::

    python examples/drag_decay_leo.py
"""

import numpy as np

from cislunar.physics import Action, make_lunar_craft
from cislunar.physics.constants import Body

R_E = Body.R_EARTH
MU = Body.MU_EARTH

_LEO_ALT_M = 400e3  # ISS-like altitude: strong drag signal


def run(days: float = 5.0, alt_km: float = 400.0) -> float:
    """
    Coast a sail spacecraft at low altitude and measure SMA decay from drag.

    Returns
    -------
    delta_sma_km : float
        Change in semi-major axis [km].  Negative means decay (expected).
    """
    alt_m = alt_km * 1e3
    r = R_E + alt_m
    v_c = np.sqrt(MU / r)

    sc = make_lunar_craft(
        position_m=[r, 0.0, 0.0],
        velocity_ms=[0.0, v_c, 0.0],
        sail_area_m2=32.0,
        propellant_kg=0.0,  # pure coast — no thrust
        start_date="2025-06-01",
    )

    sma_initial = r  # circular orbit → SMA = radius

    dt = 60.0
    n_steps = int(86400 * days / dt)

    for _ in range(n_steps):
        # Hold sail edge-on to velocity to minimise sail drag contribution;
        # this isolates the body cross-section drag (0.077 m² ram area).
        vel_hat = sc.state.velocity / np.linalg.norm(sc.state.velocity)
        orb_normal = np.cross(sc.state.position, sc.state.velocity)
        orb_normal /= np.linalg.norm(orb_normal)
        edge_on = orb_normal  # sail normal perpendicular to velocity → cos(θ)=0

        sc.step(
            Action(attitude_dir_cmd=edge_on, thrust_dir=vel_hat, throttle=0.0),
            dt_requested=dt,
        )

    r_f = np.linalg.norm(sc.state.position)
    v_f = np.linalg.norm(sc.state.velocity)
    sma_final = 1.0 / (2.0 / r_f - v_f**2 / MU)

    return (sma_final - sma_initial) / 1e3


if __name__ == "__main__":
    DAYS = 5
    ALT = 400.0

    r = R_E + ALT * 1e3
    v_c = np.sqrt(MU / r)

    sc = make_lunar_craft(
        position_m=[r, 0.0, 0.0],
        velocity_ms=[0.0, v_c, 0.0],
        sail_area_m2=32.0,
        propellant_kg=0.0,
        start_date="2025-06-01",
    )

    sma_initial = r
    dt = 60.0
    n_steps = int(86400 * DAYS / dt)
    report = n_steps // DAYS

    print(f"Drag decay at {ALT:.0f} km LEO, {DAYS} days, edge-on attitude\n")
    print(f"  {'Day':>5}  {'Alt [km]':>10}  {'ΔSMA [km]':>10}")
    print("  " + "-" * 30)

    for i in range(n_steps):
        vel_hat = sc.state.velocity / np.linalg.norm(sc.state.velocity)
        orb_normal = np.cross(sc.state.position, sc.state.velocity)
        orb_normal /= np.linalg.norm(orb_normal)
        edge_on = orb_normal

        sc.step(
            Action(attitude_dir_cmd=edge_on, thrust_dir=vel_hat, throttle=0.0),
            dt_requested=dt,
        )

        if (i + 1) % report == 0:
            r_f = np.linalg.norm(sc.state.position)
            v_f = np.linalg.norm(sc.state.velocity)
            sma = 1.0 / (2.0 / r_f - v_f**2 / MU)
            day = sc.state.time_s / 86400.0
            print(f"  {day:>5.1f}  {(r_f - R_E) / 1e3:>10.1f}  {(sma - sma_initial) / 1e3:>+10.3f}")

    r_f = np.linalg.norm(sc.state.position)
    v_f = np.linalg.norm(sc.state.velocity)
    sma_final = 1.0 / (2.0 / r_f - v_f**2 / MU)
    delta_sma_km = (sma_final - sma_initial) / 1e3

    print(f"\n  ΔSMA after {DAYS} days: {delta_sma_km:+.3f} km  (expect < 0 km from drag)")
