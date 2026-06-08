"""
Solar sail orbit raising from LEO — force-isolation demo.

Simulates a LightSail 2-class spacecraft (32 m² sail, 585 km circular LEO)
using the bang-bang orbit-raising steering law.  The sail raises its
semi-major axis over 30 days in spite of atmospheric drag.

Setup note: this is an idealised, force-isolation scenario.  The dry mass
(12 kg) is heavier than the real LightSail 2 (~5 kg), and F10.7 = 150 (solar
mean) produces more drag than the 2019 solar-minimum conditions in which
LightSail 2 flew.  Despite these conservative choices, SRP dominates at
585 km and yields a net positive ΔSMA.

Steering law
────────────
The optimal flat-plate steering for orbit raising bisects the *toward-Sun*
and *retrograde* directions.  This orients the sail so that the SRP normal
force has a prograde component everywhere in the retrograde-Sun half of the
orbit.  During the prograde-Sun half the sail is turned edge-on (no force)
to avoid any retrograde penalty and to reduce drag cross-section.

Common pitfall: using ``sail_normal = sun_hat + vel_hat`` (bisect toward-Sun
and *prograde*) gives a retrograde SRP force everywhere and will produce
neutral or decaying orbits.

Run::

    python examples/solar_sail_leo.py
"""

import numpy as np

from cislunar.physics import Action, build_cislunar_ephemeris, make_lunar_craft

# ── Orbital constants ─────────────────────────────────────────────────────────

R_E = 6.371e6
MU = 3.986004418e14

START_DATE = "2025-06-01"


def run(days: float = 30.0, alt_km: float = 585.0) -> float:
    """
    Simulate solar-sail orbit raising from circular LEO.

    Parameters
    ----------
    days :
        Simulation duration [days].
    alt_km :
        Initial circular orbit altitude [km].

    Returns
    -------
    delta_sma_km : float
        Change in semi-major axis over the simulation [km].
        Positive → orbit raised; negative → orbit decayed.
    """
    r = R_E + alt_km * 1e3
    v_c = np.sqrt(MU / r)

    _, sun_pos = build_cislunar_ephemeris(start_time=START_DATE, duration_days=days + 5.0)

    sc = make_lunar_craft(
        position_m=[r, 0.0, 0.0],
        velocity_ms=[0.0, v_c, 0.0],
        sail_area_m2=32.0,
        propellant_kg=0.0,  # pure sail — no ion thruster
        start_date=START_DATE,
    )

    sma_initial = r
    DT = 60.0
    N = int(86400 * days / DT)

    for _ in range(N):
        t = sc.state.time_s

        # Unit vectors
        sun_hat = sun_pos(t) - sc.state.position  # toward Sun
        sun_hat /= np.linalg.norm(sun_hat)
        vel_hat = sc.state.velocity / np.linalg.norm(sc.state.velocity)

        if np.dot(sun_hat, vel_hat) < 0:
            # Sun in retrograde hemisphere: bisect toward-Sun and retrograde.
            # The SRP normal force points in -sail_normal, which has a
            # prograde component proportional to sin(φ/2) where φ is the
            # Sun–prograde angle (always positive in this arc).
            sail_normal = sun_hat - vel_hat
            sail_normal /= np.linalg.norm(sail_normal)
        else:
            # Sun in prograde hemisphere: turn edge-on to avoid retrograde
            # SRP impulse and to reduce drag cross-section.
            orb_normal = np.cross(sc.state.position, sc.state.velocity)
            orb_normal /= np.linalg.norm(orb_normal)
            sail_normal = np.cross(orb_normal, sun_hat)
            sail_normal /= np.linalg.norm(sail_normal)

        sc.step(
            Action(attitude_dir_cmd=sail_normal, thrust_dir=sail_normal, throttle=0.0),
            dt_requested=DT,
        )

    r_f = np.linalg.norm(sc.state.position)
    v_f = np.linalg.norm(sc.state.velocity)
    sma_f = 1.0 / (2.0 / r_f - v_f**2 / MU)

    return (sma_f - sma_initial) / 1e3


if __name__ == "__main__":
    DAYS = 30
    ALT_KM = 585.0
    print(f"Simulating {DAYS}-day solar sail arc from {ALT_KM:.0f} km …")
    delta_km = run(days=DAYS, alt_km=ALT_KM)
    print(f"  SMA change : {delta_km:+.2f} km over {DAYS} days")
    print("  (positive → orbit raised; negative → drag-dominated decay)")
