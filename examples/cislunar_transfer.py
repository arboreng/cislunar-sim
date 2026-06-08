"""
Trans-lunar trajectory with combined ion + sail propulsion.

Demonstrates a spacecraft departing GEO and coasting toward lunar distance
with the Sun-Moon-Earth gravity model, real DE430 ephemeris, and atmospheric
drag active (though negligible at GEO altitude and above).

Intended effect: semi-major axis grows beyond GEO as the ion thruster
continuously accelerates the spacecraft prograde.  Final SMA is always
larger than the initial GEO SMA.

Prints periodic state snapshots and the final orbital elements.

Run::

    python examples/cislunar_transfer.py
"""

import numpy as np

from cislunar.physics import Action, make_lunar_craft
from cislunar.physics.constants import Body

# ── Initial state: GEO ────────────────────────────────────────────────────────

R_E = Body.R_EARTH
MU = Body.MU_EARTH

_GEO_ALT_M = 35_786e3
_GEO_SMA_M = R_E + _GEO_ALT_M


def run(days: float = 10.0) -> tuple[float, float]:
    """
    Simulate a prograde-thrust cislunar transfer from GEO.

    Returns
    -------
    final_sma_km : float
        Final semi-major axis [km].
    propellant_used_kg : float
        Propellant consumed [kg].
    """
    r_geo = _GEO_SMA_M
    v_geo = np.sqrt(MU / r_geo)

    sc = make_lunar_craft(
        position_m=[r_geo, 0.0, 0.0],
        velocity_ms=[0.0, v_geo, 0.0],
        sail_area_m2=64.0,  # 8 m × 8 m sail — larger than LightSail 2
        propellant_kg=5.0,
        start_date="2025-03-20",  # near vernal equinox for favourable Sun geometry
    )

    prop_initial = sc.state.propellant_kg
    DT = 600.0  # 10-minute steps
    N = int(86400 * days / DT)

    for _ in range(N):
        vel_hat = sc.state.velocity / (np.linalg.norm(sc.state.velocity) + 1e-30)
        sc.step(
            Action(
                attitude_dir_cmd=vel_hat,
                thrust_dir=vel_hat,
                throttle=1.0,
            ),
            dt_requested=DT,
        )

    r = np.linalg.norm(sc.state.position)
    v = np.linalg.norm(sc.state.velocity)
    sma = 1.0 / (2.0 / r - v**2 / MU)

    return sma / 1e3, prop_initial - sc.state.propellant_kg


if __name__ == "__main__":
    DAYS = 10
    print(f"Simulating {DAYS}-day cislunar transfer …\n")

    r_geo = _GEO_SMA_M
    v_geo = np.sqrt(MU / r_geo)

    sc = make_lunar_craft(
        position_m=[r_geo, 0.0, 0.0],
        velocity_ms=[0.0, v_geo, 0.0],
        sail_area_m2=64.0,
        propellant_kg=5.0,
        start_date="2025-03-20",
    )

    DT = 600.0
    DAYS_SIM = DAYS
    N = int(86400 * DAYS_SIM / DT)
    REPORT = N // 10

    print(f"  {'Day':>5}  {'Alt [km]':>10}  {'SMA [km]':>10}  {'Prop [kg]':>10}")
    print("  " + "-" * 42)

    for i in range(N):
        vel_hat = sc.state.velocity / (np.linalg.norm(sc.state.velocity) + 1e-30)
        sc.step(
            Action(attitude_dir_cmd=vel_hat, thrust_dir=vel_hat, throttle=1.0),
            dt_requested=DT,
        )
        if (i + 1) % REPORT == 0:
            r = np.linalg.norm(sc.state.position)
            v = np.linalg.norm(sc.state.velocity)
            sma = 1.0 / (2.0 / r - v**2 / MU)
            day = sc.state.time_s / 86400.0
            print(
                f"  {day:>5.1f}  {(r - R_E) / 1e3:>10.0f}  "
                f"{sma / 1e3:>10.0f}  {sc.state.propellant_kg:>10.3f}"
            )

    r = np.linalg.norm(sc.state.position)
    v = np.linalg.norm(sc.state.velocity)
    sma = 1.0 / (2.0 / r - v**2 / MU)
    e_vec = (
        np.cross(sc.state.velocity, np.cross(sc.state.position, sc.state.velocity)) / MU
        - sc.state.position / r
    )
    ecc = np.linalg.norm(e_vec)

    print(f"\n  Final SMA         : {sma / 1e3:.0f} km  (initial {_GEO_SMA_M / 1e3:.0f} km)")
    print(f"  Final eccentricity: {ecc:.4f}")
    print(f"  Periapsis altitude: {(sma * (1 - ecc) - R_E) / 1e3:.0f} km")
    print(f"  Apoapsis altitude : {(sma * (1 + ecc) - R_E) / 1e3:.0f} km")
    print(f"  Propellant used   : {5.0 - sc.state.propellant_kg:.3f} kg")
