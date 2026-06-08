"""
Eclipse-aware power gating for an ion thruster at 400 km LEO.

Demonstrates two effects of the eclipse/power model:
  1. **Eclipse suppresses thrust.** During Earth's shadow the solar array
     produces zero power; the battery discharges and eventually hits its
     depth-of-discharge floor, cutting the thruster.  The thruster fires on
     a smaller fraction of eclipse steps than sunlit steps.
  2. **Solar array recovers power after eclipse exit.** P_solar returns to
     its full value immediately on exiting the shadow, recharging the battery
     and restoring full thrust capability.

Battery starts at 20 % SoC (exactly at the depth-of-discharge floor).  During
the first sunlit arc the array recharges the battery, then eclipse drains it
until the thruster cuts again.

Intended effect: eclipse_duty_cycle < sunlit_duty_cycle.  Over 3 orbits at
400 km the eclipse fraction is ≈ 39 %, so sunlit_duty_cycle ≈ 1.0 and
eclipse_duty_cycle < 1.0.

Run::

    python examples/eclipse_power_gating.py
"""

import numpy as np

from cislunar.physics import Action, make_lunar_craft
from cislunar.physics.constants import Body

R_E = Body.R_EARTH
MU = Body.MU_EARTH


def run(orbits: float = 3.0, alt_km: float = 400.0) -> tuple[float, float]:
    """
    Fire thruster prograde for ``orbits`` orbits at ``alt_km``.

    Battery starts at the depth-of-discharge floor; solar panels are
    modelled as sun-tracking (decouple_attitude=True) for a clean
    P_solar signal independent of spacecraft pointing.

    Returns
    -------
    eclipse_duty_cycle : float
        Fraction of eclipse steps where P_thruster_w > 0 (< 1.0 expected).
    sunlit_duty_cycle : float
        Fraction of sunlit steps where P_thruster_w > 0 (≈ 1.0 expected).
    """
    alt_m = alt_km * 1e3
    r = R_E + alt_m
    v_c = np.sqrt(MU / r)
    period_s = 2 * np.pi * np.sqrt(r**3 / MU)

    sc = make_lunar_craft(
        position_m=[r, 0.0, 0.0],
        velocity_ms=[0.0, v_c, 0.0],
        sail_area_m2=0.0,  # no sail — isolate thruster/power behaviour
        propellant_kg=3.0,
        start_date="2025-06-01",
    )

    # Start at the depth-of-discharge floor so the first eclipse
    # demonstrates battery-floor cut rather than immediate cut.
    sc.power.reset(soc=0.20)
    # Decouple panel angle from spacecraft attitude for a clean P_solar signal.
    sc.power.array.decouple_attitude = True

    dt = 30.0
    n_steps = int(orbits * period_s / dt)

    eclipse_thrusting = 0
    eclipse_steps = 0
    sunlit_thrusting = 0
    sunlit_steps = 0

    for _ in range(n_steps):
        vel_hat = sc.state.velocity / np.linalg.norm(sc.state.velocity)
        sc.step(
            Action(attitude_dir_cmd=vel_hat, thrust_dir=vel_hat, throttle=1.0),
            dt_requested=dt,
        )
        if sc.power.in_eclipse:
            eclipse_steps += 1
            if sc.power.P_thruster_w > 0.0:
                eclipse_thrusting += 1
        else:
            sunlit_steps += 1
            if sc.power.P_thruster_w > 0.0:
                sunlit_thrusting += 1

    return (
        eclipse_thrusting / max(1, eclipse_steps),
        sunlit_thrusting / max(1, sunlit_steps),
    )


if __name__ == "__main__":
    ORBITS = 3.0
    ALT = 400.0

    alt_m = ALT * 1e3
    r = R_E + alt_m
    v_c = np.sqrt(MU / r)
    period_s = 2 * np.pi * np.sqrt(r**3 / MU)

    sc = make_lunar_craft(
        position_m=[r, 0.0, 0.0],
        velocity_ms=[0.0, v_c, 0.0],
        sail_area_m2=0.0,
        propellant_kg=3.0,
        start_date="2025-06-01",
    )
    sc.power.reset(soc=0.20)
    sc.power.array.decouple_attitude = True

    dt = 30.0
    n_steps = int(ORBITS * period_s / dt)
    report = max(1, n_steps // 20)

    print(f"Eclipse power gating — {ORBITS:.0f} orbits at {ALT:.0f} km")
    print("  Battery start: 20% SoC (at depth-of-discharge floor)\n")
    print(
        f"  {'Time [min]':>12}  {'Eclipse':>8}  {'P_solar [W]':>12}  "
        f"{'P_thr [W]':>10}  {'Batt [%]':>8}"
    )
    print("  " + "-" * 60)

    eclipse_thrusting = 0
    eclipse_steps = 0
    sunlit_thrusting = 0
    sunlit_steps = 0

    for i in range(n_steps):
        vel_hat = sc.state.velocity / np.linalg.norm(sc.state.velocity)
        sc.step(
            Action(attitude_dir_cmd=vel_hat, thrust_dir=vel_hat, throttle=1.0),
            dt_requested=dt,
        )
        if sc.power.in_eclipse:
            eclipse_steps += 1
            if sc.power.P_thruster_w > 0.0:
                eclipse_thrusting += 1
        else:
            sunlit_steps += 1
            if sc.power.P_thruster_w > 0.0:
                sunlit_thrusting += 1

        if (i + 1) % report == 0:
            t_min = sc.state.time_s / 60.0
            ecl = "yes" if sc.power.in_eclipse else "no"
            print(
                f"  {t_min:>12.1f}  {ecl:>8}  {sc.power.P_solar_w:>12.1f}  "
                f"{sc.power.P_thruster_w:>10.1f}  {sc.power.battery_pct:>8.1f}"
            )

    ecl_duty = eclipse_thrusting / max(1, eclipse_steps)
    sun_duty = sunlit_thrusting / max(1, sunlit_steps)
    print(f"\n  Eclipse duty cycle : {ecl_duty:.0%}  (< sunlit duty cycle = {sun_duty:.0%})")
    print(f"  Eclipse suppresses thrust: {ecl_duty < sun_duty}")
