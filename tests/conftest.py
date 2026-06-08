"""Shared fixtures, constants, and helpers for all physics test files."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cislunar.physics import (
    Action,
    GravityModel,
    IonThrusterModel,
    PowerBudgetModel,
    SolarSailModel,
    Spacecraft,
    SpacecraftState,
    make_lunar_craft,
)
from cislunar.physics.constants import AU, Body  # noqa: E402

# ── Shared constants ──────────────────────────────────────────────────────────

R_LEO = Body.R_EARTH + 400e3
V_CIRC_LEO = math.sqrt(Body.MU_EARTH / R_LEO)
ONE_DAY = 86_400.0
SUN_PLUS_X = np.array([AU, 0.0, 0.0])


# ── Shared helpers ────────────────────────────────────────────────────────────


def _integrate_for(
    sc: Spacecraft, total_s: float, step_s: float = 60.0, action: Action | None = None
) -> None:
    """Integrate *sc* forward by *total_s* seconds in fixed substeps."""
    act = action if action is not None else Action(throttle=0.0)
    remaining = total_s
    while remaining > 0:
        dt = min(step_s, remaining)
        sc.step(act, dt)
        remaining -= dt


# ── Shared fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def leo_spacecraft():
    """Fresh ``make_lunar_craft`` in a 400 km circular LEO.

    32 m² sail, 0.5 kg propellant — the standard reference spacecraft used
    in the majority of integration tests. Function-scoped because almost
    every caller mutates the state via ``sc.step()``.
    """
    return make_lunar_craft(
        position_m=[R_LEO, 0.0, 0.0],
        velocity_ms=[0.0, V_CIRC_LEO, 0.0],
        sail_area_m2=32.0,
        propellant_kg=0.5,
    )


@pytest.fixture
def make_leo_spacecraft():
    """Factory returning a fresh LEO spacecraft; accepts kwargs to override defaults."""

    def _make(**overrides):
        kwargs = dict(
            position_m=[R_LEO, 0.0, 0.0],
            velocity_ms=[0.0, V_CIRC_LEO, 0.0],
            sail_area_m2=32.0,
            propellant_kg=0.5,
        )
        kwargs.update(overrides)
        return make_lunar_craft(**kwargs)

    return _make


@pytest.fixture
def two_body_sc():
    """
    Minimal Earth-only spacecraft: point-mass gravity, zero SRP, no drag,
    no thrust. Used for conservation-law tests where any extra force would
    contaminate the measurement.
    """
    grav = GravityModel().add_body("Earth", mu=Body.MU_EARTH)
    state = SpacecraftState(
        position=np.array([R_LEO, 0.0, 0.0]),
        velocity=np.array([0.0, V_CIRC_LEO, 0.0]),
        propellant_kg=0.0,
        mass_dry_kg=12.0,
    )
    return Spacecraft(
        initial_state=state,
        sail_model=SolarSailModel(area_m2=0.0),
        thruster_model=IonThrusterModel(),
        gravity_model=grav,
        sun_pos_fn=lambda t: SUN_PLUS_X,
        power_model=PowerBudgetModel(),
    )
