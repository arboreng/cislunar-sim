"""
Standard Event factories for the Spacecraft integrator.

Each function returns a configured Event object ready to pass to
Spacecraft.step(events=[...]).

Usage
-----
from cislunar.physics.forces.events import CheckpointEvent, EclipseEvent, SurfaceEvent
from cislunar.physics.spacecraft import Spacecraft

events = [
    CheckpointEvent("LEO Gate", cp_pos=np.array([7e6, 0, 0]), cp_r=5000.0),
    EclipseEvent(sun_pos_fn=sun_fn, body_radius_m=R_EARTH),
    SurfaceEvent("Earth", body_pos_fn=lambda t: np.zeros(3), body_radius_m=R_EARTH),
]
new_state, triggered = sc.step(action, dt_requested=60.0, events=events)
for ev in triggered:
    print(f"{ev.name} at t={ev.t_event:.1f} s")

Design notes
------------
All event functions follow the sign convention:
  fn(y, t) > 0  →  "outside" / "before" the event
  fn(y, t) < 0  →  "inside"  / "after"  the event
  fn(y, t) = 0  →  event boundary

This means direction="falling" detects entry (outside → inside) and
direction="rising" detects exit (inside → outside).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..spacecraft import Event

# ── Checkpoint crossing ────────────────────────────────────────────────────────


def CheckpointEvent(
    name: str,
    cp_pos: NDArray,
    cp_r: float,
    tol_s: float = 1.0,
) -> Event:
    """
    Detects when the spacecraft enters the checkpoint sphere.

    fn = ‖pos − cp_pos‖ − cp_r

    Positive outside the sphere, negative inside.  direction="falling"
    catches the entry crossing (outside → inside).

    Args:
        name:   Gate label (e.g. "LEO Gate", "Lunar Gate").
        cp_pos: Checkpoint centre in ECI [m].
        cp_r:   Checkpoint radius [m].
        tol_s:  Bisection convergence tolerance [s].  Default 1 s is
                sufficient for all checkpoint gate timings.
    """
    _cp = np.asarray(cp_pos, dtype=float)
    _r = float(cp_r)

    def _fn(y: NDArray, t: float) -> float:
        return float(np.linalg.norm(y[0:3] - _cp)) - _r

    return Event(
        name=name,
        fn=_fn,
        direction="falling",  # outside → inside
        terminal=False,  # don't stop integration; just record time
        tol_s=tol_s,
    )


# ── Eclipse entry / exit ───────────────────────────────────────────────────────


def EclipseEvent(
    sun_pos_fn: Callable[[float], NDArray],
    body_pos_fn: Callable[[float], NDArray] | None = None,
    body_radius_m: float = 6.371e6,  # Earth radius default
    tol_s: float = 1.0,
    name: str = "eclipse",
    detect: str = "both",  # "entry", "exit", or "both"
) -> Event:
    """
    Detects eclipse entry and/or exit using the cylindrical shadow model.

    fn = shadow_cylinder_signed(pos, sun_pos, body_pos, body_r)

    Returns a positive value in sunlight and negative in shadow.
    The sign is the dot product of (body → spacecraft) with (body → Sun),
    weighted so that the cylindrical shadow boundary is the zero.

    Specifically:
        sun_dir = (sun_pos - body_pos) / |...|
        proj    = dot(pos - body_pos, sun_dir)          # along-sun distance
        perp    = (pos - body_pos) - proj * sun_dir     # cross-track distance
        fn      = |perp| - body_r      if proj < 0       (behind the body)
                = +inf                  if proj >= 0      (sunward side, never eclipsed)

    This is negative inside the cylindrical shadow and positive outside.

    Args:
        sun_pos_fn:    Callable(t) → Sun ECI position [m].
        body_pos_fn:   Callable(t) → Occulting body ECI position [m].
                       Pass None for Earth (body at ECI origin).
        body_radius_m: Occulting body radius [m].
        tol_s:         Bisection convergence tolerance [s].
        name:          Event label.
        detect:        "entry" (sunlight→shadow), "exit" (shadow→sunlight),
                       or "both".
    """
    _R = float(body_radius_m)
    if detect == "entry":
        _direction: Literal["rising", "falling", "both"] = "falling"
    elif detect == "exit":
        _direction = "rising"
    else:
        _direction = "both"

    def _fn(y: NDArray, t: float) -> float:
        pos = y[0:3]
        sun_pos = sun_pos_fn(t)
        body_pos = body_pos_fn(t) if body_pos_fn is not None else np.zeros(3)

        sun_dir = sun_pos - body_pos
        sun_dist = float(np.linalg.norm(sun_dir))
        if sun_dist < 1.0:
            return 1.0  # degenerate: treat as sunlit
        sun_hat = sun_dir / sun_dist

        rel = pos - body_pos
        proj = float(np.dot(rel, sun_hat))

        if proj >= 0.0:
            # Spacecraft is on the sunward side of the body — always sunlit
            return float(np.linalg.norm(rel)) - _R  # positive outside body

        perp_vec = rel - proj * sun_hat
        perp = float(np.linalg.norm(perp_vec))
        return perp - _R  # negative inside cylindrical shadow

    return Event(name=name, fn=_fn, direction=_direction, terminal=False, tol_s=tol_s)


# ── Surface / reentry crossing ─────────────────────────────────────────────────


def SurfaceEvent(
    name: str,
    body_pos_fn: Callable[[float], NDArray] | None = None,
    body_radius_m: float = 6.371e6 + 80e3,  # Earth reentry threshold default
    tol_s: float = 1.0,
) -> Event:
    """
    Detects when the spacecraft crosses inside a body's surface (or reentry
    threshold), using a terminal event to halt integration at the crossing.

    fn = ‖pos − body_pos‖ − body_radius_m

    Positive outside, negative inside.  direction="falling" catches entry.
    terminal=True means integration stops at the crossing time, giving an
    accurate impact time and state without overshooting into the body.

    Args:
        name:          Event label (e.g. "Earth reentry", "Moon impact").
        body_pos_fn:   Callable(t) → body ECI position [m].
                       Pass None for Earth (body at ECI origin).
        body_radius_m: Threshold radius.  For Earth, pass R_EARTH + 80e3
                       (Kármán line) rather than R_EARTH itself.
        tol_s:         Bisection convergence tolerance [s].
    """
    _R = float(body_radius_m)

    def _fn(y: NDArray, t: float) -> float:
        pos = y[0:3]
        body_pos = body_pos_fn(t) if body_pos_fn is not None else np.zeros(3)
        return float(np.linalg.norm(pos - body_pos)) - _R

    return Event(
        name=name,
        fn=_fn,
        direction="falling",
        terminal=True,  # stop integration at impact — no overshoot
        tol_s=tol_s,
    )
