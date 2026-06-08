"""
Regression tests for the public examples.

Each test calls the example's run() helper and asserts the signed,
feature-specific success metric that the example is named for.

Rationale: examples must positively demonstrate their named feature.
 - gto_periapsis_raise   : periapsis altitude must reach the 1 000 km target
 - solar_sail_leo        : ΔSMA must be positive (net orbit raising)
 - cislunar_transfer     : final SMA must exceed the initial GEO SMA
 - drag_decay_leo        : ΔSMA must be negative (drag-driven decay)
 - eclipse_power_gating  : eclipse duty cycle < sunlit duty cycle
"""

import sys
from pathlib import Path

# Examples are not installed as a package, so add the repo root to sys.path
# so they can be imported as plain modules.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ── Ion thruster — GTO periapsis raise ───────────────────────────────────────


def test_gto_periapsis_raise_hits_target():
    """Periapsis altitude must reach 1 000 km within 7 days."""
    from examples.gto_periapsis_raise import _TARGET_PERIAPSIS_M, R_E, run

    target_km = (_TARGET_PERIAPSIS_M - R_E) / 1e3  # 1 000 km
    periapsis_km, propellant_kg = run(days=7.0)

    assert periapsis_km >= target_km, (
        f"Periapsis {periapsis_km:.1f} km did not reach target {target_km:.0f} km"
    )
    assert propellant_kg > 0, "Thruster should have consumed propellant"
    assert propellant_kg < 3.0, "Propellant used should be less than the initial 3 kg load"


# ── Solar sail — LEO orbit raising ───────────────────────────────────────────


def test_solar_sail_raises_sma():
    """Solar sail must produce a net positive ΔSMA over 30 days at 585 km."""
    from examples.solar_sail_leo import run

    delta_sma_km = run(days=30, alt_km=585.0)

    assert delta_sma_km > 0, (
        f"SMA changed by {delta_sma_km:+.3f} km — expected positive orbit raising, got decay"
    )


# ── Ion + sail — cislunar transfer ───────────────────────────────────────────


def test_cislunar_transfer_raises_sma():
    """Final SMA must exceed the initial GEO SMA after 10 days of prograde thrust."""
    from examples.cislunar_transfer import _GEO_SMA_M, run

    geo_km = _GEO_SMA_M / 1e3
    final_sma_km, propellant_kg = run(days=10.0)

    assert final_sma_km > geo_km, (
        f"Final SMA {final_sma_km:.0f} km ≤ initial GEO SMA {geo_km:.0f} km — no orbit raising"
    )
    assert propellant_kg > 0, "Thruster should have consumed propellant"


# ── Atmospheric drag — LEO SMA decay ────────────────────────────────────────


def test_drag_decay_sma_decreases():
    """Atmospheric drag must produce a net negative ΔSMA over 5 days at 400 km."""
    from examples.drag_decay_leo import run

    delta_sma_km = run(days=5.0, alt_km=400.0)

    assert delta_sma_km < 0, (
        f"ΔSMA {delta_sma_km:+.3f} km is non-negative — drag should produce orbital decay"
    )


# ── Eclipse power gating ──────────────────────────────────────────────────────


def test_eclipse_suppresses_thrust():
    """Eclipse must cut the thruster on a larger fraction of steps than sunlit periods."""
    from examples.eclipse_power_gating import run

    eclipse_duty_cycle, sunlit_duty_cycle = run(orbits=3.0, alt_km=400.0)

    assert sunlit_duty_cycle > 0, "Thruster should fire during at least some sunlit steps"
    assert eclipse_duty_cycle < sunlit_duty_cycle, (
        f"Eclipse duty cycle {eclipse_duty_cycle:.0%} ≥ sunlit duty cycle "
        f"{sunlit_duty_cycle:.0%} — eclipse is not suppressing thrust"
    )
