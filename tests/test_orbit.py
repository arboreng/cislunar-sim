"""Full-stack orbit integration tests."""

from __future__ import annotations

import numpy as np
from conftest import V_CIRC_LEO

from cislunar.physics import Action


# ══════════════════════════════════════════════════════════════════════════════
# Section 5 — Full-stack 10-minute orbit integration
# ══════════════════════════════════════════════════════════════════════════════
class TestOrbitIntegration:
    """Does a 10-minute orbit segment obey orbital mechanics at the top level?"""

    def test_position_moves_less_than_one_arc(self, leo_spacecraft):
        state0 = leo_spacecraft.state.copy()
        action = Action(attitude_dir_cmd=np.array([-1.0, 0.0, 0.0]), throttle=0.0)
        state1, _ = leo_spacecraft.step(action, dt_requested=600.0)
        arc_length_m = V_CIRC_LEO * 600.0
        pos_change_m = np.linalg.norm(state1.position - state0.position)
        assert pos_change_m < arc_length_m, (
            f"moved {pos_change_m / 1e3:.1f} km > arc {arc_length_m / 1e3:.1f} km"
        )

    def test_orbital_radius_stable_over_10min(self, leo_spacecraft):
        state0 = leo_spacecraft.state.copy()
        action = Action(attitude_dir_cmd=np.array([-1.0, 0.0, 0.0]), throttle=0.0)
        state1, _ = leo_spacecraft.step(action, dt_requested=600.0)
        r0 = np.linalg.norm(state0.position)
        r1 = np.linalg.norm(state1.position)
        assert abs(r1 - r0) < 5000, f"Δr={abs(r1 - r0):.0f} m"

    def test_elapsed_time_matches_request(self, leo_spacecraft):
        action = Action(attitude_dir_cmd=np.array([-1.0, 0.0, 0.0]), throttle=0.0)
        state1, _ = leo_spacecraft.step(action, dt_requested=600.0)
        assert abs(state1.time_s - 600.0) < 1.0, f"t={state1.time_s:.2f} s"
