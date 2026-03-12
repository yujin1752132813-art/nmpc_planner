from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from config.defaults import SolverConfig, SimConfig
from .scenario_builder import Scenario
from .types import EgoState, RefPoint


@dataclass
class ReferenceManager:
    scenario: Scenario
    solver_cfg: SolverConfig
    sim_cfg: SimConfig
    current_s: float = 0.0

    def update_progress(self, ego: EgoState) -> float:
        """Estimate current path progress with:
        1) nearest-point search in a local s window
        2) tangential projection for sub-grid continuous progress
        3) monotonic non-decreasing guard

        This avoids the low-speed 'sticking' problem caused by using only
        discrete nearest-point indices on a coarse path grid.
        """
        s_grid = self.scenario.s_grid
        x_grid = self.scenario.x_grid
        y_grid = self.scenario.y_grid
        yaw_grid = self.scenario.yaw_grid

        # search around previous progress, but do not allow large backward window
        # backward window is kept very small only for numerical robustness
        back = 0.25
        front = self.sim_cfg.reference_search_margin

        s_lo = max(self.current_s - back, 0.0)
        s_hi = min(self.current_s + front, self.scenario.total_length)

        idx_lo = int(np.searchsorted(s_grid, s_lo))
        idx_hi = int(np.searchsorted(s_grid, s_hi))
        idx_hi = max(idx_hi, idx_lo + 2)

        xs = x_grid[idx_lo:idx_hi]
        ys = y_grid[idx_lo:idx_hi]
        dist2 = (xs - ego.x) ** 2 + (ys - ego.y) ** 2
        local_idx = int(np.argmin(dist2))
        best_idx = idx_lo + local_idx

        # nearest discrete path point
        s_ref = float(s_grid[best_idx])
        x_ref = float(x_grid[best_idx])
        y_ref = float(y_grid[best_idx])
        yaw_ref = float(yaw_grid[best_idx])

        # tangential projection to get continuous progress
        tx = float(np.cos(yaw_ref))
        ty = float(np.sin(yaw_ref))
        dx = float(ego.x - x_ref)
        dy = float(ego.y - y_ref)
        ds_tangent = dx * tx + dy * ty

        s_proj = s_ref + ds_tangent

        # clamp to valid range
        s_proj = float(np.clip(s_proj, 0.0, self.scenario.total_length))

        # keep progress approximately monotonic
        # allow no backward movement in normal forward driving
        if s_proj < self.current_s:
            s_proj = self.current_s

        self.current_s = s_proj
        return self.current_s

    def build_local_reference(self, ego: EgoState) -> List[RefPoint]:
        s0 = self.update_progress(ego)
        refs = []
        s = s0

        for _ in range(self.solver_cfg.horizon + 1):
            r = self.sample(s)
            refs.append(r)

            # IMPORTANT:
            # advance future reference with the actual reference speed,
            # not with max(ref_v, 1.0). Otherwise low-speed start/stop
            # phases become geometrically inconsistent.
            ds = max(r.v_ref, 1.0) * self.solver_cfg.dt
            s = min(s + ds, self.scenario.total_length)

        return refs

    def sample(self, s_query: float) -> RefPoint:
        s_query = float(np.clip(s_query, 0.0, self.scenario.total_length))
        i = int(np.searchsorted(self.scenario.s_grid, s_query))

        if i <= 0:
            return self.scenario.path[0]
        if i >= len(self.scenario.s_grid):
            return self.scenario.path[-1]

        s0 = self.scenario.s_grid[i - 1]
        s1 = self.scenario.s_grid[i]
        alpha = 0.0 if s1 == s0 else (s_query - s0) / (s1 - s0)

        def lerp(a, b):
            return float(a + alpha * (b - a))

        return RefPoint(
            s=s_query,
            x=lerp(self.scenario.x_grid[i - 1], self.scenario.x_grid[i]),
            y=lerp(self.scenario.y_grid[i - 1], self.scenario.y_grid[i]),
            yaw=lerp(self.scenario.yaw_grid[i - 1], self.scenario.yaw_grid[i]),
            kappa=lerp(self.scenario.kappa_grid[i - 1], self.scenario.kappa_grid[i]),
            v_ref=lerp(self.scenario.v_grid[i - 1], self.scenario.v_grid[i]),
        )