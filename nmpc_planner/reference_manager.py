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
        s_grid = self.scenario.s_grid
        x_grid = self.scenario.x_grid
        y_grid = self.scenario.y_grid
        back = 3.0
        front = self.sim_cfg.reference_search_margin

        idx_lo = int(np.searchsorted(s_grid, max(self.current_s - back, 0.0)))
        idx_hi = int(np.searchsorted(s_grid, min(self.current_s + front, self.scenario.total_length)))
        idx_hi = max(idx_hi, idx_lo + 2)

        xs = x_grid[idx_lo:idx_hi]
        ys = y_grid[idx_lo:idx_hi]
        dist2 = (xs - ego.x) ** 2 + (ys - ego.y) ** 2
        local_idx = int(np.argmin(dist2))
        best_idx = idx_lo + local_idx
        self.current_s = float(s_grid[best_idx])
        return self.current_s

    def build_local_reference(self, ego: EgoState) -> List[RefPoint]:
        s0 = self.update_progress(ego)
        refs = []
        s = s0
        for _ in range(self.solver_cfg.horizon + 1):
            refs.append(self.sample(s))
            ref_v = refs[-1].v_ref
            s = min(s + max(ref_v, 1.0) * self.solver_cfg.dt, self.scenario.total_length)
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