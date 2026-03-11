from __future__ import annotations

from typing import List

from config.defaults import SolverConfig
from .types import EgoState, PlannerOutput, SolveStatus, TrajectoryPoint
from .utils import clamp


class FallbackManager:
    def __init__(self, solver_cfg: SolverConfig):
        self.dt = solver_cfg.dt
        self.horizon = solver_cfg.horizon

    def safe_stop(self, ego: EgoState) -> PlannerOutput:
        traj: List[TrajectoryPoint] = []
        x = ego.x
        y = ego.y
        yaw = ego.yaw
        v = ego.v
        delta = ego.delta
        a = ego.a

        for k in range(self.horizon + 1):
            if k > 0:
                a = clamp(a - 0.6 * self.dt, -2.5, 0.0)
                v = max(0.0, v + a * self.dt)
                x += v * self.dt
            traj.append(
                TrajectoryPoint(
                    t=k * self.dt,
                    x=x,
                    y=y,
                    yaw=yaw,
                    v=v,
                    delta=delta,
                    a=a,
                    theta=ego.theta,
                    delta_rate=0.0,
                    jerk=-0.6,
                    theta_rate=v,
                )
            )
        return PlannerOutput(status=SolveStatus.FALLBACK, solve_time_ms=0.0, traj=traj)