from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List


@dataclass
class EgoState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    v: float = 0.0
    delta: float = 0.0
    a: float = 0.0
    theta: float = 0.0


@dataclass
class RefPoint:
    s: float
    x: float
    y: float
    yaw: float
    kappa: float
    v_ref: float


@dataclass
class PlannerInput:
    ego: EgoState
    local_ref: List[RefPoint]


@dataclass
class TrajectoryPoint:
    t: float
    x: float
    y: float
    yaw: float
    v: float
    delta: float
    a: float
    theta: float
    delta_rate: float
    jerk: float
    theta_rate: float


class SolveStatus(Enum):
    OK = "ok"
    SOLVER_FAILED = "solver_failed"
    VALIDATION_FAILED = "validation_failed"
    FALLBACK = "fallback"


@dataclass
class PlannerOutput:
    status: SolveStatus
    solve_time_ms: float
    traj: List[TrajectoryPoint] = field(default_factory=list)
    solver_status_code: int = 0
    solver_cost: float = 0.0
    solver_residuals: List[float] = field(default_factory=list)
    solver_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunLog:
    states: List[EgoState] = field(default_factory=list)
    commands: List[List[float]] = field(default_factory=list)
    references: List[List[RefPoint]] = field(default_factory=list)
    solve_times_ms: List[float] = field(default_factory=list)
    statuses: List[str] = field(default_factory=list)
    solver_status_codes: List[int] = field(default_factory=list)
    solver_costs: List[float] = field(default_factory=list)

    def to_dict(self):
        return {
            "states": [asdict(s) for s in self.states],
            "commands": self.commands,
            "solve_times_ms": self.solve_times_ms,
            "statuses": self.statuses,
            "solver_status_codes": self.solver_status_codes,
            "solver_costs": self.solver_costs,
        }