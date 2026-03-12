from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


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
class CorridorLimiter:
    """Stage-local lateral clipping in the reference Frenet frame.

    The limiter is expressed in lateral offset `ec` coordinates, where:
      - left side is positive
      - right side is negative
      - the feasible corridor is l_min <= ec <= l_max

    Future obstacle integration can use this directly:
      - close left passing space  -> reduce l_max
      - close right passing space -> increase l_min
    """

    s_start: float
    s_end: float
    l_min: Optional[float] = None
    l_max: Optional[float] = None
    reason: str = "unknown"


@dataclass
class CorridorStation:
    """One stage of the feasible corridor around the local reference."""

    s: float
    ref_x: float
    ref_y: float
    ref_yaw: float
    t_x: float
    t_y: float
    n_x: float
    n_y: float

    # raw road-space bounds before obstacle clipping
    road_l_min: float
    road_l_max: float

    # final feasible-space bounds after all limiters are fused
    l_min: float
    l_max: float

    source_tags: List[str] = field(default_factory=list)


@dataclass
class FeasibleCorridor:
    stations: List[CorridorStation] = field(default_factory=list)


@dataclass
class PlannerInput:
    ego: EgoState
    local_ref: List[RefPoint]
    corridor: Optional[FeasibleCorridor] = None


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
    corridors: List[FeasibleCorridor] = field(default_factory=list)
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