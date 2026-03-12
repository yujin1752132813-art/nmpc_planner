from __future__ import annotations

from typing import Iterable, List, Optional

from config.defaults import SimConfig
from .types import CorridorLimiter, CorridorStation, FeasibleCorridor, RefPoint


class CorridorBuilder:
    """Build feasible corridor in the local Frenet frame of the reference.

    Current implementation:
      1) road boundary is symmetric around the reference centerline
      2) optional limiters clip left/right space in s-ranges

    This keeps the interface ready for future inputs from:
      - HD map lane boundaries
      - perception/mapless drivable area
      - obstacle occupancy shrinkers
    """

    def __init__(self, sim_cfg: SimConfig):
        self.sim_cfg = sim_cfg

    def build(
        self,
        refs: List[RefPoint],
        limiters: Optional[Iterable[CorridorLimiter]] = None,
    ) -> FeasibleCorridor:
        limiters = list(limiters or [])
        stations: List[CorridorStation] = []

        base_l_min = -(self.sim_cfg.road_half_width_m - self.sim_cfg.corridor_safety_margin_m)
        base_l_max = +(self.sim_cfg.road_half_width_m - self.sim_cfg.corridor_safety_margin_m)
        min_half_width = 0.5 * self.sim_cfg.min_corridor_width_m

        for ref in refs:
            import math

            t_x = float(math.cos(ref.yaw))
            t_y = float(math.sin(ref.yaw))
            n_x = float(-math.sin(ref.yaw))
            n_y = float(math.cos(ref.yaw))

            l_min = base_l_min
            l_max = base_l_max
            tags = ["road_boundary"]

            for limiter in limiters:
                if limiter.s_start <= ref.s <= limiter.s_end:
                    if limiter.l_min is not None:
                        l_min = max(l_min, float(limiter.l_min))
                    if limiter.l_max is not None:
                        l_max = min(l_max, float(limiter.l_max))
                    tags.append(limiter.reason)

            if l_max - l_min < 2.0 * min_half_width:
                center = 0.5 * (l_min + l_max)
                l_min = center - min_half_width
                l_max = center + min_half_width
                tags.append("min_width_guard")

            stations.append(
                CorridorStation(
                    s=float(ref.s),
                    ref_x=float(ref.x),
                    ref_y=float(ref.y),
                    ref_yaw=float(ref.yaw),
                    t_x=t_x,
                    t_y=t_y,
                    n_x=n_x,
                    n_y=n_y,
                    road_l_min=float(base_l_min),
                    road_l_max=float(base_l_max),
                    l_min=float(l_min),
                    l_max=float(l_max),
                    source_tags=tags,
                )
            )

        return FeasibleCorridor(stations=stations)