from __future__ import annotations

from dataclasses import dataclass
from typing import List

from config.defaults import SimConfig
from .types import RefPoint


@dataclass
class CorridorSample:
    center_x: float
    center_y: float
    width: float


class CorridorBuilder:
    def __init__(self, sim_cfg: SimConfig):
        self.sim_cfg = sim_cfg

    def build(self, refs: List[RefPoint]) -> List[CorridorSample]:
        return [
            CorridorSample(center_x=r.x, center_y=r.y, width=self.sim_cfg.trajectory_width)
            for r in refs
        ]