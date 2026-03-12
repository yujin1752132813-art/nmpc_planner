from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np

from config.defaults import SimConfig
from .path_smoother import smooth_reference_curvature
from .types import RefPoint
from .utils import unwrap_sequence, clamp


@dataclass
class Scenario:
    path: List[RefPoint]
    s_grid: np.ndarray
    x_grid: np.ndarray
    y_grid: np.ndarray
    yaw_grid: np.ndarray
    kappa_grid: np.ndarray
    v_grid: np.ndarray
    total_length: float
    goal_x: float
    goal_y: float
    goal_yaw: float


class ScenarioBuilder:
    def __init__(self, sim_cfg: SimConfig):
        self.sim_cfg = sim_cfg

    def build(self) -> Scenario:
        ds = self.sim_cfg.path_ds
        r = 20.0
        straight_1 = 50.0
        straight_2 = 50.0
        loop_len = 2.0 * 2.0 * math.pi * r
        total_length = straight_1 + loop_len + straight_2

        s_grid = np.arange(0.0, total_length + ds, ds)
        x_vals = []
        y_vals = []
        yaw_vals = []
        kappa_vals = []
        v_vals = []

        for s in s_grid:
            if s <= straight_1:
                x = s
                y = 0.0
                yaw = 0.0
                kappa = 0.0
            elif s <= straight_1 + loop_len:
                u = s - straight_1
                alpha = -math.pi / 2.0 + u / r
                cx, cy = 50.0, 20.0
                x = cx + r * math.cos(alpha)
                y = cy + r * math.sin(alpha)
                yaw = alpha + math.pi / 2.0
                kappa = 1.0 / r
            else:
                u = s - straight_1 - loop_len
                x = 50.0 + u
                y = 0.0
                yaw = 0.0
                kappa = 0.0

            x_vals.append(x)
            y_vals.append(y)
            yaw_vals.append(yaw)
            kappa_vals.append(kappa)
            v_vals.append(self._speed_profile(s, total_length))

        yaw_vals = unwrap_sequence(yaw_vals)

        raw_path = [
            RefPoint(
                s=float(s_grid[i]),
                x=float(x_vals[i]),
                y=float(y_vals[i]),
                yaw=float(yaw_vals[i]),
                kappa=float(kappa_vals[i]),
                v_ref=float(v_vals[i]),
            )
            for i in range(len(s_grid))
        ]

        if self.sim_cfg.enable_curvature_smoothing:
            path = smooth_reference_curvature(
                raw_path,
                sigma_m=self.sim_cfg.curvature_smoothing_sigma_m,
                passes=self.sim_cfg.curvature_smoothing_passes,
            )
        else:
            path = raw_path

        s_grid = np.array([p.s for p in path], dtype=float)
        x_grid = np.array([p.x for p in path], dtype=float)
        y_grid = np.array([p.y for p in path], dtype=float)
        yaw_grid = np.array([p.yaw for p in path], dtype=float)
        kappa_grid = np.array([p.kappa for p in path], dtype=float)
        v_grid = np.array([p.v_ref for p in path], dtype=float)

        return Scenario(
            path=path,
            s_grid=s_grid,
            x_grid=x_grid,
            y_grid=y_grid,
            yaw_grid=yaw_grid,
            kappa_grid=kappa_grid,
            v_grid=v_grid,
            total_length=float(total_length),
            goal_x=float(x_grid[-1]),
            goal_y=float(y_grid[-1]),
            goal_yaw=float(yaw_grid[-1]),
        )

    def _speed_profile(self, s: float, total_length: float) -> float:
        cruise = 6.0
        accel_len = 20.0
        decel_len = 30.0

        if s < accel_len:
            return clamp(cruise * s / accel_len, 0.0, cruise)

        stop_start = total_length - decel_len
        if s >= stop_start:
            rem = max(total_length - s, 0.0)
            a_ref = 0.6
            v_stop = math.sqrt(max(2.0 * a_ref * rem, 0.0))
            if rem < 0.5:
                return 0.0
            return clamp(v_stop, 0.0, cruise)

        return cruise