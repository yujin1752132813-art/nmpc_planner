from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from config.defaults import SolverConfig, VehicleConfig
from .scenario_builder import Scenario
from .types import EgoState, RunLog
from .utils import wrap_angle


@dataclass
class SmoothnessMetrics:
    final_position_error_m: float
    final_speed_mps: float
    max_abs_steering_rate: float
    max_abs_jerk: float
    max_abs_lateral_accel: float
    rms_contour_error: float
    mean_solve_time_ms: float


class TrajectoryValidator:
    def __init__(self, vehicle_cfg: VehicleConfig, solver_cfg: SolverConfig):
        self.vehicle_cfg = vehicle_cfg
        self.dt = solver_cfg.dt

    def compute_metrics(self, log: RunLog, scenario: Scenario) -> SmoothnessMetrics:
        xs = np.array([s.x for s in log.states])
        ys = np.array([s.y for s in log.states])
        vs = np.array([s.v for s in log.states])
        deltas = np.array([s.delta for s in log.states])
        accs = np.array([s.a for s in log.states])
        yaws = np.array([s.yaw for s in log.states])

        steering_rate = np.diff(deltas) / self.dt if len(deltas) > 1 else np.array([0.0])
        jerk = np.diff(accs) / self.dt if len(accs) > 1 else np.array([0.0])
        lateral_acc = vs * vs / self.vehicle_cfg.wheel_base * np.tan(deltas)

        contour_errors = []
        for state in log.states:
            idx = int(np.argmin((scenario.x_grid - state.x) ** 2 + (scenario.y_grid - state.y) ** 2))
            ref_x = scenario.x_grid[idx]
            ref_y = scenario.y_grid[idx]
            ref_yaw = scenario.yaw_grid[idx]
            ec = -np.sin(ref_yaw) * (state.x - ref_x) + np.cos(ref_yaw) * (state.y - ref_y)
            contour_errors.append(ec)

        final_position_error = float(np.hypot(xs[-1] - scenario.goal_x, ys[-1] - scenario.goal_y))
        final_speed = float(vs[-1])
        return SmoothnessMetrics(
            final_position_error_m=final_position_error,
            final_speed_mps=final_speed,
            max_abs_steering_rate=float(np.max(np.abs(steering_rate))),
            max_abs_jerk=float(np.max(np.abs(jerk))),
            max_abs_lateral_accel=float(np.max(np.abs(lateral_acc))),
            rms_contour_error=float(np.sqrt(np.mean(np.square(contour_errors)))),
            mean_solve_time_ms=float(np.mean(log.solve_times_ms)) if log.solve_times_ms else 0.0,
        )

    def save_plots(self, log: RunLog, scenario: Scenario, out_dir: Path) -> SmoothnessMetrics:
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics = self.compute_metrics(log, scenario)

        t = np.arange(len(log.states)) * self.dt
        xs = np.array([s.x for s in log.states])
        ys = np.array([s.y for s in log.states])
        yaws = np.array([s.yaw for s in log.states])
        vs = np.array([s.v for s in log.states])
        deltas = np.array([s.delta for s in log.states])
        accs = np.array([s.a for s in log.states])
        thetas = np.array([s.theta for s in log.states])

        steering_rate = np.diff(deltas) / self.dt if len(deltas) > 1 else np.array([0.0])
        jerk = np.diff(accs) / self.dt if len(accs) > 1 else np.array([0.0])
        yaw_rate = np.diff(yaws) / self.dt if len(yaws) > 1 else np.array([0.0])
        lateral_acc = vs * vs / self.vehicle_cfg.wheel_base * np.tan(deltas)

        ref_speed = np.array([scenario.v_grid[int(np.clip(np.searchsorted(scenario.s_grid, s.theta), 0, len(scenario.s_grid)-1))] for s in log.states])

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.plot(scenario.x_grid, scenario.y_grid, linestyle="--", label="reference path")
        ax.plot(xs, ys, label="tracked trajectory")
        ax.scatter([scenario.goal_x], [scenario.goal_y], marker="x", label="goal")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title("NMPC circle-track trajectory")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(out_dir / "trajectory.png", dpi=150)
        plt.close(fig)

        fig = plt.figure(figsize=(12, 10))
        axs = fig.subplots(4, 2)
        axs = axs.reshape(-1)
        axs[0].plot(t, vs, label="speed")
        axs[0].plot(t, ref_speed, linestyle="--", label="ref")
        axs[0].set_title("Speed [m/s]")
        axs[0].grid(True)
        axs[0].legend()

        axs[1].plot(t, accs)
        axs[1].set_title("Acceleration [m/s²]")
        axs[1].grid(True)

        axs[2].plot(t, deltas)
        axs[2].set_title("Steering angle [rad]")
        axs[2].grid(True)

        axs[3].plot(t[:-1], steering_rate)
        axs[3].set_title("Steering rate [rad/s]")
        axs[3].grid(True)

        axs[4].plot(t[:-1], jerk)
        axs[4].set_title("Jerk [m/s³]")
        axs[4].grid(True)

        axs[5].plot(t, lateral_acc)
        axs[5].set_title("Lateral acceleration [m/s²]")
        axs[5].grid(True)

        axs[6].plot(t[:-1], yaw_rate)
        axs[6].set_title("Yaw rate [rad/s]")
        axs[6].grid(True)

        axs[7].plot(t, thetas)
        axs[7].set_title("Path progress theta [m]")
        axs[7].grid(True)

        fig.tight_layout()
        fig.savefig(out_dir / "smoothness.png", dpi=150)
        plt.close(fig)

        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(asdict(metrics), f, indent=2)

        return metrics