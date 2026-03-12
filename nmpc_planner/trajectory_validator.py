from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from config.defaults import SolverConfig, VehicleConfig
from .scenario_builder import Scenario
from .types import RunLog
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

    def _compute_tracking_series(self, log: RunLog, scenario: Scenario) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return:
        - contour/lateral error series
        - progress_s series
        - progress_ratio (%) series
        """
        lateral_errors = []
        progress_s = []

        for state in log.states:
            idx = int(np.argmin((scenario.x_grid - state.x) ** 2 + (scenario.y_grid - state.y) ** 2))
            ref_x = scenario.x_grid[idx]
            ref_y = scenario.y_grid[idx]
            ref_yaw = scenario.yaw_grid[idx]

            # signed lateral error in local path frame
            ec = -np.sin(ref_yaw) * (state.x - ref_x) + np.cos(ref_yaw) * (state.y - ref_y)
            lateral_errors.append(ec)

            s_val = float(np.clip(state.theta, 0.0, scenario.total_length))
            progress_s.append(s_val)

        lateral_errors = np.array(lateral_errors, dtype=float)
        progress_s = np.array(progress_s, dtype=float)
        progress_ratio = 100.0 * progress_s / max(scenario.total_length, 1e-6)
        return lateral_errors, progress_s, progress_ratio

    def compute_metrics(self, log: RunLog, scenario: Scenario) -> SmoothnessMetrics:
        xs = np.array([s.x for s in log.states])
        ys = np.array([s.y for s in log.states])
        vs = np.array([s.v for s in log.states])
        deltas = np.array([s.delta for s in log.states])
        accs = np.array([s.a for s in log.states])

        steering_rate = np.diff(deltas) / self.dt if len(deltas) > 1 else np.array([0.0])
        jerk = np.diff(accs) / self.dt if len(accs) > 1 else np.array([0.0])
        lateral_acc = vs * vs / self.vehicle_cfg.wheel_base * np.tan(deltas)

        lateral_errors, _, _ = self._compute_tracking_series(log, scenario)

        final_position_error = float(np.hypot(xs[-1] - scenario.goal_x, ys[-1] - scenario.goal_y))
        final_speed = float(vs[-1])

        return SmoothnessMetrics(
            final_position_error_m=final_position_error,
            final_speed_mps=final_speed,
            max_abs_steering_rate=float(np.max(np.abs(steering_rate))),
            max_abs_jerk=float(np.max(np.abs(jerk))),
            max_abs_lateral_accel=float(np.max(np.abs(lateral_acc))),
            rms_contour_error=float(np.sqrt(np.mean(np.square(lateral_errors)))),
            mean_solve_time_ms=float(np.mean(log.solve_times_ms)) if log.solve_times_ms else 0.0,
        )

    def save_plots(self, log: RunLog, scenario: Scenario, out_dir: Path) -> SmoothnessMetrics:
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics = self.compute_metrics(log, scenario)

        frame_id = np.arange(len(log.states))
        xs = np.array([s.x for s in log.states])
        ys = np.array([s.y for s in log.states])
        yaws_raw = np.array([s.yaw for s in log.states])
        yaws = np.array([wrap_angle(y) for y in yaws_raw])  # FIX: display wrapped yaw
        vs = np.array([s.v for s in log.states])
        deltas = np.array([s.delta for s in log.states])
        accs = np.array([s.a for s in log.states])

        steering_rate = np.diff(deltas) / self.dt if len(deltas) > 1 else np.array([0.0])
        jerk = np.diff(accs) / self.dt if len(accs) > 1 else np.array([0.0])
        yaw_rate = np.diff(yaws_raw) / self.dt if len(yaws_raw) > 1 else np.array([0.0])
        lateral_acc = vs * vs / self.vehicle_cfg.wheel_base * np.tan(deltas)

        lateral_errors, progress_s, progress_ratio = self._compute_tracking_series(log, scenario)

        # reference speed from projected progress
        ref_speed = []
        for s_val in progress_s:
            idx = int(np.clip(np.searchsorted(scenario.s_grid, s_val), 0, len(scenario.s_grid) - 1))
            ref_speed.append(scenario.v_grid[idx])
        ref_speed = np.array(ref_speed, dtype=float)

        # trajectory plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.plot(scenario.x_grid, scenario.y_grid, linestyle="--", label="reference path")
        ax.plot(xs, ys, label="tracked trajectory")
        ax.scatter([scenario.goal_x], [scenario.goal_y], marker="x", label="goal")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title("NMPC trajectory")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(out_dir / "trajectory.png", dpi=150)
        plt.close(fig)

        # smoothness / debug plot with frame_id on x-axis
        fig = plt.figure(figsize=(14, 12))
        axs = fig.subplots(5, 2)
        axs = axs.reshape(-1)

        axs[0].plot(frame_id, vs, label="speed")
        axs[0].plot(frame_id, ref_speed, linestyle="--", label="ref")
        axs[0].set_title("Speed [m/s]")
        axs[0].set_xlabel("frame id")
        axs[0].grid(True)
        axs[0].legend()

        axs[1].plot(frame_id, accs)
        axs[1].set_title("Acceleration [m/s²]")
        axs[1].set_xlabel("frame id")
        axs[1].grid(True)

        axs[2].plot(frame_id, deltas)
        axs[2].set_title("Steering angle delta [rad]")
        axs[2].set_xlabel("frame id")
        axs[2].grid(True)

        axs[3].plot(frame_id[:-1], steering_rate)
        axs[3].set_title("Steering rate [rad/s]")
        axs[3].set_xlabel("frame id")
        axs[3].grid(True)

        axs[4].plot(frame_id[:-1], jerk)
        axs[4].set_title("Jerk [m/s³]")
        axs[4].set_xlabel("frame id")
        axs[4].grid(True)

        axs[5].plot(frame_id, lateral_acc)
        axs[5].set_title("Lateral acceleration [m/s²]")
        axs[5].set_xlabel("frame id")
        axs[5].grid(True)

        axs[6].plot(frame_id, yaws)
        axs[6].set_title("Yaw (wrapped) [rad]")
        axs[6].set_xlabel("frame id")
        axs[6].grid(True)

        axs[7].plot(frame_id, lateral_errors)
        axs[7].set_title("Lateral error [m]")
        axs[7].set_xlabel("frame id")
        axs[7].grid(True)

        axs[8].plot(frame_id, progress_s, label="progress s")
        axs[8].set_title("Progress along path [m]")
        axs[8].set_xlabel("frame id")
        axs[8].grid(True)

        axs[9].plot(frame_id, progress_ratio)
        axs[9].set_title("Progress ratio [% of total path]")
        axs[9].set_xlabel("frame id")
        axs[9].grid(True)

        fig.tight_layout()
        fig.savefig(out_dir / "smoothness.png", dpi=150)
        plt.close(fig)

        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(asdict(metrics), f, indent=2)

        return metrics