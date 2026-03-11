from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_summary(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    def col(name: str):
        return np.array([float(r[name]) for r in rows], dtype=float)

    return {
        "frame_id": col("frame_id"),
        "solve_time_ms": col("solve_time_ms"),
        "solver_cost": col("solver_cost"),
        "residual_norm": col("residual_norm"),
        "position_error_m": col("position_error_m"),
        "command_jerk": col("command_jerk"),
        "command_delta_rate": col("command_delta_rate"),
        "speed_mps": col("speed_mps"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Run directory, e.g. results/debug_runs/run_20260311_120000")
    args = parser.parse_args()

    run_dir = Path(args.run)
    data = _load_summary(run_dir / "summary.csv")

    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    axs = axs.reshape(-1)

    axs[0].plot(data["frame_id"], data["solve_time_ms"])
    axs[0].set_title("solve_time_ms")
    axs[0].grid(True)

    axs[1].plot(data["frame_id"], data["solver_cost"])
    axs[1].set_title("solver_cost")
    axs[1].grid(True)

    axs[2].plot(data["frame_id"], data["residual_norm"])
    axs[2].set_title("residual_norm")
    axs[2].grid(True)

    axs[3].plot(data["frame_id"], data["position_error_m"])
    axs[3].set_title("position_error_m")
    axs[3].grid(True)

    axs[4].plot(data["frame_id"], data["command_jerk"], label="jerk")
    axs[4].plot(data["frame_id"], data["command_delta_rate"], label="delta_rate")
    axs[4].set_title("commands")
    axs[4].grid(True)
    axs[4].legend()

    axs[5].plot(data["frame_id"], data["speed_mps"])
    axs[5].set_title("speed_mps")
    axs[5].grid(True)

    fig.tight_layout()
    out = run_dir / "plots" / "run_summary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.show()


if __name__ == "__main__":
    main()