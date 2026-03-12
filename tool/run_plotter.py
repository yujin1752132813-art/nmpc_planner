from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def wrap_angle(angle: float) -> float:
    return np.arctan2(np.sin(angle), np.cos(angle))


def load_summary(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows


def load_meta(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_frames(run_dir: Path):
    frames_dir = run_dir / "frames"
    frame_files = sorted(frames_dir.glob("frame_*.json"))
    frames = []
    for p in frame_files:
        with open(p, "r", encoding="utf-8") as f:
            frames.append(json.load(f))
    return frames


def signed_lateral_error(frame) -> float:
    ego = frame["output"]["ego_after"]
    refs = frame["input"].get("local_ref", [])
    if not refs:
        return 0.0
    ref0 = refs[0]
    dx = ego["x"] - ref0["x"]
    dy = ego["y"] - ref0["y"]
    ref_yaw = ref0["yaw"]
    ec = -np.sin(ref_yaw) * dx + np.cos(ref_yaw) * dy
    return float(ec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Run directory, e.g. results/debug_runs/run_xxx")
    args = parser.parse_args()

    run_dir = Path(args.run)
    meta = load_meta(run_dir / "meta.json")
    rows = load_summary(run_dir / "summary.csv")
    frames = load_frames(run_dir)

    total_length = float(meta.get("scenario", {}).get("total_length", 1.0))

    frame_id = np.array([int(r["frame_id"]) for r in rows], dtype=int)
    solve_time_ms = np.array([float(r["solve_time_ms"]) for r in rows], dtype=float)
    solver_cost = np.array([float(r["solver_cost"]) for r in rows], dtype=float)
    residual_norm = np.array([float(r["residual_norm"]) for r in rows], dtype=float)
    position_error_m = np.array([float(r["position_error_m"]) for r in rows], dtype=float)
    speed_mps = np.array([float(r["speed_mps"]) for r in rows], dtype=float)
    command_jerk = np.array([float(r["command_jerk"]) for r in rows], dtype=float)
    command_delta_rate = np.array([float(r["command_delta_rate"]) for r in rows], dtype=float)

    lateral_error = []
    progress_s = []
    yaw_wrapped = []

    for fr in frames:
        ego = fr["output"]["ego_after"]
        lateral_error.append(signed_lateral_error(fr))
        progress_s.append(float(np.clip(ego.get("theta", 0.0), 0.0, total_length)))
        yaw_wrapped.append(float(wrap_angle(ego.get("yaw", 0.0))))

    lateral_error = np.array(lateral_error, dtype=float)
    progress_s = np.array(progress_s, dtype=float)
    progress_ratio = 100.0 * progress_s / max(total_length, 1e-6)
    yaw_wrapped = np.array(yaw_wrapped, dtype=float)

    fig, axs = plt.subplots(4, 2, figsize=(14, 12))
    axs = axs.reshape(-1)

    axs[0].plot(frame_id, solve_time_ms)
    axs[0].set_title("solve_time_ms")
    axs[0].set_xlabel("frame id")
    axs[0].grid(True)

    axs[1].plot(frame_id, solver_cost)
    axs[1].set_title("solver_cost")
    axs[1].set_xlabel("frame id")
    axs[1].grid(True)

    axs[2].plot(frame_id, residual_norm)
    axs[2].set_title("residual_norm")
    axs[2].set_xlabel("frame id")
    axs[2].grid(True)

    axs[3].plot(frame_id, lateral_error)
    axs[3].set_title("lateral_error [m]")
    axs[3].set_xlabel("frame id")
    axs[3].grid(True)

    axs[4].plot(frame_id, progress_s, label="progress_s")
    axs[4].plot(frame_id, progress_ratio, label="progress_%")
    axs[4].set_title("progress")
    axs[4].set_xlabel("frame id")
    axs[4].grid(True)
    axs[4].legend()

    axs[5].plot(frame_id, yaw_wrapped)
    axs[5].set_title("yaw_wrapped [rad]")
    axs[5].set_xlabel("frame id")
    axs[5].grid(True)

    axs[6].plot(frame_id, command_jerk, label="jerk")
    axs[6].plot(frame_id, command_delta_rate, label="delta_rate")
    axs[6].set_title("commands")
    axs[6].set_xlabel("frame id")
    axs[6].grid(True)
    axs[6].legend()

    axs[7].plot(frame_id, speed_mps, label="speed")
    axs[7].plot(frame_id, position_error_m, label="position_error")
    axs[7].set_title("speed / position_error")
    axs[7].set_xlabel("frame id")
    axs[7].grid(True)
    axs[7].legend()

    fig.tight_layout()
    out = run_dir / "plots" / "run_summary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.show()


if __name__ == "__main__":
    main()