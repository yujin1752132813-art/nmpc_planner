from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_frame(run_dir: Path, frame_id: int):
    frame_path = run_dir / "frames" / f"frame_{frame_id:06d}.json"
    with open(frame_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Run directory, e.g. results/debug_runs/run_20260311_120000")
    parser.add_argument("--frame", required=True, type=int, help="Frame id")
    args = parser.parse_args()

    run_dir = Path(args.run)
    frame = load_frame(run_dir, args.frame)

    print(json.dumps({
        "frame_id": frame["frame_id"],
        "sim_time": frame["sim_time"],
        "events": frame.get("events", []),
        "ego": frame["input"]["ego"],
        "position_error_m": frame["output"].get("position_error_m", None),
        "planner_status": frame["output"]["planner_output"].get("status", None),
        "solver_status_code": frame["output"]["planner_output"].get("solver_status_code", None),
        "solve_time_ms": frame["output"]["planner_output"].get("solve_time_ms", None),
        "solver_cost": frame["output"]["planner_output"].get("solver_cost", None),
    }, indent=2))

    refs = frame["input"]["local_ref"]
    traj = frame["output"]["planner_output"].get("traj", [])
    ego = frame["input"]["ego"]

    fig, ax = plt.subplots(figsize=(8, 6))
    if refs:
        ax.plot([r["x"] for r in refs], [r["y"] for r in refs], "--", label="local_ref")
    if traj:
        ax.plot([p["x"] for p in traj], [p["y"] for p in traj], label="predicted_traj")
    ax.scatter([ego["x"]], [ego["y"]], c="r", label="ego")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend()
    ax.set_title(f"Frame {args.frame}")
    plt.show()


if __name__ == "__main__":
    main()