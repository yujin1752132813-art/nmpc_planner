from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

_ANIM = None  # keep animation alive


def wrap_angle(angle: float) -> float:
    return np.arctan2(np.sin(angle), np.cos(angle))


def load_frames(run_dir: Path):
    frames_dir = run_dir / "frames"
    frame_files = sorted(frames_dir.glob("frame_*.json"))
    frames = []
    for p in frame_files:
        with open(p, "r", encoding="utf-8") as f:
            frames.append(json.load(f))
    return frames


def compute_plot_bounds(frames):
    xs = []
    ys = []

    for frame in frames:
        ego_after = frame["output"].get("ego_after", {})
        if ego_after:
            xs.append(ego_after["x"])
            ys.append(ego_after["y"])

        refs = frame["input"].get("local_ref", [])
        for r in refs:
            xs.append(r["x"])
            ys.append(r["y"])

        traj = frame["output"]["planner_output"].get("traj", [])
        for p in traj:
            xs.append(p["x"])
            ys.append(p["y"])

    if not xs or not ys:
        return (-10, 10), (-10, 10)

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    margin_x = max(5.0, 0.05 * (xmax - xmin + 1e-6))
    margin_y = max(5.0, 0.05 * (ymax - ymin + 1e-6))
    return (xmin - margin_x, xmax + margin_x), (ymin - margin_y, ymax + margin_y)


def main() -> None:
    global _ANIM

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Run directory, e.g. results/debug_runs/run_xxx")
    parser.add_argument("--interval-ms", type=int, default=100)
    args = parser.parse_args()

    run_dir = Path(args.run)
    frames = load_frames(run_dir)
    if not frames:
        raise RuntimeError(f"No frame files found under {run_dir / 'frames'}")

    xlim, ylim = compute_plot_bounds(frames)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    ref_line, = ax.plot([], [], "--", linewidth=1.5, label="local_ref")
    pred_line, = ax.plot([], [], linewidth=2.0, label="predicted_traj")
    hist_line, = ax.plot([], [], linewidth=2.0, label="executed_traj")
    ego_pt = ax.scatter([], [], c="r", s=50, label="ego_after")

    title = ax.set_title("")
    info_text = ax.text(
        0.02, 0.98, "", transform=ax.transAxes,
        va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.75)
    )
    ax.legend()

    hist_x = []
    hist_y = []

    def init():
        ref_line.set_data([], [])
        pred_line.set_data([], [])
        hist_line.set_data([], [])
        ego_pt.set_offsets(np.array([[np.nan, np.nan]]))
        title.set_text("Initializing...")
        info_text.set_text("")
        return ref_line, pred_line, hist_line, ego_pt, title, info_text

    def update(i):
        frame = frames[i]
        refs = frame["input"].get("local_ref", [])
        traj = frame["output"]["planner_output"].get("traj", [])
        ego_after = frame["output"]["ego_after"]

        hist_x.append(ego_after["x"])
        hist_y.append(ego_after["y"])

        if refs:
            ref_line.set_data([r["x"] for r in refs], [r["y"] for r in refs])
        else:
            ref_line.set_data([], [])

        if traj:
            pred_line.set_data([p["x"] for p in traj], [p["y"] for p in traj])
        else:
            pred_line.set_data([], [])

        hist_line.set_data(hist_x, hist_y)
        ego_pt.set_offsets(np.array([[ego_after["x"], ego_after["y"]]]))

        status = frame["output"]["planner_output"].get("status", "unknown")
        solve_time_ms = frame["output"]["planner_output"].get("solve_time_ms", 0.0)
        pos_err = frame["output"].get("position_error_m", 0.0)
        yaw_wrapped = wrap_angle(ego_after.get("yaw", 0.0))
        events = ", ".join(frame.get("events", [])) or "none"

        title.set_text(f"frame={frame['frame_id']} status={status} solve={solve_time_ms:.2f} ms")
        info_text.set_text(
            f"x={ego_after['x']:.2f}\n"
            f"y={ego_after['y']:.2f}\n"
            f"v={ego_after['v']:.2f}\n"
            f"yaw={yaw_wrapped:.3f}\n"
            f"theta={ego_after.get('theta', 0.0):.2f}\n"
            f"pos_err={pos_err:.3f}\n"
            f"events={events}"
        )

        return ref_line, pred_line, hist_line, ego_pt, title, info_text

    _ANIM = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(frames),
        interval=args.interval_ms,
        blit=False,
        repeat=False,
    )

    fig.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":
    main()