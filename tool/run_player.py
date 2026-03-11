from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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
        ego = frame["input"]["ego"]
        xs.append(ego["x"])
        ys.append(ego["y"])

        refs = frame["input"].get("local_ref", [])
        for r in refs:
            xs.append(r["x"])
            ys.append(r["y"])

        traj = frame["output"]["planner_output"].get("traj", [])
        for p in traj:
            xs.append(p["x"])
            ys.append(p["y"])

        ego_after = frame["output"].get("ego_after", None)
        if ego_after:
            xs.append(ego_after["x"])
            ys.append(ego_after["y"])

    if not xs or not ys:
        return (-10, 10), (-10, 10)

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    margin_x = max(5.0, 0.05 * (xmax - xmin + 1e-6))
    margin_y = max(5.0, 0.05 * (ymax - ymin + 1e-6))

    return (xmin - margin_x, xmax + margin_x), (ymin - margin_y, ymax + margin_y)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Run directory, e.g. results/debug_runs/run_20260311_120000")
    parser.add_argument("--interval-ms", type=int, default=100)
    args = parser.parse_args()

    run_dir = Path(args.run)
    frames = load_frames(run_dir)
    if not frames:
        raise RuntimeError(f"No frame files found under {run_dir / 'frames'}")

    xlim, ylim = compute_plot_bounds(frames)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    ref_line, = ax.plot([], [], "--", linewidth=1.5, label="local_ref")
    pred_line, = ax.plot([], [], linewidth=2.0, label="predicted_traj")
    hist_line, = ax.plot([], [], linewidth=1.8, label="executed_traj")
    ego_pt = ax.scatter([], [], c="r", s=50, label="ego")

    title = ax.set_title("")
    ax.legend()

    hist_x = []
    hist_y = []

    def init():
        ref_line.set_data([], [])
        pred_line.set_data([], [])
        hist_line.set_data([], [])
        ego_pt.set_offsets([[None, None]])
        title.set_text("Initializing...")
        return ref_line, pred_line, hist_line, ego_pt, title

    def update(i):
        frame = frames[i]

        refs = frame["input"].get("local_ref", [])
        traj = frame["output"]["planner_output"].get("traj", [])
        ego = frame["input"]["ego"]
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
        ego_pt.set_offsets([[ego["x"], ego["y"]]])

        status = frame["output"]["planner_output"].get("status", "unknown")
        solve_time_ms = frame["output"]["planner_output"].get("solve_time_ms", 0.0)
        events = ", ".join(frame.get("events", [])) or "none"
        pos_err = frame["output"].get("position_error_m", 0.0)

        title.set_text(
            f"frame={frame['frame_id']} | status={status} | "
            f"solve={solve_time_ms:.2f} ms | pos_err={pos_err:.3f} m | events={events}"
        )

        return ref_line, pred_line, hist_line, ego_pt, title

    # 关键修复：一定要保存到变量，防止动画对象被回收
    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(frames),
        interval=args.interval_ms,
        blit=False,
        repeat=False,
    )

    # 防止部分环境下被提前垃圾回收
    fig._anim = anim

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()