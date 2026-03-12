from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from nmpc_planner.planner_node import PlannerNode
from nmpc_planner.types import CorridorLimiter


CASE_LIMITERS: Dict[str, List[CorridorLimiter]] = {
    "base": [],
    "close_left": [
        CorridorLimiter(
            s_start=15.0,
            s_end=35.0,
            l_max=-0.50,
            reason="close_left_test",
        )
    ],
    "narrow_guard": [
        CorridorLimiter(
            s_start=20.0,
            s_end=30.0,
            l_min=0.20,
            l_max=0.40,
            reason="narrow_guard_test",
        )
    ],
}


def _ec_from_station(station: dict, x: float, y: float) -> float:
    dx = float(x - station["ref_x"])
    dy = float(y - station["ref_y"])
    return float(station["n_x"] * dx + station["n_y"] * dy)


def _corridor_violation(station: dict, x: float, y: float) -> float:
    ec = _ec_from_station(station, x, y)
    if ec < station["l_min"]:
        return float(station["l_min"] - ec)
    if ec > station["l_max"]:
        return float(ec - station["l_max"])
    return 0.0


def load_frames(run_dir: Path):
    frames_dir = run_dir / "frames"
    frame_files = sorted(frames_dir.glob("frame_*.json"))
    frames = []
    for p in frame_files:
        with open(p, "r", encoding="utf-8") as f:
            frames.append(json.load(f))
    return frames


def analyze_run(run_dir: Path, case_name: str, min_width_m: float) -> Tuple[bool, dict]:
    frames = load_frames(run_dir)

    max_pred_violation = 0.0
    max_exec_violation = 0.0
    max_reconstruct_error = 0.0
    min_width_seen = 1e9
    symmetry_error = 0.0

    clipped_lmax_in_window = -1e9
    right_bound_error_in_window = 0.0
    guard_hits = 0
    guard_min_width = 1e9

    for frame in frames:
        corridor = frame["input"]["corridor"]["stations"]
        traj = frame["output"]["planner_output"].get("traj", [])
        ego_after = frame["output"]["ego_after"]

        # geometry checks
        for st in corridor:
            width = float(st["l_max"] - st["l_min"])
            min_width_seen = min(min_width_seen, width)

            left_x = st["ref_x"] + st["n_x"] * st["l_max"]
            left_y = st["ref_y"] + st["n_y"] * st["l_max"]
            right_x = st["ref_x"] + st["n_x"] * st["l_min"]
            right_y = st["ref_y"] + st["n_y"] * st["l_min"]

            ec_left = _ec_from_station(st, left_x, left_y)
            ec_right = _ec_from_station(st, right_x, right_y)

            max_reconstruct_error = max(
                max_reconstruct_error,
                abs(ec_left - st["l_max"]),
                abs(ec_right - st["l_min"]),
            )

            if case_name == "base":
                symmetry_error = max(symmetry_error, abs(st["l_min"] + st["l_max"]))

            if case_name == "close_left" and 15.0 <= st["s"] <= 35.0:
                clipped_lmax_in_window = max(clipped_lmax_in_window, st["l_max"])
                right_bound_error_in_window = max(
                    right_bound_error_in_window,
                    abs(st["l_min"] - st["road_l_min"]),
                )

            if case_name == "narrow_guard" and 20.0 <= st["s"] <= 30.0:
                guard_min_width = min(guard_min_width, width)
                if "min_width_guard" in st["source_tags"]:
                    guard_hits += 1

        # predicted trajectory must stay in corridor
        K = min(len(corridor), len(traj))
        for k in range(1, K):
            st = corridor[k]
            pt = traj[k]
            max_pred_violation = max(
                max_pred_violation,
                _corridor_violation(st, pt["x"], pt["y"]),
            )

        # executed state: compare against stage-1 corridor as a practical check
        if len(corridor) > 1:
            st1 = corridor[1]
            max_exec_violation = max(
                max_exec_violation,
                _corridor_violation(st1, ego_after["x"], ego_after["y"]),
            )

    passed = True
    reasons = []

    if max_reconstruct_error > 1e-8:
        passed = False
        reasons.append(f"boundary reconstruction error too large: {max_reconstruct_error:.3e}")

    if min_width_seen + 1e-9 < min_width_m and case_name == "narrow_guard":
        passed = False
        reasons.append(f"min width guard failed: min_width_seen={min_width_seen:.3f} < {min_width_m:.3f}")

    if max_pred_violation > 1e-4:
        passed = False
        reasons.append(f"predicted trajectory violated corridor: {max_pred_violation:.6f} m")

    if case_name == "base":
        if symmetry_error > 1e-8:
            passed = False
            reasons.append(f"base corridor is not symmetric: symmetry_error={symmetry_error:.3e}")

    if case_name == "close_left":
        if clipped_lmax_in_window > 0.21:
            passed = False
            reasons.append(f"left bound not clipped correctly: max l_max in window = {clipped_lmax_in_window:.3f}")
        if right_bound_error_in_window > 1e-8:
            passed = False
            reasons.append(f"right boundary changed unexpectedly: error = {right_bound_error_in_window:.3e}")

    if case_name == "narrow_guard":
        if guard_hits == 0:
            passed = False
            reasons.append("min_width_guard tag was never triggered")
        if guard_min_width + 1e-9 < min_width_m:
            passed = False
            reasons.append(f"guarded width too small: {guard_min_width:.3f} < {min_width_m:.3f}")

    report = {
        "case": case_name,
        "run_dir": str(run_dir),
        "max_pred_violation_m": max_pred_violation,
        "max_exec_violation_m": max_exec_violation,
        "max_reconstruct_error": max_reconstruct_error,
        "min_width_seen_m": min_width_seen,
        "symmetry_error": symmetry_error,
        "clipped_lmax_in_window": clipped_lmax_in_window,
        "right_bound_error_in_window": right_bound_error_in_window,
        "guard_hits": guard_hits,
        "guard_min_width_m": guard_min_width,
        "passed": passed,
        "reasons": reasons,
    }
    return passed, report


def run_case(case_name: str) -> Tuple[bool, dict]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"corridor_{case_name}_{ts}"

    node = PlannerNode(
        corridor_limiters=CASE_LIMITERS[case_name],
        run_name=run_name,
    )
    node.run()

    run_dir = node.recorder.current_run_dir
    passed, report = analyze_run(
        run_dir=run_dir,
        case_name=case_name,
        min_width_m=node.sim_cfg.min_corridor_width_m,
    )

    print("=" * 80)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"PLAYER COMMAND: python3 tool/run_player.py --run {run_dir}")
    print("RESULT:", "PASS" if passed else "FAIL")
    return passed, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=["base", "close_left", "narrow_guard", "all"],
        default="all",
    )
    args = parser.parse_args()

    if args.case == "all":
        all_ok = True
        for case_name in ["base", "close_left", "narrow_guard"]:
            ok, _ = run_case(case_name)
            all_ok = all_ok and ok
        print("=" * 80)
        print("OVERALL:", "PASS" if all_ok else "FAIL")
    else:
        run_case(args.case)


if __name__ == "__main__":
    main()