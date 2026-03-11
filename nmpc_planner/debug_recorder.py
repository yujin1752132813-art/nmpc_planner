from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class DebugRecorder:
    """Simple offline frame recorder.

    Writes:
      - meta.json
      - summary.csv
      - frames/frame_000000.json ...
      - events.jsonl

    This first version is intentionally synchronous and simple so it is easy to
    inspect and robust for offline debugging. For hard real-time deployment,
    the same interface can later be backed by an async writer thread.
    """

    def __init__(self, root_dir: Path, run_name: Optional[str] = None, enable: bool = True):
        self.enable = enable
        self.root_dir = Path(root_dir)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name or f"run_{ts}"
        self.run_dir = self.root_dir / self.run_name
        self.frames_dir = self.run_dir / "frames"
        self.plots_dir = self.run_dir / "plots"
        self._summary_file = None
        self._summary_writer = None
        self._started = False
        self._frame_count = 0
        self._event_count = 0

    @property
    def current_run_dir(self) -> Path:
        return self.run_dir

    def start(self, meta: Dict[str, Any]) -> None:
        if not self.enable or self._started:
            return
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        with open(self.run_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(self._to_jsonable(meta), f, indent=2)

        self._summary_file = open(self.run_dir / "summary.csv", "w", newline="", encoding="utf-8")
        self._summary_writer = csv.DictWriter(
            self._summary_file,
            fieldnames=[
                "frame_id",
                "sim_time",
                "status",
                "solve_time_ms",
                "solver_status_code",
                "solver_cost",
                "residual_norm",
                "position_error_m",
                "speed_mps",
                "command_delta_rate",
                "command_jerk",
                "command_theta_rate",
                "event_count",
            ],
        )
        self._summary_writer.writeheader()
        self._started = True

    def record_frame(
        self,
        *,
        frame_id: int,
        sim_time: float,
        ego_before: Any,
        local_ref: List[Any],
        corridor: Any,
        obstacles: Any,
        planner_output: Any,
        command: List[float],
        ego_after: Any,
        position_error_m: float,
        events: Optional[List[str]] = None,
    ) -> None:
        if not self.enable:
            return
        if not self._started:
            raise RuntimeError("DebugRecorder.start() must be called before record_frame().")

        events = events or []
        frame_json = {
            "frame_id": frame_id,
            "sim_time": sim_time,
            "events": events,
            "input": {
                "ego": ego_before,
                "local_ref": local_ref,
                "corridor": corridor,
                "obstacles": obstacles,
            },
            "output": {
                "planner_output": planner_output,
                "command": command,
                "ego_after": ego_after,
                "position_error_m": position_error_m,
            },
        }

        frame_path = self.frames_dir / f"frame_{frame_id:06d}.json"
        with open(frame_path, "w", encoding="utf-8") as f:
            json.dump(self._to_jsonable(frame_json), f, indent=2)

        residuals = getattr(planner_output, "solver_residuals", []) or []
        residual_norm = float(sum(abs(float(v)) for v in residuals)) if residuals else 0.0
        status = getattr(getattr(planner_output, "status", None), "value", str(getattr(planner_output, "status", "unknown")))
        solver_status_code = int(getattr(planner_output, "solver_status_code", 0) or 0)
        solver_cost = float(getattr(planner_output, "solver_cost", 0.0) or 0.0)
        solve_time_ms = float(getattr(planner_output, "solve_time_ms", 0.0) or 0.0)
        speed_mps = float(getattr(ego_after, "v", 0.0) or 0.0)

        self._summary_writer.writerow(
            {
                "frame_id": frame_id,
                "sim_time": sim_time,
                "status": status,
                "solve_time_ms": solve_time_ms,
                "solver_status_code": solver_status_code,
                "solver_cost": solver_cost,
                "residual_norm": residual_norm,
                "position_error_m": float(position_error_m),
                "speed_mps": speed_mps,
                "command_delta_rate": float(command[0]) if len(command) > 0 else 0.0,
                "command_jerk": float(command[1]) if len(command) > 1 else 0.0,
                "command_theta_rate": float(command[2]) if len(command) > 2 else 0.0,
                "event_count": len(events),
            }
        )
        self._summary_file.flush()

        if events:
            with open(self.run_dir / "events.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"frame_id": frame_id, "events": events}) + "\n")
            self._event_count += 1

        self._frame_count += 1

    def close(self) -> None:
        if not self.enable or not self._started:
            return
        summary = {
            "frame_count": self._frame_count,
            "event_count": self._event_count,
        }
        with open(self.run_dir / "recording_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        if self._summary_file is not None:
            self._summary_file.close()
            self._summary_file = None
        self._started = False

    def _to_jsonable(self, obj: Any) -> Any:
        if is_dataclass(obj):
            return {k: self._to_jsonable(v) for k, v in asdict(obj).items()}
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {str(k): self._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_jsonable(v) for v in obj]
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return obj