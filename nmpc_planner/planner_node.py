from __future__ import annotations

from pathlib import Path

from config.defaults import CostConfig, SimConfig, SolverConfig, VehicleConfig
from .corridor_builder import CorridorBuilder
from .debug_recorder import DebugRecorder
from .fallback_manager import FallbackManager
from .obstacle_predictor import ObstaclePredictor
from .reference_manager import ReferenceManager
from .scenario_builder import ScenarioBuilder
from .solver_wrapper import SolverWrapper
from .trajectory_validator import TrajectoryValidator
from .types import EgoState, PlannerInput, RunLog, SolveStatus


class PlannerNode:
    def __init__(self):
        self.vehicle_cfg = VehicleConfig()
        self.solver_cfg = SolverConfig()
        self.cost_cfg = CostConfig()
        self.sim_cfg = SimConfig()

        self.scenario = ScenarioBuilder(self.sim_cfg).build()
        self.ref_manager = ReferenceManager(self.scenario, self.solver_cfg, self.sim_cfg)
        self.corridor_builder = CorridorBuilder(self.sim_cfg)
        self.obstacle_predictor = ObstaclePredictor()
        self.solver = SolverWrapper(self.vehicle_cfg, self.solver_cfg, self.cost_cfg)
        self.fallback = FallbackManager(self.solver_cfg)
        self.validator = TrajectoryValidator(self.vehicle_cfg, self.solver_cfg)
        self.log = RunLog()

        results_root = Path(__file__).resolve().parents[1] / "results"
        self.recorder = DebugRecorder(results_root / "debug_runs", enable=True)

    def _detect_events(self, output, ego_after: EgoState, position_error_m: float):
        events = []
        if output.status != SolveStatus.OK:
            events.append("SOLVER_NOT_OK")
        if output.solve_time_ms > 0.8 * self.solver_cfg.dt * 1000.0:
            events.append("HIGH_SOLVE_TIME")
        if output.solver_status_code != 0:
            events.append("SOLVER_STATUS_NONZERO")
        if output.solver_residuals:
            residual_norm = sum(abs(float(v)) for v in output.solver_residuals)
            if residual_norm > 1e-2:
                events.append("HIGH_RESIDUAL")
        if abs(ego_after.a) >= self.vehicle_cfg.a_max - 1e-6 or abs(ego_after.a) <= self.vehicle_cfg.a_min + 1e-6:
            events.append("ACC_HIT_LIMIT")
        if abs(ego_after.delta) >= self.vehicle_cfg.delta_max - 1e-6:
            events.append("STEER_HIT_LIMIT")
        if position_error_m > 1.0:
            events.append("LARGE_POSITION_ERROR")
        return events

    def run(self):
        ego = EgoState(x=0.0, y=0.0, yaw=0.0, v=0.0, delta=0.0, a=0.0, theta=0.0)
        self.log.states.append(ego)

        self.recorder.start(
            {
                "vehicle_cfg": self.vehicle_cfg,
                "solver_cfg": self.solver_cfg,
                "cost_cfg": self.cost_cfg,
                "sim_cfg": self.sim_cfg,
                "scenario": {
                    "goal_x": self.scenario.goal_x,
                    "goal_y": self.scenario.goal_y,
                    "goal_yaw": self.scenario.goal_yaw,
                    "total_length": self.scenario.total_length,
                },
            }
        )

        try:
            for frame_id in range(self.sim_cfg.max_steps):
                sim_time = frame_id * self.solver_cfg.dt
                ego_before = ego
                local_ref = self.ref_manager.build_local_reference(ego_before)
                corridor = self.corridor_builder.build(local_ref)
                obstacles = self.obstacle_predictor.predict()
                planner_input = PlannerInput(ego=ego_before, local_ref=local_ref)
                output = self.solver.solve(planner_input)

                if output.status != SolveStatus.OK or not output.traj:
                    output = self.fallback.safe_stop(ego_before)

                cmd = [
                    output.traj[0].delta_rate,
                    output.traj[0].jerk,
                    output.traj[0].theta_rate,
                ]
                ego = self.solver.propagate(ego_before, tuple(cmd))

                self.log.states.append(ego)
                self.log.commands.append(cmd)
                self.log.references.append(local_ref)
                self.log.solve_times_ms.append(output.solve_time_ms)
                self.log.statuses.append(output.status.value)
                self.log.solver_status_codes.append(output.solver_status_code)
                self.log.solver_costs.append(output.solver_cost)

                pos_err = ((ego.x - self.scenario.goal_x) ** 2 + (ego.y - self.scenario.goal_y) ** 2) ** 0.5
                events = self._detect_events(output, ego, pos_err)
                self.recorder.record_frame(
                    frame_id=frame_id,
                    sim_time=sim_time,
                    ego_before=ego_before,
                    local_ref=local_ref,
                    corridor=corridor,
                    obstacles=obstacles,
                    planner_output=output,
                    command=cmd,
                    ego_after=ego,
                    position_error_m=pos_err,
                    events=events,
                )

                if pos_err <= self.sim_cfg.stop_tolerance_m and ego.v <= self.sim_cfg.stop_speed_mps:
                    break
        finally:
            self.recorder.close()

        out_dir = Path(__file__).resolve().parents[1] / "results"
        metrics = self.validator.save_plots(self.log, self.scenario, out_dir)

        print("Simulation finished")
        print(f"final position error [m]: {metrics.final_position_error_m:.3f}")
        print(f"final speed [m/s]: {metrics.final_speed_mps:.3f}")
        print(f"max |steering rate| [rad/s]: {metrics.max_abs_steering_rate:.3f}")
        print(f"max |jerk| [m/s^3]: {metrics.max_abs_jerk:.3f}")
        print(f"max |lateral accel| [m/s^2]: {metrics.max_abs_lateral_accel:.3f}")
        print(f"RMS contour error [m]: {metrics.rms_contour_error:.3f}")
        print(f"mean solve time [ms]: {metrics.mean_solve_time_ms:.2f}")
        print(f"outputs saved to: {out_dir}")
        print(f"debug run saved to: {self.recorder.current_run_dir}")


if __name__ == "__main__":
    PlannerNode().run()