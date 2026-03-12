from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

from config.defaults import CostConfig, SolverConfig, VehicleConfig
from .acados_ocp import NH, NX, NU, build_acados_ocp
from .types import EgoState, FeasibleCorridor, PlannerInput, PlannerOutput, SolveStatus, TrajectoryPoint
from .utils import unwrap_to_near

try:
    from acados_template import AcadosOcpSolver
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "acados_template is not available. Install acados first and then install "
        "the Python interface with: pip install -e $ACADOS_SOURCE_DIR/interfaces/acados_template"
    ) from exc


class SolverWrapper:
    def __init__(self, vehicle_cfg: VehicleConfig, solver_cfg: SolverConfig, cost_cfg: CostConfig):
        self.vehicle_cfg = vehicle_cfg
        self.solver_cfg = solver_cfg
        self.cost_cfg = cost_cfg
        self.nx = NX
        self.nu = NU
        self.N = solver_cfg.horizon
        self.dt = solver_cfg.dt
        self.project_root = Path(__file__).resolve().parents[1]
        self.json_file = str((self.project_root / solver_cfg.json_file).resolve())
        self._ocp = build_acados_ocp(vehicle_cfg, solver_cfg, cost_cfg, self.project_root)
        self.solver = AcadosOcpSolver(
            self._ocp,
            json_file=self.json_file,
            build=True,
            generate=True,
            verbose=False,
        )

        self.x_guess = np.zeros((self.N + 1, self.nx), dtype=float)
        self.u_guess = np.zeros((self.N, self.nu), dtype=float)
        self._initialized = False

    def solve(self, planner_input: PlannerInput) -> PlannerOutput:
        if not self._initialized:
            self._initialize_warm_start_from_reference(planner_input)
            self._initialized = True

        self._set_initial_state(planner_input.ego)
        self._set_stage_params(planner_input.local_ref)
        self._set_stage_path_constraints(planner_input.corridor)
        self._apply_warm_start()

        wall_t0 = time.perf_counter()
        status_code = int(self.solver.solve())
        wall_solve_ms = (time.perf_counter() - wall_t0) * 1000.0

        try:
            solve_time_ms = float(self.solver.get_stats("time_tot")) * 1000.0
        except Exception:
            solve_time_ms = wall_solve_ms

        try:
            solver_cost = float(self.solver.get_cost())
        except Exception:
            solver_cost = 0.0

        try:
            solver_residuals = np.array(self.solver.get_residuals(recompute=True), dtype=float).tolist()
        except Exception:
            solver_residuals = []

        solver_stats = {}
        for key in ("nlp_iter", "sqp_iter", "qp_iter", "time_qp", "time_lin", "time_sim"):
            try:
                value = self.solver.get_stats(key)
                if hasattr(value, "tolist"):
                    value = value.tolist()
                solver_stats[key] = value
            except Exception:
                pass

        residual_norm = float(sum(abs(float(v)) for v in solver_residuals)) if solver_residuals else 0.0
        if status_code != 0 or residual_norm > 1e3 or solver_cost > 1e6:
            return PlannerOutput(
                status=SolveStatus.SOLVER_FAILED,
                solve_time_ms=solve_time_ms,
                traj=[],
                solver_status_code=status_code,
                solver_cost=solver_cost,
                solver_residuals=solver_residuals,
                solver_stats=solver_stats,
            )

        traj = self._read_solution_trajectory()
        self._shift_warm_start_from_solution()
        return PlannerOutput(
            status=SolveStatus.OK,
            solve_time_ms=solve_time_ms,
            traj=traj,
            solver_status_code=status_code,
            solver_cost=solver_cost,
            solver_residuals=solver_residuals,
            solver_stats=solver_stats,
        )

    def _initialize_warm_start_from_reference(self, planner_input: PlannerInput) -> None:
        ego = planner_input.ego
        self.x_guess[0, :] = np.array([ego.x, ego.y, ego.yaw, ego.v, ego.delta, ego.a, ego.theta], dtype=float)
        for k in range(1, self.N + 1):
            ref = planner_input.local_ref[k]
            self.x_guess[k, :] = np.array([ref.x, ref.y, ref.yaw, ref.v_ref, 0.0, 0.0, ref.s], dtype=float)
        self._make_yaw_guess_continuous()
        self.u_guess.fill(0.0)

    def _set_initial_state(self, ego: EgoState) -> None:
        x0 = np.array([ego.x, ego.y, ego.yaw, ego.v, ego.delta, ego.a, ego.theta], dtype=float)
        x0[2] = unwrap_to_near(x0[2], self.x_guess[1, 2] if self.N >= 1 else x0[2])

        self.solver.constraints_set(0, "lbx", x0)
        self.solver.constraints_set(0, "ubx", x0)
        self.x_guess[0, :] = x0
        self._make_yaw_guess_continuous()

    def _set_stage_params(self, refs) -> None:
        for k in range(self.N + 1):
            ref = refs[k]
            pk = np.array([ref.x, ref.y, ref.yaw, ref.v_ref, ref.s, ref.kappa], dtype=float)
            self.solver.set(k, "p", pk)

    def _set_stage_path_constraints(self, corridor: FeasibleCorridor | None) -> None:
        wide = 1.0e3
        default_lh = np.array([-self.vehicle_cfg.a_lat_max, -wide], dtype=float)
        default_uh = np.array([self.vehicle_cfg.a_lat_max, wide], dtype=float)

        # acados nonlinear constraints h are not active at stage 0 by default.
        # So we only set bounds for stages 1..N.
        if corridor is None or not corridor.stations:
            for k in range(1, self.N + 1):
                self.solver.constraints_set(k, "lh", default_lh)
                self.solver.constraints_set(k, "uh", default_uh)
            return

        if len(corridor.stations) != self.N + 1:
            raise ValueError(
                f"corridor length mismatch: got {len(corridor.stations)}, expected {self.N + 1}"
            )

        # stage 0: do nothing
        # stage 1..N: apply corridor bounds
        for k in range(1, self.N + 1):
            station = corridor.stations[k]
            lh = np.array([-self.vehicle_cfg.a_lat_max, station.l_min], dtype=float)
            uh = np.array([self.vehicle_cfg.a_lat_max, station.l_max], dtype=float)
            self.solver.constraints_set(k, "lh", lh)
            self.solver.constraints_set(k, "uh", uh)

    def _apply_warm_start(self) -> None:
        self._make_yaw_guess_continuous()
        for k in range(self.N + 1):
            self.solver.set(k, "x", self.x_guess[k])
        for k in range(self.N):
            self.solver.set(k, "u", self.u_guess[k])

    def _read_solution_trajectory(self) -> List[TrajectoryPoint]:
        traj: List[TrajectoryPoint] = []
        xs = [np.array(self.solver.get(k, "x"), dtype=float).reshape(-1) for k in range(self.N + 1)]
        us = [np.array(self.solver.get(k, "u"), dtype=float).reshape(-1) for k in range(self.N)]

        for k in range(1, len(xs)):
            xs[k][2] = unwrap_to_near(xs[k][2], xs[k - 1][2])

        for k in range(self.N + 1):
            xk = xs[k]
            uk = us[k] if k < self.N else us[-1]
            traj.append(
                TrajectoryPoint(
                    t=k * self.dt,
                    x=float(xk[0]),
                    y=float(xk[1]),
                    yaw=float(xk[2]),
                    v=float(xk[3]),
                    delta=float(xk[4]),
                    a=float(xk[5]),
                    theta=float(xk[6]),
                    delta_rate=float(uk[0]),
                    jerk=float(uk[1]),
                    theta_rate=float(uk[2]),
                )
            )
        return traj

    def _shift_warm_start_from_solution(self) -> None:
        xs = [np.array(self.solver.get(k, "x"), dtype=float).reshape(-1) for k in range(self.N + 1)]
        us = [np.array(self.solver.get(k, "u"), dtype=float).reshape(-1) for k in range(self.N)]

        for k in range(1, len(xs)):
            xs[k][2] = unwrap_to_near(xs[k][2], xs[k - 1][2])

        for k in range(self.N):
            self.x_guess[k, :] = xs[k + 1]
        self.x_guess[self.N, :] = xs[self.N]

        for k in range(self.N - 1):
            self.u_guess[k, :] = us[k + 1]
        self.u_guess[self.N - 1, :] = us[self.N - 1]

        self._make_yaw_guess_continuous()

    def _make_yaw_guess_continuous(self) -> None:
        for k in range(1, self.N + 1):
            self.x_guess[k, 2] = unwrap_to_near(self.x_guess[k, 2], self.x_guess[k - 1, 2])

    def _integrate_step_numeric(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        dt = self.dt
        wb = self.vehicle_cfg.wheel_base

        def f(xv, uv):
            px, py, psi, v, delta, a, theta = xv
            delta_rate, jerk, theta_rate = uv
            return np.array(
                [
                    v * np.cos(psi),
                    v * np.sin(psi),
                    v / wb * np.tan(delta),
                    a,
                    delta_rate,
                    jerk,
                    theta_rate,
                ],
                dtype=float,
            )

        k1 = f(x, u)
        k2 = f(x + 0.5 * dt * k1, u)
        k3 = f(x + 0.5 * dt * k2, u)
        k4 = f(x + dt * k3, u)
        xn = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        xn[3] = float(np.clip(xn[3], self.vehicle_cfg.v_min, self.vehicle_cfg.v_max))
        xn[4] = float(np.clip(xn[4], -self.vehicle_cfg.delta_max, self.vehicle_cfg.delta_max))
        xn[5] = float(np.clip(xn[5], self.vehicle_cfg.a_min, self.vehicle_cfg.a_max))
        return xn

    def propagate(self, ego: EgoState, command: Tuple[float, float, float]) -> EgoState:
        x = np.array([ego.x, ego.y, ego.yaw, ego.v, ego.delta, ego.a, ego.theta], dtype=float)
        u = np.array(command, dtype=float)
        xn = self._integrate_step_numeric(x, u)
        return EgoState(
            x=float(xn[0]),
            y=float(xn[1]),
            yaw=float(xn[2]),
            v=float(xn[3]),
            delta=float(xn[4]),
            a=float(xn[5]),
            theta=float(xn[6]),
        )