from __future__ import annotations

from pathlib import Path

import numpy as np
import casadi as ca

from config.defaults import CostConfig, SolverConfig, VehicleConfig

try:
    from acados_template import AcadosModel, AcadosOcp
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "acados_template is not available. Install acados first and then install "
        "the Python interface with: pip install -e $ACADOS_SOURCE_DIR/interfaces/acados_template"
    ) from exc


NX = 7
NU = 3
NP = 5

# stage residual dimension:
# [ec, el, eyaw, v-ref_v, theta-ref_s, a, delta_rate, jerk, theta_rate-ref_v]
NY = 9

# terminal residual dimension:
# [px-ref_x, py-ref_y, eyaw, v-ref_v, theta-ref_s, a,
#  stop_gate*(v-ref_v), stop_gate*(theta-ref_s)]
NY_E = 8


def build_acados_ocp(
    vehicle_cfg: VehicleConfig,
    solver_cfg: SolverConfig,
    cost_cfg: CostConfig,
    project_root: Path,
) -> AcadosOcp:
    model = _build_model(vehicle_cfg)

    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.N_horizon = solver_cfg.horizon
    ocp.solver_options.tf = solver_cfg.horizon * solver_cfg.dt

    export_dir = project_root / solver_cfg.code_export_directory
    export_dir.parent.mkdir(parents=True, exist_ok=True)
    ocp.code_gen_opts.code_export_directory = str(export_dir)

    # =========================
    # COST
    # =========================
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    ocp.cost.W = np.diag(
        [
            cost_cfg.w_contour,      # ec
            cost_cfg.w_lag,          # el
            cost_cfg.w_yaw,          # eyaw
            cost_cfg.w_v,            # v - ref_v
            cost_cfg.w_theta,        # theta - ref_s
            cost_cfg.w_a,            # a
            cost_cfg.w_delta_rate,   # delta_rate
            cost_cfg.w_jerk,         # jerk
            cost_cfg.w_theta_rate,   # theta_rate - ref_v
        ]
    )

    ocp.cost.W_e = np.diag(
        [
            cost_cfg.w_terminal_xy,         # px - ref_x
            cost_cfg.w_terminal_xy,         # py - ref_y
            cost_cfg.w_terminal_yaw,        # eyaw
            cost_cfg.w_terminal_v,          # v - ref_v
            cost_cfg.w_terminal_theta,      # theta - ref_s
            cost_cfg.w_terminal_a,          # a
            cost_cfg.w_terminal_stop_v,     # stop_gate * (v - ref_v)
            cost_cfg.w_terminal_stop_theta, # stop_gate * (theta - ref_s)
        ]
    )

    ocp.cost.yref = np.zeros((NY,))
    ocp.cost.yref_e = np.zeros((NY_E,))

    # =========================
    # INPUT BOUNDS
    # =========================
    ocp.constraints.idxbu = np.array([0, 1, 2], dtype=int)
    ocp.constraints.lbu = np.array(
        [-vehicle_cfg.delta_rate_max, -vehicle_cfg.jerk_max, 0.0], dtype=float
    )
    ocp.constraints.ubu = np.array(
        [vehicle_cfg.delta_rate_max, vehicle_cfg.jerk_max, vehicle_cfg.v_max], dtype=float
    )

    # =========================
    # STATE BOUNDS
    # =========================
    ocp.constraints.idxbx = np.array([3, 4, 5], dtype=int)
    ocp.constraints.lbx = np.array(
        [vehicle_cfg.v_min, -vehicle_cfg.delta_max, vehicle_cfg.a_min], dtype=float
    )
    ocp.constraints.ubx = np.array(
        [vehicle_cfg.v_max, vehicle_cfg.delta_max, vehicle_cfg.a_max], dtype=float
    )

    ocp.constraints.x0 = np.zeros((NX,), dtype=float)

    # =========================
    # NONLINEAR CONSTRAINTS
    # =========================
    ocp.constraints.lh = np.array([-vehicle_cfg.a_lat_max], dtype=float)
    ocp.constraints.uh = np.array([vehicle_cfg.a_lat_max], dtype=float)
    ocp.constraints.lh_e = np.array([-vehicle_cfg.a_lat_max], dtype=float)
    ocp.constraints.uh_e = np.array([vehicle_cfg.a_lat_max], dtype=float)

    ocp.parameter_values = np.zeros((NP,), dtype=float)

    # =========================
    # SOLVER OPTIONS
    # =========================
    ocp.solver_options.qp_solver = solver_cfg.qp_solver
    ocp.solver_options.hessian_approx = solver_cfg.hessian_approx
    ocp.solver_options.integrator_type = solver_cfg.integrator_type
    ocp.solver_options.nlp_solver_type = solver_cfg.nlp_solver_type
    ocp.solver_options.nlp_solver_max_iter = solver_cfg.nlp_solver_max_iter
    ocp.solver_options.qp_solver_cond_N = solver_cfg.qp_cond_N
    ocp.solver_options.levenberg_marquardt = solver_cfg.levenberg_marquardt
    ocp.solver_options.print_level = 0

    return ocp


def _build_model(vehicle_cfg: VehicleConfig) -> AcadosModel:
    model = AcadosModel()
    model.name = "circle_track_nmpc"

    # states
    px = ca.SX.sym("px")
    py = ca.SX.sym("py")
    psi = ca.SX.sym("psi")
    v = ca.SX.sym("v")
    delta = ca.SX.sym("delta")
    a = ca.SX.sym("a")
    theta = ca.SX.sym("theta")
    x = ca.vertcat(px, py, psi, v, delta, a, theta)

    # controls
    delta_rate = ca.SX.sym("delta_rate")
    jerk = ca.SX.sym("jerk")
    theta_rate = ca.SX.sym("theta_rate")
    u = ca.vertcat(delta_rate, jerk, theta_rate)

    # xdot
    xdot = ca.SX.sym("xdot", NX)

    # parameters
    # [ref_x, ref_y, ref_yaw, ref_v, ref_s]
    p = ca.SX.sym("p", NP)
    ref_x = p[0]
    ref_y = p[1]
    ref_yaw = p[2]
    ref_v = p[3]
    ref_s = p[4]

    wb = vehicle_cfg.wheel_base

    # dynamics
    f_expl = ca.vertcat(
        v * ca.cos(psi),
        v * ca.sin(psi),
        v / wb * ca.tan(delta),
        a,
        delta_rate,
        jerk,
        theta_rate,
    )
    f_impl = xdot - f_expl

    # geometric tracking errors
    ec = -ca.sin(ref_yaw) * (px - ref_x) + ca.cos(ref_yaw) * (py - ref_y)
    el =  ca.cos(ref_yaw) * (px - ref_x) + ca.sin(ref_yaw) * (py - ref_y)
    eyaw = ca.atan2(ca.sin(psi - ref_yaw), ca.cos(psi - ref_yaw))

    # -------------------------
    # stage residual
    # -------------------------
    model.cost_y_expr = ca.vertcat(
        ec,
        el,
        eyaw,
        v - ref_v,
        theta - ref_s,
        a,                    # NEW: penalize acceleration itself
        delta_rate,
        jerk,
        theta_rate - ref_v,
    )

    # -------------------------
    # terminal stop gate
    # -------------------------
    # ref_v -> 0 near stop, so stop_gate -> 1
    # ref_v large in cruise, so stop_gate -> small
    v_stop_scale = 0.5
    stop_gate = 1.0 / (1.0 + (ref_v / v_stop_scale) ** 2)

    # -------------------------
    # terminal residual
    # -------------------------
    model.cost_y_expr_e = ca.vertcat(
        px - ref_x,
        py - ref_y,
        eyaw,
        v - ref_v,
        theta - ref_s,
        a,                            # NEW: terminal acceleration penalty
        stop_gate * (v - ref_v),      # NEW: explicitly strengthen terminal stop speed near stop
        stop_gate * (theta - ref_s),  # NEW: explicitly strengthen stop position/progress near stop
    )

    # lateral acceleration constraint
    alat = v * v / wb * ca.tan(delta)
    model.con_h_expr = ca.vertcat(alat)
    model.con_h_expr_e = ca.vertcat(alat)

    model.x = x
    model.xdot = xdot
    model.u = u
    model.p = p
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl

    return model