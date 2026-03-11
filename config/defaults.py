from dataclasses import dataclass


@dataclass(frozen=True)
class VehicleConfig:
    wheel_base: float = 2.8
    v_min: float = 0.0
    v_max: float = 8.0
    delta_max: float = 0.6
    delta_rate_max: float = 0.5
    a_min: float = -3.0
    a_max: float = 2.0
    jerk_max: float = 2.5
    a_lat_max: float = 3.0


@dataclass(frozen=True)
class SolverConfig:
    dt: float = 0.3
    horizon: int = 12
    nlp_solver_type: str = "SQP_RTI"
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM"
    hessian_approx: str = "GAUSS_NEWTON"
    integrator_type: str = "ERK"
    nlp_solver_max_iter: int = 50
    qp_cond_N: int = 6
    levenberg_marquardt: float = 1e-4
    json_file: str = "generated/acados_ocp_circle_track.json"
    code_export_directory: str = "generated/acados_circle_track_solver"
    use_cython_wrapper: bool = False


@dataclass(frozen=True)
class CostConfig:
    w_contour: float = 45.0
    w_lag: float = 6.0
    w_yaw: float = 8.0
    w_v: float = 5.0
    w_theta: float = 3.5
    w_delta_rate: float = 0.25
    w_jerk: float = 0.10
    w_theta_rate: float = 0.80
    w_du: float = 0.60
    w_terminal_xy: float = 120.0
    w_terminal_yaw: float = 20.0
    w_terminal_v: float = 15.0
    w_terminal_theta: float = 12.0


@dataclass(frozen=True)
class SimConfig:
    max_steps: int = 260
    stop_tolerance_m: float = 0.5
    stop_speed_mps: float = 0.25
    reference_search_margin: float = 12.0
    path_ds: float = 0.25
    trajectory_width: float = 4.0