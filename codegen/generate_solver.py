from __future__ import annotations

from pathlib import Path

from config.defaults import CostConfig, SolverConfig, VehicleConfig
from nmpc_planner.acados_ocp import build_acados_ocp

try:
    from acados_template import AcadosOcpSolver
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "acados_template is not available. Install acados first and then install "
        "the Python interface with: pip install -e $ACADOS_SOURCE_DIR/interfaces/acados_template"
    ) from exc


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    ocp = build_acados_ocp(VehicleConfig(), SolverConfig(), CostConfig(), project_root)
    json_file = str((project_root / SolverConfig().json_file).resolve())
    solver = AcadosOcpSolver(
        ocp,
        json_file=json_file,
        build=True,
        generate=True,
        verbose=True,
    )
    del solver
    print(f"Generated acados solver under: {project_root / SolverConfig().code_export_directory}")
    print(f"JSON description written to: {json_file}")


if __name__ == "__main__":
    main()