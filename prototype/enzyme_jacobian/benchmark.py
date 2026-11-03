"""Benchmark of the Enzyme Jacobian and the KLU solver of CVODE on the four benchmark models.

Per model and configuration (dense + difference quotient, dense + Enzyme, KLU + Enzyme), all from
the same build: the median wall time of `simulate(0, end, 201)` at rtol = atol = 1e-10 and the
CVODE counters of that simulation. Per model: the median load time with recompilation, with and
without Config.LLVM_ENZYME_JACOBIAN, and, from the C++ tool enzyme_jacobian_timing, the time of
one Enzyme Jacobian against the difference quotient of CVODE (n evaluations of the state vector
rate) at the state in the middle of the timecourse.

`--baseline-only` measures only the load time and the dense + difference quotient simulation with
the Enzyme Jacobian switched off; it runs with the Python of a `develop` build as well, which is
the check that the prototype costs nothing when switched off.

Usage:
    python benchmark.py [--reps 20] [--out benchmark.csv]
    python benchmark.py --baseline-only --label develop --out baseline-develop.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path
from statistics import median

import roadrunner

from models import benchmark_cases

HERE = Path(__file__).resolve().parent
TIMING = Path(os.environ.get("RR_ENZYME_TIMING", Path.home() / "rr-enzyme" / "build" / "bin" / "enzyme_jacobian_timing"))
CONFIGS = [("difference_quotient", "dense"), ("enzyme", "dense"), ("enzyme", "klu")]
STATS = ["steps", "rhs_evals", "rhs_evals_jac", "jac_evals", "nonlin_iters", "nonlin_conv_fails", "err_test_fails", "lin_setups"]


def load_time(path: Path, enzyme: bool, reps: int) -> float:
    """Median time of loading and compiling the model (the model cache is bypassed)."""
    roadrunner.Config.setValue(roadrunner.Config.LOADSBMLOPTIONS_RECOMPILE, True)
    if hasattr(roadrunner.Config, "LLVM_ENZYME_JACOBIAN"):
        roadrunner.Config.setValue(roadrunner.Config.LLVM_ENZYME_JACOBIAN, enzyme)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        roadrunner.RoadRunner(str(path))
        times.append(time.perf_counter() - t0)
    roadrunner.Config.setValue(roadrunner.Config.LOADSBMLOPTIONS_RECOMPILE, False)
    return median(times)


def configure(rr, jacobian: str, solver: str) -> None:
    """Set the Jacobian and the linear solver ("klu" needs "enzyme" at every moment)."""
    rr.integrator.setValue("linear_solver", "dense")
    rr.integrator.setValue("jacobian", jacobian)
    rr.integrator.setValue("linear_solver", solver)


def simulations(rr, case, configs: list[tuple[str, str] | None], reps: int) -> list[tuple[float, dict]]:
    """Median wall time of the simulation and the CVODE counters of the last one, per configuration.
    The configurations take turns in every repetition, so a change of the load of the machine
    affects all of them alike."""
    rr.integrator.setValue("relative_tolerance", 1e-10)
    rr.integrator.setValue("absolute_tolerance", 1e-10)
    times: list[list[float]] = [[] for _ in configs]
    stats: list[dict] = [{} for _ in configs]
    for _ in range(reps):
        for k, config in enumerate(configs):
            if config is not None:
                configure(rr, *config)
            rr.resetAll()
            case.setup(rr)
            t0 = time.perf_counter()
            rr.simulate(0, case.end, 201)
            times[k].append(time.perf_counter() - t0)
            if hasattr(rr.integrator, "getSolverStatistics"):
                stats[k] = rr.integrator.getSolverStatistics()
    return [(median(t), s) for t, s in zip(times, stats)]


def jacobian_timing(case) -> dict:
    """Run the C++ timing tool at the state in the middle of the timecourse."""
    args = [str(TIMING), str(case.sbml), str(case.end / 2)]
    if case.dose is not None:
        args += [case.dose[0], str(case.dose[1])]
    args += ["21"]
    out = subprocess.run(args, capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def width_experiment(out: Path, widths: tuple[int, ...] = (1, 2, 4, 16)) -> None:
    """Load time and time per Enzyme Jacobian against the vector width (RR_ENZYME_MAX_WIDTH)."""
    rows = []
    for name, case in benchmark_cases().items():
        for width in widths:
            env = {**os.environ, "RR_ENZYME_MAX_WIDTH": str(width)}
            args = [str(TIMING), str(case.sbml), str(case.end / 2)]
            if case.dose is not None:
                args += [case.dose[0], str(case.dose[1])]
            args += ["21"]
            result = json.loads(subprocess.run(args, capture_output=True, text=True, check=True, env=env)
                                .stdout.strip().splitlines()[-1])
            width_used = min(width, result["colors"])
            rows.append({"model": name, "max_width": width, "width": width_used,
                         "passes": -(-result["colors"] // width_used), **result})
            print(rows[-1], flush=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--load-reps", type=int, default=5)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--label", default="feature")
    parser.add_argument("--out", default=str(HERE / "benchmark.csv"))
    parser.add_argument("--widths", action="store_true", help="only the vector width experiment (width.csv)")
    args = parser.parse_args()
    if args.widths:
        width_experiment(HERE / "width.csv")
        return

    roadrunner.Logger.setLevel(roadrunner.Logger.LOG_ERROR)
    rows = []
    for name, case in benchmark_cases().items():
        print(f"{name} ...", flush=True)
        if args.baseline_only:
            if hasattr(roadrunner.Config, "LLVM_ENZYME_JACOBIAN"):
                roadrunner.Config.setValue(roadrunner.Config.LLVM_ENZYME_JACOBIAN, False)
            rr = roadrunner.RoadRunner(str(case.sbml))
            (wall, _), = simulations(rr, case, [None], args.reps)
            rows.append({"build": args.label, "model": name, "config": "dense+difference_quotient",
                         "load_s": load_time(case.sbml, False, args.load_reps), "simulate_s": wall})
            continue

        load_plain = load_time(case.sbml, False, args.load_reps)
        load_enzyme = load_time(case.sbml, True, args.load_reps)
        timing = jacobian_timing(case)
        roadrunner.Config.setValue(roadrunner.Config.LLVM_ENZYME_JACOBIAN, True)
        rr = roadrunner.RoadRunner(str(case.sbml))
        for (jacobian, solver), (wall, stats) in zip(CONFIGS, simulations(rr, case, CONFIGS, args.reps)):
            rows.append({
                "build": args.label, "model": name, "config": f"{solver}+{jacobian}",
                "states": timing["states"], "jac_nnz": timing["nnz"], "colors": timing["colors"],
                "load_s": load_plain, "load_enzyme_s": load_enzyme,
                "t_rate_s": timing["t_rate"], "t_jac_dq_s": timing["t_dq"], "t_jac_enzyme_s": timing["t_enzyme"],
                "simulate_s": wall, **{k: stats.get(k) for k in STATS},
                "petab_commit": case.petab_commit or "",
            })
        roadrunner.Config.setValue(roadrunner.Config.LLVM_ENZYME_JACOBIAN, False)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if args.baseline_only else
                                sorted({k for r in rows for k in r}, key=list(rows[0].keys()).index))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
