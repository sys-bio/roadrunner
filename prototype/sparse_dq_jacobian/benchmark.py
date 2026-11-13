"""Benchmark of the sparse difference quotient Jacobian and the KLU solver of CVODE.

The same four models and simulations as the Enzyme Jacobian prototype (sys-bio/roadrunner#1418):
`simulate(0, end, 201)` at rtol = atol = 1e-10, after a reset and the dose of the model. Every
model is measured with two builds, each in a process of its own, with LLJit at the optimization
level DEFAULT:

- this build (SPARSE_BUILD): dense + difference quotient (CVODE's own), dense + sparse
  difference quotient, and KLU + sparse difference quotient;
- the build of the Enzyme prototype of #1418 (ENZYME_BUILD): dense + difference quotient and
  KLU + Enzyme. It has the local patch enzyme-counter-totals.patch, which sums its counters over
  the reinitializations of CVODE like getStatistics of this build does.

Per model and configuration: the median wall time of the simulation (the configurations take
turns), its CVODE counters, its error against a reference simulation, and the time of one
Jacobian. The counters of both builds are summed over the reinitializations of CVODE (at the roots
of events and piecewise conditions). The reference is dense + difference quotient at rtol = 1e-12, atol = 1e-14; the error is
the largest |x - x_ref| / (atol + rtol * |x_ref|) over the output, in units of the tolerances of
the benchmark. The rounds take turns between the builds and the models, and every number in the
table is the median over the rounds.

The time of one Jacobian: for the sparse difference quotient, the time CVODE spends in the
Jacobian callback per call (setting collect_timing); for CVODE's dense difference quotient, n
times the time of one evaluation of the state vector rate (as in #1418); for Enzyme, the C++
timing tool of the Enzyme prototype at the state in the middle of the timecourse.

Usage:
    python benchmark.py [--rounds 3] [--reps 10] [--out benchmark.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
HOME = Path.home()
SPARSE_BUILD = HOME / "rr-enzyme" / "build-sparsedq"
# the Enzyme prototype of #1418 with enzyme-counter-totals.patch
ENZYME_BUILD = HOME / "rr-enzyme" / "build-enzyme-totals"
LLJIT_DEFAULT = 2

SPARSE_CONFIGS = [("difference_quotient", "dense"), ("sparse_difference_quotient", "dense"),
                  ("sparse_difference_quotient", "klu")]
ENZYME_CONFIGS = [("difference_quotient", "dense"), ("enzyme", "klu")]


def worker(build: str, model: str, reps: int) -> None:
    """Measure one model with the roadrunner on sys.path; print one JSON object."""
    import numpy as np
    import roadrunner
    from models import benchmark_cases

    roadrunner.Logger.setLevel(roadrunner.Logger.LOG_ERROR)
    C = roadrunner.Config
    C.setValue(C.LLVM_BACKEND, C.LLJIT)
    C.setValue(C.LLJIT_OPTIMIZATION_LEVEL, LLJIT_DEFAULT)
    enzyme = build == "enzyme"
    if enzyme:
        C.setValue(C.LLVM_ENZYME_JACOBIAN, True)
    case = benchmark_cases()[model]

    t0 = time.perf_counter()
    rr = roadrunner.RoadRunner(str(case.sbml))
    load = time.perf_counter() - t0

    def configure(jacobian: str, solver: str) -> None:
        # KLU needs the matching Jacobian at every moment
        rr.integrator.setValue("linear_solver", "dense")
        rr.integrator.setValue("jacobian", jacobian)
        rr.integrator.setValue("linear_solver", solver)

    def run(jacobian: str, solver: str, rtol: float = 1e-10, atol: float = 1e-10):
        configure(jacobian, solver)
        rr.integrator.setValue("relative_tolerance", rtol)
        rr.integrator.setValue("absolute_tolerance", atol)
        rr.resetAll()
        case.setup(rr)
        t0 = time.perf_counter()
        result = np.array(rr.simulate(0, case.end, 201))
        return time.perf_counter() - t0, result

    configs = ENZYME_CONFIGS if enzyme else SPARSE_CONFIGS
    _, reference = run("difference_quotient", "dense", 1e-12, 1e-14)
    times = [[] for _ in configs]
    results = [None for _ in configs]
    for _ in range(reps):
        for k, config in enumerate(configs):
            wall, results[k] = run(*config)
            times[k].append(wall)

    rows = []
    for k, (jacobian, solver) in enumerate(configs):
        if enzyme:
            # with the patch, the counters only grow; the counters of the run are the difference
            configure(jacobian, solver)
            before = rr.integrator.getSolverStatistics()
            run(jacobian, solver)
            after = rr.integrator.getSolverStatistics()
            s = {key: after[key] - before[key] for key in after}
            stats = {"steps": s["steps"], "err_test_fails": s["err_test_fails"], "jac_evals": s["jac_evals"],
                     "rhs_evals": s["rhs_evals"], "rhs_evals_jac": s["rhs_evals_jac"], "lin_setups": s["lin_setups"]}
        else:
            run(jacobian, solver)
            s = rr.integrator.getStatistics()
            stats = {"steps": s["num_steps"], "err_test_fails": s["num_err_test_fails"],
                     "jac_evals": s["num_jac_evals"], "rhs_evals": s["num_rhs_evals"],
                     "rhs_evals_jac": s["num_lin_rhs_evals"] + s["num_jac_model_evals"],
                     "lin_setups": s["num_lin_solv_setups"], "colors": s["num_jac_colors"],
                     "reinits": s["num_reinits"]}
        error = np.max(np.abs(results[k][:, 1:] - reference[:, 1:]) / (1e-10 + 1e-10 * np.abs(reference[:, 1:])))
        rows.append({"build": build, "model": model, "config": f"{solver}+{jacobian}", "load_s": load,
                     "simulate_s": median(times[k]), "error_per_tol": float(error), **stats})

    # the time of one Jacobian
    # the absolute tolerance vector has one entry per state of CVODE
    states = len(rr.integrator.getAbsoluteToleranceVector())
    if not enzyme:
        rr.integrator.setValue("collect_timing", True)
        run("difference_quotient", "dense")
        s = rr.integrator.getStatistics()
        t_rate = s["time_rhs_evals"] / (s["num_rhs_evals"] + s["num_lin_rhs_evals"])
        run("sparse_difference_quotient", "dense")
        s = rr.integrator.getStatistics()
        t_sparse = s["time_jac_evals"] / max(s["num_jac_evals"], 1)
        rr.integrator.setValue("collect_timing", False)
        for row in rows:
            row.update({"states": states, "t_rate_s": t_rate, "t_jac_dq_s": states * t_rate, "t_jac_s": (
                t_sparse if "sparse" in row["config"] else states * t_rate)})
    print(json.dumps(rows))


def enzyme_jacobian_time(case) -> dict:
    """The C++ timing tool of the Enzyme prototype, at the state in the middle of the timecourse."""
    args = [str(ENZYME_BUILD / "bin" / "enzyme_jacobian_timing"), str(case.sbml), str(case.end / 2)]
    if case.dose is not None:
        args += [case.dose[0], str(case.dose[1])]
    args += ["21"]
    env = {**os.environ, "RR_LLVM_BACKEND": "lljit", "RR_LLJIT_OPT": "default"}
    out = subprocess.run(args, capture_output=True, text=True, check=True, env=env).stdout
    return json.loads(out.strip().splitlines()[-1])


def driver(rounds: int, reps: int, out: Path) -> None:
    sys.path.insert(0, str(HERE))
    from models import benchmark_cases
    cases = benchmark_cases()
    runs: dict[tuple[str, str, str], list[dict]] = {}
    for r in range(rounds):
        for model, case in cases.items():
            for build, prefix in (("sparse", SPARSE_BUILD), ("enzyme", ENZYME_BUILD)):
                env = {**os.environ, "PYTHONPATH": str(prefix / "lib" / "site-packages") + os.pathsep + str(HERE)}
                proc = subprocess.run([sys.executable, __file__, "--worker", build, model, "--reps", str(reps)],
                                      capture_output=True, text=True, env=env)
                if proc.returncode != 0:
                    raise RuntimeError(f"{build} {model}: {proc.stderr[-2000:]}")
                rows = json.loads(proc.stdout.strip().splitlines()[-1])
                if build == "enzyme":
                    timing = enzyme_jacobian_time(case)
                    for row in rows:
                        row.update({"states": timing["states"], "t_rate_s": timing["t_rate"],
                                    "t_jac_dq_s": timing["t_dq"],
                                    "t_jac_s": timing["t_enzyme"] if "enzyme" in row["config"] else timing["t_dq"]})
                for row in rows:
                    runs.setdefault((row["build"], row["model"], row["config"]), []).append(row)
                    print(r, row["build"], row["model"], row["config"], f"{row['simulate_s'] * 1000:.1f} ms",
                          flush=True)
    rows = []
    for group in runs.values():
        row = dict(group[0])
        for column in ("load_s", "simulate_s", "t_rate_s", "t_jac_dq_s", "t_jac_s"):
            row[column] = median(float(g[column]) for g in group)
        row["rounds"] = len(group)
        rows.append(row)
    columns = sorted({k for r in rows for k in r}, key=lambda k: (list(rows[0].keys()) + ["colors", "reinits"]).index(k))
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--out", default=str(HERE / "benchmark.csv"))
    parser.add_argument("--worker", nargs=2, metavar=("BUILD", "MODEL"))
    args = parser.parse_args()
    if args.worker:
        worker(args.worker[0], args.worker[1], args.reps)
    else:
        driver(args.rounds, args.reps, Path(args.out))


if __name__ == "__main__":
    main()
