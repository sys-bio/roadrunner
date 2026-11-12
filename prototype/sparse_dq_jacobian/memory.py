"""Memory use of the sparse difference quotient, KLU and the Enzyme Jacobian.

For every model, build and configuration, a fresh process (Linux) measures from /proc/self/status:

- load: the resident set size (RSS) after loading the model minus the RSS before, and the peak RSS
  during the load (code generation and compilation) minus the RSS before;
- configured: the RSS after setting the Jacobian and the linear solver of CVODE (which creates
  them) minus the RSS before the load;
- simulate: the peak RSS during `simulate(0, end, 201)` at rtol = atol = 1e-10 minus the RSS before
  it (the Jacobian values, the factorization and the workspace of the difference quotient);
- total peak: the peak RSS from the load to the end of the simulation minus the RSS before the load;
- heap: the memory allocated with malloc and in use after the simulation (glibc mallinfo2), which
  holds the CVODE memory, the matrices and factorizations of the linear solver and the workspace of
  the Jacobian; its difference between the configurations is their difference in memory, which
  the RSS does not show, since freed memory stays in the process. The code that LLVM compiles is
  not in the heap but in the RSS of the load.

The peak RSS (VmHWM) is reset before every phase by writing 5 to /proc/self/clear_refs. Every
number is the median over `--reps` processes. The builds and the JIT are those of benchmark.py:
this build, and the #1418 build with the model loaded with Config.LLVM_ENZYME_JACOBIAN ("enzyme")
and without it ("enzyme-off").

Usage:
    python memory.py [--reps 3] [--out memory.csv]
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import median

from benchmark import ENZYME_BUILD, ENZYME_CONFIGS, HERE, LLJIT_DEFAULT, SPARSE_BUILD, SPARSE_CONFIGS


def status_kib(field: str) -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1])
    raise KeyError(field)


class _Mallinfo2(ctypes.Structure):
    _fields_ = [(name, ctypes.c_size_t) for name in
                ("arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks", "fsmblks", "uordblks", "fordblks",
                 "keepcost")]


def heap_kib() -> float:
    """The bytes allocated with malloc and in use (glibc mallinfo2), in KiB. Unlike the RSS, it
    drops when memory is freed, so it shows the memory of the linear solver that replaces another."""
    libc = ctypes.CDLL("libc.so.6")
    libc.mallinfo2.restype = _Mallinfo2
    info = libc.mallinfo2()
    return (info.uordblks + info.hblkhd) / 1024


def reset_peak() -> None:
    with open("/proc/self/clear_refs", "w") as f:
        f.write("5")


def worker(build: str, model: str, jacobian: str, solver: str) -> None:
    """Measure one model and configuration; print one JSON object (MiB)."""
    import roadrunner
    from models import benchmark_cases

    roadrunner.Logger.setLevel(roadrunner.Logger.LOG_ERROR)
    C = roadrunner.Config
    C.setValue(C.LLVM_BACKEND, C.LLJIT)
    C.setValue(C.LLJIT_OPTIMIZATION_LEVEL, LLJIT_DEFAULT)
    if build == "enzyme":
        C.setValue(C.LLVM_ENZYME_JACOBIAN, True)
    case = benchmark_cases()[model]

    reset_peak()
    start = status_kib("VmRSS")
    rr = roadrunner.RoadRunner(str(case.sbml))
    load_peak = status_kib("VmHWM") - start
    load = status_kib("VmRSS") - start

    rr.integrator.setValue("linear_solver", "dense")
    rr.integrator.setValue("jacobian", jacobian)
    rr.integrator.setValue("linear_solver", solver)
    rr.integrator.setValue("relative_tolerance", 1e-10)
    rr.integrator.setValue("absolute_tolerance", 1e-10)
    rr.resetAll()
    case.setup(rr)
    configured = status_kib("VmRSS") - start
    heap_loaded = heap_kib()
    reset_peak()
    before = status_kib("VmRSS")
    rr.simulate(0, case.end, 201)
    simulate_peak = status_kib("VmHWM") - before
    total_peak = max(load_peak, before - start + simulate_peak)
    heap_simulated = heap_kib()
    print(json.dumps({"load_mib": load / 1024, "load_peak_mib": load_peak / 1024,
                      "configured_mib": configured / 1024, "simulate_peak_mib": simulate_peak / 1024,
                      "total_peak_mib": total_peak / 1024, "heap_mib": heap_simulated / 1024,
                      "heap_simulate_mib": (heap_simulated - heap_loaded) / 1024}))


def driver(reps: int, out: Path) -> None:
    sys.path.insert(0, str(HERE))
    from models import benchmark_cases
    rows = []
    for model in benchmark_cases():
        for build, prefix, configs in (("sparse", SPARSE_BUILD, SPARSE_CONFIGS),
                                       ("enzyme", ENZYME_BUILD, ENZYME_CONFIGS),
                                       ("enzyme-off", ENZYME_BUILD, [("difference_quotient", "dense")])):
            env = {**os.environ, "PYTHONPATH": str(prefix / "lib" / "site-packages") + os.pathsep + str(HERE)}
            for jacobian, solver in configs:
                runs = []
                for _ in range(reps):
                    proc = subprocess.run([sys.executable, __file__, "--worker", build, model, jacobian, solver],
                                          capture_output=True, text=True, env=env)
                    if proc.returncode != 0:
                        raise RuntimeError(f"{build} {model} {jacobian} {solver}: {proc.stderr[-2000:]}")
                    runs.append(json.loads(proc.stdout.strip().splitlines()[-1]))
                row = {"build": build, "model": model, "config": f"{solver}+{jacobian}"}
                for key in runs[0]:
                    row[key] = median(r[key] for r in runs)
                rows.append(row)
                print(row, flush=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--out", default=str(HERE / "memory.csv"))
    parser.add_argument("--worker", nargs=4, metavar=("BUILD", "MODEL", "JACOBIAN", "SOLVER"))
    args = parser.parse_args()
    if args.worker:
        worker(*args.worker)
    else:
        driver(args.reps, Path(args.out))


if __name__ == "__main__":
    main()
