"""Load time of the Enzyme Jacobian against the number of reactions per differentiated block.

For Chen_MSB2009 (827 reactions) and each block size (RR_ENZYME_BLOCK_SIZE) and JIT backend: the
time of the Enzyme pass, of the IR setup before it (sparsity, copies of evalReactionRates reduced
to their block), of the compilation, and of the whole load, from the log of roadrunner, as medians
over fresh processes; and the time of one Enzyme Jacobian from enzyme_jacobian_timing.

Usage: python blocks.py [--reps 3]    # writes blocks.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import median

from benchmark import TIMING
from models import backend_environment, benchmark_cases

HERE = Path(__file__).resolve().parent
BLOCK_SIZES = (100000, 256, 128, 64, 32, 16)
BACKENDS = ("mcjit", "lljit-none")

LOAD = """
import sys, time, roadrunner
sys.path.insert(0, {here!r})
from models import apply_backend
apply_backend()
C = roadrunner.Config
C.setValue(C.LOADSBMLOPTIONS_RECOMPILE, True)
C.setValue(C.LLVM_ENZYME_JACOBIAN, True)
roadrunner.Logger.setLevel(roadrunner.Logger.LOG_INFORMATION)
roadrunner.Logger.enableFileLogging({log!r})
roadrunner.Logger.disableConsoleLogging()
t0 = time.perf_counter()
roadrunner.RoadRunner({path!r})
print(time.perf_counter() - t0)
"""


def load_phases(path: Path, env: dict[str, str]) -> dict[str, float]:
    """Load the model once in a fresh process; the phases from roadrunner's log."""
    with tempfile.TemporaryDirectory() as tmp:
        log = str(Path(tmp) / "rr.log")
        out = subprocess.run([sys.executable, "-c", LOAD.format(here=str(HERE), log=log, path=str(path))],
                             capture_output=True, text=True, check=True, env=env).stdout
        text = Path(log).read_text()
    blocks, setup, enzyme = re.search(
        r"Enzyme Jacobian: (\d+) reaction rate blocks, sparsity and IR setup ([\d.e-]+) s, Enzyme pass ([\d.e-]+) s", text
    ).groups()
    compilation = re.search(r"Enzyme Jacobian: preparation [\d.e-]+ s, compilation ([\d.e-]+) s", text).group(1)
    return {"blocks": int(blocks), "setup_s": float(setup), "enzyme_pass_s": float(enzyme),
            "compile_s": float(compilation), "load_s": float(out.strip().splitlines()[-1])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()
    case = benchmark_cases()["Chen_MSB2009"]
    rows = []
    for backend in BACKENDS:
        for size in BLOCK_SIZES:
            env = {**backend_environment(backend), "RR_ENZYME_BLOCK_SIZE": str(size)}
            runs = [load_phases(case.sbml, env) for _ in range(args.reps)]
            timing = json.loads(subprocess.run([str(TIMING), str(case.sbml), str(case.end / 2), "21"], capture_output=True,
                                               text=True, check=True, env=env).stdout.strip().splitlines()[-1])
            row = {"backend": backend, "block_size": size, "blocks": runs[0]["blocks"],
                   **{k: median(r[k] for r in runs) for k in ("setup_s", "enzyme_pass_s", "compile_s", "load_s")},
                   "t_jac_enzyme_s": timing["t_enzyme"]}
            rows.append(row)
            print(row, flush=True)
    with open(HERE / "blocks.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
