"""Print the tables of RESULTS.md from benchmark.csv and memory.csv.

Usage:
    python table.py [benchmark.csv [memory.csv]]
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = ["hctz_body", "glimepiride_body", "Smith_BMCSystBiol2013", "Chen_MSB2009"]
CONFIGS = [("sparse", "dense+difference_quotient", "dense + DQ"),
           ("sparse", "dense+sparse_difference_quotient", "dense + sparse DQ"),
           ("sparse", "klu+sparse_difference_quotient", "KLU + sparse DQ"),
           ("enzyme", "dense+difference_quotient", "dense + DQ (Enzyme build)"),
           ("enzyme", "klu+enzyme", "KLU + Enzyme")]


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "benchmark.csv"
    memory = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "memory.csv"
    rows = {(r["build"], r["model"], r["config"]): r for r in csv.DictReader(open(path, newline=""))}

    print("| model | states | colors | configuration | simulate (ms) | speedup | steps | error test fails "
          "| Jacobians | model evaluations for Jacobians | one Jacobian (µs) | error / tol |")
    print("| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for model in MODELS:
        first = True
        for build, config, label in CONFIGS:
            r = rows[(build, model, config)]
            base = rows[(build, model, "dense+difference_quotient")]
            speedup = float(base["simulate_s"]) / float(r["simulate_s"])
            colors = rows[("sparse", model, "klu+sparse_difference_quotient")]["colors"]
            print(f"| {model if first else ''} | {r['states'] if first else ''} | {colors if first else ''} "
                  f"| {label} | {float(r['simulate_s']) * 1e3:.3g} | {speedup:.2f}x | {r['steps']} "
                  f"| {r['err_test_fails']} | {r['jac_evals']} | {r['rhs_evals_jac']} "
                  f"| {float(r['t_jac_s']) * 1e6:.3g} | {float(r['error_per_tol']):.3g} |")
            first = False

    print()
    print("| model | load, this build (s) | load with Enzyme (s) |")
    print("| --- | ---: | ---: |")
    for model in MODELS:
        print(f"| {model} | {float(rows[('sparse', model, 'klu+sparse_difference_quotient')]['load_s']):.2f} "
              f"| {float(rows[('enzyme', model, 'klu+enzyme')]['load_s']):.2f} |")

    print()
    memory_table(memory)


MEMORY_CONFIGS = [("sparse", "dense+difference_quotient", "dense + DQ"),
                  ("sparse", "dense+sparse_difference_quotient", "dense + sparse DQ"),
                  ("sparse", "klu+sparse_difference_quotient", "KLU + sparse DQ"),
                  ("enzyme-off", "dense+difference_quotient", "#1418 build without Enzyme: dense + DQ"),
                  ("enzyme", "dense+difference_quotient", "#1418 build with Enzyme: dense + DQ"),
                  ("enzyme", "klu+enzyme", "#1418 build with Enzyme: KLU + Enzyme")]


def memory_table(path: Path) -> None:
    rows = {(r["build"], r["model"], r["config"]): r for r in csv.DictReader(open(path, newline=""))}
    print("| model | build and configuration | RSS after the load (MiB) | peak RSS of the load (MiB) "
          "| heap after the simulation (MiB) | heap, change by the simulation (MiB) |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    for model in MODELS:
        first = True
        for build, config, label in MEMORY_CONFIGS:
            r = rows[(build, model, config)]
            print(f"| {model if first else ''} | {label} | {float(r['load_mib']):.1f} | {float(r['load_peak_mib']):.1f} "
                  f"| {float(r['heap_mib']):.2f} | {float(r['heap_simulate_mib']):.2f} |")
            first = False


if __name__ == "__main__":
    main()
