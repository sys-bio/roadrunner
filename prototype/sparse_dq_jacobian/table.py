"""Print the tables of RESULTS.md from benchmark.csv.

Usage:
    python table.py [benchmark.csv]
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


if __name__ == "__main__":
    main()
