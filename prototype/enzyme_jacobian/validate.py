"""Validation of the Enzyme Jacobian of roadrunner against finite differences.

For the four benchmark models (at the start, the middle and the end of their timecourse) and for
every case of the SBML test suite that roadrunner simulates, the exact Jacobian of the state
vector rate (`RoadRunner.getStateVectorJacobian`) is compared with finite differences of the
state vector rate, and every nonzero of the finite differences must lie inside the sparsity
pattern. Each model is also simulated with the three solver configurations, whose results must
agree. Models the Enzyme Jacobian does not cover are counted by their reason, not failed.

The reference: central differences with the steps 1e-6 * max(|y_j|, floor) for the floors 1e-6,
1e-3, 1 and 1e3; an entry passes if it matches one of them within a relative tolerance of 1e-6
(plus 1e-8 of the largest entry of the row). Several steps are needed because a state at zero
next to large terms of its rate (Chen_MSB2009: a rate of 1.3e5 and derivatives of 0.02) needs a
step far above 1e-12 to rise above their rounding, while small states need small steps. At a kink
(max, min or a piecewise switching exactly at the state) the exact derivative is that of the
active branch, which matches a one-sided difference, so an entry also passes if it matches a
one-sided difference within 1e-4.

Usage: python validate.py [--sts-only | --benchmark-only] [--jobs N]
Writes validation.md and validation.json next to this script; exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import roadrunner

HERE = Path(__file__).resolve().parent
STS = HERE.parent.parent / "test" / "sbml-test-suite" / "semantic"
CONFIGS = [("difference_quotient", "dense"), ("enzyme", "dense"), ("enzyme", "klu")]
FD_FLOORS = (1e-6, 1e-3, 1.0, 1e3)
# The configurations take different steps, so their results differ by the global integration
# error, which at rtol = atol = 1e-10 is about 1e-9 of the scale of a column and up to about 1e-8
# in absolute terms around zero (Chen_MSB2009). Each configuration is compared with a reference
# at rtol = 1e-12, atol = 1e-14, per column within 1e-6 of the largest absolute value of the
# column plus 1e-7.
SIM_COLUMN_RTOL = 1e-6
SIM_ATOL = 1e-7


def fd_jacobians(model, floor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Central, forward and backward difference Jacobians of the state vector rate."""
    y = np.array(model.getStateVectorValues())
    t = model.getTime()
    n = len(y)
    f0 = np.array(model.getStateVectorRateAt(t, list(y)))
    central, forward, backward = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
    for j in range(n):
        h = 1e-6 * max(abs(y[j]), floor)
        yp, ym = y.copy(), y.copy()
        yp[j] += h
        ym[j] -= h
        fp = np.array(model.getStateVectorRateAt(t, list(yp)))
        fm = np.array(model.getStateVectorRateAt(t, list(ym)))
        central[:, j] = (fp - fm) / (2 * h)
        forward[:, j] = (fp - f0) / h
        backward[:, j] = (f0 - fm) / h
    return central, forward, backward


def check_jacobian(rr) -> dict:
    """Compare the Enzyme Jacobian at the current state with finite differences."""
    model = rr.model
    n = len(model.getStateVectorValues())
    # a 1 x 1 matrix comes back from the bindings with fewer dimensions
    J = np.asarray(rr.getStateVectorJacobian(), dtype=float).reshape(n, n)
    P = np.asarray(rr.getStateVectorJacobianPattern(), dtype=float).reshape(n, n)
    if n == 0:
        return {"states": 0, "nnz": 0, "bad": 0, "outside": 0, "max_rel_err": 0.0}
    fds = [fd_jacobians(model, floor) for floor in FD_FLOORS]
    c1 = fds[0][0]
    # the largest entry of each row, over all step sizes
    # (a large step can leave the domain of the rate, e.g. a negative argument of sqrt: NaN)
    scale = np.nan_to_num(np.nanmax(np.abs(np.stack([f[0] for f in fds])), axis=(0, 2)))[:, None]

    def close(ref: np.ndarray) -> np.ndarray:
        return np.abs(J - ref) <= 1e-6 * np.abs(ref) + 1e-8 * scale + 1e-14

    def one_sided(ref: np.ndarray) -> np.ndarray:
        return np.abs(J - ref) <= 1e-4 * np.abs(ref) + 1e-6 * scale + 1e-14

    ok = np.zeros(J.shape, dtype=bool)
    for central, forward, backward in fds:
        ok |= close(central) | one_sided(forward) | one_sided(backward)
    # a nonzero of any central difference above the noise floor must be in the pattern
    outside = np.zeros(J.shape, dtype=bool)
    for central, _, _ in fds:
        outside |= (np.abs(central) > 1e-6 * scale) & (np.abs(central) > 1e-12) & (P == 0)
    # relative error against the closest reference, over the entries above the noise floor
    refs = np.stack([m for f in fds for m in f])
    best = np.take_along_axis(refs, np.argmin(np.nan_to_num(np.abs(refs - J), nan=np.inf), axis=0)[None], axis=0)[0]
    significant = np.abs(best) > 1e-8 * scale
    rel = np.where(significant, np.abs(J - best) / np.maximum(np.abs(best), 1e-300), 0.0)
    bad_entries = [(int(i), int(j), float(J[i, j]), float(c1[i, j])) for i, j in zip(*np.nonzero(~ok))]
    return {
        "states": int(n),
        "nnz": int(P.sum()),
        "bad": int((~ok).sum()),
        "outside": int(outside.sum()),
        "max_rel_err": float(rel.max()),
        "examples": bad_entries[:5],
    }


def reset_all(rr) -> None:
    """RoadRunner.resetAll through the SWIG function: the Python bindings turn a model symbol
    named `reset` into a property of the RoadRunner class, which hides the method and removes it
    for the rest of the process (18 test suite cases, sys-bio/roadrunner#1415)."""
    flags = (roadrunner.SelectionRecord.TIME | roadrunner.SelectionRecord.RATE | roadrunner.SelectionRecord.FLOATING
             | roadrunner.SelectionRecord.BOUNDARY | roadrunner.SelectionRecord.COMPARTMENT
             | roadrunner.SelectionRecord.GLOBAL_PARAMETER | roadrunner.SelectionRecord.STOICHIOMETRY)
    roadrunner._roadrunner.RoadRunner_reset(rr, flags)


def configure(rr, jacobian: str, solver: str) -> None:
    """Set the Jacobian and the linear solver of CVODE. "klu" needs "enzyme" at every moment, so
    the solver goes to "dense" first."""
    rr.integrator.setValue("linear_solver", "dense")
    rr.integrator.setValue("jacobian", jacobian)
    rr.integrator.setValue("linear_solver", solver)


def simulate(rr, setup, start: float, end: float, points: int, jacobian: str, solver: str,
             rtol: float = 1e-10, atol: float = 1e-10) -> np.ndarray:
    """Simulate from the initial state with the given Jacobian and linear solver."""
    reset_all(rr)
    configure(rr, jacobian, solver)
    rr.integrator.setValue("relative_tolerance", rtol)
    rr.integrator.setValue("absolute_tolerance", atol)
    setup(rr)
    return np.array(rr.simulate(start, end, points))


def compare_simulations(results: dict[str, np.ndarray], reference: np.ndarray) -> tuple[bool, dict[str, float]]:
    """Deviation of every configuration from the tight reference, per column in units of
    1e-6 of the largest absolute value of the column plus 1e-7. A configuration with the Enzyme
    Jacobian passes if its deviation is at most 1 or at most that of the difference quotient
    (which at times is the less accurate one: test suite case 00197)."""
    scale = np.max(np.abs(reference), axis=0)
    deviation = {}
    for key, values in results.items():
        if values.shape != reference.shape:
            deviation[key] = float("inf")
            continue
        diff = np.max(np.abs(values - reference), axis=0)
        deviation[key] = float(np.max(diff / (SIM_COLUMN_RTOL * scale + SIM_ATOL)))
    baseline = deviation[f"{CONFIGS[0][0]}+{CONFIGS[0][1]}"]
    agree = all(d <= max(1.0, baseline) for key, d in deviation.items() if key.startswith("enzyme"))
    return agree, deviation


def validate_model(rr, setup, start: float, end: float, points: int, check_times: list[float]) -> dict:
    """Jacobian checks at the given times and the simulation comparison of one model."""
    checks = []
    reset_all(rr)
    configure(rr, "difference_quotient", "dense")
    setup(rr)
    t0 = start
    for t in check_times:
        if t > t0:
            rr.simulate(t0, t, 2)
            t0 = t
        checks.append({"time": t, **check_jacobian(rr)})
    results = {}
    for jacobian, solver in CONFIGS:
        results[f"{jacobian}+{solver}"] = simulate(rr, setup, start, end, points, jacobian, solver)
    reference = simulate(rr, setup, start, end, points, "difference_quotient", "dense", rtol=1e-12, atol=1e-14)
    agree, deviation = compare_simulations(results, reference)
    ok = agree and all(c["bad"] == 0 and c["outside"] == 0 for c in checks)
    return {"ok": ok, "checks": checks, "simulations_agree": agree, "sim_deviation": deviation}


# --- SBML test suite -------------------------------------------------------------------------


def sts_settings(case: Path) -> dict:
    """The start, duration and steps of a test suite case."""
    settings = {}
    for line in (case / f"{case.name}-settings.txt").read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            settings[key.strip()] = value.strip()
    return settings


def sts_sbml(case: Path) -> Path | None:
    """The SBML file of the highest level and version of a case."""
    for lv in ("l3v2", "l3v1", "l2v5", "l2v4", "l2v3", "l2v2", "l2v1", "l1v2"):
        f = case / f"{case.name}-sbml-{lv}.xml"
        if f.exists():
            return f
    return None


def validate_sts_case(case: Path) -> dict:
    """Validate one test suite case; the status is ok, failed, fallback or skipped."""
    result = {"case": case.name}
    sbml = sts_sbml(case)
    if sbml is None:
        return {**result, "status": "skipped", "reason": "no SBML file"}
    settings = sts_settings(case)
    try:
        start = float(settings["start"])
        end = start + float(settings["duration"])
        points = int(settings["steps"]) + 1
    except (KeyError, ValueError):
        return {**result, "status": "skipped", "reason": "no timecourse in the settings"}
    try:
        rr = roadrunner.RoadRunner(str(sbml))
        rr.simulate(start, end, points)
    except Exception as e:  # noqa: BLE001 - any failure means roadrunner does not simulate it
        return {**result, "status": "skipped", "reason": f"roadrunner cannot simulate it: {str(e)[:80]}"}
    if not rr.model.hasStateVectorJacobian():
        return {**result, "status": "fallback", "reason": rr.model.getStateVectorJacobianUnavailableReason()}
    if "RandomEventExecution" in (case / f"{case.name}-model.m").read_text():
        # events in random order: two simulations differ whatever the Jacobian
        try:
            checks = [{"time": rr.model.getTime(), **check_jacobian(rr)}]
        except Exception as e:  # noqa: BLE001
            return {**result, "status": "failed", "reason": f"exception: {str(e)[:200]}"}
        ok = all(c["bad"] == 0 and c["outside"] == 0 for c in checks)
        return {**result, "status": "ok" if ok else "failed", "checks": checks, "simulations_agree": None,
                "sim_deviation": {}, "note": "random event execution: simulations not compared"}
    try:
        v = validate_model(rr, lambda r: None, start, end, points, [start, (start + end) / 2, end])
    except Exception as e:  # noqa: BLE001
        return {**result, "status": "failed", "reason": f"exception: {str(e)[:200]}"}
    return {**result, "status": "ok" if v["ok"] else "failed", **v}


def sts_worker(cases: list[str]) -> None:
    """Validate the given cases in this process, one JSON line per case on stdout."""
    roadrunner.Logger.setLevel(roadrunner.Logger.LOG_ERROR)
    roadrunner.Config.setValue(roadrunner.Config.LLVM_ENZYME_JACOBIAN, True)
    for name in cases:
        print(json.dumps(validate_sts_case(STS / name)), flush=True)


def run_batch(cases: list[str]) -> list[dict]:
    """Run a batch in a subprocess; a crash reruns its cases one by one to find the culprit."""
    proc = subprocess.run([sys.executable, __file__, "--sts-worker", *cases], capture_output=True, text=True)
    results = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
    if proc.returncode == 0:
        return results
    done = {r["case"] for r in results}
    rest = [c for c in cases if c not in done]
    if len(cases) == 1:
        return results + [{"case": cases[0], "status": "failed", "reason": f"crash (exit {proc.returncode})"}]
    out = list(results)
    for c in rest:
        out.extend(run_batch([c]))
    return out


def validate_sts(jobs: int) -> list[dict]:
    """Validate the whole semantic test suite in parallel batches."""
    cases = sorted(p.name for p in STS.iterdir() if p.is_dir())
    batches = [cases[i:i + 40] for i in range(0, len(cases), 40)]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return [r for batch in pool.map(run_batch, batches) for r in batch]


# --- benchmark models and report ---------------------------------------------------------------


def validate_benchmark() -> dict[str, dict]:
    """Validate the four benchmark models."""
    from models import benchmark_cases

    roadrunner.Logger.setLevel(roadrunner.Logger.LOG_ERROR)
    roadrunner.Config.setValue(roadrunner.Config.LLVM_ENZYME_JACOBIAN, True)
    out = {}
    for name, case in benchmark_cases().items():
        rr = roadrunner.RoadRunner(str(case.sbml))
        if not rr.model.hasStateVectorJacobian():
            out[name] = {"ok": False, "reason": rr.model.getStateVectorJacobianUnavailableReason()}
            continue
        out[name] = validate_model(rr, case.setup, 0.0, case.end, 201, [0.0, case.end / 2, case.end])
        out[name]["petab_commit"] = case.petab_commit
    return out


def write_report(bench: dict[str, dict], sts: list[dict], seconds: float) -> None:
    """Write validation.md and validation.json."""
    lines = ["# Validation of the Enzyme Jacobian", ""]
    lines.append(
        "Generated by `validate.py`. Reference: finite differences of the state vector rate, see the "
        "module docstring for the tolerances. `bad` counts entries that match no reference, `outside` "
        "finite difference nonzeros outside the sparsity pattern. Simulations: `difference_quotient+dense`, "
        "`enzyme+dense` and `enzyme+klu` at `rtol = atol = 1e-10`, each compared with a reference at "
        "`rtol = 1e-12, atol = 1e-14`. The deviation is the largest difference of a column in units of "
        f"{SIM_COLUMN_RTOL:g} of the largest absolute value of the column plus {SIM_ATOL:g} (1 = at the "
        "limit); an Enzyme configuration passes at a deviation of at most 1 or at most that of the "
        "difference quotient."
    )
    if bench:
        lines += ["", "## Benchmark models", ""]
        lines.append("| model | states | pattern nnz | time | bad | outside | max rel. error | simulations agree (deviation dq / enzyme / klu) |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for name, r in bench.items():
            if "checks" not in r:
                lines.append(f"| {name} | | | | | | | not available: {r.get('reason')} |")
                continue
            for c in r["checks"]:
                lines.append(
                    f"| {name} | {c['states']} | {c['nnz']} | {c['time']:g} | {c['bad']} | {c['outside']} | "
                    f"{c['max_rel_err']:.1e} | {'yes' if r['simulations_agree'] else 'NO'} "
                    f"({' / '.join(f'{d:.1e}' for d in r['sim_deviation'].values())}) |"
                )
    if sts:
        by_status: dict[str, list[dict]] = {}
        for r in sts:
            by_status.setdefault(r["status"], []).append(r)
        lines += ["", "## SBML test suite (semantic)", ""]
        lines.append(f"{len(sts)} cases: " + ", ".join(f"{len(v)} {k}" for k, v in sorted(by_status.items())) + ".")
        ok_cases = by_status.get("ok", [])
        if ok_cases:
            worst = max(ok_cases, key=lambda r: max(c["max_rel_err"] for c in r["checks"]))
            lines.append(
                f"Largest relative error of a passing case: {max(c['max_rel_err'] for c in worst['checks']):.1e} "
                f"(case {worst['case']})."
            )
        if by_status.get("failed"):
            lines += ["", "### Failed cases", ""]
            for r in by_status["failed"]:
                detail = r.get("reason") or "; ".join(
                    f"t={c['time']:g}: {c['bad']} bad, {c['outside']} outside {c.get('examples')}" for c in r.get("checks", [])
                    if c["bad"] or c["outside"]
                ) or f"simulations differ (deviation {r.get('sim_deviation')})"
                lines.append(f"- {r['case']}: {detail}")
        for status in ("fallback", "skipped"):
            reasons: dict[str, list[str]] = {}
            for r in by_status.get(status, []):
                key = re.sub(r"'[^']*'", "'...'", r["reason"]) if status == "fallback" else r["reason"][:60]
                reasons.setdefault(key, []).append(r["case"])
            if reasons:
                lines += ["", f"### {status.capitalize()} by reason", ""]
                for reason, cases in sorted(reasons.items(), key=lambda kv: -len(kv[1])):
                    lines.append(f"- {len(cases)}: {reason} (e.g. {', '.join(cases[:5])})")
    lines += ["", f"Run time: {seconds:.0f} s.", ""]
    (HERE / "validation.md").write_text("\n".join(lines))
    (HERE / "validation.json").write_text(json.dumps({"benchmark": bench, "sts": sts}, indent=1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sts-only", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--jobs", type=int, default=20)
    parser.add_argument("--sts-worker", nargs="*")
    args = parser.parse_args()
    if args.sts_worker is not None:
        sts_worker(args.sts_worker)
        return 0
    t0 = time.time()
    bench = {} if args.sts_only else validate_benchmark()
    sts = [] if args.benchmark_only else validate_sts(args.jobs)
    write_report(bench, sts, time.time() - t0)
    failed = [n for n, r in bench.items() if not r["ok"]] + [r["case"] for r in sts if r["status"] == "failed"]
    print((HERE / "validation.md").read_text())
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
