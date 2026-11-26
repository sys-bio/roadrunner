"""Compare two Linux RoadRunner builds in paired ABBA blocks, or summarize saved runs."""

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

ORDER = ("baseline", "candidate", "candidate", "baseline")
CLOCKS = ("wall_ns", "thread_cpu_ns")
OPERATIONS = ("simulate", "reset", "reset+simulate")


def digest(contents):
    return hashlib.sha256(contents).hexdigest()


def file_hash(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def read_json(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    opener = gzip.open if path.suffix == ".gz" else open
    tmp = path.with_suffix(path.suffix + ".tmp")
    with opener(tmp, "wt", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False)
    tmp.replace(path)


def setup_model(rr, sbml, opts):
    load = rr.LoadSBMLOptions()
    load.setLLVMBackend(rr.LoadSBMLOptions.LLJIT)
    load.setLLJitOptimizationLevel(rr.LoadSBMLOptions.DEFAULT)
    runner = rr.RoadRunner(sbml.decode("utf-8"), load)
    state_ids = [
        runner.model.getStateVectorId(i) for i in range(len(runner.getRatesOfChange()))
    ]
    for sid, value in opts["preinit"]:
        runner.setValue(f"init({sid})", value)
    runner.timeCourseSelections = ["time", *state_ids]
    solver = runner.getIntegrator()
    for key, value in {
        "jacobian": "sparse_difference_quotient",
        "linear_solver": "klu",
        "stiff": True,
        "relative_tolerance": 1e-6,
        "absolute_tolerance": 1e-10,
        "maximum_num_steps": 20000,
        "variable_step_size": False,
        "collect_timing": False,
    }.items():
        solver.setValue(key, value)
    return runner, solver, state_ids


def certificate(np, result, solver, points, end, state_ids):
    array = np.ascontiguousarray(result)
    if array.dtype != np.float64 or array.shape != (points, len(state_ids) + 1):
        raise AssertionError(
            f"unexpected simulation result: {array.dtype}, {array.shape}"
        )
    if (
        not np.isfinite(array).all()
        or array[0, 0] != 0
        or not np.isclose(array[-1, 0], end)
    ):
        raise AssertionError("nonfinite or incomplete trajectory")
    counters = {
        k: v for k, v in solver.getStatistics().items() if not k.startswith("time_")
    }
    if not counters["sparse_linear_solver"] or counters["num_steps"] <= 0:
        raise AssertionError(f"unexpected solver output: {counters}")
    return {
        "sha256": digest(array.view(np.uint8)),
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "colnames_sha256": digest(json.dumps(result.colnames).encode()),
        "solver_statistics": counters,
    }


def measure_case(np, runner, solver, state_ids, name, end, points, api, args, fixed):
    if api == "raw":
        from roadrunner import SimulateOptions

        opts = SimulateOptions()
        opts.start = 0
        opts.end = end
        opts.steps = points - 1
        opts.structured_result = False
        opts.copy_result = True

        def simulate():
            return runner._simulate(opts)
    else:

        def simulate():
            return runner.simulate(0, end, points)

    case = f"{name}/{points}/{api}"
    samples = []
    cert = None
    reps = fixed.get(case)
    while reps is None or len(samples) < args.warmups + reps:
        wall = time.perf_counter_ns()
        cpu = time.thread_time_ns()
        runner.resetAll()
        reset_cpu = time.thread_time_ns() - cpu
        reset_wall = time.perf_counter_ns() - wall
        wall = time.perf_counter_ns()
        cpu = time.thread_time_ns()
        result = simulate()
        sim_cpu = time.thread_time_ns() - cpu
        sim_wall = time.perf_counter_ns() - wall
        proof = certificate(np, result, solver, points, end, state_ids)
        del result
        if cert is None:
            cert = proof
        elif proof != cert:
            raise AssertionError(
                f"repeated execution differs for {case}: {cert} vs {proof}"
            )
        samples.append(
            {"reset": [reset_wall, reset_cpu], "simulate": [sim_wall, sim_cpu]}
        )
        if reps is None and len(samples) == args.warmups:
            warm_median = statistics.median(x["simulate"][1] for x in samples)
            reps = max(4, math.ceil(args.target_ms * 1_000_000 / max(1, warm_median)))
    raw = samples[args.warmups :]
    return case, {
        "proof": cert,
        "repetitions": reps,
        "warmup_samples": samples[: args.warmups],
        "samples": raw,
    }


def mapped_libraries():
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/") and ".so" in fields[5]:
            paths.add(Path(fields[5]).resolve())
    return paths


def worker(job):
    os.sched_setaffinity(0, {job.cpu})
    site = Path(job.site)
    sys.path.insert(0, str(site))
    import numpy as np
    import roadrunner as rr
    from roadrunner import _roadrunner as ext
    from roadrunner import named_array

    extension = Path(ext.__file__).resolve()
    for module in [rr, rr.roadrunner, ext, named_array]:
        if not Path(module.__file__).resolve().is_relative_to(site):
            raise RuntimeError(f"unexpected package path: {module.__file__}")
    libraries = mapped_libraries()
    extensions = {p for p in libraries if p.name.startswith("_roadrunner.")}
    if extensions != {extension} or extension != Path(job.extension):
        raise RuntimeError(f"unexpected RoadRunner extension(s): {extensions}")
    if Path(sys.executable).absolute() != Path(job.python) or os.sched_getaffinity(
        0
    ) != {job.cpu}:
        raise RuntimeError("unexpected worker interpreter or CPU affinity")
    rr.Logger.setLevel(rr.Logger.LOG_ERROR)
    meta = {
        "variant": job.variant,
        "python": sys.version,
        "executable": sys.executable,
        "affinity": sorted(os.sched_getaffinity(0)),
        "numpy": np.__version__,
        "library_version": rr.__version__,
        "build_revision": rr.getVersionStr(rr.VERSIONSTR_GIT_COMMIT),
        "module": rr.__file__,
        "extension": str(extension),
        "extension_sha256": file_hash(extension),
        "proxy_sha256": file_hash(Path(rr.roadrunner.__file__)),
        "named_array_sha256": file_hash(Path(named_array.__file__)),
        "native_libraries": sorted(
            str(p)
            for p in libraries
            if any(s in p.name for s in ("roadrunner", "sundials", "LLVM", "librr"))
        ),
    }
    models = {}
    cases = {}
    for name, model in job.models.items():
        sbml = Path(model["path"]).read_bytes()
        fingerprint = digest(sbml)
        if model.get("sha256", fingerprint) != fingerprint:
            raise ValueError(f"{name}: SBML SHA-256 differs from model config")
        wall = time.perf_counter_ns()
        cpu = time.thread_time_ns()
        runner, solver, state_ids = setup_model(rr, sbml, model)
        models[name] = {
            "setup_thread_cpu_ns": time.thread_time_ns() - cpu,
            "setup_wall_ns": time.perf_counter_ns() - wall,
            "sbml": model["path"],
            "sbml_sha256": fingerprint,
            "end": model["end"],
            "selection_ids": ["time", *state_ids],
            "preinit": model["preinit"],
        }
        for points in [101] if name == "perfusion" else [101, 1001]:
            for api in ("public", "raw"):
                case, run = measure_case(
                    np,
                    runner,
                    solver,
                    state_ids,
                    name,
                    model["end"],
                    points,
                    api,
                    job,
                    job.counts,
                )
                cases[case] = run
            if (
                cases[f"{name}/{points}/public"]["proof"]
                != cases[f"{name}/{points}/raw"]["proof"]
            ):
                raise AssertionError(
                    f"{name}/{points}: public and raw results or solver statistics differ"
                )
        del solver, runner
    write_json(
        Path(job.worker_output), {"metadata": meta, "models": models, "cases": cases}
    )


def show_table(report, operation, clock):
    """Use the original samples, including when reading the published JSON files."""
    workers = report["workers"]
    if not workers or len(workers) % 4:
        raise ValueError("a summary requires complete ABBA blocks")
    is_control = bool(report.get("control"))
    heading = "Control B" if is_control else "Candidate"
    print(f"{operation}, {clock}; median of {len(workers) // 4} paired ABBA block(s)")
    print(f"| Case | Baseline (ms) | {heading} (ms) | Speedup | Block range |")
    print("| --- | ---: | ---: | ---: | ---: |")
    index = CLOCKS.index(clock)
    for case in workers[0]["cases"]:
        baseline = []
        candidate = []
        for i in range(0, len(workers), 4):
            medians = []
            for item in workers[i : i + 4]:
                rows = item["cases"][case]["samples"]
                medians.append(
                    statistics.median(
                        r["reset"][index] + r["simulate"][index]
                        if operation == "reset+simulate"
                        else r[operation][index]
                        for r in rows
                    )
                )
            baseline.append(statistics.geometric_mean([medians[0], medians[3]]))
            candidate.append(statistics.geometric_mean([medians[1], medians[2]]))
        ratios = [a / b for a, b in zip(baseline, candidate)]
        print(
            f"| {case} | {statistics.median(baseline) / 1e6:.6f} | {statistics.median(candidate) / 1e6:.6f} | "
            f"{statistics.median(ratios):.4f} | {min(ratios):.4f}-{max(ratios):.4f} |"
        )


def equivalence(run):
    return {
        "cases": {k: (v["proof"], v["repetitions"]) for k, v in run["cases"].items()},
        "inputs": {
            k: (v["sbml_sha256"], v["selection_ids"], v["preinit"], v["end"])
            for k, v in run["models"].items()
        },
        "software": [
            run["metadata"][k]
            for k in (
                "python",
                "numpy",
                "library_version",
                "proxy_sha256",
                "named_array_sha256",
            )
        ],
    }


def run_blocks(args, sites, cpu):
    model_config = read_json(args.model_config)
    models = {}
    for name in args.models or model_config:
        spec = model_config[name]
        models[name] = {
            **spec,
            "path": str((args.model_config.parent / spec["path"]).resolve(strict=True)),
            "preinit": spec.get("preinit", []),
        }
    if not models:
        raise ValueError("select at least one model")
    extensions = {}
    for label, site in sites.items():
        matches = list((site / "roadrunner").glob("_roadrunner*.so"))
        if len(matches) != 1:
            raise ValueError(
                f"expected one native extension under {site}/roadrunner, found {matches}"
            )
        extensions[label] = matches[0].resolve(strict=True)
    fingerprints = {label: file_hash(path) for label, path in extensions.items()}
    if (
        fingerprints["baseline"] == fingerprints["candidate"]
    ) != args.same_binary_control:
        raise ValueError(
            "expect distinct binary hashes for a comparison and identical hashes for a same-binary control"
        )

    counts = {}
    if args.counts_from:
        previous = read_json(args.counts_from)["workers"][0]["cases"]
        counts = {
            f"{name}/{n}/{mode}": previous[f"{name}/{n}/{mode}"]["repetitions"]
            for name in models
            for n in ([101] if name == "perfusion" else [101, 1001])
            for mode in ("public", "raw")
        }
    evidence = {
        "driver_sha256": file_hash(Path(__file__)),
        "platform": platform.platform(),
        "cpu": cpu,
        "cpu_model": next(
            (
                v.split(":", 1)[1].strip()
                for v in Path("/proc/cpuinfo").read_text().splitlines()
                if v.startswith("model name")
            ),
            "",
        ),
        "blocks": args.blocks,
        "worker_order": ORDER,
        "control": args.same_binary_control,
        "sites": {k: str(v) for k, v in sites.items()},
        "extension_sha256": fingerprints,
        "model_config_sha256": file_hash(args.model_config),
        "target_ms": args.target_ms,
        "minimum_reps": 4,
        "warmups_per_case": args.warmups,
        "sample_clocks_in_order": CLOCKS,
        "configuration": "LLJIT DEFAULT (O2), CVODE BDF, KLU sparse_difference_quotient, rtol=1e-6, base atol=1e-10, fixed output, collect_timing=False; materialized array; resetAll measured separately",
        "workers": [],
    }
    env = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    expected = None
    with tempfile.TemporaryDirectory() as directory:
        worker_output = Path(directory) / "worker.json"
        for block in range(args.blocks):
            for variant in ORDER:
                print(f"block {block + 1}/{args.blocks}, {variant}", flush=True)
                job = {
                    "cpu": cpu,
                    "variant": variant,
                    "site": str(sites[variant]),
                    "extension": str(extensions[variant]),
                    "python": str(args.python.absolute()),
                    "worker_output": str(worker_output),
                    "models": models,
                    "warmups": args.warmups,
                    "target_ms": args.target_ms,
                    "counts": counts,
                }
                subprocess.run(
                    [job["python"], "-I", str(Path(__file__).resolve()), "--worker"],
                    input=json.dumps(job),
                    text=True,
                    check=True,
                    env=env,
                )
                run = read_json(worker_output)
                if run["metadata"]["extension_sha256"] != fingerprints[variant]:
                    raise RuntimeError(f"{variant}: binary changed during the run")
                actual = equivalence(run)
                if expected is None:
                    expected = actual
                    counts = {k: v["repetitions"] for k, v in run["cases"].items()}
                elif actual != expected:
                    mismatches = [
                        k
                        for k, v in actual["cases"].items()
                        if v != expected["cases"].get(k)
                    ]
                    raise AssertionError(
                        f"worker input, software or output mismatch: {mismatches}"
                    )
                evidence["workers"].append(run)
                write_json(args.output, evidence)
    show_table(evidence, args.operation, args.clock)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summarize",
        type=Path,
        metavar="RESULTS",
        help="print table from a saved JSON or JSON.gz without RoadRunner or NumPy",
    )
    parser.add_argument("--operation", choices=OPERATIONS, default="simulate")
    parser.add_argument("--clock", choices=CLOCKS, default="thread_cpu_ns")
    parser.add_argument("--baseline-site-packages", type=Path)
    parser.add_argument("--candidate-site-packages", type=Path)
    parser.add_argument(
        "--same-binary-control",
        action="store_true",
        help="run the baseline under both labels to measure timing variability",
    )
    parser.add_argument(
        "--counts-from",
        type=Path,
        help="reuse repetition counts from an earlier run's JSON or JSON.gz",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="interpreter with NumPy compatible with both extensions",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        help="JSON mapping model names to path, end and optional preinit pairs or sha256; paths relative to config",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="subset of model names (default: all config entries)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="save raw timings and verification data to JSON or JSON.gz",
    )
    parser.add_argument(
        "--cpu",
        type=int,
        help="Linux CPU to pin workers to (default: first available CPU)",
    )
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument(
        "--target-ms",
        type=float,
        default=30,
        help="target simulation CPU milliseconds per case during baseline calibration",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.summarize:
        show_table(read_json(args.summarize), args.operation, args.clock)
        return
    if args.worker:
        worker(SimpleNamespace(**json.load(sys.stdin)))
        return
    if not sys.platform.startswith("linux"):
        parser.error(
            "running timings requires Linux CPU affinity and /proc; --summarize is portable"
        )
    if args.blocks < 1 or args.warmups < 1 or not 0 < args.target_ms < math.inf:
        parser.error("blocks, warmups and target-ms must be positive")
    if not (args.baseline_site_packages and args.model_config and args.output):
        parser.error(
            "a run requires --baseline-site-packages, --model-config and --output"
        )
    if bool(args.candidate_site_packages) == args.same_binary_control:
        parser.error(
            "choose exactly one of --candidate-site-packages or --same-binary-control"
        )
    allowed = os.sched_getaffinity(0)
    cpu = min(allowed) if args.cpu is None else args.cpu
    if cpu not in allowed:
        parser.error(f"--cpu {cpu} is unavailable; allowed CPUs are {sorted(allowed)}")
    sites = {
        "baseline": args.baseline_site_packages.resolve(strict=True),
        "candidate": (
            args.candidate_site_packages or args.baseline_site_packages
        ).resolve(strict=True),
    }
    run_blocks(args, sites, cpu)


if __name__ == "__main__":
    main()
