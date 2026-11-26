"""Compare defined Setting conversions with a baseline header from Git."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="f35126ec5")
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--compiler", default=os.environ.get("CXX", "c++"))
    parser.add_argument("--sanitize", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional JSON verification report")
    args = parser.parse_args()
    repo = args.repo.resolve()
    baseline = subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{args.baseline_ref}:source/Setting.h"]
    )
    candidate = (repo / "source/Setting.h").read_bytes()
    probe = Path(__file__).with_name("check_conversions.cpp")
    report: dict[str, object] = {
        "baseline_ref": args.baseline_ref,
        "baseline_header_sha256": hashlib.sha256(baseline).hexdigest(),
        "candidate_header_sha256": hashlib.sha256(candidate).hexdigest(),
        "compiler": subprocess.check_output(
            [args.compiler, "--version"], text=True
        ).splitlines()[0],
    }
    with tempfile.TemporaryDirectory(prefix="rr-setting-conversions-") as scratch:
        work = Path(scratch)
        outputs = {}
        variants = {"baseline": baseline, "candidate": candidate}
        if args.sanitize:
            variants["candidate-sanitized"] = candidate
        for name, header in variants.items():
            variant = work / name
            variant.mkdir()
            (variant / "Setting.h").write_bytes(header)
            binary = variant / "check"
            command = [args.compiler, "-std=c++17", "-O0", "-g0"]
            if name.endswith("-sanitized"):
                command.extend(
                    [
                        "-fsanitize=undefined,float-cast-overflow",
                        "-fno-sanitize-recover=all",
                    ]
                )
            command.extend(
                [
                    "-I",
                    str(variant),
                    "-I",
                    str(repo / "source"),
                    str(probe),
                    "-o",
                    str(binary),
                ]
            )
            subprocess.run(command, check=True)
            records = variant / "records.tsv"
            with records.open("wb") as output:
                subprocess.run([str(binary)], stdout=output, check=True)
            outputs[name] = records
        reference = outputs["baseline"].read_bytes()
        for name, path in outputs.items():
            if path.read_bytes() != reference:
                raise AssertionError(f"{name} conversion values or exceptions differ")
        rows = reference.decode().splitlines()
        report.update(
            {
                "compared_records": len(rows),
                "record_class_counts": dict(
                    Counter(row.split("\t")[4] for row in rows)
                ),
                "record_sha256": {name: digest(path) for name, path in outputs.items()},
                "sanitizers_passed": args.sanitize,
                "excluded": "The probe's optional native-observations mode is not run: it includes pre-existing undefined or platform-sensitive floating conversions.",
            }
        )
    result = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(result)
    print(result, end="")


if __name__ == "__main__":
    main()
