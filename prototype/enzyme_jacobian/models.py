"""The benchmark models of the Enzyme Jacobian prototype and how each one is simulated.

The two PBPK models come from their repositories, the two PEtab models from the local cache of
the PEtab benchmark collection (Benchmark-Models-PEtab), at the nominal values of their parameter
table and without a simulation condition, as in the profiles of sys-bio/roadrunner#1409. The
nominal values are written into a copy of the SBML, so that resets and initial assignments of
roadrunner see them like any other value of the model.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import libsbml

HOME = Path.home()
WORK = HOME / "rr-enzyme" / "models"
PETAB_CACHE = HOME / ".cache" / "sbmlsim" / "petab-benchmark"
# the newer of the two cached commits (2026-10-26; the other one is from 2026-09-28)
PETAB_COMMIT = "c44d5ede9bf979db5aa8ae4e94fc31fba9776675"


LLJIT_OPT_LEVELS = ("none", "less", "default", "aggressive")


def backend_label() -> str:
    """The JIT backend chosen by RR_LLVM_BACKEND (mcjit, the default of roadrunner, or lljit) and,
    for LLJit, RR_LLJIT_OPT (its codegen level, none by default as in roadrunner)."""
    backend = os.environ.get("RR_LLVM_BACKEND", "mcjit").lower()
    if backend == "mcjit":
        return "mcjit"
    if backend != "lljit":
        raise ValueError(f"RR_LLVM_BACKEND must be mcjit or lljit, not {backend!r}")
    opt = os.environ.get("RR_LLJIT_OPT", "none").lower()
    if opt not in LLJIT_OPT_LEVELS:
        raise ValueError(f"RR_LLJIT_OPT must be one of {LLJIT_OPT_LEVELS}, not {opt!r}")
    return f"lljit-{opt}"


def set_backend(label: str) -> None:
    """Set the JIT backend of roadrunner for the next loads: mcjit or lljit-<level>."""
    import roadrunner

    C = roadrunner.Config
    if label == "mcjit":
        C.setValue(C.LLVM_BACKEND, C.MCJIT)
        return
    backend, _, opt = label.partition("-")
    if backend != "lljit" or opt not in LLJIT_OPT_LEVELS:
        raise ValueError(f"unknown backend {label!r}: mcjit or lljit-<{'|'.join(LLJIT_OPT_LEVELS)}>")
    C.setValue(C.LLVM_BACKEND, C.LLJIT)
    C.setValue(C.LLJIT_OPTIMIZATION_LEVEL, LLJIT_OPT_LEVELS.index(opt))


def backend_environment(label: str) -> dict[str, str]:
    """The environment that selects the backend `label` in a subprocess (see backend_label)."""
    env = {k: v for k, v in os.environ.items() if k not in ("RR_LLVM_BACKEND", "RR_LLJIT_OPT")}
    if label != "mcjit":
        env["RR_LLVM_BACKEND"], _, env["RR_LLJIT_OPT"] = label.partition("-")
    return env


def apply_backend() -> str:
    """Set the JIT backend of roadrunner from the environment (see backend_label), return its label."""
    label = backend_label()
    set_backend(label)
    return label


@dataclass
class Case:
    """A benchmark model: its SBML, the dose set after a reset, and the end of its timecourse."""

    name: str
    sbml: Path
    end: float
    dose: tuple[str, float] | None = None
    petab_commit: str | None = None

    def setup(self, rr: object) -> None:
        """Give the dose (a model parameter) after a reset; a model without a dose runs as it is."""
        if self.dose is not None:
            rr.setValue(*self.dose)


def _petab_case(name: str) -> Case:
    """The PEtab problem `name` at its nominal parameters, until its last measurement."""
    problem = PETAB_CACHE / PETAB_COMMIT / "problems" / name / "v1"
    doc = libsbml.readSBMLFromFile(str(problem / "model.xml"))
    if doc.getNumErrors(libsbml.LIBSBML_SEV_ERROR) + doc.getNumErrors(libsbml.LIBSBML_SEV_FATAL) > 0:
        raise ValueError(f"cannot read {problem / 'model.xml'}")
    model = doc.getModel()
    with open(problem / "parameters.tsv", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            p = model.getParameter(row["parameterId"])
            if p is not None:
                # PEtab nominal values are on the linear scale
                p.setValue(float(row["nominalValue"]))
    with open(problem / "measurements.tsv", newline="") as f:
        end = max(float(row["time"]) for row in csv.DictReader(f, delimiter="\t"))
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / f"{name}_nominal.xml"
    libsbml.writeSBMLToFile(doc, str(path))
    return Case(name=name, sbml=path, end=end, petab_commit=PETAB_COMMIT)


def benchmark_cases() -> dict[str, Case]:
    """The four benchmark models, by name, smallest first."""
    cases = [
        Case(
            name="hctz_body",
            sbml=HOME / "git" / "hctz-model" / "models" / "hctz_body_flat.xml",
            end=1440.0,
            dose=("PODOSE_hctz", 25.0),
        ),
        Case(
            name="glimepiride_body",
            sbml=HOME / "git" / "glimepiride-model" / "models" / "glimepiride_body_flat.xml",
            end=1440.0,
            dose=("PODOSE_gli", 4.0),
        ),
        _petab_case("Smith_BMCSystBiol2013"),
        _petab_case("Chen_MSB2009"),
    ]
    return {c.name: c for c in cases}
