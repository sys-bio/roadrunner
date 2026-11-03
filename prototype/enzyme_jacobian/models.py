"""The benchmark models of the Enzyme Jacobian prototype and how each one is simulated.

The two PBPK models come from their repositories, the two PEtab models from the local cache of
the PEtab benchmark collection (Benchmark-Models-PEtab), at the nominal values of their parameter
table and without a simulation condition, as in the profiles of sys-bio/roadrunner#1409. The
nominal values are written into a copy of the SBML, so that resets and initial assignments of
roadrunner see them like any other value of the model.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import libsbml

HOME = Path.home()
WORK = HOME / "rr-enzyme" / "models"
PETAB_CACHE = HOME / ".cache" / "sbmlsim" / "petab-benchmark"
# the newer of the two cached commits (2026-10-26; the other one is from 2026-09-28)
PETAB_COMMIT = "c44d5ede9bf979db5aa8ae4e94fc31fba9776675"


@dataclass
class Case:
    """A benchmark model: its SBML, what is set after a reset, and the end of its timecourse."""

    name: str
    sbml: Path
    setup: Callable[[object], None]
    end: float
    petab_commit: str | None = None


def _dose(parameter: str, value: float) -> Callable[[object], None]:
    """Setup that gives one oral dose by setting the dose parameter after the reset."""

    def setup(rr: object) -> None:
        rr.setValue(parameter, value)

    return setup


def _no_setup(rr: object) -> None:
    """Setup of a model that is simulated as it is."""


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
    return Case(name=name, sbml=path, setup=_no_setup, end=end, petab_commit=PETAB_COMMIT)


def benchmark_cases() -> dict[str, Case]:
    """The four benchmark models, by name, smallest first."""
    cases = [
        Case(
            name="hctz_body",
            sbml=HOME / "git" / "hctz-model" / "models" / "hctz_body_flat.xml",
            setup=_dose("PODOSE_hctz", 25.0),
            end=1440.0,
        ),
        _petab_case("Smith_BMCSystBiol2013"),
        Case(
            name="glimepiride_body",
            sbml=HOME / "git" / "glimepiride-model" / "models" / "glimepiride_body_flat.xml",
            setup=_dose("PODOSE_gli", 4.0),
            end=1440.0,
        ),
        _petab_case("Chen_MSB2009"),
    ]
    return {c.name: c for c in cases}
