"""The code of an initial assignment is generated once per symbol.

Every initial assignment which another initial assignment reads was inlined
into the code of the reader again, so a chain in which each value reads the
one before twice grew as 2^n. The initial value resolvers cache the value of
an initial assignment (and of an assignment rule) per symbol, within the
blocks of a piecewise and per case of the generated accessors.
"""

import os
import sys
import unittest

thisDir = os.path.dirname(os.path.realpath(__file__))
rr_site_packages = os.path.dirname(os.path.dirname(thisDir))
sys.path += [rr_site_packages]

from roadrunner.roadrunner import RoadRunner  # noqa: E402

LEVELS = 40
"""The length of the chain; without the cache its code has 2^40 terms."""


def _math(k: int) -> str:
    """`p_k = p_(k-1) + piecewise(p_(k-1), AGE < 50, 2*p_(k-1))`."""
    p = f"<ci> p{k - 1} </ci>"
    return (
        "<math xmlns='http://www.w3.org/1998/Math/MathML'><apply><plus/>"
        f"{p}<piecewise><piece>{p}<apply><lt/><ci> AGE </ci><cn> 50 </cn></apply>"
        f"</piece><otherwise><apply><times/><cn> 2 </cn>{p}</apply></otherwise>"
        "</piecewise></apply></math>"
    )


def _sbml() -> str:
    """A model of a chain of initial assignments, read by a species and a rule."""
    params = "<parameter id='AGE' value='35' constant='true'/>"
    params += "<parameter id='p0' value='1' constant='true'/>"
    params += "".join(
        f"<parameter id='p{k}' constant='true'/>" for k in range(1, LEVELS + 1)
    )
    params += "<parameter id='q' constant='false'/>"
    assignments = "".join(
        f"<initialAssignment symbol='p{k}'>{_math(k)}</initialAssignment>"
        for k in range(1, LEVELS + 1)
    )
    assignments += (
        "<initialAssignment symbol='S'><math xmlns='http://www.w3.org/1998/Math/MathML'>"
        f"<apply><divide/><ci> p{LEVELS} </ci><ci> p{LEVELS - 1} </ci></apply>"
        "</math></initialAssignment>"
    )
    rules = (
        "<listOfRules><assignmentRule variable='q'>"
        "<math xmlns='http://www.w3.org/1998/Math/MathML'>"
        f"<apply><plus/><ci> S </ci><ci> p{LEVELS - 2} </ci></apply></math>"
        "</assignmentRule></listOfRules>"
    )
    return (
        "<?xml version='1.0' encoding='UTF-8'?>"
        "<sbml xmlns='http://www.sbml.org/sbml/level3/version2/core' level='3' version='2'>"
        "<model id='chain'>"
        "<listOfCompartments><compartment id='c' size='1' constant='true' "
        "spatialDimensions='3'/></listOfCompartments>"
        "<listOfSpecies><species id='S' compartment='c' hasOnlySubstanceUnits='false' "
        "boundaryCondition='false' constant='false'/></listOfSpecies>"
        f"<listOfParameters>{params}</listOfParameters>"
        f"<listOfInitialAssignments>{assignments}</listOfInitialAssignments>"
        f"{rules}</model></sbml>"
    )


class InitialAssignmentCacheTests(unittest.TestCase):
    def test_chain_of_initial_assignments(self) -> None:
        """The chain loads and every value is that of its branch."""
        r = RoadRunner(_sbml())
        self.assertAlmostEqual(r["p10"], 2.0**10)
        self.assertAlmostEqual(r[f"p{LEVELS}"] / 2.0**LEVELS, 1.0)
        self.assertAlmostEqual(r["S"], 2.0)
        self.assertAlmostEqual(r["q"], 2.0 + 2.0 ** (LEVELS - 2))

    def test_changed_initial_value_takes_the_other_branch(self) -> None:
        """An initial value changed before a reset is read by every assignment."""
        r = RoadRunner(_sbml())
        r.setValue("init(AGE)", 60.0)
        r.reset()
        self.assertAlmostEqual(r["p10"] / 3.0**10, 1.0)
        self.assertAlmostEqual(r["S"], 3.0)
        self.assertAlmostEqual(r["init([S])"], 3.0)
        self.assertAlmostEqual(r["q"] / (3.0 + 3.0 ** (LEVELS - 2)), 1.0)


if __name__ == "__main__":
    unittest.main()
