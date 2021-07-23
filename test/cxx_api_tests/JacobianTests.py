import os
import sys
import unittest

thisDir = os.path.dirname(os.path.realpath(__file__))
rr_site_packages = os.path.dirname(os.path.dirname(thisDir))

sys.path += [
    rr_site_packages,
]

try:
    from roadrunner.roadrunner import (
        RoadRunner, Integrator, SteadyStateSolver,
        ExecutableModel, RoadRunnerOptions, Config
    )
    import roadrunner
except ImportError:
    from roadrunner import (
        RoadRunner, Integrator, SteadyStateSolver,
        ExecutableModel, RoadRunnerOptions
    )
    import roadrunner

from roadrunner.testing import TestModelFactory as tmf
from roadrunner.testing.RoadRunnerTest import RoadRunnerTest


class JacobianTests(RoadRunnerTest):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def _checkJacFull(self, modelName: str, rrJacMode: Config.ROADRUNNER_JACOBIAN_MODE):
        """
        check the values of TestModel jacobian interface against those computed by the current model

        If full is set to false, then we turn on moiety conservation

        :param modelName: A valid test model name (See TestModelFactory::getAvailableTestModelNames()
        :param tol:  degree of tolerance, allowed error
        :param rrJacMode: either Config::ROADRUNNER_JACOBIAN_MODE_AMOUNTS or Config::ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS
        :param full: true if we are looking at full jacobian, false if were looking at reduced.
        """
        self.checkValidTestModelName(modelName)
        testModel = tmf.TestModelFactory(modelName)
        self.checkTestModelImplements(testModel, tmf.JacobianResultFull)
        rr = RoadRunner(testModel.str())
        Config.setValue(Config.ROADRUNNER_JACOBIAN_MODE, rrJacMode)

        expectedJac = None
        if rrJacMode == Config.ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS:
            expectedJac = testModel.fullJacobianConc()
        elif rrJacMode == Config.ROADRUNNER_JACOBIAN_MODE_AMOUNTS:
            expectedJac = testModel.fullJacobianAmt()

        else:
            raise ValueError("Should never reach here")

        jacSettings = testModel.jacobianSettings()
        if "steady_state" in jacSettings.keys():
            rr.steadyState()
        actualJac = rr.getFullJacobian()

        print("\n", expectedJac)
        print(actualJac)

        self.checkMatricesEqual(expectedJac, actualJac)

    def _checkJacReduced(self, modelName: str, rrJacMode: Config.ROADRUNNER_JACOBIAN_MODE):
        """
        check the values of TestModel jacobian interface against those computed by the current model

        If full is set to false, then we turn on moiety conservation

        :param modelName: A valid test model name (See TestModelFactory::getAvailableTestModelNames()
        :param tol:  degree of tolerance, allowed error
        :param rrJacMode: either Config::ROADRUNNER_JACOBIAN_MODE_AMOUNTS or Config::ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS
        :param full: true if we are looking at full jacobian, false if were looking at reduced.
        """
        self.checkValidTestModelName(modelName)
        testModel = tmf.TestModelFactory(modelName)
        self.checkTestModelImplements(testModel, tmf.JacobianResultReduced)
        rr = RoadRunner(testModel.str())
        Config.setValue(Config.ROADRUNNER_JACOBIAN_MODE, rrJacMode)

        expectedJac = None
        if rrJacMode == Config.ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS:
            expectedJac = testModel.reducedJacobianConc()
        elif rrJacMode == Config.ROADRUNNER_JACOBIAN_MODE_AMOUNTS:
            expectedJac = testModel.reducedJacobianAmt()
        else:
            raise ValueError("Should never reach here")

        rr.conservedMoietyAnalysis = True
        jacSettings = testModel.jacobianSettings()
        print("jacSettings", jacSettings)
        if "steady_state" in jacSettings.keys():
            rr.steadyState()

        actualJac = rr.getReducedJacobian()

        self.checkMatricesEqual(expectedJac, actualJac)

    def test_BimolecularEnd_FullAmt(self):
        self._checkJacFull("BimolecularEnd", Config.ROADRUNNER_JACOBIAN_MODE_AMOUNTS)

    def test_BimolecularEnd_FullConc(self):
        self._checkJacFull("BimolecularEnd", Config.ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS)

    def test_SimpleFlux_FullAmt(self):
        self._checkJacFull("SimpleFlux", Config.ROADRUNNER_JACOBIAN_MODE_AMOUNTS)

    def test_SimpleFlux_FullConc(self):
        self._checkJacFull("SimpleFlux", Config.ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS)

    def test_SimpleFlux_ReducedAmt(self):
        self._checkJacReduced("SimpleFlux", Config.ROADRUNNER_JACOBIAN_MODE_AMOUNTS)

    def test_SimpleFlux_ReducedConc(self):
        self._checkJacReduced("SimpleFlux", Config.ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS)


if __name__ == "__main__":
    unittest.main()
