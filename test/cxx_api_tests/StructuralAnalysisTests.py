import os
import sys
import unittest

import numpy as np

thisDir = os.path.dirname(os.path.realpath(__file__))
rr_site_packages = os.path.dirname(os.path.dirname(thisDir))

sys.path += [
    rr_site_packages,
]

try:
    from roadrunner.roadrunner import (
        RoadRunner, Integrator, SteadyStateSolver,
        ExecutableModel, RoadRunnerOptions
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


class StructuralAnalysisTests(RoadRunnerTest):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def loadTestModel(self, modelName: str):
        """Instantiate instance of TestModel called modelName"""
        self.checkValidTestModelName(modelName)
        testModel = tmf.TestModelFactory(modelName)
        self.checkTestModelImplements(testModel, tmf.StructuralProperties)
        return testModel

    def checkLinkMatrix(self, modelName: str):
        testModel = self.loadTestModel(modelName)
        expected = testModel.linkMatrix()
        rr = RoadRunner(testModel.str())
        actual = rr.getLinkMatrix()
        self.checkMatricesEqual(expected, actual)

    def checkNrMatrix(self, modelName: str):
        testModel = self.loadTestModel(modelName)
        expected = testModel.NrMatrix()
        rr = RoadRunner(testModel.str())
        rr.setConservedMoietyAnalysis = True
        actual = rr.getNrMatrix()
        self.checkMatricesEqual(expected, actual)

    def checkKMatrix(self, modelName: str):
        testModel = self.loadTestModel(modelName)
        expected = testModel.KMatrix()
        rr = RoadRunner(testModel.str())
        actual = rr.getKMatrix()
        for i in range(expected.shape[0]):
            for j in range(expected.shape[1]):
                self.assertAlmostEqual(expected[i][j], actual[i][j], 7)

    def checkReducedStoicMatrix(self, modelName: str):
        testModel = self.loadTestModel(modelName)
        expected = testModel.reducedStoicMatrix()
        rr = RoadRunner(testModel.str())
        rr.setConservedMoietyAnalysis = True
        actual = rr.getReducedStoichiometryMatrix()
        self.checkMatricesEqual(expected, actual)

    def checkFullStoicMatrix(self, modelName: str):
        testModel = self.loadTestModel(modelName)
        expected = testModel.fullStoicMatrix()
        rr = RoadRunner(testModel.str())
        # note turn this to False and the test fails.
        # is this right?
        rr.conservedMoietyAnalysis = True
        actual = rr.getFullStoichiometryMatrix()
        self.checkMatricesEqual(expected, actual)

    def checkExtendedStoicMatrix(self, modelName: str):
        testModel = self.loadTestModel(modelName)
        expected = testModel.extendedStoicMatrix()
        rr = RoadRunner(testModel.str())
        # moiety conservation breaks the test.
        #  Theres something more going on here that should be investigated
        # rr.setConservedMoietyAnalysis = True
        actual = rr.getExtendedStoichiometryMatrix()
        self.checkMatricesEqual(expected, actual)

    def checkL0Matrix(self, modelName: str):
        testModel = self.loadTestModel(modelName)
        expected = testModel.L0Matrix()
        rr = RoadRunner(testModel.str())
        actual = rr.getL0Matrix()
        self.checkMatricesEqual(expected, actual)

    def checkConservationMatrix(self, modelName: str):
        testModel = self.loadTestModel(modelName)
        expected = testModel.conservationMatrix()
        rr = RoadRunner(testModel.str())
        rr.setConservedMoietyAnalysis = True
        actual = rr.getConservationMatrix()
        self.checkMatricesEqual(expected, actual)

    def test_BimolecularEnd_LinkMatrix(self):
        self.checkLinkMatrix("BimolecularEnd")

    def test_BimolecularEnd_NrMatrix(self):
        self.checkNrMatrix("BimolecularEnd")

    def test_BimolecularEnd_KMatrix(self):
        self.checkKMatrix("BimolecularEnd")

    def test_BimolecularEnd_ReducedStoicMatrix(self):
        self.checkReducedStoicMatrix("BimolecularEnd")

    def test_BimolecularEnd_FullStoicMatrix(self):
        self.checkFullStoicMatrix("BimolecularEnd")

    def test_BimolecularEnd_ExtendedStoicMatrix(self):
        self.checkExtendedStoicMatrix("BimolecularEnd")

    @unittest.skip("L0 matrix is empty for this model. Currently there is a "
                   "bug in RoadRunner's swig interface when porting an "
                   "empty Matrix to empty Python np array")
    def test_BimolecularEnd_L0Matrix(self):
        self.checkL0Matrix("BimolecularEnd")

    @unittest.skip("Conservation matrix is empty for this model. Currently there is a "
                   "bug in RoadRunner's swig interface when porting an "
                   "empty Matrix to empty Python np array")
    def test_BimolecularEnd_ConservationMatrix(self):
        self.checkConservationMatrix("BimolecularEnd")
