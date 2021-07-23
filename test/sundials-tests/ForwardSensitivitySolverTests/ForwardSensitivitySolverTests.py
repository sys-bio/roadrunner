import os
import sys

import numpy as np

thisDir = os.path.dirname(os.path.realpath(__file__))
rr_site_packages = os.path.dirname(os.path.dirname(thisDir))

sys.path += [
    rr_site_packages,
]
from roadrunner.roadrunner import RoadRunner, ForwardSensitivitySolver
import roadrunner.testing.TestModelFactory as tmf

import unittest


class FFSUnitTests(unittest.TestCase):
    maxDiff = None

    def setUp(self) -> None:

        self.testModel = tmf.TestModelFactory("OpenLinearFlux")
        self.rr = RoadRunner(self.testModel.str())
        self.model = self.rr.getModel()
        self.sens = ForwardSensitivitySolver(self.model)

    def test_getGlobalParameterNames(self):
        actual = self.sens.getGlobalParameterNames()
        expected = ('kin', 'kf', 'kout', 'kb')
        self.assertEqual(expected, actual)

    def test_getVariableNames(self):
        actual = self.sens.getVariableNames()
        expected = ('S1', 'S2')
        self.assertEqual(expected, actual)

    def test_CheckValueOfNp(self):
        actual = self.sens.Np
        expected = 4
        self.assertEqual(expected, actual)

    def test_CheckValueOfNsWhenDefaultToAllParameters(self):
        actual = self.sens.Ns
        expected = 4
        self.assertEqual(expected, actual)

    def test_CheckValueOfNsWhenSelectingParameters(self):
        sens2 = ForwardSensitivitySolver(self.model, ["kf"])
        actual = sens2.Ns
        expected = 1
        self.assertEqual(expected, actual)

    def test_CheckValueOfNumModelVariables(self):
        actual = self.sens.numModelVariables
        expected = 2
        self.assertEqual(expected, actual)

    def test_getModelParametersAsMap(self):
        actual = self.sens.getModelParametersAsMap()
        expected = {'kb': 0.01, 'kf': 0.1, 'kin': 1.0, 'kout': 0.2}
        self.assertEqual(expected, actual)

    def test_getModelParametersAsVector(self):
        actual = self.sens.getModelParametersAsVector()
        expected = np.array([1., 0.1, 0.2, 0.01])
        self.assertTrue((expected == actual).all())

    def test_deducePlistDefaultToAllParameters(self):
        self.sens.deducePlist()
        actual = self.sens.plist
        expected = (0, 1, 2, 3)
        self.assertEqual(expected, actual)

    def test_deducePlistFirstParameter(self):
        sens2 = ForwardSensitivitySolver(self.model, ["kin"])
        sens2.deducePlist()
        actual = sens2.plist
        expected = (0,)
        self.assertEqual(expected, actual)

    def test_deducePlistSecondParameter(self):
        sens2 = ForwardSensitivitySolver(self.model, ["kf"])
        sens2.deducePlist()
        actual = sens2.plist
        expected = (1,)
        self.assertEqual(expected, actual)

    def test_SettingsNonLinearSolver(self):
        self.sens.setValue("nonlinear_solver", "fixed_point")
        actual = self.sens.getValue("nonlinear_solver")
        expected = "fixed_point"
        self.assertEqual(expected, actual)

class FFSIntegrationTests(unittest.TestCase):
    maxDiff = None

    def setUp(self) -> None:
        self.testModel = tmf.TestModelFactory("OpenLinearFlux")
        self.rr = RoadRunner(self.testModel.str())
        self.model = self.rr.getModel()
        self.sens = ForwardSensitivitySolver(self.model)

    def checkMatrices3DEqual(self, expected, actual, places=7):
        for k in range(expected.shape[0]):
            for j in range(expected.shape[1]):
                for i in range(expected.shape[2]):
                    print(k, j, i)
                    print(expected[k, j, i], actual[k, j, i])
                    self.assertAlmostEqual(expected[k, j, i], actual[k, j, i], places=places)

    def checkSensWithModel(self, modelName: str, places: int = 7):
        """Check that we can integrate a model in Python without the RoadRunner object

        Args:
            modelName: The name of a test model to integrate. The test model must implement th e
                        timeSeriesResult interface. See TestModelFactor.h
            places: number of decimal places to use for floating point comparisons
        """
        if modelName not in tmf.getAvailableTestModels():
            raise ValueError(f"Model name \"{modelName}\" not available. Here are a list of "
                             f"available model names: {tmf.getAvailableTestModels()}")
        testModel = tmf.TestModelFactory(modelName)
        if tmf.TimeSeriesSensitivityResult not in testModel.__class__.mro():
            raise ValueError("This TestModel does not implement the TimeSeriesSensitivityResult interface. ")

        # we need a RoadRunner object to create a model for us
        rr = RoadRunner(testModel.str())
        model = rr.getModel()
        solver = ForwardSensitivitySolver(model)

        # all tests are done with stiff set to False
        # this is probably a weakness
        solver.setValue("stiff", False)

        settingsMap = testModel.timeSeriesSensitivityResultSettings()
        start = settingsMap["start"]
        duration = settingsMap["duration"]
        numSteps = settingsMap["steps"]
        numIntervals = numSteps - 1

        # always have one more number of data points (in the row direction) than there are number of steps / intervals
        stepSize = (duration - start) / numIntervals

        # collect true values from the TestModel type
        expectedTime, expectedResults = testModel.timeSeriesSensitivityResult()

        # a place to store actual test result data
        results = np.zeros(expectedResults.shape)
        idx = np.zeros(expectedTime.shape)


        numStates = solver.numModelVariables
        numParameters = solver.Ns

        # sensitivities for t=0 are always 0
        if start != 0.0:
            idx[0] = rr.getCurrentTime()
            results[0] = solver.getSensitivityMatrix()

        # integrate, collect data for comparison against actual data
        t = start
        for i in range(1, numSteps):
            t = solver.integrate(t, stepSize)
            idx[i] = t
            results[i] = solver.getSensitivityMatrix()


        print("Expected Results")
        print(expectedResults)
        print("Actual Results")
        print(results)
        # check matrices "almost" equal
        self.checkMatrices3DEqual(expectedResults, results, places=places)

        # check index the same
        self.assertTrue((expectedTime == idx).all())

    def checkSensWithRoadRunner(self, modelName: str, places: int = 7):
        """Check that we can integrate a model in Python using the main RoadRunner api

        Args:
            modelName: The name of a test model to integrate. The test model must implement th e
                        timeSeriesResult interface. See TestModelFactor.h
            places: number of decimal places to use for floating point comparisons
        """
        if modelName not in tmf.getAvailableTestModels():
            raise ValueError(f"Model name \"{modelName}\" not available. Here are a list of "
                             f"available model names: {tmf.getAvailableTestModels()}")
        testModel = tmf.TestModelFactory(modelName)
        if tmf.TimeSeriesSensitivityResult not in testModel.__class__.mro():
            raise ValueError("This TestModel does not implement the TimeSeriesSensitivityResult interface. ")

        rr = RoadRunner(testModel.str())

        # all tests are done with stiff set to False
        # this is probably a weakness
        rr.getSensitivitySolver().setValue("stiff", False)

        settingsMap = testModel.timeSeriesSensitivityResultSettings()
        start = settingsMap["start"]
        duration = settingsMap["duration"]
        numSteps = settingsMap["steps"]
        numIntervals = numSteps - 1
        # always have one more number of data points (in the row direction) than there are number of steps / intervals
        stepSize = (duration - start) / numIntervals

        stop = duration # slightly clearer
        idx, results = rr.timeSeriesSensitivities(start, stop, numSteps)

        # collect true values from the TestModel type
        expectedTime, expectedResults = testModel.timeSeriesSensitivityResult()

        print("Expected Results")
        print(expectedResults)
        print("Actual Results")
        print(results)
        # check matrices "almost" equal
        self.checkMatrices3DEqual(expectedResults, results, places=places)

        # check index the same
        self.assertTrue((expectedTime == idx).all())

    def testOpenLinearFluxWithoutRoadRunnerInstance(self):
        self.checkSensWithModel("OpenLinearFlux", 1)

    def testSimpleFluxWithoutRoadRunnerInstance(self):
        self.checkSensWithModel("SimpleFlux", -1)

    def testModel269WithoutRoadRunnerInstance(self):
        self.checkSensWithModel("Model269", 1)

    def testModel28WithoutRoadRunnerInstance(self):
        self.checkSensWithModel("Model28", 1)

    def testOpenLinearWithRoadRunnerInstance(self):
        self.checkSensWithModel("OpenLinearFlux", 1)

    def testSimpleFluxWithRoadRunnerInstance(self):
        self.checkSensWithModel("SimpleFlux", -1)

    def testModel269WithRoadRunnerInstance(self):
        self.checkSensWithModel("Model269", 1)

    def testModel28WithRoadRunnerInstance(self):
        self.checkSensWithModel("Model28", 1)








