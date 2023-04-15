import os
import sys

import numpy as np

thisDir = os.path.dirname(os.path.realpath(__file__))
rr_site_packages = os.path.dirname(os.path.dirname(thisDir))

sys.path += [
    rr_site_packages,
]
from roadrunner.roadrunner import RoadRunner, CVODEIntegrator, ForwardSensitivitySolver
import roadrunner.tests.TestModelFactory as tmf

import unittest


class CVODEIntegratorTests(unittest.TestCase):
    maxDiff = None

    def setUp(self) -> None:

        self.testModel = tmf.TestModelFactory("OpenLinearFlux")
        self.rr = RoadRunner(self.testModel.str())
        self.model = self.rr.getModel()
        self.integrator = CVODEIntegrator(self.model)

    def checkMatricesEqual(self, expected, actual, places=7):
        for i in range(expected.shape[0]):
            for j in range(expected.shape[1]):
                self.assertAlmostEqual(expected[i, j], actual[i, j], places=places)

    def checkModelIntegrates(self, modelName: str, SolverType=CVODEIntegrator, places: int = 7):
        """Check that we can integrate a model in Python without the RoadRunner object

        Args:
            modelName: The name of a test model to integrate. The test model must implement th e
                        timeSeriesResult interface. See TestModelFactor.h
            SolverType: Either CVODEIntegrator or ForwardSensitivitySolver, since this method doubles for both
            places: number of decimal places to use for floating point comparisons
        """
        if modelName not in tmf.getAvailableTestModels():
            raise ValueError(f"Model name \"{modelName}\" not available. Here are a list of "
                             f"available model names: {tmf.getAvailableTestModels()}")
        testModel = tmf.TestModelFactory(modelName)
        if tmf.TimeSeriesResult not in testModel.__class__.mro():
            raise ValueError("This TestModel does not implement the TimeSeriesResult interface. ")

        # we need a RoadRunner object to create a model for us
        rr = RoadRunner(testModel.str())
        model = rr.getModel()
        integrator = SolverType(model)

        # all tests are done with stiff set to False
        # this is probably a weakness
        integrator.setValue("stiff", False)

        settingsMap = testModel.timeSeriesSettings()
        start = settingsMap["start"]
        duration = settingsMap["duration"]
        steps = settingsMap["steps"]

        # always have one more number of data points (in the row direction) than there are number of steps / intervals
        num = steps + 1
        stepSize = (duration - start) / steps

        # collect true values from the TestModel type
        expectedResults = testModel.timeSeriesResult()

        # a place to store actual test result data
        results = np.zeros(expectedResults.shape)

        # fill in 0th time point
        results[0, 0] = rr.getCurrentTime()
        numStates = len(rr.getFloatingSpeciesInitialConcentrationIds())

        for j in range(1, numStates + 1):
            results[0, j] = rr.getFloatingSpeciesAmountsNamedArray()[0][j - 1]

        t = start
        for i in range(1, num):
            t = integrator.integrate(t, stepSize)
            results[i, 0] = rr.getCurrentTime()
            for j in range(1, numStates + 1):
                results[i, j] = rr.getFloatingSpeciesAmountsNamedArray()[0][j - 1]

        print(expectedResults)
        print(results)
        self.checkMatricesEqual(expectedResults, results, places=places)

    def checkRoadRunnerSimulates(self, modelName: str, places: int = 5):
        """Check that the RoadRunner.simulate produces correct output"""

        if modelName not in tmf.getAvailableTestModels():
            raise ValueError(f"Model name \"{modelName}\" not available. Here are a list of "
                             f"available model names: {tmf.getAvailableTestModels()}")
        testModel = tmf.TestModelFactory(modelName)

        if tmf.TimeSeriesResult not in testModel.__class__.mro():
            raise ValueError("This TestModel does not implement the TimeSeriesResult interface. ")

        # we need a RoadRunner object to create a model for us
        rr = RoadRunner(testModel.str())

        # all tests are done with stiff set to False
        # this is probably a weakness
        rr.getIntegrator().setValue("stiff", False)

        settingsMap = testModel.timeSeriesSettings()
        start = settingsMap["start"]
        duration = settingsMap["duration"]
        steps = settingsMap["steps"]

        # always have one more number of data points (in the row direction) than there are number of steps / intervals
        num = steps + 1
        stepSize = (duration - start) / steps

        # collect true values from the TestModel type
        expectedResults = testModel.timeSeriesResult()

        results = rr.simulate(start, duration, num)
        self.checkMatricesEqual(expectedResults, results, places=places)

    # Integration tests

    # With CVODEIntegrator only

    def testOpenLinearFluxWithoutRoadRunnerInstance(self):
        self.checkModelIntegrates("OpenLinearFlux", CVODEIntegrator, 5)

    def testSimpleFluxWithoutRoadRunnerInstance(self):
        self.checkModelIntegrates("SimpleFlux", CVODEIntegrator, 5)

    def testModel269WithoutRoadRunnerInstance(self):
        self.checkModelIntegrates("Model269", CVODEIntegrator, 5)

    def testModel28WithoutRoadRunnerInstance(self):
        self.checkModelIntegrates("Model28", CVODEIntegrator, 5)

    def testFactorialInRateLawWithoutRoadRunnerInstance(self):
        self.checkModelIntegrates("FactorialInRateLaw", CVODEIntegrator, 4)

    # with ForwardSensitivitySolver only
    #    Note: This is slight abuse of classification, but the test method
    #    doubles nicely for the FFS

    def testOpenLinearFluxWithoutRoadRunnerInstanceFFS(self):
        self.checkModelIntegrates("OpenLinearFlux", ForwardSensitivitySolver, 5)

    def testSimpleFluxWithoutRoadRunnerInstanceFFS(self):
        self.checkModelIntegrates("SimpleFlux", ForwardSensitivitySolver, 5)

    def testModel269WithoutRoadRunnerInstanceFFS(self):
        self.checkModelIntegrates("Model269", ForwardSensitivitySolver, 5)

    def testModel28WithoutRoadRunnerInstanceFFS(self):
        self.checkModelIntegrates("Model28", ForwardSensitivitySolver, 5)

    def testFactorialInRateLawWithoutRoadRunnerInstanceFFS(self):
        self.checkModelIntegrates("FactorialInRateLaw", ForwardSensitivitySolver, 4)

    def testOpenLinearFluxWithRoadRunnerInstance(self):
        self.checkRoadRunnerSimulates("OpenLinearFlux", 5)

    def testSimpleFluxWithRoadRunnerInstance(self):
        self.checkRoadRunnerSimulates("SimpleFlux", 5)

    def testModel269WithRoadRunnerInstance(self):
        self.checkRoadRunnerSimulates("Model269", 5)

    def testModel28WithRoadRunnerInstance(self):
        self.checkRoadRunnerSimulates("Model28", 5)

    def testFactorialInRateLawWithRoadRunnerInstance(self):
        self.checkRoadRunnerSimulates("FactorialInRateLaw", 4)

    # Unit tests

    def test_relative_tolerance(self):
        self.integrator.setValue("relative_tolerance", 1e-07)
        self.assertEqual(1e-07, self.integrator.getValue("relative_tolerance"))

    def test_absolute_tolerance(self):
        self.integrator.setValue("absolute_tolerance", 1e-13)
        self.assertEqual(1e-13, self.integrator.getValue("absolute_tolerance"))

    def test_stiff(self):
        self.integrator.setValue("stiff", False)
        self.assertFalse(self.integrator.getValue("stiff"))

    def test_set_maximum_bdf_order(self):
        self.integrator.setValue("maximum_bdf_order", 3)
        self.assertEqual(3, self.integrator.getValue("maximum_bdf_order"))

    def test_set_maximum_adams_order(self):
        self.integrator.setValue("maximum_adams_order", 2)
        self.assertEqual(2, self.integrator.getValue("maximum_adams_order"))

    def test_set_maximum_num_steps(self):
        self.integrator.setValue("maximum_num_steps", 500)
        self.assertEqual(500, self.integrator.getValue("maximum_num_steps"))

    def test_set_maximum_time_step(self):
        self.integrator.setValue("maximum_time_step", 5)
        self.assertEqual(5, self.integrator.getValue("maximum_time_step"))

    def test_set_minimum_time_step(self):
        self.integrator.setValue("minimum_time_step", 0.1)
        self.assertEqual(0.1, self.integrator.getValue("minimum_time_step"))

    def test_set_initial_time_step(self):
        self.integrator.setValue("initial_time_step", 3.56)
        self.assertEqual(3.56, self.integrator.getValue("initial_time_step"))

    def test_set_multiple_steps(self):
        self.integrator.setValue("multiple_steps", True)
        self.assertTrue(self.integrator.getValue("multiple_steps"))

    def test_set_variable_step_size(self):
        self.integrator.setValue("variable_step_size", True)
        self.assertTrue(self.integrator.getValue("variable_step_size"))

    def test_set_max_output_rows(self):
        self.integrator.setValue("max_output_rows", 50)
        self.assertEqual(50, self.integrator.getValue("max_output_rows"))

    def test_simulate_sequence_with_times(self):
        results = self.rr.simulate()
        self.assertEqual(len(results), 51);
        results = self.rr.simulate(0, 10, 11);
        self.assertEqual(len(results), 11);
        results = self.rr.simulate(times=[3, 5, 11]);
        self.assertEqual(len(results), 3);
        results = self.rr.simulate(0, 10, 11);
        self.assertEqual(len(results), 11);
        results = self.rr.simulate()
        self.assertEqual(len(results), 51);

if __name__ == "__main__":
    unittest.main()
