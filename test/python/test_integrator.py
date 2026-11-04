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


class IntegratorStatisticsTests(unittest.TestCase):
    """The counters of the work of the integrators (issue 1411)."""

    CVODE_COUNTERS = [
        "num_steps", "num_rhs_evals", "num_lin_solv_setups", "num_err_test_fails",
        "num_nonlin_solv_iters", "num_nonlin_solv_conv_fails", "num_step_solve_fails",
        "num_jac_evals", "num_lin_rhs_evals", "num_lin_iters", "num_lin_conv_fails",
        "num_g_evals", "num_stab_lim_order_reds", "num_reinits", "num_events",
    ]

    TIMES = ["time_total", "time_rhs_evals", "time_g_evals", "time_lin_solv_setups", "time_lin_solv_solves"]

    def setUp(self) -> None:
        self.rr = RoadRunner(tmf.TestModelFactory("OneEvent10Triggers").str())

    def numEventsInResult(self, result):
        """The event of OneEvent10Triggers raises S1, which otherwise decays."""
        s1 = result[:, 1]
        return int(np.sum(np.diff(s1) > 0))

    def test_cvode_counters(self):
        result = self.rr.simulate(0, 10, 101)
        stats = self.rr.integrator.getStatistics()
        for key in self.CVODE_COUNTERS:
            self.assertIsInstance(stats[key], int, key)
            self.assertGreaterEqual(stats[key], 0, key)
        self.assertGreater(stats["num_steps"], 0)
        self.assertGreaterEqual(stats["num_rhs_evals"], stats["num_steps"])
        self.assertGreater(stats["num_jac_evals"], 0)
        self.assertGreater(stats["num_lin_rhs_evals"], 0)
        self.assertGreater(stats["num_g_evals"], 0)
        num_events = self.numEventsInResult(result)
        self.assertGreater(num_events, 1)
        self.assertEqual(num_events, stats["num_events"])
        self.assertEqual(num_events, stats["num_reinits"])
        for key in ["actual_init_step", "last_step", "current_step", "current_time"]:
            self.assertIsInstance(stats[key], float, key)
        for key in ["last_order", "current_order"]:
            self.assertIsInstance(stats[key], int, key)
        for key in self.TIMES:
            self.assertNotIn(key, stats)

    def test_cvode_counters_are_summed_over_reinitializations(self):
        """CVODE sets its counters to zero after every event; the statistics are the totals."""
        self.rr.simulate(0, 10, 101)
        total = self.rr.integrator.getStatistics()["num_steps"]
        self.rr.simulate(0, 1, 11)
        beforeFirstEvent = self.rr.integrator.getStatistics()["num_steps"]
        self.assertGreater(total, 5 * beforeFirstEvent)

    def test_each_simulation_has_its_own_statistics(self):
        self.rr.simulate(0, 10, 101)
        first = self.rr.integrator.getStatistics()
        self.rr.resetAll()
        self.rr.simulate(0, 10, 101)
        second = self.rr.integrator.getStatistics()
        for key in self.CVODE_COUNTERS:
            self.assertEqual(first[key], second[key], key)

    def test_reset_statistics(self):
        self.rr.simulate(0, 10, 101)
        self.rr.integrator.resetStatistics()
        stats = self.rr.integrator.getStatistics()
        for key in self.CVODE_COUNTERS:
            self.assertEqual(0, stats[key], key)

    def test_non_stiff_solver_has_no_linear_solver(self):
        self.rr.integrator.stiff = False
        self.rr.simulate(0, 10, 101)
        stats = self.rr.integrator.getStatistics()
        self.assertGreater(stats["num_steps"], 0)
        for key in ["num_lin_solv_setups", "num_jac_evals", "num_lin_rhs_evals"]:
            self.assertEqual(0, stats[key], key)

    def test_cvode_timing(self):
        self.rr.integrator.collect_timing = True
        self.rr.simulate(0, 10, 101)
        stats = self.rr.integrator.getStatistics()
        for key in self.TIMES:
            self.assertIsInstance(stats[key], float, key)
            self.assertGreaterEqual(stats[key], 0.0, key)
        self.assertGreater(stats["time_total"], 0.0)
        self.assertGreater(stats["time_rhs_evals"], 0.0)
        self.assertGreater(stats["time_lin_solv_setups"], 0.0)
        self.assertGreater(stats["time_lin_solv_solves"], 0.0)
        parts = stats["time_rhs_evals"] + stats["time_g_evals"] + stats["time_lin_solv_setups"] \
            + stats["time_lin_solv_solves"]
        self.assertLessEqual(parts, stats["time_total"])

    def test_one_step(self):
        self.rr.oneStep(0, 0.5)
        stats = self.rr.integrator.getStatistics()
        self.assertGreater(stats["num_steps"], 0)
        self.rr.oneStep(0.5, 0.5, False)
        self.assertGreater(self.rr.integrator.getStatistics()["num_steps"], stats["num_steps"])

    def test_rk4(self):
        self.rr.setIntegrator("rk4")
        self.rr.simulate(0, 10, 101)
        stats = self.rr.integrator.getStatistics()
        self.assertEqual({"num_steps", "num_rhs_evals", "num_events"}, set(stats))
        self.assertEqual(100, stats["num_steps"])
        self.assertEqual(400, stats["num_rhs_evals"])

    def test_euler(self):
        self.rr.setIntegrator("euler")
        self.rr.simulate(0, 10, 101)
        stats = self.rr.integrator.getStatistics()
        self.assertEqual({"num_steps", "num_rhs_evals", "num_events"}, set(stats))
        self.assertEqual(100, stats["num_steps"])
        self.assertEqual(100, stats["num_rhs_evals"])

    def test_rk45(self):
        self.rr.setIntegrator("rk45")
        self.rr.simulate(0, 10, 101)
        stats = self.rr.integrator.getStatistics()
        self.assertEqual({"num_steps", "num_rhs_evals", "num_err_test_fails", "num_events"}, set(stats))
        self.assertGreaterEqual(stats["num_steps"], 100)
        self.assertEqual(6 * (stats["num_steps"] + stats["num_err_test_fails"]), stats["num_rhs_evals"])

    def test_gillespie(self):
        rr = RoadRunner(tmf.TestModelFactory("SimpleFlux").str())
        rr.setIntegrator("gillespie")
        rr.integrator.seed = 1
        rr.simulate(0, 10, 11)
        stats = rr.integrator.getStatistics()
        self.assertEqual({"num_steps", "num_rhs_evals", "num_events"}, set(stats))
        self.assertGreater(stats["num_steps"], 0)
        self.assertGreaterEqual(stats["num_rhs_evals"], stats["num_steps"])


if __name__ == "__main__":
    unittest.main()
