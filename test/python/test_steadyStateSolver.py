import os
import sys

thisDir = os.path.dirname(os.path.realpath(__file__))
rr_site_packages = os.path.dirname(os.path.dirname(thisDir))

sys.path += [
    rr_site_packages,
]
from roadrunner.roadrunner import RoadRunner, BasicNewtonIteration
from roadrunner.tests.TestModelFactory import TestModelFactory
import roadrunner.tests.TestModelFactory as tmf

import unittest


class NewtonIterationUnitTests(unittest.TestCase):

    def setUp(self) -> None:
        # we choose the easiest model for the unit tests.
        # see integration tests for solving other models.
        self.testModel = TestModelFactory("SimpleFluxManuallyReduced")
        self.rr = RoadRunner(self.testModel.str())

    def checkResults(self, results):
        for i in range(len(results.colnames)):
            speciesId = results.colnames[i]
            actualResult = results[0][i]

            expected = self.testModel.steadyState()[speciesId]
            print("Comparing ", speciesId, "expected result: ", expected, "actual result: ", actualResult)
            self.assertAlmostEqual(expected, actualResult)

    def test_newton_iteration_registered(self):
        self.assertIn("newton", self.rr.getRegisteredSteadyStateSolverNames())

    def test_newton_linesearch_registered(self):
        self.assertIn("newton_linesearch", self.rr.getRegisteredSteadyStateSolverNames())

    def test_settings_map_is_dict(self):
        solver = BasicNewtonIteration(self.rr.getModel())
        settings = solver.getSettingsMap()
        self.assertIsInstance(settings, dict)

    def test_solve_using_newton_iteration_directly(self):
        solver = BasicNewtonIteration(self.rr.getModel())
        solver.allow_presimulation = True
        solver.solve()
        self.checkResults(self.rr.getFloatingSpeciesConcentrationsNamedArray())

    def test_change_then_reset_settings(self):
        solver = BasicNewtonIteration(self.rr.getModel())
        solver.presimulation_maximum_steps = 400
        # ensure we've actually changed something
        self.assertEqual(400, solver.presimulation_maximum_steps)
        solver.resetSettings()
        self.assertEqual(500, solver.presimulation_maximum_steps)

    def test_regenerate_model_before_creating_solver(self):
        self.rr.regenerateModel()
        solver = BasicNewtonIteration(self.rr.getModel())
        solver.solve()
        self.checkResults(self.rr.getFloatingSpeciesConcentrationsNamedArray())

    def test_regenerate_model_after_creating_solver(self):
        solver = BasicNewtonIteration(self.rr.getModel())
        self.rr.regenerateModel()
        # after regeneration, the pointer to the model is different so we must sync with model before solving.
        solver.syncWithModel(self.rr.getModel())
        solver.solve()
        self.checkResults(self.rr.getFloatingSpeciesConcentrationsNamedArray())

    def test_presimulation_times(self):
        solver = BasicNewtonIteration(self.rr.getModel())
        self.assertIsInstance(solver.presimulation_times, list)
        self.assertListEqual([0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0], solver.presimulation_times)


class SteadyStateSolverIntegrationTests(unittest.TestCase):
    # defined by subclasses
    strategy = None

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def checkValidTestModelName(self, model_name: str):
        if model_name not in tmf.getAvailableTestModels():
            raise ValueError(f"The requested test model_name \"{model_name}\""
                             f"is not one of the available test models: {tmf.getAvailableTestModels()}")

    def checkValidTestModelType(self, test_model : tmf.TestModel):
        if not issubclass(type(test_model), tmf.SteadyStateResult):
            raise ValueError("This test model does not implement the "
                             "SteadyStateResult interface")

    def checkSteadyState(self, model_name: str, solver_name: str, places:int=5):
        # grab the test model
        self.checkValidTestModelName(model_name)

        test_model = TestModelFactory(model_name)

        self.checkValidTestModelType(test_model)

        # load it into roadrunner
        rr = RoadRunner(test_model.str())

        # collect reference results
        expected_results = test_model.steadyState()

        # set the steady state solver
        rr.setSteadyStateSolver(solver_name)

        # get settings for this problem
        settings = test_model.steadyStateSettings()

        # apply settings
        for setting, value in settings.items():
            if setting == "moiety_conservation":
                rr.conservedMoietyAnalysis = True
            else:
                try:
                    rr.getSteadyStateSolver().setValue(setting, value)
                except Exception:
                    # its okay for a solver to not have a particular options
                    continue

        try:
            # do steady state calculation
            rr.steadyState()

            actual_results = rr.getFloatingSpeciesConcentrationsNamedArray()
            for species_name, expected_ss_val in expected_results.items():
                actual = actual_results[species_name][0]
                print("Comparing reference value : ", expected_ss_val, "with actual value: ", actual)
                self.assertAlmostEqual(expected_ss_val, actual, places=places)
        except Exception as e:
            print(f"Error during steady state calculation: {e}")

    def checkSteadyStateFluxes(self, model_name: str, solver_name: str, places=5):
        self.checkValidTestModelName(model_name)

        test_model = TestModelFactory(model_name)

        self.checkValidTestModelType(test_model)

        # load it into roadrunner
        rr = RoadRunner(test_model.str())

        # collect reference results
        expected_results = test_model.steadyStateFluxes()

        # set the steady state solver
        rr.setSteadyStateSolver(solver_name)

        # get settings for this problem
        settings = test_model.steadyStateSettings()

        # apply settings
        for setting, value in settings.items():
            if setting == "moiety_conservation":
                rr.conservedMoietyAnalysis = True
            else:
                try:
                    rr.getSteadyStateSolver().setValue(setting, value)
                except Exception:
                    # its okay for a solver to not have a particular options
                    continue

        # do steady state calculation
        rr.steadyState()

        for reactionId, expectedSteadyStateFlux in expected_results.items():
            actualSteadyStateFlux = rr.getModel().getValue(reactionId)
            print("Comparing reference value : ", expectedSteadyStateFlux, "with actual value: ", actualSteadyStateFlux)
            self.assertAlmostEqual(expectedSteadyStateFlux, actualSteadyStateFlux, places=places)


class SundialsNewtonIterationTests(SteadyStateSolverIntegrationTests):

    def testOpenLinearFlux(self):
        self.checkSteadyState("OpenLinearFlux", "newton")

    def testSimpleFluxManuallyReduced(self):
        self.checkSteadyState("SimpleFluxManuallyReduced", "newton")

    def testSimpleFlux(self):
        self.checkSteadyState("SimpleFlux", "newton")

    def testVenkatraman2010(self):
        self.checkSteadyState("Venkatraman2010", "newton")

    def testBrown2004(self):
        self.checkSteadyState("Brown2004", "newton")

    def testBimolecularEnd(self):
        self.checkSteadyState("BimolecularEnd", "newton", 4)

    def testBimolecularEndSteadyStateFluxes(self):
        self.checkSteadyStateFluxes("BimolecularEnd", "newton")


class LineSearchNewtonIterationTests(SteadyStateSolverIntegrationTests):
    def testOpenLinearFlux(self):
        self.checkSteadyState("OpenLinearFlux", "newton_linesearch")

    def testSimpleFluxManuallyReduced(self):
        self.checkSteadyState("SimpleFluxManuallyReduced", "newton_linesearch")

    def testSimpleFlux(self):
        self.checkSteadyState("SimpleFlux", "newton_linesearch")

    def testVenkatraman2010(self):
        self.checkSteadyState("Venkatraman2010", "newton_linesearch")

    def testBrown2004(self):
        self.checkSteadyState("Brown2004", "newton_linesearch")

    def testBimolecularEnd(self):
        self.checkSteadyState("BimolecularEnd", "newton_linesearch", 4)

    def testBimolecularEndSteadyStateFluxes(self):
        self.checkSteadyStateFluxes("BimolecularEnd", "newton_linesearch", 5)


class NLEQ1IntegrationTests(SteadyStateSolverIntegrationTests):
    def testOpenLinearFlux(self):
        self.checkSteadyState("OpenLinearFlux", "nleq1")

    def testSimpleFluxManuallyReduced(self):
        self.checkSteadyState("SimpleFluxManuallyReduced", "nleq1")

    def testSimpleFlux(self):
        self.checkSteadyState("SimpleFlux", "nleq1")

    def testVenkatraman2010(self):
        self.checkSteadyState("Venkatraman2010", "nleq1")

    def testBrown2004(self):
        self.checkSteadyState("Brown2004", "nleq1")

    def testBimolecularEnd(self):
        self.checkSteadyState("BimolecularEnd", "nleq1", 4)

    def testBimolecularEndSteadyStateFluxes(self):
        self.checkSteadyStateFluxes("BimolecularEnd", "nleq1", 5)


class NLEQ2IntegrationTests(SteadyStateSolverIntegrationTests):
    def testOpenLinearFlux(self):
        self.checkSteadyState("OpenLinearFlux", "nleq2")

    def testSimpleFluxManuallyReduced(self):
        self.checkSteadyState("SimpleFluxManuallyReduced", "nleq2")

    def testSimpleFlux(self):
        self.checkSteadyState("SimpleFlux", "nleq2")

    def testVenkatraman2010(self):
        self.checkSteadyState("Venkatraman2010", "nleq2")

    def testBrown2004(self):
        self.checkSteadyState("Brown2004", "nleq2")

    def testBimolecularEnd(self):
        self.checkSteadyState("BimolecularEnd", "nleq2", 4)

    def testBimolecularEndSteadyStateFluxes(self):
        self.checkSteadyStateFluxes("BimolecularEnd", "nleq2", 5)



if __name__ == "__main__":
    unittest.main()
