"""Unit test style tests for roadrunner Python API"""

import os
import sys
import unittest
import numpy

thisDir = os.path.dirname(os.path.realpath(__file__))
rr_site_packages = os.path.dirname(os.path.dirname(thisDir))

sys.path += [
    rr_site_packages,
]

try:
    from roadrunner.roadrunner import (
        RoadRunner, Integrator, SteadyStateSolver,
        ExecutableModel, RoadRunnerOptions, Config,
        SensitivitySolver,
        BasicNewtonIteration,
        LinesearchNewtonIteration,
        NLEQ1Solver,
        NLEQ2Solver,
    )


except ImportError:
    from roadrunner import (
        RoadRunner, Integrator, SteadyStateSolver,
        ExecutableModel,
        RoadRunnerOptions,
        Config,
        SensitivitySolver,
        BasicNewtonIteration,
        LinesearchNewtonIteration,
        NLEQ1Solver,
        NLEQ2Solver,
    )

# note, we can also just import test models directly
#   aka import OpenLinearFlux
# Note: Importing "TestModelFactory" into global namespaces
#  tricks some Python unit testing frameworks (such as nose)
#  into thinking the function TestModelFactory is a test, which will fail.
#  So we import TestModelFactory as an alias.
import roadrunner.tests.TestModelFactory as tmf

Config.setValue(Config.LLJIT_OPTIMIZATION_LEVEL, Config.NONE)

class RoadRunnerTests(unittest.TestCase):
    maxDiff = None

    def setUp(self) -> None:
        # print(getAvailableTestModels())
        self.testModel = tmf.TestModelFactory("SimpleFlux")
        self.rr = RoadRunner(self.testModel.str())

    def tearDown(self) -> None:
        pass

    def checkMatricesAlmostEqual(self, expected, actual, places=7):
        if len(expected) != len(actual):
            raise ValueError("len(expected) != len(actual)")

        if len(expected[0]) != len(actual[0]):
            raise ValueError("len(expected[0]) != len(actual[0])")

        for i in range(len(expected)):
            for j in range(len(expected[0])):
                self.assertAlmostEqual(expected[i][j], actual[i][j], places=places)

    def test_species_attribute(self):
        self.assertEqual(self.rr.S1, 10)

    def test_species_attribute_amount(self):
        self.assertEqual(self.rr.S1_amt, 10)

    def test_species_attribute_conc(self):
        self.assertEqual(self.rr.S1_conc, 10)

    def test_reaction_flux(self):
        self.assertEqual(self.rr._J0, 1)

    def test_attribute_access_protocols_setattr(self):
        self.rr.__setattr__("cheese", 6)
        self.assertTrue(hasattr(self.rr, "cheese"))
        self.assertEqual(getattr(self.rr, "cheese"), 6)

    def test_attribute_access_protocols_getattr(self):
        self.assertEqual(self.rr.__getattr__("S1"), 10)

    def test___dict__(self):
        self.assertEqual(
            ["this"],
            list(self.rr.__dict__.keys())
        )

    @unittest.skip("__eq__ may be implemented, may not. If not we should do")
    def test___eq__(self):
        pass

    def test___getitem__(self):
        self.assertEqual(self.rr.__getitem__("S1"), 10.0)

    def test___len__(self):
        self.assertEqual(len(self.rr), 62)  # Enumerates all possible rr[x] arguments.

    def test___repr__(self):
        l = repr([self.rr])
        self.assertIn("roadrunner.RoadRunner()", l)

    def test___setitem__fails_when_key_not_exist(self):
        with self.assertRaises(RuntimeError):
            self.rr.__setitem__("five", 5)

    def test___setitem__when_key_exists(self):
        self.rr.__setitem__("S1", 12)
        self.assertEqual(self.rr.__getitem__("S1"), 12)

    # def test___sizeof__(self):
    #     self.assertEqual(32, self.rr.__sizeof__())

    def test___str__(self):
        s = str(self.rr)
        print(s)
        self.assertIn("roadrunner", s)

    @unittest.skip("Method not implemented yet (but should be)")
    def test_getCompartmentIds(self):
        """This method is missing from the rr api"""
        pass

    def test_addCompartment(self):
        """The rr._addCompartment() method is called by the
        rr.addCompartment() method. The difference, is the
        latter adds the compartment to the __dict__ as an
        attribute.
        """
        self.rr.addCompartment("C1", 1, True)
        self.assertEqual(self.rr.C1, 1)

    def testAddParameter(self):
        self.rr.addParameter("k_new", 25, True)
        self.assertEqual(25, self.rr.k_new)

    def testAddReaction(self):
        self.rr.addReaction("NewReaction", ["S1"], ["S2"], "kf*S1", True)
        self.rr.S1 = 10
        self.assertEqual(self.rr.NewReaction, 1.0)

    def testAddSpeciesConcentrations(self):
        self.rr.addSpeciesConcentration("NewSpecies", "default_compartment", 123.3, False, False, "", True)
        self.assertEqual(self.rr.NewSpecies, 123.3)

    def testAddSpecies(self):
        self.rr.addSpeciesAmount("NewSpecies", "default_compartment", 123.3, False, False, "", True)
        self.assertEqual(self.rr.NewSpecies, 123.3)

    @unittest.skip
    def test_removeInitialAssignment(self):
        self.assertEqual(
            self.rr.removeInitialAssignment(),
        )

    @unittest.skip
    def test_removeParameter(self):
        self.assertEqual(
            self.rr.removeParameter(), )

    @unittest.skip
    def test_removeReaction(self):
        self.assertEqual(
            self.rr.removeReaction(),
        )

    @unittest.skip
    def test_removeRules(self):
        self.assertEqual(
            self.rr.removeRules(),

        )

    @unittest.skip
    def test_removeSpecies(self):
        self.assertEqual(
            self.rr.removeSpecies(),
        )

    def test_diffstep_getter(self):
        self.assertEqual(self.rr._diffstep_getter(), 0.02)

    def test_diffstep_stter(self):
        self.rr._diffstep_stter(0.01)
        self.assertEqual(self.rr._diffstep_getter(), 0.01)

    def test_getConservedMoietyAnalysis(self):
        self.assertFalse(self.rr._getConservedMoietyAnalysis())

    def test_setConservedMoietyAnalysis(self):
        self.rr.conservedMoietyAnalysis = True
        self.assertTrue(self.rr._getConservedMoietyAnalysis())

    def test_addAssignmentRule(self):
        self.rr.addSpeciesConcentration("SNew", "default_compartment", 2.5, False, True, "", True)
        self.rr.addAssignmentRule("SNew", "S1+5", True)
        self.assertEqual(self.rr.SNew, 15)

    @unittest.skip("Not sure how to test this right now")
    def test_addDelay(self):
        pass

    @unittest.skip("Not sure how to test this right now")
    def test_addEvent(self):
        pass

    @unittest.skip("Not sure how to test this right now")
    def test_addEventAssignment(self):
        pass

    @unittest.skip("Not sure how to test this right now")
    def test_addInitialAssignment(self):
        self.rr.addSpeciesConcentration("NewSpec", "default_compartment", 13.3, False, True, "", False)
        self.rr.addInitialAssignment("NewSpec", "S1+S2", True)

    @unittest.skip("Not sure how to test this right now")
    def test_addPriority(self):
        pass

    @unittest.skip("Not sure how to test this right now")
    def test_addRateRule(self):
        pass

    @unittest.skip("Not sure how to test this right now")
    def test_addTrigger(self):
        pass

    def test_clearModel(self):
        self.assertTrue(self.rr.clearModel())

    @unittest.skip("Not sure how to test this right now. Docs are not clear")
    def test_createSelection(self):
        pass

    def test_default_compartment(self):
        self.assertEqual(self.rr.default_compartment, 1)

    def test_diffstep(self):
        self.assertEqual(self.rr.diffstep, 0.02)

    @unittest.skip("This API is unclear. ")
    def test_ensureSolversRegistered(self):
        self.rr.ensureSolversRegistered()

    def test_getAvailableIntegrators(self):
        self.assertEqual(self.rr.getAvailableIntegrators(), ("cvode",))

    def test_getBoundarySpeciesAmountsNamedArray(self):
        self.rr.setBoundary("S1", True)
        bs = self.rr.getBoundarySpeciesAmountsNamedArray()
        self.assertEqual(bs["S1"], 10)

    def test_getBoundarySpeciesConcentrationsNamedArray(self):
        self.rr.setBoundary("S1", True)
        self.rr.default_compartment = 0.5
        bs = self.rr.getBoundarySpeciesConcentrationsNamedArray()
        self.assertEqual(bs["S1"], 20)

    def test_getCC(self):
        self.assertAlmostEqual(self.rr.getCC("_J0", "kf"), 0.009090905110435235)

    def test_getCompiler(self):
        compiler_string = str(self.rr.getCompiler())
        self.assertIn("Compiler", compiler_string)

    def test_getConservationMatrix(self):
        rr2 = RoadRunner(tmf.TestModelFactory("Brown2004").str())
        rr2.conservedMoietyAnalysis = True
        print(rr2.getConservationMatrix().shape)
        self.assertEqual(rr2.getConservationMatrix().shape, (15, 28))

    def test_getConservedMoietyIds(self):
        """This mode has no conserved moieties so poor model for testing this feature"""
        rr2 = RoadRunner(tmf.TestModelFactory("Brown2004").str())
        rr2.conservedMoietyAnalysis = True
        expected = tuple([f"_CSUM{i}" for i in range(15)])
        self.assertEqual(rr2.getConservedMoietyIds(), expected)

    def test_getConservedMoietyValues(self):
        """This mode has no conserved moieties so poor model for testing this feature"""
        rr2 = RoadRunner(tmf.TestModelFactory("Brown2004").str())
        rr2.conservedMoietyAnalysis = True
        print(rr2.getConservedMoietyValues())
        expected = [1.2000e+05, 6.0000e+05, 1.2000e+05, 1.2000e+05, 6.0000e+05, 1.2000e+05, 1.0000e+04, 1.2000e+05,
                    1.2000e+05, 4.4600e+05, 1.2000e+05, 1.0002e+07, 1.2000e+05, 1.2000e+05, 8.0000e+04]
        actual = rr2.getConservedMoietyValues()
        for i in range(len(expected)):
            self.assertAlmostEqual(expected[i], actual[i])

    def test_getCurrentSBML(self):
        self.assertIsInstance(self.rr.getCurrentSBML(), str)

    def test_getDependentFloatingSpeciesIds(self):
        """No dependents in this model"""
        self.assertTrue(callable(self.rr.getDependentFloatingSpeciesIds))

    def test_getDependentRatesOfChange(self):
        """No dependents in this model so we'll just check callable"""
        self.assertTrue(callable(self.rr.getDependentRatesOfChange))

    def test_getDependentRatesOfChangeNamedArray(self):
        """No dependents in this model"""
        self.assertTrue(callable(self.rr.getDependentRatesOfChangeNamedArray))

    def test_getDiffStepSize(self):
        self.assertEqual(0.02, self.rr.getDiffStepSize())

    def test_getEE(self):
        self.assertAlmostEqual(self.rr.getEE("_J0", "kf"), 1.0000000000000024)

    def test_getEigenValueIds(self):
        expected = ('eigen(S1)', 'eigenReal(S1)', 'eigenImag(S1)', 'eigen(S2)', 'eigenReal(S2)', 'eigenImag(S2)')
        self.assertEqual(self.rr.getEigenValueIds(), expected)

    def test_getExistingIntegratorNames(self):
        expected = ('cvode',)
        self.assertEqual(expected, self.rr.getExistingIntegratorNames())

    def test_getExtendedStoichiometryMatrix(self):
        rr2 = RoadRunner(tmf.TestModelFactory("OpenLinearFlux").str())
        mat = rr2.getExtendedStoichiometryMatrix()
        print(mat)
        expected = [
            [1, -1, 0],
            [0, 1, -1],
            [-1, 0, 0],
            [0, 0, 1],
        ]
        # todo implement proper equality operators
        print(mat.shape)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                self.assertEqual(expected[i][j], mat[i][j])

    def test_getFloatingSpeciesAmountsNamedArray(self):
        x = self.rr.getFloatingSpeciesAmountsNamedArray()
        self.assertEqual(x["S2"], 1)

    def test_getFloatingSpeciesConcentrationsNamedArray(self):
        self.rr.default_compartment = 0.33
        x = self.rr.getFloatingSpeciesConcentrationsNamedArray()
        self.assertAlmostEqual(3.0303030303030303, x["S2"][0])

    def test_getFloatingSpeciesInitialConcentrationIds(self):
        x = self.rr.getFloatingSpeciesInitialConcentrationIds()
        self.assertEqual(x, ("init([S1])", "init([S2])",))

    @unittest.skip("figure out how to use")
    def test_getFrequencyResponse(self):
        pass

    def test_getFullEigenValues(self):
        print(dir(self.testModel))
        expected = self.testModel.fullEigenValues()
        actual = self.rr.getFullEigenValues()
        print(expected)
        print(actual)
        self.assertAlmostEqual(expected[0], actual[0])
        self.assertAlmostEqual(expected[1], actual[1])

    def test_getFullJacobian(self):
        print(dir(self.testModel))
        expected = self.testModel.fullJacobianAmt()
        actual = self.rr.getFullJacobian()
        for i in range(expected.shape[0]):
            for j in range(expected.shape[1]):
                self.assertAlmostEqual(expected[i][j], actual[i][j])

    def test_getFullJacobianConc(self):
        Config.setValue(Config.ROADRUNNER_JACOBIAN_MODE, Config.ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS)
        expected = self.testModel.fullJacobianConc()
        actual = self.rr.getFullJacobian()
        for i in range(expected.shape[0]):
            for j in range(expected.shape[1]):
                self.assertAlmostEqual(expected[i][j], actual[i][j])

    def test_getFullJacobianAmt(self):
        expected = self.testModel.fullJacobianAmt()
        Config.setValue(Config.ROADRUNNER_JACOBIAN_MODE, Config.ROADRUNNER_JACOBIAN_MODE_AMOUNTS)
        actual = self.rr.getFullJacobian()
        for i in range(expected.shape[0]):
            for j in range(expected.shape[1]):
                self.assertAlmostEqual(expected[i][j], actual[i][j])

    def test_getFullStoichiometryMatrix(self):
        expected = self.testModel.fullStoicMatrix()
        self.checkMatricesAlmostEqual(expected, self.rr.getFullStoichiometryMatrix())

    def test_getGlobalParameterByName(self):
        self.assertEqual(self.rr.getGlobalParameterByName("kf"), 0.1)

    @unittest.skip("No way to use this method from Python - it "
                   "requires an out parameter as input")
    def test_getIds(self):
        pass

    def test_getIndependentFloatingSpeciesIds(self):
        self.assertEqual(self.rr.getIndependentFloatingSpeciesIds(), ["S1"])

    def test_getIndependentRatesOfChange(self):
        self.assertEqual(-0.99, self.rr.getIndependentRatesOfChange())

    def test_getIndependentRatesOfChangeNamedArray(self):
        self.assertEqual(
            self.rr.getIndependentRatesOfChangeNamedArray(),
            [[-0.99]]
        )

    def test_getInfo(self):
        self.assertIn(
            "jacobianStepSize",
            self.rr.getInfo(),
        )

    def test_getIntegrator(self):
        self.assertEqual(
            self.rr.getIntegrator().getName(),
            "cvode"
        )

    def test_getIntegratorByName(self):
        self.assertIsInstance(
            self.rr.getIntegratorByName("cvode"),
            Integrator
        )

    def test_getKMatrix(self):
        expected = [[1], [1]]
        self.checkMatricesAlmostEqual(
            expected,
            self.rr.getKMatrix(),
        )

    def test_getKineticLaw(self):
        self.assertEqual(
            self.rr.getKineticLaw("_J0"),
            "kf * S1"
        )

    def test_getL0Matrix(self):
        print(dir(self.testModel))
        expected = self.testModel.L0Matrix()
        actual = self.rr.getL0Matrix()
        self.assertEqual(expected, actual)

    def test_getLinkMatrix(self):
        # roadrunner automatically trasposes column vectors
        # into row vectors
        expected = self.testModel.linkMatrix()
        actual = self.rr.getLinkMatrix().T[0]
        print(expected)
        print(actual)
        self.assertTrue((expected == actual).all())

    def test_getModel(self):
        self.assertIsInstance(
            self.rr.getModel(),
            ExecutableModel
        )

    def test_getNrMatrix(self):
        expected = self.testModel.NrMatrix()
        self.checkMatricesAlmostEqual(
            expected,
            self.rr.getNrMatrix(),
        )

    @unittest.skip("No idea what the argument for this method should be?")
    def test_getParamPromotedSBML(self):
        self.assertEqual(
            self.rr.getParamPromotedSBML(),
        )

    def test_getRatesOfChange(self):
        print(self.rr.getRatesOfChange())
        self.assertSequenceEqual(
            list(self.rr.getRatesOfChange()),
            [-0.99, 0.99]
        )

    def test_getRatesOfChangeNamedArray(self):
        print(self.rr.getRatesOfChangeNamedArray())
        self.assertSequenceEqual(
            list(self.rr.getRatesOfChangeNamedArray()[0]),
            [-0.99, 0.99]
        )

    def test_getReactionRates(self):
        self.assertEqual(1.0, self.rr.getReactionRates()[0])
        self.assertEqual(0.01, self.rr.getReactionRates()[1])

    def test_getReducedEigenValues(self):
        expected = self.testModel.reducedEigenValues()
        self.rr.conservedMoietyAnalysis = True
        actual = self.rr.getReducedEigenValues()
        print("expected: ", expected)
        print("actual: ", actual)
        self.assertEqual(expected.shape, actual.shape)
        self.assertAlmostEqual(expected[0], actual[0])

    def test_getReducedJacobian(self):
        self.rr.conservedMoietyAnalysis = True
        expected = self.testModel.reducedJacobianConc()
        actual = self.rr.getReducedJacobian()
        print(expected)
        print(actual)

        self.assertAlmostEqual(
            -0.11,
            self.rr.getReducedJacobian()["S1"][0],
        )

    def test_getReducedStoichiometryMatrix(self):
        self.rr.conservedMoietyAnalysis = True
        expected = self.testModel.reducedStoicMatrix()
        self.checkMatricesAlmostEqual(
            expected,
            self.rr.getReducedStoichiometryMatrix(),
        )

    def test_getRegisteredIntegratorNames(self):
        self.assertEqual(
            ('cvode', 'gillespie', 'rk4', 'rk45', 'euler'),
            self.rr.getRegisteredIntegratorNames(),
        )

    def test_getRegisteredSteadyStateSolverNames(self):
        self.assertEqual(
            ('nleq1', 'nleq2', 'newton', "newton_linesearch"),
            self.rr.getRegisteredSteadyStateSolverNames(),
        )

    def test_getRegisteredSensitivitySolverNames(self):
        print(dir(self.rr))
        self.assertEqual(
            ("forward",),
            self.rr.getRegisteredSensitivitySolverNames(),
        )

    def test_getSBML(self):
        self.assertIsInstance(
            self.rr.getSBML(),
            str
        )

    def test_getScaledConcentrationControlCoefficientMatrix(self):
        expected = self.testModel.scaledConcentrationControlCoefficientMatrix()
        actual = self.rr.getScaledConcentrationControlCoefficientMatrix()
        self.checkMatricesAlmostEqual(
            expected, actual, 5
        )

    def test_getScaledElasticityMatrix(self):
        actual = self.rr.getScaledElasticityMatrix()
        expected = self.testModel.scaledElasticityMatrix()
        print(actual)
        self.checkMatricesAlmostEqual(
            expected, actual
        )

    def test_getScaledFloatingSpeciesElasticity(self):
        self.assertAlmostEqual(
            1.0,
            self.rr.getScaledFloatingSpeciesElasticity("_J1", "S2"),
        )

    def test_getScaledFluxControlCoefficientMatrix(self):
        expected = self.testModel.scaledFluxControlCoefficientMatrix()
        actual = self.rr.getScaledFluxControlCoefficientMatrix()
        self.checkMatricesAlmostEqual(
            expected, actual, 5
        )

    def test_getSelectedValues(self):
        self.assertEqual(self.rr.getSelectedValues()[0], 0)
        self.assertEqual(self.rr.getSelectedValues()[1], 10)

    def test_getSimulationData(self):
        self.assertTrue(
            self.rr.getSimulationData().size == 0
        )
        self.rr.simulate(0, 10, 11)
        actual = self.rr.getSimulationData()
        expected = self.testModel.timeSeriesResult()
        self.checkMatricesAlmostEqual(actual, expected, places=4)

    def test_getSteadyStateSelectionStrings(self):
        self.assertEqual(
            self.rr.getSteadyStateSelectionStrings(),
            ("[S1]", "[S2]")
        )

    def test_getSteadyStateSolver(self):
        solver = self.rr.getSteadyStateSolver()
        self.assertEqual("nleq2", solver.getName())

    def test_getSteadyStateThreshold(self):
        self.assertEqual(
            self.rr.getSteadyStateThreshold(),
            0.01
        )

    def test_getSteadyStateValues(self):
        expected = self.testModel.steadyState()
        actual = self.rr.getSteadyStateValues()
        print(expected)
        print(actual)
        self.assertEqual(expected["S1"], actual[0])
        self.assertEqual(expected["S2"], actual[1])

    def test_getSteadyStateValuesNamedArray(self):
        expected = self.testModel.steadyState()
        actual = self.rr.getSteadyStateValuesNamedArray()
        print(expected)
        self.assertAlmostEqual(expected["S1"], actual[0, 0])
        self.assertAlmostEqual(expected["S2"], actual[0, 1])

    def test_getSensitivitySolver(self):
        self.assertEqual(
            "forward",
            self.rr.getSensitivitySolver().getName()
        )

    @unittest.skip("This method should be not be visible in Python")
    def test_getSupportedIdTypes(self):
        """returns a bit field of the ids that this class supports."""

    def test_getUnscaledConcentrationControlCoefficientMatrix(self):
        expected = self.testModel.unscaledConcentrationControlCoefficientMatrix()
        actual = self.rr.getUnscaledConcentrationControlCoefficientMatrix()
        self.checkMatricesAlmostEqual(
            expected, actual, 5
        )

    def test_getUnscaledElasticityMatrix(self):
        expected = self.testModel.unscaledElasticityMatrix()
        actual = self.rr.getUnscaledElasticityMatrix()
        self.checkMatricesAlmostEqual(
            expected, actual, 5
        )

    def test_getUnscaledFluxControlCoefficientMatrix(self):
        expected = self.testModel.unscaledFluxControlCoefficientMatrix()
        actual = self.rr.getUnscaledFluxControlCoefficientMatrix()
        self.checkMatricesAlmostEqual(
            expected, actual, 5
        )

    def test_getUnscaledParameterElasticity(self):
        self.assertAlmostEqual(
            10,
            self.rr.getUnscaledParameterElasticity("_J0", "kf"),
        )

    def test_getUnscaledSpeciesElasticity(self):
        self.assertEqual(
            0.10000000000000007,
            self.rr.getUnscaledSpeciesElasticity(0, 0),
        )

    def test_getValue(self):
        self.assertEqual(
            self.rr.getValue("kb"),
            0.01
        )

    def test_getuCC(self):
        self.assertAlmostEqual(
            0.09090905110435235,
            self.rr.getuCC("_J0", "kf"),
        )

    def test_getuEE(self):
        self.assertAlmostEqual(
            self.rr.getuEE("_J0", "kf"),
            10
        )

    def test_integrator(self):
        self.assertIsInstance(
            self.rr.integrator,
            Integrator
        )

    def test_integratorExists(self):
        self.assertTrue(
            self.rr.integratorExists("cvode"),
        )

    def test_internalOneStep(self):
        before = self.rr.S2
        self.rr.internalOneStep(0, 10)
        after = self.rr.S2
        self.assertNotEqual(before, after)

    def test_isModelLoaded(self):
        self.assertTrue(
            self.rr.isModelLoaded(),
        )

    def test_items(self):
        expected = [('S1', 10.0),
                    ('S2', 1.0),
                    ('[S1]', 10.0),
                    ('[S2]', 1.0),
                    ('default_compartment', 1.0),
                    ('kf', 0.1),
                    ('kb', 0.01),
                    ('_J0', 1.0),
                    ('_J1', 0.01),
                    ('init([S1])', 10.0),
                    ('init([S2])', 1.0),
                    ('init(S1)', 10.0),
                    ('init(S2)', 1.0),
                    ('init(default_compartment)', 1.0),
                    ('init(kf)', 0.1),
                    ('init(kb)', 0.01),
                    ("S1'", -0.99),
                    ("S2'", 0.99),
                    ("[S1]'", -0.99),
                    ("[S2]'", 0.99),
                    ('stoich(S1, _J0)', 1.0),
                    ('stoich(S1, _J1)', 1.0),
                    ('stoich(S2, _J0)', 1.0),
                    ('stoich(S2, _J1)', 1.0),
                    ('uec(_J0, S1)', 0.100000),
                    ('uec(_J0, S2)', 0.0),
                    ('uec(_J0, kf)', 10.000000),
                    ('uec(_J0, kb)', 0.0),
                    ('uec(_J1, S1)', 0.0),
                    ('uec(_J1, S2)', 0.009999),
                    ('uec(_J1, kf)', 0.0),
                    ('uec(_J1, kb)', 1.000000),
                    ('ec(_J0, S1)', 1.000000),
                    ('ec(_J0, S2)', 0.0),
                    ('ec(_J0, kf)', 1.000000),
                    ('ec(_J0, kb)', 0.0),
                    ('ec(_J1, S1)', 0.0),
                    ('ec(_J1, S2)', 0.999999),
                    ('ec(_J1, kf)', 0.0),
                    ('ec(_J1, kb)', 1.000000),
                    ('ucc(_J0, kf)', 0.090909),
                    ('ucc(_J0, kb)', 9.090909),
                    ('ucc(_J1, kf)', 0.090909),
                    ('ucc(_J1, kb)', 9.090909),
                    ('ucc(S1, kf)', -9.090905),
                    ('ucc(S1, kb)', 90.909090),
                    ('ucc(S2, kf)', 9.090905),
                    ('ucc(S2, kb)', -90.909090),
                    ('cc(_J0, kf)', 0.009090),
                    ('cc(_J0, kb)', 0.090909),
                    ('cc(_J1, kf)', 0.909090),
                    ('cc(_J1, kb)', 9.090909),
                    ('cc(S1, kf)', -0.090909),
                    ('cc(S1, kb)', 0.090909),
                    ('cc(S2, kf)', 0.909090),
                    ('cc(S2, kb)', -0.909090),
                    ('eigen(S1)', (-0.110000+0j)),
                    ('eigenReal(S1)', -0.110000),
                    ('eigenImag(S1)', 0.0),
                    ('eigen(S2)', 0j),
                    ('eigenReal(S2)', 0.0),
                    ('eigenImag(S2)', 0.0)]
        items = self.rr.items()
        for item_index in range(len(expected)):
            self.assertEqual(expected[item_index][0], items[item_index][0])
            self.assertAlmostEqual(expected[item_index][1], items[item_index][1], None, "", 1e-6)

    def test_iteritems(self):

        working_iterator = False
        for i in self.rr.iteritems():
            working_iterator = True
        self.assertTrue(working_iterator)

    def test_iterkeys(self):

        working_iterator = False
        for i in self.rr.iterkeys():
            working_iterator = True
        self.assertTrue(working_iterator)

    def test_itervalues(self):

        working_iterator = False
        for i in self.rr.itervalues():
            working_iterator = True
        self.assertTrue(working_iterator)

    def test_parameter_value(self):
        self.assertEqual(
            self.rr.kf,
            0.1
        )

    def test_keys(self):

        expected = ['S1',
                    'S2',
                    '[S1]',
                    '[S2]',
                    'default_compartment',
                    'kf',
                    'kb',
                    '_J0',
                    '_J1',
                    'init([S1])',
                    'init([S2])',
                    'init(S1)',
                    'init(S2)',
                    'init(default_compartment)',
                    'init(kf)',
                    'init(kb)',
                    "S1'",
                    "S2'",
                    "[S1]'",
                    "[S2]'",
                    'stoich(S1, _J0)',
                    'stoich(S1, _J1)',
                    'stoich(S2, _J0)',
                    'stoich(S2, _J1)',
                    'uec(_J0, S1)',
                    'uec(_J0, S2)',
                    'uec(_J0, kf)',
                    'uec(_J0, kb)',
                    'uec(_J1, S1)',
                    'uec(_J1, S2)',
                    'uec(_J1, kf)',
                    'uec(_J1, kb)',
                    'ec(_J0, S1)',
                    'ec(_J0, S2)',
                    'ec(_J0, kf)',
                    'ec(_J0, kb)',
                    'ec(_J1, S1)',
                    'ec(_J1, S2)',
                    'ec(_J1, kf)',
                    'ec(_J1, kb)',
                    'ucc(_J0, kf)',
                    'ucc(_J0, kb)',
                    'ucc(_J1, kf)',
                    'ucc(_J1, kb)',
                    'ucc(S1, kf)',
                    'ucc(S1, kb)',
                    'ucc(S2, kf)',
                    'ucc(S2, kb)',
                    'cc(_J0, kf)',
                    'cc(_J0, kb)',
                    'cc(_J1, kf)',
                    'cc(_J1, kb)',
                    'cc(S1, kf)',
                    'cc(S1, kb)',
                    'cc(S2, kf)',
                    'cc(S2, kb)',
                    'eigen(S1)',
                    'eigenReal(S1)',
                    'eigenImag(S1)',
                    'eigen(S2)',
                    'eigenReal(S2)',
                    'eigenImag(S2)',]
        self.assertEqual(
            expected,
            self.rr.keys(),
        )

    def test_load(self):
        self.rr.load(self.testModel.str()),
        self.assertTrue(self.rr.isModelLoaded())

    def test_loadState(self):
        this_dir = os.path.dirname(__file__)
        fname = os.path.join(this_dir, "roadrunner_state.rr")
        self.rr.saveState(fname)
        self.assertTrue(os.path.isfile(fname))
        self.rr.loadState(fname)
        self.assertTrue(self.rr.isModelLoaded())
        os.remove(fname)

    def test_loadStateS(self):
        x = self.rr.saveStateS()
        rr2 = RoadRunner()
        rr2.loadStateS(x)
        data = rr2.simulate(0, 10, 11)
        print(data)
        self.assertEqual((11, 3), data.shape)

    def test_makeIntegrator(self):
        integrator = self.rr.makeIntegrator("gillespie")
        self.assertIsInstance(
            integrator,
            Integrator
        )

    def test_mcaSteadyState(self):
        self.assertAlmostEqual(
            self.rr.mcaSteadyState(),
            1.3877787807814457e-17
        )

    def test_model(self):
        self.assertIsInstance(
            self.rr.model,
            ExecutableModel
        )

    def test_oneStep(self):
        before = self.rr.S2
        self.rr.internalOneStep(0, 10)
        after = self.rr.S2
        self.assertNotEqual(before, after)

    def test_options(self):
        self.assertIsInstance(
            self.rr.options,
            RoadRunnerOptions
        )

    def test_regenerateModel(self):
        self.rr.regenerateModel()
        self.assertTrue(self.rr.isModelLoaded())

    @unittest.skip("unclear on how to use this method")
    def test_removeCompartment(self):
        self.rr.removeCompartment("default_compartment", True)
        with self.assertRaises(AttributeError):
            self.rr.default_compartment

    @unittest.skip("no events in model - add one ")
    def test_removeEvent(self):
        pass

    @unittest.skip("no events in model - add one ")
    def test_removeEventAssignments(self):
        print(self.rr.removeEventAssignments())

    def test_reset(self):
        self.rr.reset()
        self.assertTrue(
            self.rr.isModelLoaded()
        )

    def test_resetAll(self):
        """difference between reset and resetAll?"""
        self.rr.resetAll()
        self.assertTrue(
            self.rr.isModelLoaded()
        )

    def test_resetParameter(self):
        self.assertEqual(self.rr.kf, 0.1)  # starting value
        self.rr.kf = 15  # uses setattr
        self.assertEqual(self.rr.kf, 15)  # starting value
        self.rr.resetParameter()
        self.assertEqual(self.rr.kf, 0.1)

    def test_resetSelectionLists(self):
        self.rr.selections = ["[S2]"]
        self.rr.resetSelectionLists()
        self.assertEqual(
            ['time', "[S1]", '[S2]'],
            self.rr.selections,
        )

    def test_resetToOrigin(self):
        self.kf = 16
        self.rr.resetToOrigin()
        self.assertEqual(
            self.rr.kf, 0.1
        )

    def test_getCurrentTime(self):
        self.assertEqual(0, self.rr.getCurrentTime())
        self.rr.simulate(0, 100, 101)
        self.assertEqual(100, self.rr.getCurrentTime())

    def test_selections(self):
        self.assertEqual(
            self.rr.selections,
            ['time', '[S1]', '[S2]']
        )

    def test_setBoundary(self):
        self.rr.setBoundary("S1", False)
        self.assertEqual(0, self.rr.getBoundarySpeciesAmountsNamedArray().size)
        self.rr.setBoundary("S1", True)
        self.assertEqual(1, self.rr.getBoundarySpeciesAmountsNamedArray().size)

    @unittest.skip("We can setConstant but how to getConstant?")
    def test_setConstant(self):
        self.rr.setConstant("S1", False)

    def test_setDiffStepSize(self):
        self.rr.setDiffStepSize(1e-6)
        self.assertEqual(
            self.rr.getDiffStepSize(),
            1e-6
        )

    def test_setGlobalParameterByName(self):
        self.rr.setGlobalParameterByName("kf", 6),
        self.assertEqual(
            self.rr.kf, 6
        )

    @unittest.skip("How to getHasOnlySubstanceUnits?")
    def test_setHasOnlySubstanceUnits(self):
        self.rr.setHasOnlySubstanceUnits("S1", True)

    @unittest.skip("This function looks broken?")
    def test_setInitAmount(self):
        self.rr.setInitAmount("S1", 45.6, True)
        self.assertEqual(
            self.rr.S1, 45.6
        )

    @unittest.skip("This function looks broken?")
    def test_setInitConcentration(self):
        self.rr.setInitConcentration("S1", 45.6, True)
        self.assertEqual(
            self.rr.S1, 45.6
        )

    def test_setIntegrator(self):
        self.rr.setIntegrator("gillespie")
        self.assertEqual(
            "gillespie",
            self.rr.integrator.getName()
        )

    def test_setIntegratorSetting(self):
        self.assertTrue(
            self.rr.integrator.getValue("stiff")
        )

    def test_setKineticLaw(self):
        self.rr.setKineticLaw("_J0", "1+2+3+4+5")
        self.assertEqual(
            self.rr.getKineticLaw("_J0"),
            "1 + 2 + 3 + 4 + 5"
        )

    @unittest.skip("What should go in RoadRunnerOptions?")
    def test_setOptions(self):
        opt = RoadRunnerOptions()
        self.assertEqual(
            self.rr.setOptions(),
        )

    @unittest.skip("how to use?")
    def test_setPersistent(self):
        self.assertEqual(
            self.rr.setPersistent(),
        )

    @unittest.skip("How to get the reversible flag back after we've set it?")
    def test_setReversible(self):
        pass

    def test_setSteadyStateSolver(self):
        self.rr.setSteadyStateSolver("newton")
        solver = self.rr.getSteadyStateSolver()
        print(solver, type(solver))
        self.assertEqual(
            solver.getName(),
            "newton"
        )

    def test_setSteadyStateThreshold(self):
        self.rr.setSteadyStateThreshold(6)
        self.assertEqual(
            self.rr.getSteadyStateThreshold(), 6
        )

    @unittest.skip("how to use?")
    def test_setTriggerInitialValue(self):
        self.assertEqual(
            self.rr.setTriggerInitialValue(),
        )

    @unittest.skip("What values are set on a RoadRunner instance?")
    def test_setValue(self):
        self.assertEqual(
            self.rr.setValue(),
        )

    def test_simulate(self):
        expected = self.testModel.timeSeriesResult()
        settings = self.testModel.timeSeriesSettings()
        start = settings["start"]
        duration = settings["duration"]
        steps = settings["steps"]
        num = steps + 1
        self.rr.simulate(start, duration, num)
        print(expected)
        actual = self.rr.getSimulationData()
        print(actual)
        self.checkMatricesAlmostEqual(expected, actual, 4)

    def test_steadyState(self):
        expected = self.testModel.steadyState()
        self.rr.steadyState()
        actual = self.rr.getFloatingSpeciesAmountsNamedArray()
        print(expected)
        print(actual)
        self.assertAlmostEqual(expected["S1"], actual[0][0])
        self.assertAlmostEqual(expected["S2"], actual[0][1])

    def test_steadyStateNamedArray(self):
        expected = self.testModel.steadyState()
        self.rr.steadyState()
        actual = self.rr.steadyStateNamedArray()
        print(expected)
        print(actual)
        self.assertAlmostEqual(expected["S1"], actual[0][0])
        self.assertAlmostEqual(expected["S2"], actual[0][1])

    def test_steadyStateSelections(self):
        self.assertEqual(
            self.rr.steadyStateSelections,
            ["[S1]", "[S2]"]
        )

    def test_steadyStateSolver(self):
        self.assertIsInstance(
            self.rr.steadyStateSolver,
            SteadyStateSolver
        )

    def test_steadyStateSolverExists(self):
        self.assertTrue(self.rr.steadyStateSolverExists("nleq2"))
        self.assertFalse(self.rr.steadyStateSolverExists("newton"))  # not instantiated yet

    def test_steadyStateThresh(self):
        self.assertEqual(
            self.rr.steadyStateThresh,
            0.01
        )

    def test_timeCourseSelections(self):
        self.assertEqual(
            self.rr.timeCourseSelections,
            ['time', "[S1]", '[S2]']
        )

    @unittest.skip("unclear how to test. Should this method be private?")
    def test_validateCurrentSBML(self):
        self.assertTrue(self.rr.validateCurrentSBML())

    def test_load_model_with_large_recursion_limit(self):
        """
        This was a fun one. Tellurium crashed with version roadrunner 2.0.6
        because it imports plotly. Plotly imports Ipython.display, which
        imports jedi which changes the recursion depth with sys.setrecursionlimit.
        This caused roadrunner to crash. Here we write a test to resolve this problem.
        :return:
        """

        """
        Okay, so we can't call getIds() inside __getattr__. Why not? 
        No, actually - we can't call *anything. infinite recursion
        """
        import sys
        sys.setrecursionlimit(3000)
        m = RoadRunner(self.testModel.str())
        self.assertIsInstance(m, RoadRunner)

    def test_getitem_when_exists(self):
        import sys
        sys.setrecursionlimit(3000)
        m = RoadRunner(self.testModel.str())
        self.assertEqual(10.0, m["S1"])

    def test_getitem_when_not_exists(self):
        import sys
        sys.setrecursionlimit(3000)
        m = RoadRunner(self.testModel.str())
        with self.assertRaises(RuntimeError):
            s = m["S100"]

    def test_getitem_when_created_after_instantiation(self):
        import sys
        sys.setrecursionlimit(3000)
        m = RoadRunner(self.testModel.str())
        m.addSpeciesConcentration("S100", "default_compartment", 123.3, False, False, "", True)
        val = m["S100"]
        self.assertAlmostEqual(val, 123.3)

    def test_getitem_when_created_after_instantiation_dot_notation(self):
        import sys
        sys.setrecursionlimit(3000)
        m = RoadRunner(self.testModel.str())
        m.addSpeciesConcentration("S100", "default_compartment", 123.3, False, False, "", True)
        m._makeProperties()
        self.assertAlmostEqual(m.S100, 123.3)

    def test_simulateWithTimes(self):
        self.rr.resetToOrigin()
        self.rr.simulate(times = [0, 1, 5, 10])
        result = self.rr.getSimulationData()
        self.assertEqual(list(result[:,0]), [0, 1, 5, 10])

    def test_simulateWithNumpyIntTimes(self):
        self.rr.resetToOrigin()
        numpy_intvec = numpy.array([0, 1, 5, 10])
        
        self.rr.simulate(times = numpy_intvec)
        result = self.rr.getSimulationData()
        self.assertEqual(list(result[:,0]), [0, 1, 5, 10])


if __name__ == "__main__":
    unittest.main()
