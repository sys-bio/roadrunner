//
// Created by Ciaran on 07/05/2021.
//

#include "gtest/gtest.h"

#include "RoadRunnerTest.h"
#include "TestModelFactory.h"
#include "SteadyStateSolver.h"
#include "rrExecutableModel.h"
#include "rrRoadRunner.h"
#include "rrConfig.h"

using namespace rr;


class RoadRunnerAPITests : public RoadRunnerTest {

public:
    std::string sbml = SimpleFlux().str();

    RoadRunnerAPITests() = default;
};

TEST_F(RoadRunnerAPITests, DefaultJacobianMode) {
    Setting x = Config::getValue(
            Config::ROADRUNNER_JACOBIAN_MODE
    );
    ASSERT_TRUE(x.get<int>() == Config::ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS);
}

TEST_F(RoadRunnerAPITests, SetJacobianModeToAmt) {
    /**
     * Implicit type conversion has got us into a spot of trouble here.
     * The Config keys are being implicitly converted into an int,
     * rather than their original type, an unsigned int.
     */
    Config::setValue(Config::ROADRUNNER_JACOBIAN_MODE, Config::ROADRUNNER_JACOBIAN_MODE_AMOUNTS);
    Setting x = Config::getValue(
            Config::ROADRUNNER_JACOBIAN_MODE
    );
    ASSERT_TRUE(x.get<int>() == Config::ROADRUNNER_JACOBIAN_MODE_AMOUNTS);
}

TEST_F(RoadRunnerAPITests, GetFullJacobianDefaultConfigSettings) {
    RoadRunner rr(sbml);
    auto matrix = rr.getFullJacobian();
    // no modification of roadrunner Config
    checkMatrixEqual(matrix, SimpleFlux().fullJacobianConc());
}

TEST_F(RoadRunnerAPITests, GetFullJacobianUsingConcMode) {
    Config::setValue(Config::ROADRUNNER_JACOBIAN_MODE, Config::ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS);
    RoadRunner rr(sbml);
    auto matrix = rr.getFullJacobian();
    checkMatrixEqual(matrix, SimpleFlux().fullJacobianConc());
}

TEST_F(RoadRunnerAPITests, GetFullJacobianUsingAmtMode) {
    Config::setValue(Config::ROADRUNNER_JACOBIAN_MODE, Config::ROADRUNNER_JACOBIAN_MODE_AMOUNTS);
    RoadRunner rr(sbml);
    auto matrix = rr.getFullJacobian();
    checkMatrixEqual(matrix, SimpleFlux().fullJacobianAmt());
}

TEST_F(RoadRunnerAPITests, GetFullJacobianUsingAmtModeAsLong) {
    /**
     * Some context for developers:
     * Python uses long for int values. So when
     * Config::setValue is used from Python, the Setting that
     * is implicitly created (second argument to Config::setValue) is a
     * std::int64_t, not an int! The Setting
     * is completely capable of handling an int64, but in other
     * places (getFullJacobian, for example) an int is expected,
     * generating an std::bad_variant_access error.
     * The solution is to use the Setting::getAs<int> command
     * instead of Setting::get<int>, which will do the
     * conversion, if one is needed.
     */
    Config::setValue(Config::ROADRUNNER_JACOBIAN_MODE, std::int64_t(Config::ROADRUNNER_JACOBIAN_MODE_AMOUNTS));
    RoadRunner rr(sbml);
    auto matrix = rr.getFullJacobian();
    checkMatrixEqual(matrix, SimpleFlux().fullJacobianAmt());
}


TEST_F(RoadRunnerAPITests, getIds) {
    std::list<std::string> expected(
            { "S1", "S2", "[S1]", "[S2]",
              "default_compartment", "kf", "kb", "_J0",
              "_J1", "init([S1])", "init([S2])", "init(S1)",
              "init(S2)", "S1'", "S2'", "eigen(S1)", "eigenReal(S1)",
              "eigenImag(S1)", "eigen(S2)", "eigenReal(S2)",
              "eigenImag(S2)" }
            );
    RoadRunner rr(sbml);
    std::list<std::string> l;
    rr.getIds(SelectionRecord::ALL, l);
    ASSERT_EQ(expected, l);
}

TEST_F(RoadRunnerAPITests, LoadModelWithOnlyLayoutInformation) {
    LayoutOnly layoutOnly;
    ASSERT_NO_THROW(RoadRunner rr(layoutOnly.str()););
}

TEST_F(RoadRunnerAPITests, loadSBMLWithLeadingNewlineCharacter) {
    sbml = "\n" + sbml;
    ASSERT_NO_THROW(RoadRunner rr(sbml););
}

/**
 * It appears that support for various SBML versions is not a functional
 * feature at the moment.
 */
TEST_F(RoadRunnerAPITests, DISABLED_RoadRunnerConstructorVersion) {
    RoadRunner rr(3, 2);
    rr.load(sbml);
    std::cout << rr.getSBML(3, 2) << std::endl;
}

/**
 * Support non functional for this test. The responsibility for downloading
 * the sbml from the link is passed on to libsbml, but sbml raises the following:
 *  C++ exception with description "Could not open stream: HTTPS transport not supported" thrown in the test body.
 */
TEST_F(RoadRunnerAPITests, DISABLED_RoadRunnerConstructorFromURL) {
    std::string url = "https://www.ebi.ac.uk/biomodels/model/download/BIOMD0000000373.3?filename=BIOMD0000000373_url.xml";
    RoadRunner rr(url);
    std::cout << rr.getSBML(3, 2) << std::endl;

}

/**
 * The third roadrunner constructor has the signature :
 *      RoadRunner(const std::string& compiler, const std::string& tempDir,
                   const std::string& supportCodeDir);
   But it s not clear how to use this feature? It is likely that this constructor is no longer
   needed and probably doesn't work anyway.
 */
TEST_F(RoadRunnerAPITests, DISABLED_CompilerTempDirSupportCodeDir) {

}

/**
 * The copy constructor errors out on CVODE, for somereason the
 * maximum adams order is too large. disable until fixed.
 */
TEST_F(RoadRunnerAPITests, DISABLED_CopyConstructor) {
    RoadRunner rr1(TestModelFactory("SimpleFlux")->str());
    RoadRunner rr2 = rr1; // logs a error from cvode
    // todo, use equality operators once they are built
    ASSERT_STREQ(rr1.getSBML().c_str(), rr2.getSBML().c_str());

}

/**
 * Equality operators are not actually supported at the moment, this
 * is something that we should add
 */
TEST_F(RoadRunnerAPITests, DISABLED_EqualityOperators) {
    RoadRunner rr1(TestModelFactory("SimpleFlux")->str());
    RoadRunner rr2(TestModelFactory("SimpleFlux")->str());
//    ASSERT_TRUE(rr1 == rr2);
}

/**
 * Inequality operators are not actually supported at the moment, this
 * is something that we should add
 */
TEST_F(RoadRunnerAPITests, DISABLED_InequalityOperators) {
    RoadRunner rr1(TestModelFactory("SimpleFlux")->str());
    RoadRunner rr2(TestModelFactory("OpenLinearFlux")->str());
//    ASSERT_TRUE(rr1 != rr2);
}

TEST_F(RoadRunnerAPITests, getInstanceID) {
    RoadRunner rr1(TestModelFactory("SimpleFlux")->str());
    RoadRunner rr2(TestModelFactory("SimpleFlux")->str());
    ASSERT_EQ(1, rr1.getInstanceID());
    ASSERT_EQ(2, rr2.getInstanceID());
}

TEST_F(RoadRunnerAPITests, getInstanceCount) {
    RoadRunner rr1(TestModelFactory("SimpleFlux")->str());
    RoadRunner rr2(TestModelFactory("SimpleFlux")->str());
    ASSERT_EQ(2, rr2.getInstanceCount());
}

TEST_F(RoadRunnerAPITests, getParamPromotedSBML) {
    std::string s = RoadRunner::getParamPromotedSBML(TestModelFactory("ModelWithLocalParameters")->str());
    std::string expected = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                           "<sbml xmlns=\"http://www.sbml.org/sbml/level3/version1/core\" xmlns:layout=\"http://www.sbml.org/sbml/level3/version1/layout/version1\" xmlns:render=\"http://www.sbml.org/sbml/level3/version1/render/version1\" level=\"3\" version=\"1\" layout:required=\"false\" render:required=\"false\">\n"
                           "  <model id=\"ModelWithLocalParameters\" name=\"ModelWithLocalParameters\" substanceUnits=\"substance\" timeUnits=\"time\" volumeUnits=\"volume\" areaUnits=\"area\" lengthUnits=\"length\" extentUnits=\"substance\">\n"
                           "    <listOfCompartments>\n"
                           "      <compartment id=\"compartment\" name=\"compartment\" spatialDimensions=\"3\" size=\"1\" units=\"volume\" constant=\"true\"/>\n"
                           "    </listOfCompartments>\n"
                           "    <listOfSpecies>\n"
                           "      <species id=\"S1\" name=\"S1\" compartment=\"compartment\" initialConcentration=\"1\" substanceUnits=\"substance\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
                           "      <species id=\"S2\" name=\"S2\" compartment=\"compartment\" initialConcentration=\"1\" substanceUnits=\"substance\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
                           "      <species id=\"S3\" name=\"S3\" compartment=\"compartment\" initialConcentration=\"1\" substanceUnits=\"substance\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
                           "    </listOfSpecies>\n"
                           "    <listOfParameters>\n"
                           "      <parameter id=\"R1_k1\" name=\"k1\" value=\"0.1\" constant=\"true\"/>\n"
                           "      <parameter id=\"R2_k1\" name=\"k1\" value=\"0.1\" constant=\"true\"/>\n"
                           "    </listOfParameters>\n"
                           "    <listOfReactions>\n"
                           "      <reaction id=\"R1\" name=\"R1\" reversible=\"false\" fast=\"false\">\n"
                           "        <listOfReactants>\n"
                           "          <speciesReference species=\"S1\" stoichiometry=\"1\" constant=\"true\"/>\n"
                           "        </listOfReactants>\n"
                           "        <listOfProducts>\n"
                           "          <speciesReference species=\"S2\" stoichiometry=\"1\" constant=\"true\"/>\n"
                           "        </listOfProducts>\n"
                           "        <kineticLaw>\n"
                           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
                           "            <apply>\n"
                           "              <times/>\n"
                           "              <ci> compartment </ci>\n"
                           "              <ci> R1_k1 </ci>\n"
                           "              <ci> S1 </ci>\n"
                           "            </apply>\n"
                           "          </math>\n"
                           "        </kineticLaw>\n"
                           "      </reaction>\n"
                           "      <reaction id=\"R2\" name=\"R2\" reversible=\"false\" fast=\"false\">\n"
                           "        <listOfReactants>\n"
                           "          <speciesReference species=\"S2\" stoichiometry=\"1\" constant=\"true\"/>\n"
                           "        </listOfReactants>\n"
                           "        <listOfProducts>\n"
                           "          <speciesReference species=\"S3\" stoichiometry=\"1\" constant=\"true\"/>\n"
                           "        </listOfProducts>\n"
                           "        <kineticLaw>\n"
                           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
                           "            <apply>\n"
                           "              <times/>\n"
                           "              <ci> compartment </ci>\n"
                           "              <ci> R2_k1 </ci>\n"
                           "              <ci> S2 </ci>\n"
                           "            </apply>\n"
                           "          </math>\n"
                           "        </kineticLaw>\n"
                           "      </reaction>\n"
                           "    </listOfReactions>\n"
                           "  </model>\n"
                           "</sbml>\n";
    ASSERT_STREQ(expected.c_str(), s.c_str());
}

/**
 * getCompiler is only used in the C backend, which I'm
 * no sure even compiles anymore.
 */
TEST_F(RoadRunnerAPITests, DISABLED_getCompiler) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
}

TEST_F(RoadRunnerAPITests, IsModelLoadedWhenFalse) {
    RoadRunner rr;
    ASSERT_FALSE(rr.isModelLoaded());
}

TEST_F(RoadRunnerAPITests, IsModelLoadedWhenTrue) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    ASSERT_TRUE(rr.isModelLoaded());
}


TEST_F(RoadRunnerAPITests, getConservedMoietyAnalysisWhenFalse) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    ASSERT_FALSE(rr.getConservedMoietyAnalysis());
}

TEST_F(RoadRunnerAPITests, getConservedMoietyAnalysisWhenTrue) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    rr.setConservedMoietyAnalysis(true);
    ASSERT_TRUE(rr.getConservedMoietyAnalysis());
}

TEST_F(RoadRunnerAPITests, tmpDir) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    rr.setTempDir(fs::current_path().string());
    ASSERT_STREQ(rr.getTempDir().c_str(), fs::current_path().string().c_str());
}

TEST_F(RoadRunnerAPITests, getIntegratorByNameCVODE) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    ASSERT_STREQ(rr.getIntegratorByName("cvode")->getName().c_str(), "cvode");
}

TEST_F(RoadRunnerAPITests, makeGillespieIntegrator) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    ASSERT_STREQ(rr.makeIntegrator("gillespie")->getName().c_str(), "gillespie");
}

TEST_F(RoadRunnerAPITests, getSteadyStateSolver) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    SteadyStateSolver *solver = rr.getSteadyStateSolver();
    ASSERT_STREQ(solver->getName().c_str(), "nleq2");
}

TEST_F(RoadRunnerAPITests, getExistingIntegratorNames) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    std::vector<std::string> expected = std::vector<std::string>({"cvode"});
    ASSERT_EQ(expected, rr.getExistingIntegratorNames());
}

TEST_F(RoadRunnerAPITests, getRegisteredIntegratorNames) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    std::vector<std::string> expected = std::vector<std::string>({"cvode", "gillespie", "rk4", "rk45", "euler"});
    ASSERT_EQ(expected, rr.getRegisteredIntegratorNames());
}

TEST_F(RoadRunnerAPITests, getRegisteredSteadyStateSolverNames) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    std::vector<std::string> expected = std::vector<std::string>({"nleq1", "nleq2", "newton", "newton_linesearch"});
    ASSERT_EQ(expected, rr.getRegisteredSteadyStateSolverNames());
}

TEST_F(RoadRunnerAPITests, setIntegrator) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    rr.setIntegrator("gillespie");
    ASSERT_STREQ(rr.getIntegrator()->getName().c_str(), "gillespie");
}

TEST_F(RoadRunnerAPITests, integratorExistsWhenTrue) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    ASSERT_TRUE(rr.integratorExists("cvode")); // default
}

TEST_F(RoadRunnerAPITests, integratorExistsWhenFalse) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    ASSERT_FALSE(rr.integratorExists("gillespie")); // not yet created
}

TEST_F(RoadRunnerAPITests, setSteadyStateSolver) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    rr.setSteadyStateSolver("newton");
    ASSERT_STREQ(rr.getSteadyStateSolver()->getName().c_str(), "newton");

}

TEST_F(RoadRunnerAPITests, steadyStateSolverExistsWhenTrue) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    ASSERT_TRUE(rr.steadyStateSolverExists("nleq2"));
}

TEST_F(RoadRunnerAPITests, steadyStateSolverExistsWhenFalse) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    ASSERT_FALSE(rr.steadyStateSolverExists("newton"));
}

/**
 * getModelName doesn't seem to work. But it should be tested
 * in LLVMExecutableModel
 */
TEST_F(RoadRunnerAPITests, DISABLED_getModelName) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    ASSERT_STREQ(rr.getModelName().c_str(), "name");
}

TEST_F(RoadRunnerAPITests, clearModel) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    rr.clearModel();
    ASSERT_FALSE(rr.isModelLoaded());
}

TEST_F(RoadRunnerAPITests, oneStep) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    rr.oneStep(0, 10, true);
    ASSERT_EQ(rr.getModel()->getTime(), 10);
}

/**
 * Difference between oneStep and internalOneStep
 */
TEST_F(RoadRunnerAPITests, internalOneStep) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    rr.oneStep(0, 10, true);
    ASSERT_EQ(rr.getModel()->getTime(), 10);
}

/**
 * State saving needs to be tested properly somehwere else.
 * The save / load features should be a class in their own
 * right and that should be fully tested.
 * This is just a superficial test
 */
TEST_F(RoadRunnerAPITests, saveState) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    fs::path fname = fs::current_path() / "testSaveState.rr";
    rr.saveState(fname.string());
    ASSERT_TRUE(fs::exists(fname));
    fs::remove(fname); // clear up written files
}

TEST_F(RoadRunnerAPITests, loadState) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    fs::path fname = fs::current_path() / "testSaveState.rr";
    rr.saveState(fname.string());
    ASSERT_TRUE(fs::exists(fname));
    RoadRunner loaded;
    loaded.loadState(fname.string());
    ASSERT_EQ(loaded.getNumberOfFloatingSpecies(), 2);
    fs::remove(fname); // clear up written files
}

TEST_F(RoadRunnerAPITests, getSimulationData) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    rr.simulate(0, 10, 11);
    auto x = rr.getSimulationData();
    ASSERT_EQ(11 * 3, x->size()); // 11 time steps, three variables (time, s1, s2)
}

TEST_F(RoadRunnerAPITests, setSimulateOptions) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    SimulateOptions options;
    options.steps = 101;
    options.duration = 100;
    options.start = 0;
    rr.setSimulateOptions(options);
    auto x = rr.getSimulateOptions();
    ASSERT_EQ(x.steps, 101);
    ASSERT_EQ(x.duration, 100);
    ASSERT_EQ(x.start, 0);
}

/**
 * The accuracy of the sbml should be tested somewhere else.
 * This is a superficial API test
 */
TEST_F(RoadRunnerAPITests, getSBML) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    ASSERT_EQ(typeid(std::string), typeid(rr.getSBML()));
}

/**
 * Similar to getSBML, supficial test and the real tests
 * should be within the sbml generating code.
 */
TEST_F(RoadRunnerAPITests, getCurrentSBML) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    ASSERT_EQ(typeid(std::string), typeid(rr.getCurrentSBML()));

}

TEST_F(RoadRunnerAPITests, reset) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    rr.simulate(0, 100, 101);
    ASSERT_EQ(rr.getModel()->getTime(), 100);
    rr.reset();
    ASSERT_EQ(rr.getModel()->getTime(), 0);
}

TEST_F(RoadRunnerAPITests, resetSelectionLists) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    SelectionRecord record = rr.createSelection("S1");
    ASSERT_STREQ(record.to_string().c_str(), "S1");
    rr.resetSelectionLists();
    ASSERT_STREQ(record.to_string().c_str(), "S1");
}

TEST_F(RoadRunnerAPITests, getSelections) {
    std::vector<std::string> expected({"time", "[S1]", "[S2]"});
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    auto x = rr.getSelections();
    std::vector<std::string> actual;
    actual.reserve(x.size());
    for (auto i : x) {
        actual.push_back(i.to_string());
    }
    ASSERT_EQ(expected, actual);
}

TEST_F(RoadRunnerAPITests, getValue) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    auto x = rr.getValue("[S1]");
    ASSERT_EQ(x, 10);
}

TEST_F(RoadRunnerAPITests, setSelections) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    rr.setSelections(std::vector<std::string>({"S1", "[S1]"}));
}

TEST_F(RoadRunnerAPITests, getSelectedValues) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    std::vector<double> x = rr.getSelectedValues();
    std::vector<double> expected({0.0, 10.0, 1.0});
    ASSERT_EQ(x, expected);
}

TEST_F(RoadRunnerAPITests, getIndependentFloatingSpeciesIds) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    auto x = rr.getIndependentFloatingSpeciesIds();
    std::vector<std::string> expected({"S1"});
    ASSERT_EQ(x, expected);
}

TEST_F(RoadRunnerAPITests, getDependentFloatingSpeciesIds) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    auto x = rr.getDependentFloatingSpeciesIds();
    std::vector<std::string> expected({"S2"});
    ASSERT_EQ(x, expected);
}

TEST_F(RoadRunnerAPITests, getFloatingSpeciesConcentrationIds) {
    std::cout << TestModelFactory("SimpleFlux")->str() << std::endl;
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    auto x = rr.getDependentFloatingSpeciesIds();
    std::vector<std::string> expected({"S2"});
    ASSERT_EQ(x, expected);
}

TEST_F(RoadRunnerAPITests, getFloatingSpeciesInitialConcentrationIds) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    auto x = rr.getFloatingSpeciesInitialConcentrations();
    std::vector<double> expected({10.0, 1.0});
    ASSERT_EQ(x, expected);
}

/**
 * I don't understand what this is or how to test it.
 */
TEST_F(RoadRunnerAPITests, DISABLED_getSupportedIdTypes) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    auto x = rr.getSupportedIdTypes();
}

TEST_F(RoadRunnerAPITests, setValue) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    rr.setValue("[S1]", 12.34);
    ASSERT_EQ(rr.getValue("[S1]"), 12.34);
}

TEST_F(RoadRunnerAPITests, getFloatingSpeciesAmountsNamedArray) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    auto x = rr.getFloatingSpeciesAmountsNamedArray();
    std::cout << x << std::endl;
    ls::DoubleMatrix expected({
                                      {10, 1}
                              });
    checkMatrixEqual(expected, x);
    // why does DoubleMatrix not implement equality ops?
}

TEST_F(RoadRunnerAPITests, getFloatingSpeciesConcentrationsNamedArray) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    auto x = rr.getFloatingSpeciesConcentrationsNamedArray();
    std::cout << x << std::endl;
    ls::DoubleMatrix expected({
                                      {10, 1}
                              });
    checkMatrixEqual(expected, x);
}

TEST_F(RoadRunnerAPITests, getBoundarySpeciesAmountsNamedArray) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    rr.setBoundary("S1", true, true);
    auto x = rr.getBoundarySpeciesAmountsNamedArray();
    std::cout << x << std::endl;
    ls::DoubleMatrix expected({{10}});
    checkMatrixEqual(expected, x);
}

TEST_F(RoadRunnerAPITests, getBoundarySpeciesConcentrationsNamedArray) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    rr.setBoundary("S1", true, true);
    auto x = rr.getBoundarySpeciesConcentrationsNamedArray();
    std::cout << x << std::endl;
    ls::DoubleMatrix expected({{10}});
    checkMatrixEqual(expected, x);
}


TEST_F(RoadRunnerAPITests, getRatesOfChange) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    auto x = rr.getRatesOfChange();
    std::vector<double> expected({-0.99, 0.99});
    ASSERT_EQ(x, expected);
}

TEST_F(RoadRunnerAPITests, getRatesOfChangeNamedArray) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    auto x = rr.getRatesOfChangeNamedArray();
    ls::DoubleMatrix expected({{-0.99, 0.99}});
    checkMatrixEqual(x, expected);
}

TEST_F(RoadRunnerAPITests, getIndependentRatesOfChange) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    auto x = rr.getIndependentRatesOfChange();
    std::vector<double> expected({-0.99});
    ASSERT_EQ(x, expected);
}

TEST_F(RoadRunnerAPITests, getIndependentRatesOfChangeNamedArray) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    auto x = rr.getIndependentRatesOfChangeNamedArray();
    ls::DoubleMatrix expected({{-0.99}});
    checkMatrixEqual(x, expected);
}

TEST_F(RoadRunnerAPITests, getDependentRatesOfChange) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    auto x = rr.getDependentRatesOfChange();
    std::vector<double> expected({0.99});
    ASSERT_EQ(x, expected);
}


TEST_F(RoadRunnerAPITests, getDependentRatesOfChangeNamedArray) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    auto x = rr.getDependentRatesOfChangeNamedArray();
    ls::DoubleMatrix expected({{0.99}});
    checkMatrixEqual(x, expected);
}

TEST_F(RoadRunnerAPITests, getFullJacobian) {
    auto *venk = dynamic_cast<Venkatraman2010 *>(TestModelFactory("Venkatraman2010"));
    RoadRunner rr(venk->str());
    auto x = rr.getFullJacobian();
    std::cout << x << std::endl;
    checkMatrixEqual(venk->fullJacobianConc(), x, 1e-3);
}

TEST_F(RoadRunnerAPITests, getReducedJacobian) {
    SimpleFlux simpleFlux;
    RoadRunner rr(simpleFlux.str());
    rr.setConservedMoietyAnalysis(true);
    auto x = rr.getReducedJacobian();
    std::cout << x << std::endl;
    checkMatrixEqual(simpleFlux.reducedJacobianConc(), x, 1e-3);
}


TEST_F(RoadRunnerAPITests, getFullEigenValues) {
    SimpleFlux simpleFlux;
    RoadRunner rr(simpleFlux.str());
    rr.setConservedMoietyAnalysis(true);
    std::vector<std::complex<double>> expected = rr.getFullEigenValues();
    auto eig = rr.getFullEigenValues();
    for (int i = 0; i < expected.size(); i++) {
        ASSERT_EQ(expected[i], eig[i]);
    }
}

TEST_F(RoadRunnerAPITests, getReducedEigenValues) {
    SimpleFlux simpleFlux;
    RoadRunner rr(simpleFlux.str());
    rr.setConservedMoietyAnalysis(true);
    std::vector<std::complex<double>> expected = rr.getReducedEigenValues();
    auto eig = rr.getReducedEigenValues();
    for (int i = 0; i < expected.size(); i++) {
        ASSERT_EQ(expected[i], eig[i]);
    }
}


TEST_F(RoadRunnerAPITests, getLinkMatrix) {
    SimpleFlux simpleFlux;
    RoadRunner rr(simpleFlux.str());
    std::cout << rr.getLinkMatrix() << std::endl;
    checkMatrixEqual(simpleFlux.linkMatrix(), rr.getLinkMatrix());
}

TEST_F(RoadRunnerAPITests, getNrMatrix) {
    SimpleFlux simpleFlux;
    RoadRunner rr(simpleFlux.str());
    std::cout << rr.getNrMatrix() << std::endl;
    checkMatrixEqual(simpleFlux.NrMatrix(), rr.getNrMatrix());
}

TEST_F(RoadRunnerAPITests, getKMatrix) {
    SimpleFlux simpleFlux;
    RoadRunner rr(simpleFlux.str());
    std::cout << rr.getKMatrix() << std::endl;
    checkMatrixEqual(simpleFlux.KMatrix(), rr.getKMatrix());
}

TEST_F(RoadRunnerAPITests, getReducedStoichiometryMatrix) {
    SimpleFlux simpleFlux;
    RoadRunner rr(simpleFlux.str());
    rr.setConservedMoietyAnalysis(true);
    std::cout << rr.getReducedStoichiometryMatrix() << std::endl;
    checkMatrixEqual(simpleFlux.reducedStoicMatrix(), rr.getReducedStoichiometryMatrix());
}

TEST_F(RoadRunnerAPITests, getFullStoichiometryMatrix) {
    SimpleFlux simpleFlux;
    RoadRunner rr(simpleFlux.str());
    rr.setConservedMoietyAnalysis(true);
    std::cout << rr.getFullStoichiometryMatrix() << std::endl;
    checkMatrixEqual(simpleFlux.fullStoicMatrix(), rr.getFullStoichiometryMatrix());
}

TEST_F(RoadRunnerAPITests, getExtendedStoichiometryMatrix) {
    SimpleFlux simpleFlux;
    RoadRunner rr(simpleFlux.str());
    rr.setConservedMoietyAnalysis(true);
    std::cout << rr.getExtendedStoichiometryMatrix() << std::endl;
    checkMatrixEqual(simpleFlux.extendedStoicMatrix(), rr.getExtendedStoichiometryMatrix());
}

TEST_F(RoadRunnerAPITests, getL0Matrix) {
    SimpleFlux simpleFlux;
    RoadRunner rr(simpleFlux.str());
    rr.setConservedMoietyAnalysis(true);
    std::cout << rr.getL0Matrix() << std::endl;
    checkMatrixEqual(simpleFlux.L0Matrix(), rr.getL0Matrix());
}

TEST_F(RoadRunnerAPITests, getConservationMatrix) {
    SimpleFlux simpleFlux;
    RoadRunner rr(simpleFlux.str());
    rr.setConservedMoietyAnalysis(true);
    std::cout << rr.getConservationMatrix() << std::endl;
    checkMatrixEqual(simpleFlux.conservationMatrix(), rr.getConservationMatrix());
}

TEST_F(RoadRunnerAPITests, getUnscaledConcentrationControlCoefficientMatrix) {
    SimpleFlux simpleFlux;
    RoadRunner rr(simpleFlux.str());
    rr.setConservedMoietyAnalysis(true);
    std::cout << rr.getUnscaledConcentrationControlCoefficientMatrix() << std::endl;
    checkMatrixEqual(simpleFlux.unscaledConcentrationControlCoefficientMatrix(),
                     rr.getUnscaledConcentrationControlCoefficientMatrix());
}

TEST_F(RoadRunnerAPITests, getScaledConcentrationControlCoefficientMatrix) {
    SimpleFlux simpleFlux;
    RoadRunner rr(simpleFlux.str());
    rr.setConservedMoietyAnalysis(true);
    std::cout << rr.getScaledConcentrationControlCoefficientMatrix() << std::endl;
    checkMatrixEqual(simpleFlux.scaledConcentrationControlCoefficientMatrix(),
                     rr.getScaledConcentrationControlCoefficientMatrix());

}

TEST_F(RoadRunnerAPITests, getUnscaledFluxControlCoefficientMatrix) {
    SimpleFlux simpleFlux;
    RoadRunner rr(simpleFlux.str());
    rr.setConservedMoietyAnalysis(true);
    std::cout << rr.getUnscaledFluxControlCoefficientMatrix() << std::endl;
    checkMatrixEqual(simpleFlux.unscaledFluxControlCoefficientMatrix(),
                     rr.getUnscaledFluxControlCoefficientMatrix());

}

TEST_F(RoadRunnerAPITests, getScaledFluxControlCoefficientMatrix) {
    SimpleFlux simpleFlux;
    RoadRunner rr(simpleFlux.str());
    rr.setConservedMoietyAnalysis(true);
    std::cout << rr.getScaledFluxControlCoefficientMatrix() << std::endl;
    checkMatrixEqual(simpleFlux.scaledFluxControlCoefficientMatrix(),
                     rr.getScaledFluxControlCoefficientMatrix());

}

TEST_F(RoadRunnerAPITests, getEigenValueIds) {
    RoadRunner rr(SimpleFlux().str());
    auto actual = rr.getEigenValueIds();
    std::vector<std::string> expected(
            {"eigen(S1)", "eigenReal(S1)", "eigenImag(S1)", "eigen(S2)", "eigenReal(S2)", "eigenImag(S2)"});
    ASSERT_EQ(expected, actual);
}

TEST_F(RoadRunnerAPITests, getUnscaledParameterElasticity) {
    SimpleFlux simpleFlux;
    RoadRunner rr(simpleFlux.str());
    double v = rr.getUnscaledParameterElasticity("_J1", "kb");
    ASSERT_NEAR(v, 1.0, 1e-7);
}

/**
 * Unsure how to use
 */
TEST_F(RoadRunnerAPITests, DISABLED_getFrequencyResponse) {
    SimpleFlux simpleFlux;
    RoadRunner rr(simpleFlux.str());
}

TEST_F(RoadRunnerAPITests, setConservedMoietyAnalysis) {
    SimpleFlux simpleFlux;
    RoadRunner rr(simpleFlux.str());
    rr.setConservedMoietyAnalysis(true);
    ASSERT_TRUE(rr.getConservedMoietyAnalysis());
}

TEST_F(RoadRunnerAPITests, getDiffStepSize) {
    SimpleFlux simpleFlux;
    RoadRunner rr(simpleFlux.str());
    rr.setDiffStepSize(0.3);
    ASSERT_NEAR(0.3, rr.getDiffStepSize(), 1e-7);
}

TEST_F(RoadRunnerAPITests, getSteadyStateThreshold) {
    SimpleFlux simpleFlux;
    RoadRunner rr(simpleFlux.str());
    rr.setSteadyStateThreshold(0.1234);
    ASSERT_NEAR(0.1234, rr.getSteadyStateThreshold(), 1e-7);
}

TEST_F(RoadRunnerAPITests, getuCC) {
    SimpleFlux simpleFlux;
    RoadRunner rr(simpleFlux.str());
    double actual = rr.getuCC("_J1", "kb");
    double expected = 9.0909090753844755;
    ASSERT_NEAR(expected, actual, 1e-7);
}

TEST_F(RoadRunnerAPITests, getCC) {
    RoadRunner rr(SimpleFlux().str());
    double actual = rr.getCC("_J1", "kb");
    double expected = 9.0909090753844755;
    ASSERT_NEAR(expected, actual, 1e-7);
}

TEST_F(RoadRunnerAPITests, getuEE) {
    RoadRunner rr(SimpleFlux().str());
    double actual = rr.getuEE("_J1", "kb");
    double expected = 1.0;
    ASSERT_NEAR(expected, actual, 1e-7);
}

TEST_F(RoadRunnerAPITests, getEE) {
    RoadRunner rr(SimpleFlux().str());
    double actual = rr.getEE("_J1", "kb");
    double expected = 1.0;
    ASSERT_NEAR(expected, actual, 1e-7);
}

TEST_F(RoadRunnerAPITests, getUnscaledElasticityMatrix) {
    SimpleFlux simpleFlux;
    RoadRunner rr(simpleFlux.str());
    rr.setConservedMoietyAnalysis(true);
    std::cout << rr.getUnscaledElasticityMatrix() << std::endl;
    checkMatrixEqual(simpleFlux.unscaledElasticityMatrix(),
                     rr.getUnscaledElasticityMatrix());
}

TEST_F(RoadRunnerAPITests, getScaledElasticityMatrix) {
    SimpleFlux simpleFlux;
    RoadRunner rr(simpleFlux.str());
    rr.setConservedMoietyAnalysis(true);
    std::cout << rr.getScaledElasticityMatrix() << std::endl;
    checkMatrixEqual(simpleFlux.scaledElasticityMatrix(),
                     rr.getScaledElasticityMatrix());
}

TEST_F(RoadRunnerAPITests, getScaledFloatingSpeciesElasticity) {
    SimpleFlux simpleFlux;
    RoadRunner rr(simpleFlux.str());
    double actual = rr.getScaledFloatingSpeciesElasticity("_J1", "S2");
    double expected = 1.0;
    ASSERT_NEAR(expected, actual, 1e-7);
}

TEST_F(RoadRunnerAPITests, getUnscaledSpeciesElasticity) {
    SimpleFlux simpleFlux;
    RoadRunner rr(simpleFlux.str());
    // why is args int in this case and all similar methods a string
    double actual = rr.getUnscaledSpeciesElasticity(0, 0);
    double expected = 0.09999999;
    ASSERT_NEAR(expected, actual, 1e-7);
}

TEST_F(RoadRunnerAPITests, addSpeciesConcentration) {
    RoadRunner rr(SimpleFlux().str());
    rr.setValue("default_compartment", 0.33); // set volume so we see a change vs amounts
    ASSERT_EQ(2, rr.getFloatingSpeciesConcentrationIds().size());
    rr.addSpeciesConcentration("S3", "default_compartment", 123.4, false, false, "mol", true);
    ls::DoubleMatrix expectedConc({{30.303, 3.0303, 373.939}});
    ls::DoubleMatrix expectedAmt({{10, 1, 123.4}});
    checkMatrixEqual(expectedConc, rr.getFloatingSpeciesConcentrationsNamedArray());
    checkMatrixEqual(expectedAmt, rr.getFloatingSpeciesAmountsNamedArray());
}

TEST_F(RoadRunnerAPITests, addSpeciesAmount) {
    RoadRunner rr(SimpleFlux().str());
    rr.setValue("default_compartment", 0.33); // set volume so we see a change vs amounts
    ASSERT_EQ(2, rr.getFloatingSpeciesConcentrationIds().size());
    rr.addSpeciesAmount("S3", "default_compartment", 123.4, true, false, "mol", true);
    ls::DoubleMatrix expectedConc({{30.303, 3.0303, 373.939}});
    ls::DoubleMatrix expectedAmt({{10, 1, 123.4}});
    checkMatrixEqual(expectedConc, rr.getFloatingSpeciesConcentrationsNamedArray());
    checkMatrixEqual(expectedAmt, rr.getFloatingSpeciesAmountsNamedArray());
}


TEST_F(RoadRunnerAPITests, removeSpecies) {
    RoadRunner rr(SimpleFlux().str());
    rr.removeSpecies("S1");
    ASSERT_EQ(1, rr.getFloatingSpeciesAmountsNamedArray().size());

}

TEST_F(RoadRunnerAPITests, setBoundary){
    RoadRunner rr(SimpleFlux().str());
    ASSERT_EQ(0, rr.getBoundarySpeciesIds().size());
    rr.setBoundary("S1", true, true);
    ASSERT_EQ(1, rr.getBoundarySpeciesIds().size());


}

/**
 * There is no "getHasOnlySubstanceUnits" and the flag doesn't appear to
 * do anything so I'm not sure how to test this.
 */
TEST_F(RoadRunnerAPITests, DISABLED_setHasOnlySubstanceUnits){
    RoadRunner rr(SimpleFlux().str());
    rr.setHasOnlySubstanceUnits("S1", true, true);
}

TEST_F(RoadRunnerAPITests, setInitAmount){
    RoadRunner rr(SimpleFlux().str());
    rr.setInitAmount("S1", 1234.5);
    std::cout << rr.getFloatingSpeciesAmountsNamedArray() << std::endl;
    ls::DoubleMatrix expected({{1234.5, 1}});
    auto actual = rr.getFloatingSpeciesAmountsNamedArray();
    checkMatrixEqual(expected, actual);
}

TEST_F(RoadRunnerAPITests, setInitConcentration){
    RoadRunner rr(SimpleFlux().str());
    rr.setInitConcentration("S1", 1234.5);
    std::cout << rr.getFloatingSpeciesAmountsNamedArray() << std::endl;
    ls::DoubleMatrix expected({{1234.5, 1}});
    auto actual = rr.getFloatingSpeciesConcentrationsNamedArray();
    checkMatrixEqual(expected, actual);
}

/**
 * We can set something to constant, but how can
 * we verify that something *is* constant?
 */
TEST_F(RoadRunnerAPITests, DISABLED_setConstant){
    RoadRunner rr(SimpleFlux().str());
    rr.setConstant("S1", true, true);
    auto x = rr.getBoundarySpeciesIds();
    std::vector<std::string> expected({"S1"});
    ASSERT_EQ(expected, x);
}

TEST_F(RoadRunnerAPITests, addReactionFromSBML){
    std::string reaction = "<reaction id=\"r3\" reversible=\"false\" fast=\"false\">\n"
                           "        <listOfReactants>\n"
                           "          <speciesReference species=\"S1\" stoichiometry=\"1\" constant=\"true\"/>\n"
                           "        </listOfReactants>\n"
                           "        <listOfProducts>\n"
                           "          <speciesReference species=\"S2\" stoichiometry=\"1\" constant=\"true\"/>\n"
                           "        </listOfProducts>\n"
                           "        <kineticLaw>\n"
                           "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
                           "            <apply>\n"
                           "              <times/>\n"
                           "              <cn type=\"integer\"> 5 </cn>\n"
                           "              <cn type=\"integer\"> 4 </cn>\n"
                           "              <cn type=\"integer\"> 3 </cn>\n"
                           "              <ci> S1 </ci>\n"
                           "            </apply>\n"
                           "          </math>\n"
                           "        </kineticLaw>\n"
                           "      </reaction>";
    RoadRunner rr(SimpleFlux().str());
    rr.addReaction(reaction, true);
    ASSERT_EQ(3, rr.getFullStoichiometryMatrix().numCols());
}

TEST_F(RoadRunnerAPITests, addReactionDirect){
    RoadRunner rr(SimpleFlux().str());
    rr.addReaction("r3", {"S1"}, {"S2"}, "5*4*3*S1", true);
    ASSERT_EQ(3, rr.getFullStoichiometryMatrix().numCols());
}

/*
 * this test fails with
 *  unknown file: error: C++ exception with description "Roadrunner::removeParameter failed, no parameter with ID S2 existed in the model" thrown in the test body.
 */
TEST_F(RoadRunnerAPITests, DISABLED_removeReactionWithUnusedParameters){
    RoadRunner rr(SimpleFlux().str());
    rr.removeReaction("_J1", true, true);
    ASSERT_EQ(1, rr.getFullStoichiometryMatrix().numCols());
    ASSERT_EQ(1, rr.getGlobalParameterIds().size());
}

TEST_F(RoadRunnerAPITests, removeReactionWithoutUnusedParameters){
    RoadRunner rr(SimpleFlux().str());
    rr.removeReaction("_J1", false, true);
    ASSERT_EQ(1, rr.getFullStoichiometryMatrix().numCols());
    ASSERT_EQ(2, rr.getGlobalParameterIds().size());
}

/**
 * We cannot "getReversible" with current API
 */
TEST_F(RoadRunnerAPITests, DISABLED_setReversible){
    RoadRunner rr(SimpleFlux().str());
    rr.setReversible("_J1",  true, true);
}

TEST_F(RoadRunnerAPITests, setKineticLaw){
    RoadRunner rr(SimpleFlux().str());
    rr.setKineticLaw("_J1", "(kf/(kb+1))*S1", true);
    std::string k = rr.getKineticLaw("_J1");
    ASSERT_STREQ(k.c_str(), "(kf / (kb + 1)) * S1");
}

TEST_F(RoadRunnerAPITests, addParameter){
    RoadRunner rr(SimpleFlux().str());
    rr.addParameter("NewParameter", 65, true);
    ASSERT_EQ(65, rr.getGlobalParameterByName("NewParameter"));
}

TEST_F(RoadRunnerAPITests, removeParameter){
    // todo what happens when you remove a parameter that is
    // in use in a reaction? Should throw an error, or at least emit warning
    RoadRunner rr(SimpleFlux().str());
    rr.removeParameter("kf");
    ASSERT_EQ(1, rr.getGlobalParameterIds().size());
}

TEST_F(RoadRunnerAPITests, addCompartment){
    RoadRunner rr(SimpleFlux().str());
    rr.addCompartment("newComp", 0.33, true);
    ASSERT_EQ(2, rr.getNumberOfCompartments());
}

TEST_F(RoadRunnerAPITests, removeCompartment){
    RoadRunner rr(SimpleFlux().str());
    rr.addCompartment("newComp", 0.33, true);
    rr.removeCompartment("newComp", true);
    ASSERT_EQ(1, rr.getNumberOfCompartments());

}

/**
 * How can we verify that we have a new assignment rule?
 */
TEST_F(RoadRunnerAPITests, DISABLED_addAssignmentRule){
    RoadRunner rr(SimpleFlux().str());
    rr.addAssignmentRule("STotal", "S1+S2", true);
}

/**
 * Works, but how to verify that we have a new rate rule?
 */
TEST_F(RoadRunnerAPITests, DISABLED_addRateRule){
    RoadRunner rr(SimpleFlux().str());
    rr.setBoundary("S2", true);
    rr.addRateRule("S2", "(kf/kb)*S1", true);
    std::cout << rr.getSBML() << std::endl;
    std::cout << *rr.simulate(0, 100, 101) << std::endl;

}

/**
 * How to test?
 */
TEST_F(RoadRunnerAPITests, DISABLED_removeRules){
    RoadRunner rr(SimpleFlux().str());
    rr.setBoundary("S2", true);
    rr.addRateRule("S2", "(kf/kb)*S1", true);
    rr.removeRules("S2", false, true);
}

/**
 * Works but cannot currently getInitialAssignments with
 * current API, so can't test.
 */
TEST_F(RoadRunnerAPITests, addInitialAssignment){
    RoadRunner rr(SimpleFlux().str());
    rr.addInitialAssignment("S1", "0.5*S2", true);
}

/**
 * Again, how to test?
 */
TEST_F(RoadRunnerAPITests, DISABLED_removeInitialAssignment){
    RoadRunner rr(SimpleFlux().str());
    rr.addInitialAssignment("S1", "0.5*S2", true);
}

/**
 * Ideally we should have another test suite to deal with events
 *
 */
TEST_F(RoadRunnerAPITests, addEvent){
    RoadRunner rr(SimpleFlux().str());
    rr.addEvent("E1", true, "time > 4.0", true);
    std::list<std::string> out;
    std::list<std::string> expected({"E1"});
    rr.getModel()->getEventIds(out);
    ASSERT_EQ(expected, out);
}

/**
 * How to verify that the trigger has changed?
 */
TEST_F(RoadRunnerAPITests, DISABLED_addTrigger){
    RoadRunner rr(SimpleFlux().str());
    rr.addEvent("E1", true, "time > 4.0", true);
    rr.addTrigger("E1", "time > 5.0", true);
}

/**
 * How to verify that setPersistant is true with current API?
 */
TEST_F(RoadRunnerAPITests, DISABLED_setPersistent){
    RoadRunner rr(SimpleFlux().str());
    rr.addEvent("E1", true, "time > 4.0", true);
    rr.setPersistent("E1", true, true);
}

/**
 * How to test using current API?
 */
TEST_F(RoadRunnerAPITests, DISABLED_setTriggerInitialValue){
    RoadRunner rr(SimpleFlux().str());
    rr.addEvent("E1", true, "time > 4.0", true);
    rr.setTriggerInitialValue("E1", true, true);
}

/**
 * Not sure what this is. Docs?
 */
TEST_F(RoadRunnerAPITests, DISABLED_addPriority){
    RoadRunner rr(SimpleFlux().str());
    rr.addEvent("E1", true, "time > 4.0", true);
//    rr.addPriority("E1")
}

/*
 * What is the math formula of event delay?
 */
TEST_F(RoadRunnerAPITests, DISABLED_addDelay){
    RoadRunner rr(SimpleFlux().str());
    rr.addEvent("E1", true, "time > 4.0", true);
//    rr.addDelay("E1", )
}

/**
 * What is the math formula of the event assignment?
 */
TEST_F(RoadRunnerAPITests, DISABLED_addEventAssignment){
    RoadRunner rr(SimpleFlux().str());
    rr.addEvent("E1", true, "time > 4.0", true);
//    rr.addEventAssignment()

}

/**
 * What is the math formula of event assignment?
 */
TEST_F(RoadRunnerAPITests, DISABLED_removeEventAssignments){
    RoadRunner rr(SimpleFlux().str());
    rr.addEvent("E1", true, "time > 4.0", true);
}

TEST_F(RoadRunnerAPITests, removeEvent){
    RoadRunner rr(SimpleFlux().str());
    rr.addEvent("E1", true, "time > 4.0", true);
    rr.removeEvent("E1", true);
    std::list<std::string> out ;
    rr.getModel()->getEventIds(out );
    ASSERT_EQ(0, out.size());
}

TEST_F(RoadRunnerAPITests, validateCurrentSBML){
    RoadRunner rr(SimpleFlux().str());
    ASSERT_NO_THROW(rr.validateCurrentSBML(););
}

/**
 * I'm not completley clear on why this method does
 * what it does (i..e presimulate to time 10 then
 * do steady state), but this feature is tested comprehensively
 * elsewhere.
 */
TEST_F(RoadRunnerAPITests, DISABLED_mcaSteadyState){
    RoadRunner rr(SimpleFlux().str());
    rr.mcaSteadyState();
}

/**
 * This is fully tested elsewhere
 * anyway
 */
TEST_F(RoadRunnerAPITests, steadyState){
    SimpleFlux simpleFlux;
    RoadRunner rr(simpleFlux.str());
    simpleFlux.checkSteadyState(&rr);
}

/**
 * todo consolidate the steadyState computation
 * so that we only have one version.
 */
TEST_F(RoadRunnerAPITests, steadyStateNamedArray){
    SimpleFlux simpleFlux;
    RoadRunner rr(simpleFlux.str());
    auto x = rr.steadyStateNamedArray();
    ASSERT_EQ(2, x.size()); // steady state tested elsewhere.
}

TEST_F(RoadRunnerAPITests, getSteadyStateSelections){
    SimpleFlux simpleFlux;
    RoadRunner rr(simpleFlux.str());
    auto selections = rr.getSteadyStateSelections();
    ASSERT_EQ(2, selections.size()); // S1 and S2
}

TEST_F(RoadRunnerAPITests, setSteadyStateSelections){
    SimpleFlux simpleFlux;
    RoadRunner rr(simpleFlux.str());
    rr.setSteadyStateSelections(std::vector<std::string>({"time", "S1"}));
    auto selections = rr.getSteadyStateSelections();
    ASSERT_EQ(2, selections.size());
}

/**
 * todo How is this any different to steadyStateNamedArray?
 * We should remove steadyStateNamedArray and getSteadyStateValues
 * and then make steadyState return the values we want.
 */
TEST_F(RoadRunnerAPITests, getSteadyStateValues){
    SimpleFlux simpleFlux;
    RoadRunner rr(simpleFlux.str());
    auto x = rr.getSteadyStateValues();
    std::vector<double> expected({1, 10});
    ASSERT_EQ(expected, x);
}

TEST_F(RoadRunnerAPITests, getSteadyStateSelectionStrings){
    SimpleFlux simpleFlux;
    RoadRunner rr(simpleFlux.str());
    std::vector<std::string> expected({"[S1]", "[S2]"});
    ASSERT_EQ(expected, rr.getSteadyStateSelectionStrings());
}

/**
 * Another way to get steadystate values?
 *  Consolidate the API!
 */
TEST_F(RoadRunnerAPITests, getSteadyStateValuesNamedArray){
    SimpleFlux simpleFlux;
    RoadRunner rr(simpleFlux.str());
    auto x = rr.getSteadyStateValuesNamedArray();
    ls::DoubleMatrix expected({{1, 10}});
    checkMatrixEqual(expected, x);
}

/**
 * Unclear how to test here.
 * But this should really be in the unit test for llvm model
 * (if they existed)
 */
TEST_F(RoadRunnerAPITests, DISABLED_regenerateModel){
    SimpleFlux simpleFlux;
    RoadRunner rr(simpleFlux.str());
    rr.regenerateModel();
}

/**
 * The remaining methods are deprecated, sometimes
 * with no obvious replacement however.
 * We will not test deprecated methods given that
 * they will be removed.
 */

//TEST_F(RoadRunnerAPITests, setBoundarySpeciesByIndex){
//
//}

//TEST_F(RoadRunnerAPITests, getNumberOfIndependentSpecies){
//
//}

//TEST_F(RoadRunnerAPITests, getGlobalParameterIds){
//
//}

//TEST_F(RoadRunnerAPITests, getBoundarySpeciesIds){
//
//}

//TEST_F(RoadRunnerAPITests, getBoundarySpeciesConcentrationIds){
//
//}

//TEST_F(RoadRunnerAPITests, getBoundarySpeciesByIndex){
//
//}

//TEST_F(RoadRunnerAPITests, getGlobalParameterByIndex){
//
//}

//TEST_F(RoadRunnerAPITests, setGlobalParameterByName){
//
//}

//TEST_F(RoadRunnerAPITests, getGlobalParameterByName){
//
//}

//TEST_F(RoadRunnerAPITests, getConservedMoietyValues){
//
//}

//TEST_F(RoadRunnerAPITests, getConservedMoietyIds){
//
//}

//TEST_F(RoadRunnerAPITests, getNumberOfReactions){
//
//}

//TEST_F(RoadRunnerAPITests, getReactionRate){
//
//}

//TEST_F(RoadRunnerAPITests, getRateOfChange){
//
//}

//TEST_F(RoadRunnerAPITests, getRateOfChangeIds){
//
//}

//TEST_F(RoadRunnerAPITests, getNumberOfCompartments){
//
//}

//TEST_F(RoadRunnerAPITests, setCompartmentByIndex){
//
//}

//TEST_F(RoadRunnerAPITests, getCompartmentByIndex){
//
//}

//TEST_F(RoadRunnerAPITests, getCompartmentIds){
//
//}

//TEST_F(RoadRunnerAPITests, getNumberOfBoundarySpecies){
//
//}

//TEST_F(RoadRunnerAPITests, getBoundarySpeciesConcentrations){
//
//}

//TEST_F(RoadRunnerAPITests, setBoundarySpeciesConcentrations){
//
//}

//TEST_F(RoadRunnerAPITests, getNumberOfFloatingSpecies){
//
//}

//TEST_F(RoadRunnerAPITests, getFloatingSpeciesByIndex){
//
//}

//TEST_F(RoadRunnerAPITests, setFloatingSpeciesByIndex){
//
//}

//TEST_F(RoadRunnerAPITests, getFloatingSpeciesConcentrationsV){
//
//}

//TEST_F(RoadRunnerAPITests, getFloatingSpeciesAmountsV){
//
//}

//TEST_F(RoadRunnerAPITests, getBoundarySpeciesConcentrationsV){
//
//}

//TEST_F(RoadRunnerAPITests, getBoundarySpeciesAmountsV){
//
//}

//TEST_F(RoadRunnerAPITests, getFloatingSpeciesInitialConcentrations){
//
//}

//TEST_F(RoadRunnerAPITests, setFloatingSpeciesConcentrations){
//
//}

//TEST_F(RoadRunnerAPITests, setFloatingSpeciesInitialConcentrationByIndex){
//
//}

//TEST_F(RoadRunnerAPITests, setFloatingSpeciesInitialConcentrations){
//
//}

//TEST_F(RoadRunnerAPITests, getFloatingSpeciesIds){
//
//}

//TEST_F(RoadRunnerAPITests, getFloatingSpeciesInitialConditionIds){
//
//}

//TEST_F(RoadRunnerAPITests, getNumberOfGlobalParameters){
//
//}

//TEST_F(RoadRunnerAPITests, setGlobalParameterByIndex){
//
//}

//TEST_F(RoadRunnerAPITests, getGlobalParameterValues){
//
//}

//TEST_F(RoadRunnerAPITests, evalModel){
//
//}

//TEST_F(RoadRunnerAPITests, getNumberOfDependentSpecies){
//
//}

//TEST_F(RoadRunnerAPITests, getReactionRates){
//
//}

//TEST_F(RoadRunnerAPITests, getReactionIds){
//
//}





























