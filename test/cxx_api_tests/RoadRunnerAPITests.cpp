//
// Created by Ciaran on 07/05/2021.
//

#include "gtest/gtest.h"

// todo move test model factor up one directory
#include "RoadRunnerTest.h"
#include "TestModelFactory.h"

#include "rrRoadRunner.h"
#include "rrConfig.h"

using namespace rr;

/**
 * This is a more of a stub test suite
 * than a full test suite at the moment.
 * It will eventually fully test every
 * aspect of the rr::RoadRunner API
 * For now, it only tests the things
 * that need a new test to fix a bug.
 */

class RoadRunnerAPITests : public RoadRunnerTest {

public:
    OpenLinearFlux openLinearFlux;

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
    RoadRunner rr(openLinearFlux.str());
    auto matrix = rr.getFullJacobian();
    // no modification of roadrunner Config
    checkMatrixEqual(matrix, openLinearFlux.fullJacobianConc());
}

TEST_F(RoadRunnerAPITests, GetFullJacobianUsingConcMode) {
    Config::setValue(Config::ROADRUNNER_JACOBIAN_MODE, Config::ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS);
    RoadRunner rr(openLinearFlux.str());
    auto matrix = rr.getFullJacobian();
    checkMatrixEqual(matrix, openLinearFlux.fullJacobianConc());
}

TEST_F(RoadRunnerAPITests, GetFullJacobianUsingAmtMode) {
    Config::setValue(Config::ROADRUNNER_JACOBIAN_MODE, Config::ROADRUNNER_JACOBIAN_MODE_AMOUNTS);
    RoadRunner rr(openLinearFlux.str());
    auto matrix = rr.getFullJacobian();
    checkMatrixEqual(matrix, openLinearFlux.fullJacobianAmt());
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
    RoadRunner rr(openLinearFlux.str());
    auto matrix = rr.getFullJacobian();
    checkMatrixEqual(matrix, openLinearFlux.fullJacobianAmt());
}


TEST_F(RoadRunnerAPITests, getIds) {
    std::list<std::string> expected(
            {
                    "S1", "S2", "[S1]", "[S2]", "default_compartment",
                    "kin", "kf", "kout", "kb", "_J0", "_J1", "_J2",
                    "init([S1])", "init([S2])", "init(S1)", "init(S2)",
                    "S1'", "S2'", "eigen(S1)", "eigenReal(S1)",
                    "eigenImag(S1)", "eigen(S2)", "eigenReal(S2)", "eigenImag(S2)",
            });
    RoadRunner rr(openLinearFlux.str());
    std::list<std::string> l;
    rr.getIds(SelectionRecord::ALL, l);
    ASSERT_EQ(expected, l);
}

TEST_F(RoadRunnerAPITests, LoadModelWithOnlyLayoutInformation) {
    LayoutOnly layoutOnly;
    ASSERT_NO_THROW(RoadRunner rr(layoutOnly.str()););
}

TEST_F(RoadRunnerAPITests, loadSBMLWithLeadingNewlineCharacter) {
    std::string sbml = OpenLinearFlux().str();
    sbml = "\n" + sbml;
    RoadRunner rr(sbml);
}

/**
 * It appears that support for various SBML versions is not a functional
 * feature at the moment.
 */
TEST_F(RoadRunnerAPITests, DISABLED_RoadRunnerConstructorVersion) {
    RoadRunner rr(3,2);
    rr.load(TestModelFactory("OpenLinearFlux")->str());
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
    std::string s =  RoadRunner::getParamPromotedSBML(TestModelFactory("ModelWithLocalParameters")->str());
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
                           "</sbml>";
    ASSERT_STREQ(expected.c_str(), s.c_str());
}

TEST_F(RoadRunnerAPITests, getIntegrator) {
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    Integrator* integrator = rr.getIntegrator();
    auto before = rr.getFloatingSpeciesAmountsNamedArray();
    integrator->integrate(0, 1);
    auto after = rr.getFloatingSpeciesAmountsNamedArray();
    // no equality operators on ls::Matrix?

//    ASSERT_TRUE(before != after);

}




























/**
 * getCompiler is only used in the C backend, which I'm
 * no sure even compiles anymore.
 */
TEST_F(RoadRunnerAPITests, DISABLED_getCompiler){
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
}

TEST_F(RoadRunnerAPITests, IsModelLoadedWhenFalse){
    RoadRunner rr;
    ASSERT_FALSE(rr.isModelLoaded());
}

TEST_F(RoadRunnerAPITests, IsModelLoadedWhenTrue){
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    ASSERT_TRUE(rr.isModelLoaded());
}


TEST_F(RoadRunnerAPITests, getConservedMoietyAnalysisWhenFalse){
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    ASSERT_FALSE(rr.getConservedMoietyAnalysis());
}

TEST_F(RoadRunnerAPITests, getConservedMoietyAnalysisWhenTrue){
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    rr.setConservedMoietyAnalysis(true);
    ASSERT_TRUE(rr.getConservedMoietyAnalysis());
}

TEST_F(RoadRunnerAPITests, tmpDir){
    RoadRunner rr(TestModelFactory("SimpleFlux")->str());
    rr.setTempDir(fs::current_path().string());
    ASSERT_STREQ(rr.getTempDir().c_str(), fs::current_path().string().c_str());
}

































