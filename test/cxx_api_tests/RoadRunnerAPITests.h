//
// Created by Ciaran on 04/11/2021.
//

#ifndef ROADRUNNER_ROADRUNNERAPITESTS_H
#define ROADRUNNER_ROADRUNNERAPITESTS_H


#include "gtest/gtest.h"

#include "RoadRunnerTest.h"
#include "TestModelFactory.h"
#include "SteadyStateSolver.h"
#include "Integrator.h"
#include "SensitivitySolver.h"
#include "rrExecutableModel.h"
#include "rrRoadRunner.h"
#include "rrConfig.h"

using namespace rr;


class RoadRunnerAPITests : public RoadRunnerTest {

public:
    std::string sbml = SimpleFlux().str();

    RoadRunnerAPITests() {
        Logger::setFormattingPattern("%U:%u:%l: %t");
        //Logger::setLevel(Logger::LOG_DEBUG);
    };

    void DefaultJacobianMode() {
        Setting x = Config::getValue(
                Config::ROADRUNNER_JACOBIAN_MODE
        );
        ASSERT_TRUE(x.get<int>() == Config::ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS);
    }

    void SetJacobianModeToAmt() {
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

    void GetFullJacobianDefaultConfigSettings() {
        RoadRunner rr(sbml);
        auto matrix = rr.getFullJacobian();
        // no modification of roadrunner Config
        checkMatrixEqual(matrix, SimpleFlux().fullJacobianConc());
    }

    void GetFullJacobianUsingConcMode() {
        Config::setValue(Config::ROADRUNNER_JACOBIAN_MODE, Config::ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS);
        RoadRunner rr(sbml);
        auto matrix = rr.getFullJacobian();
        checkMatrixEqual(matrix, SimpleFlux().fullJacobianConc());
    }

    void GetFullJacobianUsingAmtMode() {
        Config::setValue(Config::ROADRUNNER_JACOBIAN_MODE, Config::ROADRUNNER_JACOBIAN_MODE_AMOUNTS);
        RoadRunner rr(sbml);
        auto matrix = rr.getFullJacobian();
        checkMatrixEqual(matrix, SimpleFlux().fullJacobianAmt());
    }

    void GetFullJacobianUsingAmtModeAsLong() {
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


    void getIds() {
        std::list<std::string> expected(
                {"S1", "S2", "[S1]", "[S2]", "default_compartment", "kf", "kb", "_J0", "_J1", "init([S1])",
                 "init([S2])", "init(S1)", "init(S2)", "init(default_compartment)", "init(kf)", "init(kb)", "S1'",
                 "S2'", "[S1]'", "[S2]'", "stoich(S1, _J0)", "stoich(S1, _J1)", "stoich(S2, _J0)", "stoich(S2, _J1)",
                 "uec(_J0, S1)", "uec(_J0, S2)", "uec(_J0, kf)", "uec(_J0, kb)", "uec(_J1, S1)", "uec(_J1, S2)",
                 "uec(_J1, kf)", "uec(_J1, kb)", "ec(_J0, S1)", "ec(_J0, S2)", "ec(_J0, kf)", "ec(_J0, kb)",
                 "ec(_J1, S1)", "ec(_J1, S2)", "ec(_J1, kf)", "ec(_J1, kb)", "ucc(_J0, kf)", "ucc(_J0, kb)",
                 "ucc(_J1, kf)", "ucc(_J1, kb)", "ucc(S1, kf)", "ucc(S1, kb)", "ucc(S2, kf)", "ucc(S2, kb)",
                 "cc(_J0, kf)", "cc(_J0, kb)", "cc(_J1, kf)", "cc(_J1, kb)", "cc(S1, kf)", "cc(S1, kb)", "cc(S2, kf)",
                 "cc(S2, kb)", "eigen(S1)", "eigenReal(S1)", "eigenImag(S1)", "eigen(S2)", "eigenReal(S2)",
                 "eigenImag(S2)",
                }
        );
        RoadRunner rr(sbml);
        std::list<std::string> l;
        rr.getIds(SelectionRecord::ALL, l);
        l.sort();
        expected.sort();
        ASSERT_EQ(expected, l);
    }

    void getIdsAfterRegenerate() {
        std::list<std::string> expected(
                {"S1", "S2", "[S1]", "[S2]", "default_compartment", "kf", "kb", "_J0", "_J1", "init([S1])",
                 "init([S2])", "init(S1)", "init(S2)", "init(default_compartment)", "init(kf)", "init(kb)", "S1'",
                 "S2'", "[S1]'", "[S2]'", "stoich(S1, _J0)", "stoich(S1, _J1)", "stoich(S2, _J0)", "stoich(S2, _J1)",
                 "uec(_J0, S1)", "uec(_J0, S2)", "uec(_J0, kf)", "uec(_J0, kb)", "uec(_J1, S1)", "uec(_J1, S2)",
                 "uec(_J1, kf)", "uec(_J1, kb)", "ec(_J0, S1)", "ec(_J0, S2)", "ec(_J0, kf)", "ec(_J0, kb)",
                 "ec(_J1, S1)", "ec(_J1, S2)", "ec(_J1, kf)", "ec(_J1, kb)", "ucc(_J0, kf)", "ucc(_J0, kb)",
                 "ucc(_J1, kf)", "ucc(_J1, kb)", "ucc(S1, kf)", "ucc(S1, kb)", "ucc(S2, kf)", "ucc(S2, kb)",
                 "cc(_J0, kf)", "cc(_J0, kb)", "cc(_J1, kf)", "cc(_J1, kb)", "cc(S1, kf)", "cc(S1, kb)", "cc(S2, kf)",
                 "cc(S2, kb)", "eigen(S1)", "eigenReal(S1)", "eigenImag(S1)", "eigen(S2)", "eigenReal(S2)",
                 "eigenImag(S2)",
                }
        );
        expected.sort();
        RoadRunner rr(sbml);
        rr.regenerateModel();

        std::list<std::string> l;
        rr.getIds(SelectionRecord::ALL, l);
        l.sort();
        ASSERT_EQ(expected, l);
    }

    void getIdsAfterSetConservedMoiety() {
        std::list<std::string> expected(
                {
                        "S1", "S1'", "S2", "[S1]", "[S1]'", "[S2]", "_CSUM0", "_J0", "_J1", "cc(S1, _CSUM0)",
                        "cc(S1, kb)", "cc(S1, kf)", "cc(S2, _CSUM0)", "cc(S2, kb)", "cc(S2, kf)", "cc(_J0, _CSUM0)",
                        "cc(_J0, kb)", "cc(_J0, kf)", "cc(_J1, _CSUM0)", "cc(_J1, kb)", "cc(_J1, kf)",
                        "default_compartment", "ec(_J0, S1)", "ec(_J0, S2)", "ec(_J0, kb)", "ec(_J0, kf)",
                        "ec(_J1, S1)", "ec(_J1, S2)", "ec(_J1, kb)", "ec(_J1, kf)", "eigen(S1)", "eigen(S2)",
                        "eigenImag(S1)", "eigenImag(S2)", "eigenReal(S1)", "eigenReal(S2)", "init(S1)", "init(S2)",
                        "init([S1])", "init([S2])", "init(_CSUM0)", "init(default_compartment)", "init(kb)", "init(kf)",
                        "kb", "kf", "stoich(S1, _J0)", "stoich(S1, _J1)", "ucc(S1, _CSUM0)", "ucc(S1, kb)",
                        "ucc(S1, kf)", "ucc(S2, _CSUM0)", "ucc(S2, kb)", "ucc(S2, kf)", "ucc(_J0, _CSUM0)",
                        "ucc(_J0, kb)", "ucc(_J0, kf)", "ucc(_J1, _CSUM0)", "ucc(_J1, kb)", "ucc(_J1, kf)",
                        "uec(_J0, S1)", "uec(_J0, S2)", "uec(_J0, kb)", "uec(_J0, kf)", "uec(_J1, S1)", "uec(_J1, S2)",
                        "uec(_J1, kb)", "uec(_J1, kf)",}
        );
        RoadRunner rr(sbml);
        rr.setConservedMoietyAnalysis(true);
        std::list<std::string> l;
        rr.getIds(SelectionRecord::ALL, l);
        expected.sort();
        l.sort();
        auto it = l.begin();
        //while (it != l.end()) {
        //    std::cout << *it << std::endl;
        //    it++;
        //}

        ASSERT_EQ(expected, l);
    }

    void LoadModelWithOnlyLayoutInformation() {
        LayoutOnly layoutOnly;
        ASSERT_NO_THROW(RoadRunner rr(layoutOnly.str()););
    }

    void loadSBMLWithLeadingNewlineCharacter() {
        sbml = "\n" + sbml;
        ASSERT_NO_THROW(RoadRunner rr(sbml););
    }

    /**
     * It appears that support for various SBML versions is not a functional
     * feature at the moment.
     */
    void RoadRunnerConstructorVersion() {
        RoadRunner rr(1, 2);
        rr.addCompartment("C1", 1.0);
        std::cout << rr.getSBML() << std::endl;
        rr = RoadRunner(2, 1);
        rr.addCompartment("C1", 1.0);
        std::cout << rr.getSBML() << std::endl;
        rr = RoadRunner(2, 2);
        rr.addCompartment("C1", 1.0);
        std::cout << rr.getSBML() << std::endl;
        rr = RoadRunner(2, 3);
        rr.addCompartment("C1", 1.0);
        std::cout << rr.getSBML() << std::endl;
        rr = RoadRunner(2, 4);
        rr.addCompartment("C1", 1.0);
        std::cout << rr.getSBML() << std::endl;
        rr = RoadRunner(2, 5);
        rr.addCompartment("C1", 1.0);
        std::cout << rr.getSBML() << std::endl;
        rr = RoadRunner(3, 1);
        rr.addCompartment("C1", 1.0);
        std::cout << rr.getSBML() << std::endl;
        rr = RoadRunner(3, 2);
        rr.addCompartment("C1", 1.0);
        std::cout << rr.getSBML() << std::endl;
    }

    /**
     * Support non functional for this test. The responsibility for downloading
     * the sbml from the link is passed on to libsbml, but sbml raises the following:
     *  C++ exception with description "Could not open stream: HTTPS transport not supported" thrown in the test body.
     */
    void DISABLED_RoadRunnerConstructorFromURL() {
        std::string url = "https://raw.githubusercontent.com/sys-bio/roadrunner/develop/test/models/CAPICore/Test_1.xml";
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
    void DISABLED_CompilerTempDirSupportCodeDir() {

    }

    void CopyConstructor() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr1(testModel->str());
        RoadRunner rr2 = rr1;
        // todo, use equality operators once they are built
        ASSERT_STREQ(rr1.getSBML().c_str(), rr2.getSBML().c_str());
        delete testModel;
    }

    /**
     * Equality operators are not actually supported at the moment, this
     * is something that we should add
     */
    void DISABLED_EqualityOperators() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr1(testModel->str());
        RoadRunner rr2(testModel->str());
        delete testModel;
        //    ASSERT_TRUE(rr1 == rr2);
    }

    /**
     * Inequality operators are not actually supported at the moment, this
     * is something that we should add
     */
    void DISABLED_InequalityOperators() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        TestModel *testModel2 = TestModelFactory("OpenLinearFlux");
        RoadRunner rr1(testModel->str());
        RoadRunner rr2(testModel2->str());
        delete testModel;
        delete testModel2;
        //    ASSERT_TRUE(rr1 != rr2);
    }

    void getInstanceID() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr1(testModel->str());
        RoadRunner rr2(testModel->str());
        ASSERT_EQ(1, rr1.getInstanceID());
        ASSERT_EQ(2, rr2.getInstanceID());
        delete testModel;
    }

    void getInstanceCount() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr1(testModel->str());
        RoadRunner rr2(testModel->str());
        ASSERT_EQ(2, rr2.getInstanceCount());
        delete testModel;
    }

    void getParamPromotedSBML() {
        TestModel *testModel = TestModelFactory("ModelWithLocalParameters");
        std::string s = RoadRunner::getParamPromotedSBML(testModel->str());
        delete testModel;
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
    void DISABLED_getCompiler() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        delete testModel;
    }

    void IsModelLoadedWhenFalse() {
        RoadRunner rr;
        ASSERT_FALSE(rr.isModelLoaded());
    }

    void IsModelLoadedWhenTrue() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        ASSERT_TRUE(rr.isModelLoaded());
        delete testModel;
    }


    void getConservedMoietyAnalysisWhenFalse() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        ASSERT_FALSE(rr.getConservedMoietyAnalysis());
        delete testModel;
    }

    void getConservedMoietyAnalysisWhenTrue() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        rr.setConservedMoietyAnalysis(true);
        ASSERT_TRUE(rr.getConservedMoietyAnalysis());
        delete testModel;
    }

    void tmpDir() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        rr.setTempDir(fs::current_path().string());
        ASSERT_STREQ(rr.getTempDir().c_str(), fs::current_path().string().c_str());
        delete testModel;
    }

    /**
     * getModelName doesn't seem to work. But it should be tested
     * in LLVMExecutableModel
     */
    void getModelName() {
        RoadRunner rr(SimpleFlux().str());
        ASSERT_STREQ(rr.getModelName().c_str(), "SimpleFlux");
    }

    void clearModel() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        rr.clearModel();
        ASSERT_FALSE(rr.isModelLoaded());
        delete testModel;
    }

    void oneStep() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        rr.oneStep(0, 10, true);
        ASSERT_EQ(rr.getModel()->getTime(), 10);
        delete testModel;
    }

    /**
     * Difference between oneStep and internalOneStep
     */
    void internalOneStep() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        rr.internalOneStep(0, 10, true);
        ASSERT_LT(rr.getModel()->getTime(), 0.003);
        delete testModel;
    }

    /**
     * State saving needs to be tested properly somehwere else.
     * The save / load features should be a class in their own
     * right and that should be fully tested.
     * This is just a superficial test
     */
    void saveState() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        fs::path fname = fs::current_path() / "testSaveState.rr";
        rr.saveState(fname.string());
        ASSERT_TRUE(fs::exists(fname));
        fs::remove(fname); // clear up written files
        delete testModel;
    }

    void loadState() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        fs::path fname = fs::current_path() / "testSaveState.rr";
        rr.saveState(fname.string());
        ASSERT_TRUE(fs::exists(fname));
        RoadRunner loaded;
        loaded.loadState(fname.string());
        ASSERT_EQ(loaded.getNumberOfFloatingSpecies(), 2);
        //std::cout << *loaded.simulate(0, 10, 11) << std::endl;
        fs::remove(fname); // clear up written files
        delete testModel;
    }

    void loadStateFromCopy() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        RoadRunner rr2(rr);
        fs::path fname = fs::current_path() / "testSaveState.rr";
        rr2.saveState(fname.string());
        ASSERT_TRUE(fs::exists(fname));
        RoadRunner loaded;
        loaded.loadState(fname.string());
        ASSERT_EQ(loaded.getNumberOfFloatingSpecies(), 2);
        //std::cout << *loaded.simulate(0, 10, 11) << std::endl;
        fs::remove(fname); // clear up written files
        delete testModel;
    }

    void getSimulationData() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        rr.simulate(0, 10, 11);
        auto x = rr.getSimulationData();
        ASSERT_EQ(11 * 3, x->size()); // 11 time steps, three variables (time, s1, s2)
        delete testModel;
    }

    void setSimulateOptions() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        SimulateOptions options;
        options.steps = 101;
        options.duration = 100;
        options.start = 0;
        rr.setSimulateOptions(options);
        auto x = rr.getSimulateOptions();
        ASSERT_EQ(x.steps, 101);
        ASSERT_EQ(x.duration, 100);
        ASSERT_EQ(x.start, 0);
        delete testModel;
    }

    /**
     * The accuracy of the sbml should be tested somewhere else.
     * This is a superficial API test
     */
    void getSBML() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        ASSERT_EQ(typeid(std::string), typeid(rr.getSBML()));
        delete testModel;
    }

    /**
     * Similar to getSBML, supficial test and the real tests
     * should be within the sbml generating code.
     */
    void getCurrentSBML() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        ASSERT_EQ(typeid(std::string), typeid(rr.getCurrentSBML()));
        delete testModel;
    }

    void reset() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        rr.simulate(0, 100, 101);
        ASSERT_EQ(rr.getModel()->getTime(), 100);
        rr.reset();
        ASSERT_EQ(rr.getModel()->getTime(), 0);
        delete testModel;
    }

    void getCurrentTime() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        ASSERT_EQ(rr.getCurrentTime(), 0);
        rr.simulate(0, 100, 101);
        ASSERT_EQ(rr.getCurrentTime(), 100);
        delete testModel;
    }

    void resetSelectionLists() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        SelectionRecord record = rr.createSelection("S1");
        ASSERT_STREQ(record.to_string().c_str(), "S1");
        rr.resetSelectionLists();
        ASSERT_STREQ(record.to_string().c_str(), "S1");
        delete testModel;
    }

    void getSelections() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        std::vector<std::string> expected({"time", "[S1]", "[S2]"});
        RoadRunner rr(testModel->str());
        auto x = rr.getSelections();
        std::vector<std::string> actual;
        actual.reserve(x.size());
        for (auto i: x) {
            actual.push_back(i.to_string());
        }
        ASSERT_EQ(expected, actual);
        delete testModel;
    }

    void getValue() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        auto x = rr.getValue("[S1]");
        ASSERT_EQ(x, 10);
        delete testModel;
    }

    void setSelections() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        rr.setSelections(std::vector<std::string>({"S1", "[S1]"}));
        delete testModel;
    }

    void getSelectedValues() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        std::vector<double> x = rr.getSelectedValues();
        std::vector<double> expected({0.0, 10.0, 1.0});
        ASSERT_EQ(x, expected);
        delete testModel;
    }

    void getIndependentFloatingSpeciesIds() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        auto x = rr.getIndependentFloatingSpeciesIds();
        std::vector<std::string> expected({"S1"});
        ASSERT_EQ(x, expected);

        rr.setConservedMoietyAnalysis(true);
        x = rr.getIndependentFloatingSpeciesIds();
        ASSERT_EQ(x, expected);
        delete testModel;
    }

    void getDependentFloatingSpeciesIds() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        auto x = rr.getDependentFloatingSpeciesIds();
        std::vector<std::string> expected = { "S2" };
        ASSERT_EQ(x, expected);

        rr.setConservedMoietyAnalysis(true);
        x = rr.getDependentFloatingSpeciesIds();
        ASSERT_EQ(x, expected);
        delete testModel;
    }

    void getFloatingSpeciesInitialConcentrations() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        auto x = rr.getFloatingSpeciesInitialConcentrations();
        std::vector<double> expected({10.0, 1.0});
        ASSERT_EQ(x, expected);
        delete testModel;
    }

    /**
     * I don't understand what this is or how to test it.
     */
    void DISABLED_getSupportedIdTypes() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        auto x = rr.getSupportedIdTypes();
        delete testModel;
    }

    void setValue() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        rr.setValue("[S1]", 12.34);
        ASSERT_EQ(rr.getValue("[S1]"), 12.34);
        delete testModel;
    }

    void getFloatingSpeciesAmountsNamedArray() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        auto x = rr.getFloatingSpeciesAmountsNamedArray();
        //std::cout << x << std::endl;
        ls::DoubleMatrix expected({
                                          {10, 1}
                                  });
        checkMatrixEqual(expected, x);
        // why does DoubleMatrix not implement equality ops?
        delete testModel;
    }

    void getFloatingSpeciesConcentrationsNamedArray() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        auto x = rr.getFloatingSpeciesConcentrationsNamedArray();
        //std::cout << x << std::endl;
        ls::DoubleMatrix expected({
                                          {10, 1}
                                  });
        checkMatrixEqual(expected, x);
        delete testModel;
    }

    void getDependentFloatingSpeciesConcentrations() {
        TestModel* testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        rr.setValue("default_compartment", 10.0);
        auto x = rr.getDependentFloatingSpeciesConcentrationsV();
        std::vector<double> expected = { 0.1 };
        EXPECT_EQ(x, expected);

        rr.setConservedMoietyAnalysis(true);
        x = rr.getDependentFloatingSpeciesConcentrationsV();
        EXPECT_EQ(x, expected);
        delete testModel;
    }

    void getDependentFloatingSpeciesConcentrationsNamedArray() {
        TestModel* testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        rr.setValue("default_compartment", 10.0);
        auto x = rr.getDependentFloatingSpeciesConcentrationsNamedArray();
        vector<string> expectedIds = { "S2" };
        EXPECT_EQ(x.getColNames(), expectedIds);

        rr.setConservedMoietyAnalysis(true);
        x = rr.getDependentFloatingSpeciesConcentrationsNamedArray();
        EXPECT_EQ(x.getColNames(), expectedIds);

        delete testModel;
    }

    void getDependentFloatingSpeciesAmounts() {
        TestModel* testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        rr.setValue("default_compartment", 10.0);
        auto x = rr.getDependentFloatingSpeciesAmountsV();
        std::vector<double> expected = { 1 };
        EXPECT_EQ(x, expected);

        rr.setConservedMoietyAnalysis(true);
        x = rr.getDependentFloatingSpeciesAmountsV();
        EXPECT_EQ(x, expected);
        delete testModel;
    }

    void getDependentFloatingSpeciesAmountsNamedArray() {
        TestModel* testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        rr.setValue("default_compartment", 10.0);
        auto x = rr.getDependentFloatingSpeciesAmountsNamedArray();
        vector<string> expectedIds = { "S2" };
        EXPECT_EQ(x.getColNames(), expectedIds);

        rr.setConservedMoietyAnalysis(true);
        x = rr.getDependentFloatingSpeciesAmountsNamedArray();
        EXPECT_EQ(x.getColNames(), expectedIds);

        delete testModel;
    }

    void getIndependentFloatingSpeciesConcentrations() {
        TestModel* testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        rr.setValue("default_compartment", 10.0);
        auto x = rr.getIndependentFloatingSpeciesConcentrationsV();
        std::vector<double> expected = { 1.0 };
        EXPECT_EQ(x, expected);

        rr.setConservedMoietyAnalysis(true);
        x = rr.getIndependentFloatingSpeciesConcentrationsV();
        EXPECT_EQ(x, expected);
        delete testModel;
    }

    void getIndependentFloatingSpeciesConcentrationsNamedArray() {
        TestModel* testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        rr.setValue("default_compartment", 10.0);
        auto x = rr.getIndependentFloatingSpeciesConcentrationsNamedArray();
        vector<string> expectedIds = { "S1" };
        EXPECT_EQ(x.getColNames(), expectedIds);

        rr.setConservedMoietyAnalysis(true);
        x = rr.getIndependentFloatingSpeciesConcentrationsNamedArray();
        EXPECT_EQ(x.getColNames(), expectedIds);

        delete testModel;
    }

    void getIndependentFloatingSpeciesAmounts() {
        TestModel* testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        rr.setValue("default_compartment", 10.0);
        auto x = rr.getIndependentFloatingSpeciesAmountsV();
        std::vector<double> expected = { 10.0};
        EXPECT_EQ(x, expected);

        rr.setConservedMoietyAnalysis(true);
        x = rr.getIndependentFloatingSpeciesAmountsV();
        EXPECT_EQ(x, expected);
        delete testModel;
    }

    void getIndependentFloatingSpeciesAmountsNamedArray() {
        TestModel* testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        rr.setValue("default_compartment", 10.0);
        auto x = rr.getIndependentFloatingSpeciesAmountsNamedArray();
        vector<string> expectedIds = { "S1"};
        EXPECT_EQ(x.getColNames(), expectedIds);

        rr.setConservedMoietyAnalysis(true);
        x = rr.getIndependentFloatingSpeciesAmountsNamedArray();
        EXPECT_EQ(x.getColNames(), expectedIds);

        delete testModel;
    }

    void getBoundarySpeciesAmountsNamedArray() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        rr.regenerateModel();
        rr.setBoundary("S1", true, true);
        auto x = rr.getBoundarySpeciesAmountsNamedArray();
        //std::cout << x << std::endl;
        ls::DoubleMatrix expected({{10}});
        checkMatrixEqual(expected, x);
        delete testModel;
    }

    void getBoundarySpeciesConcentrationsNamedArray() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        rr.setBoundary("S1", true, true);
        auto x = rr.getBoundarySpeciesConcentrationsNamedArray();
        //std::cout << x << std::endl;
        ls::DoubleMatrix expected({{10}});
        checkMatrixEqual(expected, x);
        delete testModel;
    }


    void getRatesOfChange() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        auto x = rr.getRatesOfChange();
        std::vector<double> expected({-0.99, 0.99});
        ASSERT_EQ(x, expected);
        delete testModel;
    }

    void getRatesOfChangeNamedArray() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        auto x = rr.getRatesOfChangeNamedArray();
        ls::DoubleMatrix expected({{-0.99, 0.99}});
        checkMatrixEqual(x, expected);
        delete testModel;
    }

    void getIndependentRatesOfChange() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        auto x = rr.getIndependentRatesOfChange();
        std::vector<double> expected({-0.99});
        ASSERT_EQ(x, expected);
        delete testModel;
    }

    void getIndependentRatesOfChangeNamedArray() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        auto x = rr.getIndependentRatesOfChangeNamedArray();
        ls::DoubleMatrix expected({{-0.99}});
        checkMatrixEqual(x, expected);
        delete testModel;
    }

    void getDependentRatesOfChange() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        auto x = rr.getDependentRatesOfChange();
        std::vector<double> expected({0.99});
        ASSERT_EQ(x, expected);
        delete testModel;
    }


    void getDependentRatesOfChangeNamedArray() {
        TestModel *testModel = TestModelFactory("SimpleFlux");
        RoadRunner rr(testModel->str());
        auto x = rr.getDependentRatesOfChangeNamedArray();
        ls::DoubleMatrix expected({{0.99}});
        checkMatrixEqual(x, expected);
        delete testModel;
    }

    void getFullJacobian() {
        auto *venk = dynamic_cast<Venkatraman2010 *>(TestModelFactory("Venkatraman2010"));
        RoadRunner rr(venk->str());
        auto x = rr.getFullJacobian();
        //std::cout << x << std::endl;
        checkMatrixEqual(venk->fullJacobianConc(), x, 1e-3);
        delete venk;
    }

    void getReducedJacobian() {
        SimpleFlux simpleFlux;
        RoadRunner rr(simpleFlux.str());
        rr.setConservedMoietyAnalysis(true);
        auto x = rr.getReducedJacobian();
        //std::cout << x << std::endl;
        checkMatrixEqual(simpleFlux.reducedJacobianConc(), x, 1e-3);
    }


    void getFullEigenValues() {
        SimpleFlux simpleFlux;
        RoadRunner rr(simpleFlux.str());
        rr.setConservedMoietyAnalysis(true);
        std::vector<std::complex<double>> expected = rr.getFullEigenValues();
        auto eig = rr.getFullEigenValues();
        for (int i = 0; i < expected.size(); i++) {
            ASSERT_EQ(expected[i], eig[i]);
        }
    }

    void getReducedEigenValues() {
        SimpleFlux simpleFlux;
        RoadRunner rr(simpleFlux.str());
        rr.setConservedMoietyAnalysis(true);
        std::vector<std::complex<double>> expected = rr.getReducedEigenValues();
        auto eig = rr.getReducedEigenValues();
        //std::cout << "eig: " << eig.size() << std::endl;
        for (int i = 0; i < expected.size(); i++) {
            ASSERT_EQ(expected[i], eig[i]);
        }
    }


    void getLinkMatrix() {
        SimpleFlux simpleFlux;
        RoadRunner rr(simpleFlux.str());
        //std::cout << rr.getLinkMatrix() << std::endl;
        checkMatrixEqual(simpleFlux.linkMatrix(), rr.getLinkMatrix());
    }

    void getNrMatrix() {
        SimpleFlux simpleFlux;
        RoadRunner rr(simpleFlux.str());
        //std::cout << rr.getNrMatrix() << std::endl;
        checkMatrixEqual(simpleFlux.NrMatrix(), rr.getNrMatrix());
    }

    void getKMatrix() {
        SimpleFlux simpleFlux;
        RoadRunner rr(simpleFlux.str());
        //std::cout << rr.getKMatrix() << std::endl;
        checkMatrixEqual(simpleFlux.KMatrix(), rr.getKMatrix());
    }

    void getReducedStoichiometryMatrix() {
        SimpleFlux simpleFlux;
        RoadRunner rr(simpleFlux.str());
        rr.setConservedMoietyAnalysis(true);
        //std::cout << rr.getReducedStoichiometryMatrix() << std::endl;
        checkMatrixEqual(simpleFlux.reducedStoicMatrix(), rr.getReducedStoichiometryMatrix());
    }

    void getFullStoichiometryMatrix() {
        SimpleFlux simpleFlux;
        RoadRunner rr(simpleFlux.str());
        rr.setConservedMoietyAnalysis(true);
        //std::cout << rr.getFullStoichiometryMatrix() << std::endl;
        checkMatrixEqual(simpleFlux.fullStoicMatrix(), rr.getFullStoichiometryMatrix());
    }

    void getExtendedStoichiometryMatrix() {
        SimpleFlux simpleFlux;
        RoadRunner rr(simpleFlux.str());
        rr.setConservedMoietyAnalysis(true);
        //std::cout << rr.getExtendedStoichiometryMatrix() << std::endl;
        checkMatrixEqual(simpleFlux.extendedStoicMatrix(), rr.getExtendedStoichiometryMatrix());
    }

    void getL0Matrix() {
        SimpleFlux simpleFlux;
        RoadRunner rr(simpleFlux.str());
        rr.setConservedMoietyAnalysis(true);
        //std::cout << rr.getL0Matrix() << std::endl;
        checkMatrixEqual(simpleFlux.L0Matrix(), rr.getL0Matrix());
    }

    void getConservationMatrix() {
        SimpleFlux simpleFlux;
        RoadRunner rr(simpleFlux.str());
        rr.setConservedMoietyAnalysis(true);
        //std::cout << rr.getConservationMatrix() << std::endl;
        checkMatrixEqual(simpleFlux.conservationMatrix(), rr.getConservationMatrix());
    }

    void getUnscaledConcentrationControlCoefficientMatrix() {
        SimpleFlux simpleFlux;
        RoadRunner rr(simpleFlux.str());
        rr.setConservedMoietyAnalysis(true);
        //std::cout << rr.getUnscaledConcentrationControlCoefficientMatrix() << std::endl;
        checkMatrixEqual(simpleFlux.unscaledConcentrationControlCoefficientMatrix(),
                         rr.getUnscaledConcentrationControlCoefficientMatrix());
    }

    void getScaledConcentrationControlCoefficientMatrix() {
        SimpleFlux simpleFlux;
        RoadRunner rr(simpleFlux.str());
        rr.setConservedMoietyAnalysis(true);
        //std::cout << rr.getScaledConcentrationControlCoefficientMatrix() << std::endl;
        checkMatrixEqual(simpleFlux.scaledConcentrationControlCoefficientMatrix(),
                         rr.getScaledConcentrationControlCoefficientMatrix());

    }

    void getUnscaledFluxControlCoefficientMatrix() {
        SimpleFlux simpleFlux;
        RoadRunner rr(simpleFlux.str());
        rr.setConservedMoietyAnalysis(true);
        //std::cout << rr.getUnscaledFluxControlCoefficientMatrix() << std::endl;
        checkMatrixEqual(simpleFlux.unscaledFluxControlCoefficientMatrix(),
                         rr.getUnscaledFluxControlCoefficientMatrix());

    }

    void getScaledFluxControlCoefficientMatrix() {
        SimpleFlux simpleFlux;
        RoadRunner rr(simpleFlux.str());
        rr.setConservedMoietyAnalysis(true);
        //std::cout << rr.getScaledFluxControlCoefficientMatrix() << std::endl;
        checkMatrixEqual(simpleFlux.scaledFluxControlCoefficientMatrix(),
                         rr.getScaledFluxControlCoefficientMatrix());

    }

    void getEigenValueIds() {
        RoadRunner rr(SimpleFlux().str());
        auto actual = rr.getEigenValueIds();
        std::vector<std::string> expected(
                {"eigen(S1)", "eigenReal(S1)", "eigenImag(S1)", "eigen(S2)", "eigenReal(S2)", "eigenImag(S2)"});
        ASSERT_EQ(expected, actual);
    }

    void getUnscaledParameterElasticity() {
        SimpleFlux simpleFlux;
        RoadRunner rr(simpleFlux.str());
        double v = rr.getUnscaledParameterElasticity("_J1", "kb");
        ASSERT_NEAR(v, 1.0, 1e-7);
    }

    /**
     * Unsure how to use
     */
    void DISABLED_getFrequencyResponse() {
        SimpleFlux simpleFlux;
        RoadRunner rr(simpleFlux.str());
    }

    void setConservedMoietyAnalysis() {
        SimpleFlux simpleFlux;
        RoadRunner rr(simpleFlux.str());
        rr.setConservedMoietyAnalysis(true);
        ASSERT_TRUE(rr.getConservedMoietyAnalysis());
    }

    void getDiffStepSize() {
        SimpleFlux simpleFlux;
        RoadRunner rr(simpleFlux.str());
        rr.setDiffStepSize(0.3);
        ASSERT_NEAR(0.3, rr.getDiffStepSize(), 1e-7);
    }

    void getSteadyStateThreshold() {
        SimpleFlux simpleFlux;
        RoadRunner rr(simpleFlux.str());
        rr.setSteadyStateThreshold(0.1234);
        ASSERT_NEAR(0.1234, rr.getSteadyStateThreshold(), 1e-7);
    }

    void getuCC() {
        SimpleFlux simpleFlux;
        RoadRunner rr(simpleFlux.str());
        double actual = rr.getuCC("_J1", "kb");
        double expected = 9.0909090753844755;
        ASSERT_NEAR(expected, actual, 1e-7);
    }

    void getCC() {
        RoadRunner rr(SimpleFlux().str());
        double actual = rr.getCC("_J1", "kb");
        double expected = 9.0909090753844755;
        ASSERT_NEAR(expected, actual, 1e-7);
    }

    void getuEE() {
        RoadRunner rr(SimpleFlux().str());
        double actual = rr.getuEE("_J1", "kb");
        double expected = 1.0;
        ASSERT_NEAR(expected, actual, 1e-7);
    }

    void getEE() {
        RoadRunner rr(SimpleFlux().str());
        double actual = rr.getEE("_J1", "kb");
        double expected = 1.0;
        ASSERT_NEAR(expected, actual, 1e-7);
    }

    void getUnscaledElasticityMatrix() {
        SimpleFlux simpleFlux;
        RoadRunner rr(simpleFlux.str());
        rr.setConservedMoietyAnalysis(true);
        //std::cout << rr.getUnscaledElasticityMatrix() << std::endl;
        checkMatrixEqual(simpleFlux.unscaledElasticityMatrix(),
                         rr.getUnscaledElasticityMatrix());
    }

    void getScaledElasticityMatrix() {
        SimpleFlux simpleFlux;
        RoadRunner rr(simpleFlux.str());
        rr.setConservedMoietyAnalysis(true);
        //std::cout << rr.getScaledElasticityMatrix() << std::endl;
        checkMatrixEqual(simpleFlux.scaledElasticityMatrix(),
                         rr.getScaledElasticityMatrix());
    }

    void getScaledFloatingSpeciesElasticity() {
        SimpleFlux simpleFlux;
        RoadRunner rr(simpleFlux.str());
        double actual = rr.getScaledFloatingSpeciesElasticity("_J1", "S2");
        double expected = 1.0;
        ASSERT_NEAR(expected, actual, 1e-7);
    }

    void getUnscaledSpeciesElasticity() {
        SimpleFlux simpleFlux;
        RoadRunner rr(simpleFlux.str());
        // why is args int in this case and all similar methods a string
        double actual = rr.getUnscaledSpeciesElasticity(0, 0);
        double expected = 0.09999999;
        ASSERT_NEAR(expected, actual, 1e-7);
    }

    void addSpeciesConcentration() {
        RoadRunner rr(SimpleFlux().str());
        rr.setValue("default_compartment", 0.33); // set volume so we see a change vs amounts
        ASSERT_EQ(2, rr.getFloatingSpeciesConcentrationIds().size());
        rr.addSpeciesConcentration("S3", "default_compartment", 123.4, false, false, "mol", true);
        ls::DoubleMatrix expectedConc({{30.303, 3.0303, 373.939}});
        ls::DoubleMatrix expectedAmt({{10, 1, 123.4}});
        checkMatrixEqual(expectedConc, rr.getFloatingSpeciesConcentrationsNamedArray());
        checkMatrixEqual(expectedAmt, rr.getFloatingSpeciesAmountsNamedArray());
    }

    void addSpeciesAmount() {
        RoadRunner rr(SimpleFlux().str());
        rr.setValue("default_compartment", 0.33); // set volume so we see a change vs amounts
        ASSERT_EQ(2, rr.getFloatingSpeciesConcentrationIds().size());
        rr.addSpeciesAmount("S3", "default_compartment", 123.4, true, false, "mol", true);
        ls::DoubleMatrix expectedConc({{30.303, 3.0303, 373.939}});
        ls::DoubleMatrix expectedAmt({{10, 1, 123.4}});
        checkMatrixEqual(expectedConc, rr.getFloatingSpeciesConcentrationsNamedArray());
        checkMatrixEqual(expectedAmt, rr.getFloatingSpeciesAmountsNamedArray());
    }


    void removeSpecies() {
        RoadRunner rr(SimpleFlux().str());
        rr.removeSpecies("S1");
        ASSERT_EQ(1, rr.getFloatingSpeciesAmountsNamedArray().size());

    }

    void setBoundary() {
        RoadRunner rr(SimpleFlux().str());
        ASSERT_EQ(0, rr.getBoundarySpeciesIds().size());
        rr.setBoundary("S1", true, true);
        ASSERT_EQ(1, rr.getBoundarySpeciesIds().size());


    }

    void setHasOnlySubstanceUnits() {
        RoadRunner rr((rrTestModelsDir_ / "CAPICore" / "decay.xml").string());
        rr.setHasOnlySubstanceUnits("S1", true, true);
        ASSERT_NEAR(rr.getValue("J0"), 10.0, 0.0000001);
    }

    void setInitAmount() {
        RoadRunner rr(SimpleFlux().str());
        rr.setInitAmount("S1", 1234.5);
        //std::cout << rr.getFloatingSpeciesAmountsNamedArray() << std::endl;
        ls::DoubleMatrix expected({{1234.5, 1}});
        auto actual = rr.getFloatingSpeciesAmountsNamedArray();
        checkMatrixEqual(expected, actual);
    }

    void setInitConcentration() {
        RoadRunner rr(SimpleFlux().str());
        rr.setInitConcentration("S1", 1234.5);
        //std::cout << rr.getFloatingSpeciesAmountsNamedArray() << std::endl;
        ls::DoubleMatrix expected({{1234.5, 1}});
        auto actual = rr.getFloatingSpeciesConcentrationsNamedArray();
        checkMatrixEqual(expected, actual);
    }

    /**
     * We can set something to constant, but how can
     * we verify that something *is* constant?
     *
     * LS:  The answer is that you can't; it's just a validation thing.  I suppose you could invalidate a model?
     * Another option is to implement support for algebraic rules, which rely on the 'constant' flags, but
     * that seems unlikely to happen any time soon.
     */
    void DISABLED_setConstant() {
        RoadRunner rr(SimpleFlux().str());
        rr.setConstant("S1", true, true);
        auto x = rr.getBoundarySpeciesIds();
        std::vector<std::string> expected({"S1"});
        ASSERT_EQ(expected, x);
    }

    void addReactionFromSBML() {
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

    void addReactionDirect() {
        RoadRunner rr(SimpleFlux().str());
        rr.addReaction("r3", {"S1"}, {"S2"}, "5*4*3*S1", true);
        ASSERT_EQ(3, rr.getFullStoichiometryMatrix().numCols());
    }

    void removeReactionAndUnusedParameters() {
        RoadRunner rr(SimpleFlux().str());
        rr.removeReaction("_J1", true, true);
        EXPECT_EQ(1, rr.getFullStoichiometryMatrix().numCols());
        EXPECT_EQ(1, rr.getGlobalParameterIds().size());
    }

    void removeReactionButNotUnusedParameters() {
        RoadRunner rr(SimpleFlux().str());
        rr.removeReaction("_J1", false, true);
        ASSERT_EQ(1, rr.getFullStoichiometryMatrix().numCols());
        ASSERT_EQ(2, rr.getGlobalParameterIds().size());
    }

    /**
     * We cannot "getReversible" with current API
     * (Also, it doesn't affect the modeling at all.)
     */
    void DISABLED_setReversible() {
        RoadRunner rr(SimpleFlux().str());
        rr.setReversible("_J1", true, true);
    }

    void setKineticLaw() {
        RoadRunner rr(SimpleFlux().str());
        rr.setKineticLaw("_J1", "(kf/(kb+1))*S1", true);
        std::string k = rr.getKineticLaw("_J1");
        ASSERT_STREQ(k.c_str(), "(kf / (kb + 1)) * S1");
    }

    void addParameter() {
        RoadRunner rr(SimpleFlux().str());
        rr.addParameter("NewParameter", 65, true);
        ASSERT_EQ(65, rr.getGlobalParameterByName("NewParameter"));
    }

    void removeParameter() {
        // todo what happens when you remove a parameter that is
        // in use in a reaction? Should throw an error, or at least emit warning
        RoadRunner rr(SimpleFlux().str());
        rr.removeParameter("kf");
        ASSERT_EQ(1, rr.getGlobalParameterIds().size());
    }

    void addCompartment() {
        RoadRunner rr(SimpleFlux().str());
        rr.addCompartment("newComp", 0.33, true);
        ASSERT_EQ(2, rr.getNumberOfCompartments());
    }

    void removeCompartment() {
        RoadRunner rr(SimpleFlux().str());
        rr.addCompartment("newComp", 0.33, true);
        rr.removeCompartment("newComp", true);
        ASSERT_EQ(1, rr.getNumberOfCompartments());

    }

    void addAssignmentRule() {
        RoadRunner rr(SimpleFlux().str());
        rr.addParameter("STotal", 0);
        rr.addAssignmentRule("STotal", "S1+S2", true);
        vector<string> ruleids = rr.getAssignmentRuleIds();
        ASSERT_TRUE(ruleids.size() == 1);
        EXPECT_STREQ(ruleids[0].c_str(), "STotal");
    }

    void addRateRule() {
        RoadRunner rr(SimpleFlux().str());
        rr.setBoundary("S2", true);
        rr.addRateRule("S2", "(kf/kb)*S1", true);
        vector<string> ruleids = rr.getRateRuleIds();
        ASSERT_TRUE(ruleids.size() == 1);
        EXPECT_STREQ(ruleids[0].c_str(), "S2");
    }

    void removeRules() {
        RoadRunner rr(SimpleFlux().str());
        rr.setBoundary("S2", true);
        rr.addRateRule("S2", "(kf/kb)*S1", true);
        vector<string> ruleids = rr.getRateRuleIds();
        ASSERT_TRUE(ruleids.size() == 1);
        EXPECT_STREQ(ruleids[0].c_str(), "S2");

        rr.removeRules("S2", false, true);
        ruleids = rr.getRateRuleIds();
        ASSERT_TRUE(ruleids.size() == 0);
    }

    void addInitialAssignment() {
        RoadRunner rr(SimpleFlux().str());
        rr.addInitialAssignment("S1", "0.5*S2", true);
        vector<string> ruleids = rr.getInitialAssignmentIds();
        ASSERT_TRUE(ruleids.size() == 1);
        EXPECT_STREQ(ruleids[0].c_str(), "S1");
    }

    void removeInitialAssignment() {
        RoadRunner rr(SimpleFlux().str());
        rr.addInitialAssignment("S1", "0.5*S2", true);
        vector<string> ruleids = rr.getInitialAssignmentIds();
        ASSERT_TRUE(ruleids.size() == 1);
        EXPECT_STREQ(ruleids[0].c_str(), "S1");

        rr.removeInitialAssignment("S1", true);
        ruleids = rr.getInitialAssignmentIds();
        ASSERT_TRUE(ruleids.size() == 0);
    }

    /**
     * Ideally we should have another test suite to deal with events
     *
     */
    void addEvent() {
        RoadRunner rr(SimpleFlux().str());
        rr.addEvent("E1", true, "time > 4.0", true);
        std::list<std::string> out;
        std::list<std::string> expected({"E1"});
        rr.getModel()->getEventIds(out);
        ASSERT_EQ(expected, out);
    }

    void addTrigger() {
        RoadRunner rr;
        rr.addParameter("p1", 3.0, false);
        rr.addEvent("E1", true, "time > 2.0", false);
        rr.addEventAssignment("E1", "p1", "5", false);
        rr.addTrigger("E1", "time > 5.0", true);
        rr.simulate(0, 3, 2);
        EXPECT_EQ(rr.getValue("p1"), 3.0);
        rr.simulate(0, 6, 2);
        EXPECT_EQ(rr.getValue("p1"), 5.0);
    }

    void setPersistent() {
        RoadRunner rr;
        rr.addParameter("p1", 3.0, false);
        rr.addParameter("p2", 2.0, false);
        rr.addEvent("E1", true, "time > 2.0", false);
        rr.addEventAssignment("E1", "p1", "6", false);
        rr.addEvent("E2", true, "p1 > 5.0", false);
        rr.addEventAssignment("E2", "p1", "2", false);
        rr.addEvent("E3", true, "p1 > 5.0", false);
        rr.addEventAssignment("E3", "p2", "5", false);
        rr.addDelay("E3", "2.0");
        rr.setPersistent("E3", false, true);
        rr.simulate(0, 5, 2);
        EXPECT_EQ(rr.getValue("p2"), 2.0);
        rr.setPersistent("E3", true, true);
        rr.reset(rr::SelectionRecord::GLOBAL_PARAMETER);
        rr.simulate(0, 5, 2);
        EXPECT_EQ(rr.getValue("p2"), 5.0);
    }

    void setTriggerInitialValue() {
        RoadRunner rr;
        rr.addParameter("p1", 3.0, false);
        rr.addEvent("E1", true, "p1 > 2.0", false);
        rr.addEventAssignment("E1", "p1", "6", false);
        rr.setTriggerInitialValue("E1", true, true);
        EXPECT_EQ(rr.getValue("p1"), 3.0);
        rr.setTriggerInitialValue("E1", false, true);
        rr.reset(rr::SelectionRecord::GLOBAL_PARAMETER);
        //rr.simulate(0, 5, 2);
        EXPECT_EQ(rr.getValue("p1"), 6.0);
    }

    void addPriority() {
        RoadRunner rr;
        rr.addParameter("p1", 3.0, false);
        rr.addEvent("E1", true, "time > 2.0", false);
        rr.addEventAssignment("E1", "p1", "6", false);
        rr.addPriority("E1", "10", false);
        rr.addEvent("E2", true, "time > 2.0", false);
        rr.addEventAssignment("E2", "p1", "12", false);
        rr.addPriority("E2", "15", true);
        rr.simulate(0, 3, 2);
        EXPECT_EQ(rr.getValue("p1"), 6.0);
        rr.addPriority("E2", "5", true);
        rr.reset(rr::SelectionRecord::GLOBAL_PARAMETER);
        rr.simulate(0, 3, 2);
        EXPECT_EQ(rr.getValue("p1"), 12.0);
    }

    void addDelay() {
        RoadRunner rr;
        rr.addParameter("p1", 3.0, false);
        rr.addEvent("E3", true, "time > 1.0", false);
        rr.addEventAssignment("E3", "p1", "5", false);
        rr.addDelay("E3", "2.0");
        rr.simulate(0, 2, 2);
        EXPECT_EQ(rr.getValue("p1"), 3.0);
        rr.simulate(0, 4, 2);
        EXPECT_EQ(rr.getValue("p1"), 5.0);
    }

    void addEventAssignment() {
        RoadRunner rr;
        rr.addParameter("p1", 3.0, false);
        rr.addParameter("p2", 5.0, false);
        rr.addEvent("E3", true, "time > 1.0", false);
        rr.addEventAssignment("E3", "p1", "5", true);
        rr.simulate(0, 2, 2);
        EXPECT_EQ(rr.getValue("p2"), 5.0);
        rr.addEventAssignment("E3", "p2", "15", true);
        rr.reset(rr::SelectionRecord::GLOBAL_PARAMETER);
        rr.simulate(0, 2, 2);
        EXPECT_EQ(rr.getValue("p2"), 15.0);

    }

    void removeEventAssignments() {
        RoadRunner rr;
        rr.addParameter("p1", 3.0, false);
        rr.addParameter("p2", 5.0, false);
        rr.addEvent("E3", true, "time > 1.0", false);
        rr.addEventAssignment("E3", "p1", "5", false);
        rr.addEventAssignment("E3", "p2", "15", true);
        rr.simulate(0, 2, 2);
        EXPECT_EQ(rr.getValue("p2"), 15.0);
        rr.removeEventAssignments("E3", "p2", true);
        rr.reset(rr::SelectionRecord::GLOBAL_PARAMETER);
        rr.simulate(0, 2, 2);
        EXPECT_EQ(rr.getValue("p2"), 5.0);
    }

    void removeEvent() {
        RoadRunner rr(SimpleFlux().str());
        rr.addEvent("E1", true, "time > 4.0", true);
        rr.removeEvent("E1", true);
        std::list<std::string> out;
        rr.getModel()->getEventIds(out);
        ASSERT_EQ(0, out.size());
    }

    void validateCurrentSBML() {
        RoadRunner rr(SimpleFlux().str());
        ASSERT_NO_THROW(rr.validateCurrentSBML(););
    }

    /**
     * I'm not completley clear on why this method does
     * what it does (i..e presimulate to time 10 then
     * do steady state), but this feature is tested comprehensively
     * elsewhere.
     */
    void DISABLED_mcaSteadyState() {
        RoadRunner rr(SimpleFlux().str());
        rr.mcaSteadyState();
    }

    /**
     * This is fully tested elsewhere
     * anyway
     */
    void steadyState() {
        SimpleFlux simpleFlux;
        RoadRunner rr(simpleFlux.str());
        simpleFlux.checkSteadyState(&rr);
    }

    /**
     * todo consolidate the steadyState computation
     * so that we only have one version.
     */
    void steadyStateNamedArray() {
        SimpleFlux simpleFlux;
        RoadRunner rr(simpleFlux.str());
        auto x = rr.steadyStateNamedArray();
        ASSERT_EQ(2, x.size()); // steady state tested elsewhere.
    }

    void getSteadyStateSelections() {
        SimpleFlux simpleFlux;
        RoadRunner rr(simpleFlux.str());
        auto selections = rr.getSteadyStateSelections();
        ASSERT_EQ(2, selections.size()); // S1 and S2
    }

    void setSteadyStateSelections() {
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
    void getSteadyStateValues() {
        SimpleFlux simpleFlux;
        RoadRunner rr(simpleFlux.str());
        auto x = rr.getSteadyStateValues();
        std::vector<double> expected({1, 10});
        ASSERT_EQ(expected, x);
    }

    void getSteadyStateSelectionStrings() {
        SimpleFlux simpleFlux;
        RoadRunner rr(simpleFlux.str());
        std::vector<std::string> expected({"[S1]", "[S2]"});
        ASSERT_EQ(expected, rr.getSteadyStateSelectionStrings());
    }

    /**
     * Another way to get steadystate values?
     *  Consolidate the API!
     */
    void getSteadyStateValuesNamedArray() {
        SimpleFlux simpleFlux;
        RoadRunner rr(simpleFlux.str());
        auto x = rr.getSteadyStateValuesNamedArray();
        ls::DoubleMatrix expected({{1, 10}});
        checkMatrixEqual(expected, x);
    }

    void regenerateModel() {
        SimpleFlux simpleFlux;
        RoadRunner rr(simpleFlux.str());
        ASSERT_EQ(2, rr.getFloatingSpeciesAmountsNamedArray().numCols());
        rr.addSpeciesConcentration("newSpecies", "default_compartment", 4, false, false, "", false);
        ASSERT_EQ(2, rr.getFloatingSpeciesAmountsNamedArray().numCols());
        rr.regenerateModel();
        ASSERT_EQ(3, rr.getFloatingSpeciesAmountsNamedArray().numCols());
    }


    void getIntegratorDefault() {
        RoadRunner rr(SimpleFlux().str());
        ASSERT_STREQ(rr.getIntegrator()->getName().c_str(), "cvode");
    }


    void getSteadyStateSolver() {
        RoadRunner rr(SimpleFlux().str());
        ASSERT_STREQ(rr.getSteadyStateSolver()->getName().c_str(), "nleq2");

    }

    void getSensitivitySolver() {
        RoadRunner rr(SimpleFlux().str());
        ASSERT_STREQ(rr.getSensitivitySolver()->getName().c_str(), "forward");
    }

    void getIntegratorByName() {
        RoadRunner rr(SimpleFlux().str());
        Integrator *gillespie = rr.getIntegratorByName("gillespie");
        ASSERT_STREQ(rr.getIntegrator()->getName().c_str(), "cvode");
        ASSERT_STREQ(gillespie->getName().c_str(), "gillespie");
    }

    void getSteadyStateSolverByName() {
        RoadRunner rr(SimpleFlux().str());
        SteadyStateSolver *kinsol_newton = rr.getSteadyStateSolverByName("newton");
        ASSERT_STREQ(rr.getSteadyStateSolver()->getName().c_str(), "nleq2");
        ASSERT_STREQ(kinsol_newton->getName().c_str(), "newton");
    }

    /**
     * Disabled test - only 1 sensitivity solver exsits right now
     */
    void DISABLED_getSensitivitySolverByName() {
        RoadRunner rr(SimpleFlux().str());
        SensitivitySolver *forward = rr.getSensitivitySolverByName("forward");
        ASSERT_STREQ(rr.getSensitivitySolver()->getName().c_str(), "forward");
        ASSERT_STREQ(forward->getName().c_str(), "gillespie");

    }

    void makeIntegrator() {
        RoadRunner rr(SimpleFlux().str());
        rr.makeIntegrator("gillespie");
        ASSERT_TRUE(rr.integratorExists("gillespie"));
    }

    void makeSteadyStateSolver() {
        RoadRunner rr(SimpleFlux().str());
        rr.makeSteadyStateSolver("newton_linesearch");
        ASSERT_TRUE(rr.steadyStateSolverExists("newton_linesearch"));
    }

    /**
     * Only 1 sensitivity solver currently exists
     */
    void DISABLED_makeSensitivitySolver() {
        RoadRunner rr(SimpleFlux().str());
        //    rr.makeSensitivitySolver();

    }

    void getExistingIntegratorNamesOneExists() {
        RoadRunner rr(SimpleFlux().str());
        auto names = rr.getExistingIntegratorNames();
        std::vector<std::string> expected({"cvode"});
        ASSERT_EQ(expected, names);
    }

    void getExistingIntegratorNamesTwoExists() {
        RoadRunner rr(SimpleFlux().str());
        rr.makeIntegrator("gillespie");
        auto names = rr.getExistingIntegratorNames();
        std::vector<std::string> expected({"cvode", "gillespie"});
        ASSERT_EQ(expected, names);
    }

    void getExistingSteadyStateSolverNamesOneExists() {
        RoadRunner rr(SimpleFlux().str());
        rr.getExistingSteadyStateSolverNames();
        auto names = rr.getExistingSteadyStateSolverNames();
        std::vector<std::string> expected({"nleq2"});
        ASSERT_EQ(expected, names);
    }

    void getExistingSteadyStateSolverNamesTwoExists() {
        RoadRunner rr(SimpleFlux().str());
        rr.makeSteadyStateSolver("newton");
        rr.getExistingSteadyStateSolverNames();
        auto names = rr.getExistingSteadyStateSolverNames();
        std::vector<std::string> expected({"nleq2", "newton"});
        ASSERT_EQ(expected, names);
    }

    void getExistingSensitivitySolverNames() {
        RoadRunner rr(SimpleFlux().str());
        rr.getExistingSensitivitySolverNames();
        auto names = rr.getExistingSensitivitySolverNames();
        std::vector<std::string> expected({"forward"});
        ASSERT_EQ(expected, names);
    }

    void getRegisteredIntegratorNames() {
        RoadRunner rr(SimpleFlux().str());
        auto names = rr.getRegisteredIntegratorNames();
        std::vector<std::string> expected({"cvode", "gillespie", "rk4", "rk45", "euler"});
        ASSERT_EQ(expected, names);
    }

    void getRegisteredSteadyStateSolverNames() {
        RoadRunner rr(SimpleFlux().str());
        auto names = rr.getRegisteredSteadyStateSolverNames();
        std::vector<std::string> expected({"nleq1", "nleq2", "newton", "newton_linesearch"});
        ASSERT_EQ(expected, names);
    }

    void getRegisteredSensitivitySolverNames() {
        RoadRunner rr(SimpleFlux().str());
        auto names = rr.getRegisteredSensitivitySolverNames();
        std::vector<std::string> expected({"forward"});
        ASSERT_EQ(expected, names);
    }

    void setIntegrator() {
        RoadRunner rr(SimpleFlux().str());
        rr.setIntegrator("gillespie");
        ASSERT_STREQ("gillespie", rr.getIntegrator()->getName().c_str());
    }

    void setIntegratorToRk4() {
        RoadRunner rr(SimpleFlux().str());
        rr.setIntegrator("rk4");
        ASSERT_STREQ("rk4", rr.getIntegrator()->getName().c_str());
    }

    void setIntegratorToRk45() {
        RoadRunner rr(SimpleFlux().str());
        rr.setIntegrator("rk45");
        ASSERT_STREQ("rk45", rr.getIntegrator()->getName().c_str());
    }

    void setIntegratorErrorMessage() {
        RoadRunner rr(SimpleFlux().str());
        std::string expectedErrMsg = R"(No such solver called "WrongIntegrator". Available options are "cvode", "gillespie", "rk4", "rk45", "euler")";
        try {
            // we always end up in the catch block
            rr.setIntegrator("WrongIntegrator");
            // if we get this far, the test has failed.
            ASSERT_TRUE(false);
        } catch (InvalidKeyException &e) {
            const std::string &what = e.what();
            ASSERT_STREQ(expectedErrMsg.c_str(), what.c_str());
        }
    }

    void setSteadyStateSolverErrorMessage() {
        RoadRunner rr(SimpleFlux().str());
        std::string expectedErrMsg = R"(No such solver called "WrongSolver". Available options are "nleq1", "nleq2", "newton", "newton_linesearch")";
        try {
            // we always end up in the catch block
            rr.setSteadyStateSolver("WrongSolver");
            // if we get this far, the test has failed.
            ASSERT_TRUE(false);
        } catch (InvalidKeyException &e) {
            const std::string &what = e.what();
            ASSERT_STREQ(expectedErrMsg.c_str(), what.c_str());
        }
    }

    void setSensitivitySolverErrorMessage() {
        RoadRunner rr(SimpleFlux().str());
        std::string expectedErrMsg = R"(No such solver called "WrongSolver". Available options are "forward")";
        try {
            // we always end up in the catch block
            rr.setSensitivitySolver("WrongSolver");
            // if we get this far, the test has failed.
            ASSERT_TRUE(false);
        } catch (InvalidKeyException &e) {
            const std::string &what = e.what();
            ASSERT_STREQ(expectedErrMsg.c_str(), what.c_str());
        }
    }

    void setSteadyStateSolver() {
        RoadRunner rr(SimpleFlux().str());
        rr.setSteadyStateSolver("nleq1");
        ASSERT_STREQ("nleq1", rr.getSteadyStateSolver()->getName().c_str());
    }

    void setSensitivitySolver() {
        RoadRunner rr(SimpleFlux().str());
        rr.setSensitivitySolver("forward");
        ASSERT_STREQ("forward", rr.getSensitivitySolver()->getName().c_str());
    }

    void integratorExistsWhenTrue() {
        RoadRunner rr(SimpleFlux().str());
        ASSERT_TRUE(rr.integratorExists("cvode"));
    }

    void steadyStateSolverExistsWhenTrue() {
        RoadRunner rr(SimpleFlux().str());
        ASSERT_TRUE(rr.steadyStateSolverExists("nleq2"));
    }

    void sensitivitySolverExistsWhenTrue() {
        RoadRunner rr(SimpleFlux().str());
        ASSERT_TRUE(rr.sensitivitySolverExists("forward"));
    }

    void integratorExistsWhenFalse() {
        RoadRunner rr(SimpleFlux().str());
        ASSERT_FALSE(rr.integratorExists("gillespie"));
    }

    void steadyStateSolverExistsWhenFalse() {
        RoadRunner rr(SimpleFlux().str());
        ASSERT_FALSE(rr.steadyStateSolverExists("newton"));
    }

    /**
     * @brief cannot test this, since only 1 SensitivitySolver implemented at this time
     */
    void DISABLED_sensitivitySolverExistsWhenFalse() {
        RoadRunner rr(SimpleFlux().str());
        //    ASSERT_TRUE(rr.sensitivitySolverExists("forward"));
    }


    /**
     * The remaining methods are deprecated, sometimes
     * with no obvious replacement however.
     * We will not test deprecated methods given that
     * they will be removed.
     */

    //void setBoundarySpeciesByIndex(){
    //
    //}

    //void getNumberOfIndependentSpecies(){
    //
    //}

    //void getGlobalParameterIds(){
    //
    //}

    //void getBoundarySpeciesIds(){
    //
    //}

    //void getBoundarySpeciesConcentrationIds(){
    //
    //}

    //void getBoundarySpeciesByIndex(){
    //
    //}

    //void getGlobalParameterByIndex(){
    //
    //}

    //void setGlobalParameterByName(){
    //
    //}

    //void getGlobalParameterByName(){
    //
    //}

    //void getConservedMoietyValues(){
    //
    //}

    //void getConservedMoietyIds(){
    //
    //}

    //void getNumberOfReactions(){
    //
    //}

    //void getReactionRate(){
    //
    //}

    //void getRateOfChange(){
    //
    //}

    //void getRateOfChangeIds(){
    //
    //}

    //void getNumberOfCompartments(){
    //
    //}

    //void setCompartmentByIndex(){
    //
    //}

    //void getCompartmentByIndex(){
    //
    //}

    //void getCompartmentIds(){
    //
    //}

    //void getNumberOfBoundarySpecies(){
    //
    //}

    //void getBoundarySpeciesConcentrations(){
    //
    //}

    //void setBoundarySpeciesConcentrations(){
    //
    //}

    //void getNumberOfFloatingSpecies(){
    //
    //}

    //void getFloatingSpeciesByIndex(){
    //
    //}

    //void setFloatingSpeciesByIndex(){
    //
    //}

    //void getFloatingSpeciesConcentrationsV(){
    //
    //}

    //void getFloatingSpeciesAmountsV(){
    //
    //}

    //void getBoundarySpeciesConcentrationsV(){
    //
    //}

    //void getBoundarySpeciesAmountsV(){
    //
    //}

    //void getFloatingSpeciesInitialConcentrations(){
    //
    //}

    //void setFloatingSpeciesConcentrations(){
    //
    //}

    //void setFloatingSpeciesInitialConcentrationByIndex(){
    //
    //}

    //void setFloatingSpeciesInitialConcentrations(){
    //
    //}

    //void getFloatingSpeciesIds(){
    //
    //}

    //void getFloatingSpeciesInitialConditionIds(){
    //
    //}

    //void getNumberOfGlobalParameters(){
    //
    //}

    //void setGlobalParameterByIndex(){
    //
    //}

    //void getGlobalParameterValues(){
    //
    //}

    //void evalModel(){
    //
    //}

    //void getNumberOfDependentSpecies(){
    //
    //}

    //void getReactionRates(){
    //
    //}

    //void getReactionIds(){
    //
    //}
};


#endif //ROADRUNNER_ROADRUNNERAPITESTS_H
