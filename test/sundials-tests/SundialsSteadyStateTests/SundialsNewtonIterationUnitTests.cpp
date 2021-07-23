#include <fstream>
#include <rr-libstruct/lsLibStructural.h>
#include "gtest/gtest.h"

#include "SundialsSteadyStateSolverUnitTest.h"
#include "BasicNewtonIteration.h"
#include "rrRoadRunner.h"
#include "TestModelFactory.h"

using namespace rr;

/**
 * @brief tests that focus on the "newton" solver from sundials test suite
 * without linesearch globalisation
 */
class BasicNewtonIterationUnitTest : public SundialsSteadyStateSolverUnitTest {
public:

    BasicNewtonIterationUnitTest() : SundialsSteadyStateSolverUnitTest() {};

    ~BasicNewtonIterationUnitTest() override = default;
};


TEST_F(BasicNewtonIterationUnitTest, SolveUsingSolverDirectly) {
    // aka without a RoadRunner instance
    BasicNewtonIteration solver(rr->getModel());
    solver.solve();
    checkResults(rr->getFloatingSpeciesConcentrationsNamedArray());
}

TEST_F(BasicNewtonIterationUnitTest, ChangeAndResetSettings) {
    BasicNewtonIteration solver(rr->getModel());
    solver.setValue("eta_param_gamma", Setting(0.4356));
    solver.resetSettings();
    ASSERT_EQ((int) solver.getValue("eta_param_gamma"), 0);
}


TEST_F(BasicNewtonIterationUnitTest, RegenerateTheModelBeforeCreatingSolver) {
    rr->regenerateModel();
    BasicNewtonIteration solver(rr->getModel());
    solver.solve();
    checkResults(rr->getFloatingSpeciesConcentrationsNamedArray());
}

TEST_F(BasicNewtonIterationUnitTest, RegenerateTheModelAfterCreatingSolver) {
    BasicNewtonIteration solver(rr->getModel());
    rr->regenerateModel();
    // after regeneration, the pointer to the model is different
    // so we must sync with model before solving.
    solver.syncWithModel(rr->getModel());
    solver.solve();
    checkResults(rr->getFloatingSpeciesConcentrationsNamedArray());
}

/**
 * Tests each of the settings can be changed
 */
class BasicNewtonSettingsTests :
        public BasicNewtonIterationUnitTest,
        public ::testing::WithParamInterface<std::pair<std::string, Setting>> {
public:
    BasicNewtonSettingsTests() = default;


    void doSettingsTest(const std::string &settingName, Setting settingValue) {
        std::cout << "Testing setting \"" << settingName << "\"" << std::endl;

        OpenLinearFlux testModel;

        RoadRunner rr(testModel.str());
        BasicNewtonIteration solver(rr.getModel());

        // actually change the setting to our new value
        solver.setValue(settingName, settingValue);

        // make comparison between setting value and value we get from solver
        if (settingValue.isInteger()) {
            EXPECT_EQ((int) settingValue, (int) solver.getValue(settingName));
        } else if (settingValue.isNumeric()) {
            EXPECT_NEAR((double) settingValue, solver.getValue(settingName), 0.001);
        } else if (settingValue.isBool()) {
            EXPECT_TRUE((bool) settingValue == (bool) solver.getValue(settingName));
        } else if (settingValue.isString()) {
            EXPECT_STREQ(settingValue.get<std::string>().c_str(),  solver.getValue(settingName).get<std::string>().c_str());
        }
    }
};


TEST_F(BasicNewtonSettingsTests, NumMaxIters) {
    doSettingsTest("num_max_iters", 153);
}

TEST_F(BasicNewtonSettingsTests, PrintLevel) {
    doSettingsTest("print_level", 2);
}

TEST_F(BasicNewtonSettingsTests, NoInitSetup) {
    doSettingsTest("no_init_setup", true);
}

TEST_F(BasicNewtonSettingsTests, EtaForm) {
    doSettingsTest("eta_form", "eta_choice2");
}

TEST_F(BasicNewtonSettingsTests, NoResMonitoring) {
    doSettingsTest("no_res_monitoring", true);
}

TEST_F(BasicNewtonSettingsTests, MaxSetupCalls) {
    doSettingsTest("max_setup_calls", 974);
}

TEST_F(BasicNewtonSettingsTests, MaxSubsetupCalls) {
    doSettingsTest("max_subsetup_calls", 132);
}

TEST_F(BasicNewtonSettingsTests, EtaConstantValue) {
    doSettingsTest("eta_constant_value", 6);
}

TEST_F(BasicNewtonSettingsTests, EtaParamGamma) {
    doSettingsTest("eta_param_gamma", 0.5234);
}

TEST_F(BasicNewtonSettingsTests, EtaParamAlpha) {
    doSettingsTest("eta_param_alpha", 0.1234);
}

TEST_F(BasicNewtonSettingsTests, ResMonMin) {
    doSettingsTest("res_mon_min", 0.01);
}

TEST_F(BasicNewtonSettingsTests, ResMonMax) {
    doSettingsTest("res_mon_max", 0.9999);
}

TEST_F(BasicNewtonSettingsTests, ResMonConstantValue) {
    doSettingsTest("res_mon_constant_value", 0.25);
}

TEST_F(BasicNewtonSettingsTests, NoMinEps) {
    doSettingsTest("no_min_eps", true);
}

TEST_F(BasicNewtonSettingsTests, MaxNewtonStep) {
    doSettingsTest("max_newton_step", 15);
}

TEST_F(BasicNewtonSettingsTests, MaxBetaFails) {
    doSettingsTest("max_beta_fails", 3);
}

TEST_F(BasicNewtonSettingsTests, FuncNormTol) {
    doSettingsTest("func_norm_tol", 0.5);
}

TEST_F(BasicNewtonSettingsTests, ScaledStepTol) {
    doSettingsTest("scaled_step_tol", 0.3);
}





















