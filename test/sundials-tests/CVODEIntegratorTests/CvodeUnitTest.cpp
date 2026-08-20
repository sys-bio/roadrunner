
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "CVODEIntegrator.h"
#include "MockExecutableModel.h"
#include "rrConfig.h"

/**
 * In general it has been hard to retroactively
 * write unit tests for CVODEIntegrator. I've done what I can
 * but in all probability it would be better to rewrite with
 * a testing centric focus
 */

using namespace rr;
using namespace testing;

class CVODEIntegratorUnitTests : public ::testing::Test {
public:

    MockExecutableModel mockExecutableModel;

    CVODEIntegratorUnitTests() {
        //When called with no arguments, getStateVector returns
        // the size of the state vector. In this mock model,
        // we say that there are 2 states, throughput the whole unit
        EXPECT_CALL(mockExecutableModel, getStateVector)
            .WillRepeatedly(Return(2));
    };

};


//TEST_F(CVODEIntegratorUnitTests, SetConcentrationToleranceFromScalar1) {
//    EXPECT_CALL(mockExecutableModel, getNumCompartments).WillRepeatedly(Return(1));
//    EXPECT_CALL(mockExecutableModel, getCompartmentVolumes)
//            .Times(2); // once in get and once in set
//    EXPECT_CALL(mockExecutableModel, getNumIndFloatingSpecies).WillRepeatedly(Return(2));
//    EXPECT_CALL(mockExecutableModel, getCompartmentIndexForFloatingSpecies).WillRepeatedly(Return(0));
//    CVODEIntegrator cvodeIntegrator(&mockExecutableModel);
//    cvodeIntegrator.setConcentrationTolerance(0.1234);
//    auto x = cvodeIntegrator.getAbsoluteToleranceVector();
//    for (auto i: x) {
//        ASSERT_NEAR(i, 0.1234, 1e-7);
//    }
//}
//
//TEST_F(CVODEIntegratorUnitTests, SetConcentrationToleranceFromScalarSmallValue) {
//    EXPECT_CALL(mockExecutableModel, getNumCompartments).WillRepeatedly(Return(1));
//    EXPECT_CALL(mockExecutableModel, getCompartmentVolumes)
//            .Times(2); // once in get and once in set
//    EXPECT_CALL(mockExecutableModel, getNumIndFloatingSpecies).WillRepeatedly(Return(2));
//    EXPECT_CALL(mockExecutableModel, getCompartmentIndexForFloatingSpecies).WillRepeatedly(Return(0));
//    CVODEIntegrator cvodeIntegrator(&mockExecutableModel);
//    cvodeIntegrator.setConcentrationTolerance(1e-16);
//    auto x = cvodeIntegrator.getAbsoluteToleranceVector();
//    for (auto i: x) {
//        ASSERT_NEAR(i, 1e-16, 1e-7);
//    }
//}
//
//TEST_F(CVODEIntegratorUnitTests, SetConcentrationToleranceFromVector) {
//    EXPECT_CALL(mockExecutableModel, getNumCompartments).WillRepeatedly(Return(1));
//    EXPECT_CALL(mockExecutableModel, getCompartmentVolumes)
//            .Times(2); // once in get and once in set
//    EXPECT_CALL(mockExecutableModel, getNumIndFloatingSpecies).WillRepeatedly(Return(2));
//    EXPECT_CALL(mockExecutableModel, getCompartmentIndexForFloatingSpecies).WillRepeatedly(Return(0));
//    CVODEIntegrator cvodeIntegrator(&mockExecutableModel);
//    cvodeIntegrator.setConcentrationTolerance(std::vector<double>({0.1234, 1.5678}));
//    auto x = cvodeIntegrator.getAbsoluteToleranceVector();
//    ASSERT_EQ(x, std::vector<double>({0.1234, 1.5678}));
//}
//
/**
 * The only time loadConfigSettings is actually used is
 * inside CVODEIntegrator::resetSettings, so I can only
 * assume its purpose is to reload the default settings
 * from the Config.
 */
TEST_F(CVODEIntegratorUnitTests, loadConfigSettings) {
    CVODEIntegrator cvodeIntegrator(&mockExecutableModel);
    cvodeIntegrator.setValue("stiff", false);
    ASSERT_FALSE((bool) cvodeIntegrator.getValue("stiff"));
    cvodeIntegrator.loadConfigSettings();
    ASSERT_TRUE((bool) cvodeIntegrator.getValue("stiff"));
}

/**
 * It is unclear what the settings file should look like. Its
 * not documented. Is this still a supported feature?
 */
TEST_F(CVODEIntegratorUnitTests, DISABLED_loadSBMLSettings) {}

TEST_F(CVODEIntegratorUnitTests, setValue) {
    CVODEIntegrator cvodeIntegrator(&mockExecutableModel);
    auto absval = cvodeIntegrator.getValue("absolute_tolerance");
    ASSERT_NEAR(1e-12, (float) absval, 1e-7);
    cvodeIntegrator.setValue("absolute_tolerance", 1e-14);
    ASSERT_NEAR(1e-14, (float) absval, 1e-7);
}

TEST_F(CVODEIntegratorUnitTests, resetSettings) {
    CVODEIntegrator cvodeIntegrator(&mockExecutableModel);


}

TEST_F(CVODEIntegratorUnitTests, ResetSettingsRelativeTolerance) {
    CVODEIntegrator cvodeIntegrator(&mockExecutableModel);
    cvodeIntegrator.setValue("relative_tolerance", 1e-14);
    ASSERT_NEAR((double) cvodeIntegrator.getValue("relative_tolerance"), 1e-14, 1e-7);
    cvodeIntegrator.resetSettings();
    ASSERT_NEAR((double) cvodeIntegrator.getValue("relative_tolerance"), 1e-6, 1e-7);
}

TEST_F(CVODEIntegratorUnitTests, ResetSettingsAbsoluteTolerance) {
    CVODEIntegrator cvodeIntegrator(&mockExecutableModel);
    cvodeIntegrator.setValue("absolute_tolerance", 1e-14);
    ASSERT_NEAR(cvodeIntegrator.getValue("absolute_tolerance").getAs<double>(), 1e-14, 1e-7);
    cvodeIntegrator.resetSettings();
    ASSERT_NEAR(cvodeIntegrator.getValue("absolute_tolerance").getAs<double>(), 1e-12, 1e-7);
}

TEST_F(CVODEIntegratorUnitTests, ResetSettingsStiff) {
    CVODEIntegrator cvodeIntegrator(&mockExecutableModel);
    cvodeIntegrator.setValue("stiff", false);
    ASSERT_FALSE((bool) cvodeIntegrator.getValue("stiff"));
    cvodeIntegrator.resetSettings();
    ASSERT_TRUE((bool) cvodeIntegrator.getValue("stiff"));
}

TEST_F(CVODEIntegratorUnitTests, ResetSettingsMaximumBdfOrder) {
    CVODEIntegrator cvodeIntegrator(&mockExecutableModel);
    cvodeIntegrator.setValue("maximum_bdf_order", 3);
    ASSERT_EQ((int) cvodeIntegrator.getValue("maximum_bdf_order"), 3);
    cvodeIntegrator.resetSettings();
    ASSERT_EQ((int) cvodeIntegrator.getValue("maximum_bdf_order"), 5);
}

TEST_F(CVODEIntegratorUnitTests, ResetSettingsMaximumAdamsOrder) {
    CVODEIntegrator cvodeIntegrator(&mockExecutableModel);
    cvodeIntegrator.setValue("maximum_adams_order", 2);
    ASSERT_EQ((int) cvodeIntegrator.getValue("maximum_adams_order"), 2);
    cvodeIntegrator.resetSettings();
    ASSERT_EQ((int) cvodeIntegrator.getValue("maximum_adams_order"), 12);
}

TEST_F(CVODEIntegratorUnitTests, ResetSettingsMaximumNumSteps) {
    CVODEIntegrator cvodeIntegrator(&mockExecutableModel);
    cvodeIntegrator.setValue("maximum_num_steps", 1e5);
    ASSERT_EQ((int) cvodeIntegrator.getValue("maximum_num_steps"), 1e5);
    cvodeIntegrator.resetSettings();
    ASSERT_EQ((int) cvodeIntegrator.getValue("maximum_num_steps"), 2e4);
}

TEST_F(CVODEIntegratorUnitTests, ResetSettingsMaximumTimeStep) {
    CVODEIntegrator cvodeIntegrator(&mockExecutableModel);
    cvodeIntegrator.setValue("maximum_time_step", 2);
    ASSERT_EQ((double) cvodeIntegrator.getValue("maximum_time_step"), 2);
    cvodeIntegrator.resetSettings();
    ASSERT_EQ((double) cvodeIntegrator.getValue("maximum_time_step"), 0); // gets interpreted as default
}

TEST_F(CVODEIntegratorUnitTests, ResetSettingsMinimumTimeStep) {
    CVODEIntegrator cvodeIntegrator(&mockExecutableModel);
    cvodeIntegrator.setValue("minimum_time_step", 1e-3);
    ASSERT_EQ((double) cvodeIntegrator.getValue("minimum_time_step"), 1e-3);
    cvodeIntegrator.resetSettings();
    ASSERT_EQ((double) cvodeIntegrator.getValue("minimum_time_step"), 0); // gets interpreted as default
}

TEST_F(CVODEIntegratorUnitTests, ResetSettingsInitialTimeStep) {
    CVODEIntegrator cvodeIntegrator(&mockExecutableModel);
    cvodeIntegrator.setValue("initial_time_step", 1.23);
    ASSERT_NEAR((double) cvodeIntegrator.getValue("initial_time_step"), 1.23, 1e-7);
    cvodeIntegrator.resetSettings();
    ASSERT_NEAR((double) cvodeIntegrator.getValue("initial_time_step"), 0, 1e-7);
}

TEST_F(CVODEIntegratorUnitTests, ResetSettingsMultipleSteps) {
    CVODEIntegrator cvodeIntegrator(&mockExecutableModel);
    cvodeIntegrator.setValue("multiple_steps", true);
    ASSERT_TRUE((bool) cvodeIntegrator.getValue("multiple_steps"));
    cvodeIntegrator.resetSettings();
    ASSERT_FALSE((bool) cvodeIntegrator.getValue("multiple_steps"));
}

TEST_F(CVODEIntegratorUnitTests, ResetSettingsVariableStepSize) {
    CVODEIntegrator cvodeIntegrator(&mockExecutableModel);
    cvodeIntegrator.setValue("variable_step_size", true);
    ASSERT_TRUE((bool) cvodeIntegrator.getValue("variable_step_size"));
    cvodeIntegrator.resetSettings();
    ASSERT_FALSE((bool) cvodeIntegrator.getValue("variable_step_size"));
}

TEST_F(CVODEIntegratorUnitTests, ResetSettingsMaxOutputRows) {
    CVODEIntegrator cvodeIntegrator(&mockExecutableModel);
    cvodeIntegrator.setValue("max_output_rows", 15);
    ASSERT_EQ((int) cvodeIntegrator.getValue("max_output_rows"), 15);
    cvodeIntegrator.resetSettings();
    ASSERT_EQ((int) cvodeIntegrator.getValue("max_output_rows"), 1e5);
}

/**
 * Not clear how to test this. Its not clear exactly what
 * tolerance tweaking actually does but commenting out
 * its only usage causes some of the tests to fail. Yet,
 * tweaking the tolerances doesn't actually change the
 * tolerances in this test.
 */
TEST_F(CVODEIntegratorUnitTests, DISABLED_tweakTolerancess) {
    CVODEIntegrator cvodeIntegrator(&mockExecutableModel);
    std::cout << cvodeIntegrator.getValue("absolute_tolerance").toString() << std::endl;
    std::cout << cvodeIntegrator.getValue("relative_tolerance").toString() << std::endl;
    cvodeIntegrator.tweakTolerances();
    std::cout << cvodeIntegrator.getValue("absolute_tolerance").toString() << std::endl;
    std::cout << cvodeIntegrator.getValue("relative_tolerance").toString() << std::endl;

}


TEST_F(CVODEIntegratorUnitTests, restart) {
    // when we reset, setTime will be called with 0 as argument
    EXPECT_CALL(mockExecutableModel, setTime(0));
    CVODEIntegrator cvodeIntegrator(&mockExecutableModel);
    cvodeIntegrator.restart(0);
}

/**
 * Regression tests for the tolerance-vector packing order.
 *
 * ExecutableModel::getStateVector() packs the state as
 * [rate-rule variables, independent floating species]. The tolerance
 * vector must follow the same layout, but setIndividualTolerance() and
 * getAbsoluteToleranceVector() historically assumed the reverse order
 * ([species, rate rules]), silently mapping tolerances to the wrong
 * state variable whenever a model had both.
 */
TEST_F(CVODEIntegratorUnitTests, SetIndividualToleranceRateRuleComesBeforeSpecies) {
    // 2 independent species (S0, S1) and 1 rate-rule variable (R0): state
    // vector length 3, overriding the fixture's default of 2.
    EXPECT_CALL(mockExecutableModel, getStateVector).WillRepeatedly(Return(3));
    EXPECT_CALL(mockExecutableModel, getNumIndFloatingSpecies).WillRepeatedly(Return(2));
    EXPECT_CALL(mockExecutableModel, getNumRateRules).WillRepeatedly(Return(1));
    EXPECT_CALL(mockExecutableModel, getRateRuleSymbols)
        .WillRepeatedly(Return(std::vector<std::string>{"R0"}));
    EXPECT_CALL(mockExecutableModel, getFloatingSpeciesIndex("R0")).WillRepeatedly(Return(-1));
    EXPECT_CALL(mockExecutableModel, getNumCompartments).WillRepeatedly(Return(1));

    CVODEIntegrator cvodeIntegrator(&mockExecutableModel);
    cvodeIntegrator.setValue("absolute_tolerance", 1e-6);
    cvodeIntegrator.setIndividualTolerance("R0", 0.5);

    std::vector<double> tol = cvodeIntegrator.getValue("absolute_tolerance").get<std::vector<double>>();
    ASSERT_EQ(3, tol.size());
    // Rate rules occupy [0, numRateRules); R0 is the only one, so its
    // tolerance must land at index 0, not at index 2 (numIndFloatingSpecies).
    ASSERT_NEAR(0.5, tol[0], 1e-9);
    ASSERT_NEAR(1e-6, tol[1], 1e-9);
    ASSERT_NEAR(1e-6, tol[2], 1e-9);
}

TEST_F(CVODEIntegratorUnitTests, SetIndividualToleranceForSpeciesIsOffsetByRateRuleCount) {
    // 2 independent species (S0, S1) and 1 rate-rule variable: state
    // vector length 3, overriding the fixture's default of 2.
    EXPECT_CALL(mockExecutableModel, getStateVector).WillRepeatedly(Return(3));
    EXPECT_CALL(mockExecutableModel, getNumIndFloatingSpecies).WillRepeatedly(Return(2));
    EXPECT_CALL(mockExecutableModel, getNumRateRules).WillRepeatedly(Return(1));
    // getAbsoluteToleranceVector() scales rate-rule tolerances unconditionally,
    // even though this test's tolerance update itself never touches a rate
    // rule. With getRateRuleValues() left at its default of 0, that scaling
    // takes the branch that also calls getFloatingSpeciesIndex("R0") to see
    // whether the rate-rule variable is itself a species -- so that needs a
    // matching expectation too, or gmock treats it as an unexpected call to
    // a method it only has an "S1" expectation for.
    EXPECT_CALL(mockExecutableModel, getRateRuleSymbols)
        .WillRepeatedly(Return(std::vector<std::string>{"R0"}));
    EXPECT_CALL(mockExecutableModel, getFloatingSpeciesIndex("R0")).WillRepeatedly(Return(-1));
    EXPECT_CALL(mockExecutableModel, getFloatingSpeciesIndex("S1")).WillRepeatedly(Return(1));
    EXPECT_CALL(mockExecutableModel, getNumCompartments).WillRepeatedly(Return(1));

    CVODEIntegrator cvodeIntegrator(&mockExecutableModel);
    cvodeIntegrator.setValue("absolute_tolerance", 1e-6);
    cvodeIntegrator.setIndividualTolerance("S1", 0.25);

    std::vector<double> tol = cvodeIntegrator.getValue("absolute_tolerance").get<std::vector<double>>();
    ASSERT_EQ(3, tol.size());
    // S1 is independent-species index 1. With 1 rate rule packed ahead of
    // the species block, its slot is numRateRules + 1 == 2, not 1.
    ASSERT_NEAR(0.25, tol[2], 1e-9);
    ASSERT_NEAR(1e-6, tol[0], 1e-9);
    ASSERT_NEAR(1e-6, tol[1], 1e-9);
}

TEST_F(CVODEIntegratorUnitTests, AbsoluteToleranceVectorPacksRateRulesBeforeSpecies) {
    // 1 independent species (S0, amount 5) and 1 rate-rule variable
    // (R0, value 3). Both amounts are non-zero, so getAbsoluteToleranceVector
    // scales each slot by abs(value) directly, without needing compartment
    // volumes.
    EXPECT_CALL(mockExecutableModel, getNumIndFloatingSpecies).WillRepeatedly(Return(1));
    EXPECT_CALL(mockExecutableModel, getNumRateRules).WillRepeatedly(Return(1));
    EXPECT_CALL(mockExecutableModel, getNumCompartments).WillRepeatedly(Return(1));
    EXPECT_CALL(mockExecutableModel, getRateRuleSymbols)
        .WillRepeatedly(Return(std::vector<std::string>{"R0"}));
    EXPECT_CALL(mockExecutableModel, getFloatingSpeciesAmounts)
        .WillRepeatedly(DoAll(SetArgPointee<2>(5.0), Return(0)));
    EXPECT_CALL(mockExecutableModel, getRateRuleValues)
        .WillRepeatedly(SetArgPointee<0>(3.0));

    CVODEIntegrator cvodeIntegrator(&mockExecutableModel);
    cvodeIntegrator.setValue("absolute_tolerance", 1e-6);
    std::vector<double> tol = cvodeIntegrator.getAbsoluteToleranceVector();

    ASSERT_EQ(2, tol.size());
    // Packed order is [rate rules, species]: R0 (value 3) at index 0,
    // S0 (amount 5) at index 1.
    ASSERT_NEAR(1e-6 * 3.0, tol[0], 1e-12);
    ASSERT_NEAR(1e-6 * 5.0, tol[1], 1e-12);
}















