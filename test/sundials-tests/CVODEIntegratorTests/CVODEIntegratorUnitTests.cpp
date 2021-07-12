
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


TEST_F(CVODEIntegratorUnitTests, SetConcentrationToleranceFromScalar1) {
    EXPECT_CALL(mockExecutableModel, getNumCompartments).WillRepeatedly(Return(1));
    EXPECT_CALL(mockExecutableModel, getCompartmentVolumes)
            .Times(2); // once in get and once in set
    EXPECT_CALL(mockExecutableModel, getNumIndFloatingSpecies).WillRepeatedly(Return(2));
    EXPECT_CALL(mockExecutableModel, getCompartmentIndexForFloatingSpecies).WillRepeatedly(Return(0));
    CVODEIntegrator cvodeIntegrator(&mockExecutableModel);
    cvodeIntegrator.setConcentrationTolerance(0.1234);
    auto x = cvodeIntegrator.getConcentrationTolerance();
    for (auto i: x) {
        ASSERT_NEAR(i, 0.1234, 1e-7);
    }
}

TEST_F(CVODEIntegratorUnitTests, SetConcentrationToleranceFromScalarSmallValue) {
    EXPECT_CALL(mockExecutableModel, getNumCompartments).WillRepeatedly(Return(1));
    EXPECT_CALL(mockExecutableModel, getCompartmentVolumes)
            .Times(2); // once in get and once in set
    EXPECT_CALL(mockExecutableModel, getNumIndFloatingSpecies).WillRepeatedly(Return(2));
    EXPECT_CALL(mockExecutableModel, getCompartmentIndexForFloatingSpecies).WillRepeatedly(Return(0));
    CVODEIntegrator cvodeIntegrator(&mockExecutableModel);
    cvodeIntegrator.setConcentrationTolerance(1e-16);
    auto x = cvodeIntegrator.getConcentrationTolerance();
    for (auto i: x) {
        ASSERT_NEAR(i, 1e-16, 1e-7);
    }
}

TEST_F(CVODEIntegratorUnitTests, SetConcentrationToleranceFromVector) {
    EXPECT_CALL(mockExecutableModel, getNumCompartments).WillRepeatedly(Return(1));
    EXPECT_CALL(mockExecutableModel, getCompartmentVolumes)
            .Times(2); // once in get and once in set
    EXPECT_CALL(mockExecutableModel, getNumIndFloatingSpecies).WillRepeatedly(Return(2));
    EXPECT_CALL(mockExecutableModel, getCompartmentIndexForFloatingSpecies).WillRepeatedly(Return(0));
    CVODEIntegrator cvodeIntegrator(&mockExecutableModel);
    cvodeIntegrator.setConcentrationTolerance(std::vector<double>({0.1234, 1.5678}));
    auto x = cvodeIntegrator.getConcentrationTolerance();
    ASSERT_EQ(x, std::vector<double>({0.1234, 1.5678}));
}

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

TEST_F(CVODEIntegratorUnitTests, setIndividualTolerance) {
    EXPECT_CALL(mockExecutableModel, getNumFloatingSpecies)
            .WillRepeatedly(Return(2));
    EXPECT_CALL(mockExecutableModel, getFloatingSpeciesIndex)
            .WillRepeatedly(Return(0)); // we say that S1 is idx 0
    EXPECT_CALL(mockExecutableModel, getNumIndFloatingSpecies)
            .WillRepeatedly(Return(2)); // assume this model has 2 species
    CVODEIntegrator cvodeIntegrator(&mockExecutableModel);
    cvodeIntegrator.setIndividualTolerance("S1", 1e-14);
    auto x = cvodeIntegrator.getConcentrationTolerance();
    ASSERT_EQ(x, std::vector<double>({1e-14, 1e-12}));
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















