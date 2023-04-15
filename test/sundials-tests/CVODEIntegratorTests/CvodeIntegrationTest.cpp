

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "rrRoadRunner.h"
#include "rrExecutableModel.h"
#include "CVODEIntegrator.h"
#include "Solver.h"
#include "TestModelFactory.h"
#include "Matrix.h"
#include "CvodeIntegrationTest.h"


using namespace rr;
using namespace testing;


TEST_F(CVODEIntegratorTests, TestSimpleFluxWithRoadRunner) {
    checkModelSimulatesWithRoadRunner("SimpleFlux");
}

TEST_F(CVODEIntegratorTests, TestModel269WithRoadRunner) {
    checkModelSimulatesWithRoadRunner("Model269");
}

TEST_F(CVODEIntegratorTests, TestModel28WithRoadRunner) {
    checkModelSimulatesWithRoadRunner("Model28");
}

TEST_F(CVODEIntegratorTests, TestFactorialInRateLawWithRoadRunner) {
    checkModelSimulatesWithRoadRunner("FactorialInRateLaw");
}

TEST_F(CVODEIntegratorTests, SimpleFluxWithRoadRunner) {
    checkModelSimulatesWithRoadRunner("SimpleFlux");
}

TEST_F(CVODEIntegratorTests, OpenLinearFluxWithRoadRunner) {
    checkModelSimulatesWithRoadRunner("OpenLinearFlux");
}

TEST_F(CVODEIntegratorTests, OneEvent10TriggersWithRoadRunner) {
    checkModelSimulatesWithRoadRunner("OneEvent10Triggers");
}



/**
 * These tests use the LLVMExecutableModel instead of the RoadRunner interface.
 */

TEST_F(CVODEIntegratorTests, TestSimpleFluxWithModel) {
    SimpleFlux testModel;
    checkModelIntegrates<CVODEIntegrator>(&testModel);
}

TEST_F(CVODEIntegratorTests, TestModel269WithModel) {
    Model269 testModel;
    checkModelIntegrates<CVODEIntegrator>(&testModel);
}

TEST_F(CVODEIntegratorTests, TestModel28WithModel) {
    Model28 testModel;
    checkModelIntegrates<CVODEIntegrator>(&testModel);
}

TEST_F(CVODEIntegratorTests, SimpleFluxWithModel) {
    SimpleFlux testModel;
    checkModelIntegrates<CVODEIntegrator>(&testModel);
}

TEST_F(CVODEIntegratorTests, OpenLinearFluxWithModel) {
    OpenLinearFlux testModel;
    checkModelIntegrates<CVODEIntegrator>(&testModel);
}

TEST_F(CVODEIntegratorTests, FactorialInRateLawWithModel) {
    FactorialInRateLaw testModel;
    checkModelIntegrates<CVODEIntegrator>(&testModel);
}

TEST_F(CVODEIntegratorTests, OneEvent10Triggers) {
    OneEvent10Triggers testModel;
    checkModelIntegrates<CVODEIntegrator>(&testModel);
}


TEST_F(CVODEIntegratorTests, CheckAbsToleranceDefaults) {
    TestModel* testModelPtr = TestModelFactory("SimpleFlux");

    // load model
    RoadRunner r(testModelPtr->str());
    std::vector<double> atols = r.getIntegrator()->getAbsoluteToleranceVector();
    ASSERT_EQ(atols.size(), 2);
    EXPECT_NEAR(atols[0], 1e-11, 1e-16);
    EXPECT_NEAR(atols[1], 1e-12, 1e-16);

    r.setValue("init(default_compartment)", 1e-10);
    r.setValue("init([S1])", 10.0);
    r.setValue("init([S2])", 1.0);
    atols = r.getIntegrator()->getAbsoluteToleranceVector();
    ASSERT_EQ(atols.size(), 2);
    EXPECT_NEAR(atols[0], 1e-21, 1e-26);
    EXPECT_NEAR(atols[1], 1e-22, 1e-26);

    r.setValue("default_compartment", 1e-15);
    r.setValue("[S1]", 10.0);
    r.setValue("[S2]", 1.0);
    atols = r.getIntegrator()->getAbsoluteToleranceVector();
    ASSERT_EQ(atols.size(), 2);
    EXPECT_NEAR(atols[0], 1e-26, 1e-31);
    EXPECT_NEAR(atols[1], 1e-27, 1e-31);

    r.setValue("S1", 1e100);
    r.setValue("S2", 1e-100);
    atols = r.getIntegrator()->getAbsoluteToleranceVector();
    ASSERT_EQ(atols.size(), 2);
    EXPECT_NEAR(atols[0], 1e88, 1e80);
    EXPECT_NEAR(atols[1], 1e-112, 1e-120);
}

TEST_F(CVODEIntegratorTests, CheckAbsToleranceNegValues) {
    TestModel* testModelPtr = TestModelFactory("SimpleFlux");

    // load model
    RoadRunner r(testModelPtr->str());

    r.setValue("default_compartment", -1e-15);
    r.setValue("[S1]", -10.0);
    r.setValue("[S2]", -1.0);
    std::vector<double> atols = r.getIntegrator()->getAbsoluteToleranceVector();
    ASSERT_EQ(atols.size(), 2);
    EXPECT_NEAR(atols[0], 1e-26, 1e-31);
    EXPECT_NEAR(atols[1], 1e-27, 1e-31);
}

TEST_F(CVODEIntegratorTests, CheckAbsToleranceSetDouble) {
    TestModel* testModelPtr = TestModelFactory("SimpleFlux");

    // load model
    RoadRunner r(testModelPtr->str());
    r.getIntegrator()->setValue("absolute_tolerance", 1e-15);
    std::vector<double> atols = r.getIntegrator()->getAbsoluteToleranceVector();
    ASSERT_EQ(atols.size(), 2);
    EXPECT_NEAR(atols[0], 1e-14, 1e-19);
    EXPECT_NEAR(atols[1], 1e-15, 1e-19);

    r.setValue("init(default_compartment)", 1e-10);
    r.setValue("init([S1])", 10.0);
    r.setValue("init([S2])", 1.0);
    atols = r.getIntegrator()->getAbsoluteToleranceVector();
    ASSERT_EQ(atols.size(), 2);
    EXPECT_NEAR(atols[0], 1e-24, 1e-29);
    EXPECT_NEAR(atols[1], 1e-25, 1e-29);

    r.setValue("default_compartment", 1e-15);
    r.setValue("[S1]", 10.0);
    r.setValue("[S2]", 1.0);
    atols = r.getIntegrator()->getAbsoluteToleranceVector();
    ASSERT_EQ(atols.size(), 2);
    EXPECT_NEAR(atols[0], 1e-29, 1e-34);
    EXPECT_NEAR(atols[1], 1e-30, 1e-34);
}

TEST_F(CVODEIntegratorTests, CheckAbsToleranceSetVector) {
    TestModel* testModelPtr = TestModelFactory("SimpleFlux");

    // load model
    RoadRunner r(testModelPtr->str());
    std::vector<double> orig = r.getIntegrator()->getAbsoluteToleranceVector();
    r.getIntegrator()->setValue("absolute_tolerance", orig);
    std::vector<double> atols = r.getIntegrator()->getAbsoluteToleranceVector();
    ASSERT_EQ(atols.size(), 2);
    EXPECT_NEAR(atols[0], 1e-11, 1e-16);
    EXPECT_NEAR(atols[1], 1e-12, 1e-16);

    orig[0] = 1e10;
    orig[1] = 1e-10;
    r.getIntegrator()->setValue("absolute_tolerance", orig);
    atols = r.getIntegrator()->getAbsoluteToleranceVector();
    ASSERT_EQ(atols.size(), 2);
    EXPECT_NEAR(atols[0], 1e10, 1.0);
    EXPECT_NEAR(atols[1], 1e-10, 1e-20);

    //Setting the values of S1 and S2 will no longer change the tolerance vector.
    r.setValue("S1", 1e100);
    r.setValue("S2", 1e-100);
    atols = r.getIntegrator()->getAbsoluteToleranceVector();
    ASSERT_EQ(atols.size(), 2);
    EXPECT_NEAR(atols[0], 1e10, 1.0);
    EXPECT_NEAR(atols[1], 1e-10, 1e-20);
}
