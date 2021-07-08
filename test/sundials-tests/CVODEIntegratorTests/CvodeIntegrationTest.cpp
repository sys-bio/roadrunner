

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
    checkModelSimulatesWithRoadRunner<SimpleFlux>("SimpleFlux");
}

TEST_F(CVODEIntegratorTests, TestModel269WithRoadRunner) {
    checkModelSimulatesWithRoadRunner<Model269>("Model269");
}

TEST_F(CVODEIntegratorTests, TestModel28WithRoadRunner) {
    checkModelSimulatesWithRoadRunner<Model28>("Model28");
}

TEST_F(CVODEIntegratorTests, TestFactorialInRateLawWithRoadRunner) {
    checkModelSimulatesWithRoadRunner<FactorialInRateLaw>("FactorialInRateLaw");
}

TEST_F(CVODEIntegratorTests, SimpleFluxWithRoadRunner) {
    checkModelSimulatesWithRoadRunner<SimpleFlux>("SimpleFlux");
}

TEST_F(CVODEIntegratorTests, OpenLinearFluxWithRoadRunner) {
    checkModelSimulatesWithRoadRunner<OpenLinearFlux>("OpenLinearFlux");
}




TEST_F(CVODEIntegratorTests, TestSimpleFluxWithModel) {
    SimpleFlux testModel;
    testModel.toFile(R"(D:\roadrunner\roadrunner\test\sundials-tests\CVODEIntegratorTests\SimpleFlux.sbml)");
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























