
#include <rrRoadRunner.h>
#include <CVODEIntegrator.h>
#include "gtest/gtest.h"
#include "SteadyStateSolverDecorator.h"
#include "ApproxSteadyStateDecorator.h"
#include "BasicNewtonIteration.h"
#include "TestModelFactory.h"

using namespace rr;

class ApproxSteadyStateDecoratorTests : public ::testing::Test {
public:
    OpenLinearFlux testModel = OpenLinearFlux();

    ApproxSteadyStateDecoratorTests() {}

    void checkResults(RoadRunner &rr) {
        const auto &expectedResult = testModel.steadyState();
        for (auto &r: expectedResult) {
            const auto &speciesName = r.first;
            const auto &expectedValue = r.second;
            const auto &actualValue = rr.getValue(speciesName);
            ASSERT_NEAR(expectedValue, actualValue, 1e-5);
        }
    }
};

TEST_F(ApproxSteadyStateDecoratorTests, CheckModelNotNullptr) {
    // setup with rr
    RoadRunner rr(testModel.str());
    ASSERT_NE(nullptr, rr.getModel());

    // create steady state solver
    SteadyStateSolver *solver = nullptr;
    BasicNewtonIteration basicNewtonIteration(rr.getModel());
    ASSERT_NE(nullptr, basicNewtonIteration.getModel());
    solver = &basicNewtonIteration;
    ASSERT_NE(nullptr, solver->getModel());

    // wrap solver in decorator
    ApproxSteadyStateDecorator decorator(solver);
    ASSERT_NE(nullptr, decorator.getModel());
}

TEST_F(ApproxSteadyStateDecoratorTests, CheckNameField) {
    // setup with rr
    RoadRunner rr(testModel.str());

    // create steady state solver
    SteadyStateSolver *solver = nullptr;
    BasicNewtonIteration basicNewtonIteration(rr.getModel());
    solver = &basicNewtonIteration;

    ApproxSteadyStateDecorator decorator(solver);
    solver = &decorator;
    ASSERT_STREQ("Approximate(newton)", solver->getName().c_str());
}

/**
 * The approximation bits also require presimulation. Its a shame
 * we cannot just use the PresimulationDecorator for this but
 * we must have roadrunner respect the approx_time variable for
 * backwards compatibility.
 */
TEST_F(ApproxSteadyStateDecoratorTests, CheckModelTimeAfterPresimulation) {
    RoadRunner rr(testModel.str());
    ASSERT_EQ(0, rr.getModel()->getTime());

    SteadyStateSolver *solver;
    BasicNewtonIteration basicNewtonIteration(rr.getModel());

    basicNewtonIteration.setValue("approx_time", 200);
    solver = &basicNewtonIteration;
    ApproxSteadyStateDecorator presim(solver);
    solver = &presim;
    solver->solve();
    ASSERT_EQ(200, rr.getModel()->getTime());
}

/**
 * This model does not converge to steady state from starting position.
 * We can approximate the steady state at a threshold of 1e-6 using a time
 * point (approx_time) of 200
 */
TEST_F(ApproxSteadyStateDecoratorTests, CheckSteadyStateValuesAfterApproximation) {

    RoadRunner rr(testModel.str());

    SteadyStateSolver *solver;
    BasicNewtonIteration basicNewtonIteration(rr.getModel());

    basicNewtonIteration.setValue("approx_time", 200);
    basicNewtonIteration.setValue("approx_tolerance", 1e-6);

    // note, only 1 step is integrated between interval: ((approx_maximum_steps - stepsize), approx_maximum_steps)
    basicNewtonIteration.setValue("approx_maximum_steps", 1000);

    solver = &basicNewtonIteration;
    ApproxSteadyStateDecorator presim(solver);
    solver = &presim;
    solver->solve();

    checkResults(rr);
}


TEST_F(ApproxSteadyStateDecoratorTests, CheckApproxSteadyStateDecoratorHasAccessToSolverVariables) {

    RoadRunner rr(testModel.str());

    BasicNewtonIteration *basicNewtonIteration = new BasicNewtonIteration(rr.getModel());

    basicNewtonIteration->setValue("approx_time", 12.67);
    ASSERT_EQ(12.67, basicNewtonIteration->getValueAsDouble("approx_time"));

    SteadyStateSolver *solver = basicNewtonIteration;
    ApproxSteadyStateDecorator decorator(solver);
    solver = &decorator;
    solver->getValueAsDouble("approx_time");
    ASSERT_EQ(12.67, solver->getValueAsDouble("approx_time"));
    delete basicNewtonIteration;
}

TEST_F(ApproxSteadyStateDecoratorTests, CheckSolveTwiceTwoRRInstances) {

    // make independent copies of input string to ensure these
    // two rr instances are independent.
    std::string sbml1 = testModel.str();
    std::string sbml2 = testModel.str();

    RoadRunner rr1(sbml1);
    RoadRunner rr2(sbml2);

    BasicNewtonIteration basicNewtonIteration1(rr1.getModel());
    BasicNewtonIteration basicNewtonIteration2(rr2.getModel());

    basicNewtonIteration1.setValue("approx_time", 200);
    basicNewtonIteration2.setValue("approx_time", 250);

    SteadyStateSolver *solver1 = &basicNewtonIteration1;
    SteadyStateSolver *solver2 = &basicNewtonIteration2;
    ApproxSteadyStateDecorator decorator1(solver1);
    ApproxSteadyStateDecorator decorator2(solver2);
    solver1 = &decorator1;
    solver2 = &decorator2;
    solver1->solve();
    solver2->solve();
    checkResults(rr1);
    checkResults(rr2);

}

TEST_F(ApproxSteadyStateDecoratorTests, CheckSolveTwiceBackToBack) {

    RoadRunner rr(testModel.str());
    BasicNewtonIteration *basicNewtonIteration = new BasicNewtonIteration(rr.getModel());
    basicNewtonIteration->setValue("approx_time", 250);

    SteadyStateSolver *solver = basicNewtonIteration;
    ApproxSteadyStateDecorator decorator1(solver);
    solver = &decorator1;
    solver->solve();
    checkResults(rr);

    rr.setValue("S1", 0);
    rr.setValue("S2", 0);

    solver->solve();
    checkResults(rr);

    delete basicNewtonIteration;

}


























