#include "gtest/gtest.h"

#include <unordered_map>
#include "SteadyStateIntegrationTests.h"
#include "TestModelFactory.h"


/**
 * @brief tests in this fixture all solve for steady state using the
 * "basic" strategy. This corresponds to setting the KIN_NONE flag
 * in kinsol.
 */
class SundialsNewtonIterationTests : public SteadyStateIntegrationTest {
public:

    SundialsNewtonIterationTests() = default;
};

TEST_F(SundialsNewtonIterationTests, CheckDecoratorRemovedAfterSolving) {
    // setup with rr
    OpenLinearFlux testModel;
    RoadRunner rr(testModel.str());

    rr.setSteadyStateSolver("newton");
    BasicDictionary opt;

    // will add PresimulationDecorator inside rr.steadyState()
    opt.setItem("allow_presimulation", true);

    rr.steadyState(&opt);
    // this string would be "Presimulation(NewtonIteration)" if decorator was not removed before setadyState returns.
    ASSERT_STREQ("newton", rr.getSteadyStateSolver()->getName().c_str());
}

TEST_F(SundialsNewtonIterationTests, CheckDecoratorRemovedAfterFailureToSolve) {
    OpenLinearFlux testModel;
    RoadRunner rr(testModel.str());

    rr.setSteadyStateSolver("newton");
    rr.getSteadyStateSolver()->setValue("allow_presimulation", false);
    rr.getSteadyStateSolver()->setValue("allow_approx", false);

    try {
        rr.steadyState();
    } catch (std::runtime_error &err) {
        ASSERT_STREQ("newton", rr.getSteadyStateSolver()->getName().c_str());
    } catch (std::exception &e) {
        ASSERT_FALSE("Test Failed") << e.what(); // should never get here
    }

}


/**
 * The OpenLinearFlux does not converge from the initial starting position (0, 0).
 * In this situation we raise an error. Here we test that it does
 */
TEST_F(SundialsNewtonIterationTests, CheckRaiseErrorWhenNotConverge) {
    // setup with rr
    OpenLinearFlux testModel;
    RoadRunner rr(testModel.str());
    rr.setSteadyStateSolver("newton");
    rr.getSteadyStateSolver()->setValue("allow_presimulation", false);
    rr.getSteadyStateSolver()->setValue("allow_approx", false);
    ASSERT_THROW(rr.steadyState(), std::runtime_error);
}

TEST_F(SundialsNewtonIterationTests, CheckNewtonIterationIsARegisteredSolver) {
    OpenLinearFlux testModel;
    RoadRunner rr(testModel.str());
    const auto &solverNames = rr.getRegisteredSteadyStateSolverNames();

    // name of bool should be expressive; it shows up on gtest failure.
    bool foundNewtonIterationInRegisteredSolverNames = false;
    if (std::find(solverNames.begin(), solverNames.end(), "newton") != solverNames.end()) {
        foundNewtonIterationInRegisteredSolverNames = true;
    }
    ASSERT_TRUE(foundNewtonIterationInRegisteredSolverNames);
}

/**
 * OpenLinearFlux (-> S1 -> S2 ->) does not require moiety conservation
 * but the NewtonIteration algorithm does not converge when starting
 * values are S1=0, S2=0. To solve this problem we use presimulation.
 */
TEST_F(SundialsNewtonIterationTests, CheckCorrectSteadyStateOpenLinearFlux) {
    testSteadyState(
            "OpenLinearFlux",
            "newton"
    );
}

TEST_F(SundialsNewtonIterationTests, CheckCorrectSteadyStateSimpleFluxManuallyReduced) {
    testSteadyState(
            "SimpleFluxManuallyReduced",
            "newton");
}

TEST_F(SundialsNewtonIterationTests, CheckCorrectSteadyStateSimpleFlux) {
    testSteadyState("SimpleFlux", "newton");
}

TEST_F(SundialsNewtonIterationTests, CheckCorrectSteadyStateVenkatraman2010) {
    testSteadyState("Venkatraman2010", "newton");
}

/**
 *
 */
TEST_F(SundialsNewtonIterationTests, DISABLED_CheckCorrectSteadyStateBrown2004) {
    testSteadyState("Brown2004", "newton");
}

TEST_F(SundialsNewtonIterationTests, CheckCorrectSteadyStateBimolecularEnd) {
    testSteadyState("BimolecularEnd", "newton");
}

TEST_F(SundialsNewtonIterationTests, CheckCorrectSteadyStateFluxesBimolecularEnd) {
    testSteadyStateFluxes("BimolecularEnd", "newton");
}
