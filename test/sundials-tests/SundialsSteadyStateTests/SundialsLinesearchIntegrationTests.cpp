#include "gtest/gtest.h"
#include "SteadyStateIntegrationTests.h"
#include "TestModelFactory.h"

#include "Setting.h"

/**
 * @brief tests in this fixture all solve for steady state using the
 * "linesearch" strategy. This corresponds to setting the KIN_LINESEARCH flag
 * in kinsol.
 */
class LineSearchNewtonIterationTests : public SteadyStateIntegrationTest {
public:

    LineSearchNewtonIterationTests() : SteadyStateIntegrationTest() {};
};

TEST_F(LineSearchNewtonIterationTests, CheckCorrectSteadyStateOpenLinearFlux) {
    testSteadyState("OpenLinearFlux", "newton_linesearch");
}

TEST_F(LineSearchNewtonIterationTests, CheckCorrectSteadyStateSimpleFluxManuallyReduced) {
    testSteadyState("SimpleFluxManuallyReduced",
                                               "newton_linesearch");
}

TEST_F(LineSearchNewtonIterationTests, CheckCorrectSteadyStateSimpleFlux) {
    testSteadyState("SimpleFlux", "newton_linesearch");
}

TEST_F(LineSearchNewtonIterationTests, CheckCorrectSteadyStateVenkatraman2010) {
    testSteadyState("Venkatraman2010", "newton_linesearch");
}

/**
 */
TEST_F(LineSearchNewtonIterationTests, DISABLED_CheckCorrectSteadyStateBrown2004) {
    testSteadyState("Brown2004", "newton_linesearch");
}

TEST_F(LineSearchNewtonIterationTests, CheckCorrectSteadyStateBimolecularEnd) {
    testSteadyState("BimolecularEnd", "newton_linesearch");
}

TEST_F(LineSearchNewtonIterationTests, CheckCorrectSteadyStateFluxBimolecularEnd) {
    testSteadyStateFluxes("BimolecularEnd", "newton_linesearch");
}
