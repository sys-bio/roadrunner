#include "gtest/gtest.h"
#include "SteadyStateIntegrationTests.h"
#include "TestModelFactory.h"

class NLEQ2IntetrationTests : public SteadyStateIntegrationTest {
public:

    NLEQ2IntetrationTests() : SteadyStateIntegrationTest() {};
};

/**
 * OpenLinearFlux (-> S1 -> S2 ->) does not require moiety conservation
 * but the NewtonIteration algorithm does not converge when starting
 * values are S1=0, S2=0. To solve this problem we use presimulation.
 */
TEST_F(NLEQ2IntetrationTests, CheckCorrectSteadyStateOpenLinearFlux) {
    testSteadyState(
            "OpenLinearFlux",
            "nleq2"
    );
}

TEST_F(NLEQ2IntetrationTests, CheckCorrectSteadyStateSimpleFluxManuallyReduced) {
    testSteadyState(
            "SimpleFluxManuallyReduced",
            "nleq2");
}

TEST_F(NLEQ2IntetrationTests, CheckCorrectSteadyStateSimpleFlux) {
    testSteadyState("SimpleFlux", "nleq2");
}

TEST_F(NLEQ2IntetrationTests, CheckCorrectSteadyStateVenkatraman2010) {
    testSteadyState("Venkatraman2010", "nleq2");
}


TEST_F(NLEQ2IntetrationTests, CheckCorrectSteadyStateBrown2004) {
    testSteadyState("Brown2004", "nleq2");
}

TEST_F(NLEQ2IntetrationTests, CheckCorrectSteadyStateBimolecularEnd) {
    testSteadyState("BimolecularEnd", "nleq2");
}

TEST_F(NLEQ2IntetrationTests, CheckCorrectSteadyStateFluxBimolecularEnd) {
    testSteadyStateFluxes("BimolecularEnd", "nleq2");
}
