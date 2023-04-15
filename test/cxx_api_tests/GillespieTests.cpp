//
// Created by Ciaran on 07/05/2021.
//

#include <rrRoadRunner.h>
#include "gtest/gtest.h"

#include "TestModelFactory.h"

#include "GillespieIntegrator.h"
#include "rrConfig.h"
using namespace rr;

/**
 * This is a more of a stub test suite
 * than a full test suite at the moment.
 * For now, it only tests the things
 * that need a new test to fix a bug.
 */

class GillespieTests : public ::testing::Test{

public:
    OpenLinearFlux openLinearFlux;
    GillespieTests() = default;
};

TEST_F(GillespieTests, Seed){
    RoadRunner rr(openLinearFlux.str());
    rr.setIntegrator("gillespie");
    rr.getIntegrator()->setValue("seed", 1234);
    ASSERT_TRUE(rr.getIntegrator()->getValue("seed") == 1234);
}

TEST_F(GillespieTests, SetSeedUsingInactiveIntegrator){
    RoadRunner rr(openLinearFlux.str());
    Integrator* integrator = rr.getIntegratorByName("gillespie");
    integrator->setValue("seed", 4);
    rr.setIntegrator("gillespie");
    std::int64_t seed = rr.getIntegrator()->getValue("seed");
    ASSERT_EQ(4, seed);
}

TEST_F(GillespieTests, MaxStepSize) {
    RoadRunner rr(3, 1);
    rr.addCompartment("comp", 1, false);
    rr.addSpeciesAmount("S1", "comp", 2);
    rr.setIntegrator("gillespie");
    rr.getIntegrator()->setValue("maximum_time_step", 25.0);
    EXPECT_TRUE(rr.getIntegrator()->getValue("maximum_time_step") == 25.0);

    const ls::DoubleMatrix* results = rr.simulate(0, 55, 25);
    //'results' should have points at 0, 25, 50, and 55, because of the max step size.
    ASSERT_EQ(results->RSize(), 4);
    EXPECT_EQ(results->Element(0, 0), 0);
    EXPECT_NEAR(results->Element(1, 0), 25, 0.001);
    EXPECT_NEAR(results->Element(2, 0), 50, 0.001);
    EXPECT_NEAR(results->Element(3, 0), 55, 0.001);

    EXPECT_EQ(results->Element(0, 1), 2.0);
    EXPECT_EQ(results->Element(1, 1), 2.0);
    EXPECT_EQ(results->Element(2, 1), 2.0);
    EXPECT_EQ(results->Element(3, 1), 2.0);


    rr.getIntegrator()->setValue("maximum_time_step", 0.0);
    results = rr.simulate(0, 55, 25);
    ASSERT_EQ(results->RSize(), 2);
    EXPECT_EQ(results->Element(0, 0), 0);
    EXPECT_NEAR(results->Element(1, 0), 55, 0.001);


    rr.getIntegrator()->setValue("maximum_time_step", 100.0);
    results = rr.simulate(0, 55, 25);
    ASSERT_EQ(results->RSize(), 2);
    EXPECT_EQ(results->Element(0, 0), 0);
    EXPECT_NEAR(results->Element(1, 0), 55, 0.001);
}

TEST_F(GillespieTests, MaxNumSteps) {
    RoadRunner rr(openLinearFlux.str());
    rr.setIntegrator("gillespie");
    //First check if properly stops with fixed step sizes
    rr.getIntegrator()->setValue("variable_step_size", false);
    rr.getIntegrator()->setValue("maximum_num_steps", 3);
    EXPECT_TRUE(rr.getIntegrator()->getValue("maximum_num_steps") == 3);
    try
    {
        const ls::DoubleMatrix* results = rr.simulate(0, 500, 3);
        EXPECT_TRUE(false);
    }
    catch (exception& e)
    {
        EXPECT_STREQ(e.what(), "GillespieIntegrator::integrate failed:  max number of steps (3) reached before desired output at time 250.");
    }

    //Now check if properly stops with variable step sizes, but a minimum time step
    rr.reset();
    rr.getIntegrator()->setValue("variable_step_size", true);
    rr.getIntegrator()->setValue("minimum_time_step", 20.0);
    try
    {
        const ls::DoubleMatrix* results = rr.simulate(0, 500, 3);
        EXPECT_TRUE(false);
    }
    catch (exception& e)
    {
        EXPECT_STREQ(e.what(), "GillespieIntegrator::integrate failed:  max number of steps (3) reached before desired output at time 20.");
    }

    //Check to make sure if max_steps is large enough, things work.
    rr.reset();
    rr.getIntegrator()->setValue("maximum_num_steps", 100000000);
    EXPECT_TRUE(rr.getIntegrator()->getValue("maximum_num_steps") == 100000000);
    const ls::DoubleMatrix* results = rr.simulate(0, 50, 11);

    //Check to make sure max steps is effectively ignored if we have no minimum time step and variable step sizes
    rr.reset();
    rr.getIntegrator()->setValue("variable_step_size", true);
    rr.getIntegrator()->setValue("minimum_time_step", 0.0);
    rr.getIntegrator()->setValue("maximum_num_steps", 1);
    results = rr.simulate(0, 50, 11);
}

