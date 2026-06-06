//
// Created by Ciaran on 07/05/2021.
//

#include <rrRoadRunner.h>
#include "gtest/gtest.h"

#include "TestModelFactory.h"

#include "GillespieIntegrator.h"
#include "rrConfig.h"
#include "RoadRunnerTest.h"

#include <algorithm>
#include <limits>
using namespace rr;

/**
 * This is a more of a stub test suite
 * than a full test suite at the moment.
 * For now, it only tests the things
 * that need a new test to fix a bug.
 */

class GillespieTests : public RoadRunnerTest{

public:
    OpenLinearFlux openLinearFlux;
    GillespieTests() = default;
};

TEST_F(GillespieTests, OddStoichiometries) {
    RoadRunner rr1((rrTestModelsDir_ / "SBMLFeatures" / "S1_S2_S2.xml").string());
    RoadRunner rr2((rrTestModelsDir_ / "SBMLFeatures" / "S1_2S2.xml").string());
    rr1.setIntegrator("gillespie");
    rr2.setIntegrator("gillespie");
    rr1.getIntegrator()->setValue("seed", 1234);
    rr2.getIntegrator()->setValue("seed", 1234);
    rr1.getFullStoichiometryMatrix();
    const ls::DoubleMatrix* results1 = rr1.simulate(0, 5, 20);
    const ls::DoubleMatrix* results2 = rr2.simulate(0, 5, 20);

    ASSERT_EQ(results1->RSize(), results2->RSize());
    ASSERT_EQ(results1->CSize(), results2->CSize());
    for (unsigned int row = 0; row < results1->RSize(); row++) {
        for (unsigned int col = 0; col < results1->CSize(); col++) {
            EXPECT_NEAR(results1->Element(row, col), results2->Element(row, col), 0.00001);
        }
    }
}

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

/**
 * Regression test for https://github.com/sys-bio/roadrunner/issues/1320
 *
 * A single irreversible reaction consumes X at a constant (zeroth-order)
 * propensity that does not depend on X.  Once X reaches 0 a correct
 * direct-method SSA can no longer fire the reaction, so X must floor at 0.
 * Previously the reaction kept firing because reactant availability was never
 * checked, driving the molecule count below zero.
 */
static const std::string ZerothOrderSinkSBML = R"(<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4"><model id="m">
 <listOfCompartments><compartment id="c" size="1"/></listOfCompartments>
 <listOfSpecies><species id="X" compartment="c" initialAmount="20" hasOnlySubstanceUnits="true"/></listOfSpecies>
 <listOfParameters><parameter id="k" value="10"/></listOfParameters>
 <listOfReactions><reaction id="R" reversible="false">
   <listOfReactants><speciesReference species="X"/></listOfReactants>
   <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><ci>k</ci></math></kineticLaw>
 </reaction></listOfReactions></model></sbml>)";

TEST_F(GillespieTests, ZerothOrderReactantStaysNonNegative) {
    RoadRunner rr(ZerothOrderSinkSBML);
    rr.setIntegrator("gillespie");
    rr.getIntegrator()->setValue("variable_step_size", false);
    rr.setSelections({"time", "X"});

    // nonnegative is on by default: X must never drop below zero, and because
    // the 20 initial molecules are always fully consumed within t = 10, the
    // floor of exactly zero is reached in every replicate.
    double globalMin = std::numeric_limits<double>::infinity();
    for (int seed = 1; seed <= 20; ++seed) {
        rr.reset();
        rr.getIntegrator()->setValue("seed", seed);
        const ls::DoubleMatrix* results = rr.simulate(0, 10, 11);
        for (unsigned int row = 0; row < results->RSize(); ++row) {
            double x = results->Element(row, 1);
            EXPECT_GE(x, 0.0) << "X went negative (" << x << ") at seed " << seed;
            globalMin = std::min(globalMin, x);
        }
    }
    EXPECT_EQ(globalMin, 0.0);
}

TEST_F(GillespieTests, NonnegativeDisabledReproducesNegativeAmounts) {
    // The pre-fix behavior remains available as an explicit opt-out: with the
    // guard disabled the rate law is evaluated literally, so the zeroth-order
    // sink drives X below zero.
    RoadRunner rr(ZerothOrderSinkSBML);
    rr.setIntegrator("gillespie");
    rr.getIntegrator()->setValue("variable_step_size", false);
    rr.getIntegrator()->setValue("nonnegative", false);
    rr.setSelections({"time", "X"});
    rr.reset();
    rr.getIntegrator()->setValue("seed", 1);

    const ls::DoubleMatrix* results = rr.simulate(0, 10, 11);
    double minX = std::numeric_limits<double>::infinity();
    for (unsigned int row = 0; row < results->RSize(); ++row)
        minX = std::min(minX, results->Element(row, 1));
    EXPECT_LT(minX, 0.0);
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

