//
// Created by Ciaran on 07/05/2021.
//

#include <rrRoadRunner.h>
#include "gtest/gtest.h"

#include "TestModelFactory.h"

#include "GillespieIntegrator.h"
#include "rrConfig.h"
#include "RoadRunnerTest.h"
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
 * Regression test for https://github.com/sys-bio/roadrunner/issues/1317
 *
 * A floating species M starts at a non-integer amount and only degrades.  The
 * value used here, 24.999999999999996, is what an initialAssignment that should
 * give 25 evaluates to in floating point; that is how the gene-expression model
 * in #1317 set its initial mRNA amounts.  Stepping down by whole molecules, M
 * crosses below zero by a rounding-error-sized amount (about -1e-15).  The
 * integrator used to treat any negative amount as fatal and set the simulation
 * time to infinity, which silently froze the entire trajectory, including an
 * independent two-state toggle that shares the model but is dynamically
 * decoupled from M.  Different seeds froze the toggle at different points, so its
 * stationary on-fraction came out badly biased instead of kon/(kon+koff) = 0.25.
 * With the halt removed the toggle equilibrates correctly regardless of M's
 * harmless dip below zero.
 */
static const std::string NonIntegerDegraderSBML = R"(<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4"><model id="m">
 <listOfCompartments><compartment id="c" size="1"/></listOfCompartments>
 <listOfSpecies>
  <species id="M"    compartment="c" initialAmount="24.999999999999996" hasOnlySubstanceUnits="true"/>
  <species id="Soff" compartment="c" initialAmount="1" hasOnlySubstanceUnits="true"/>
  <species id="Son"  compartment="c" initialAmount="0" hasOnlySubstanceUnits="true"/>
 </listOfSpecies>
 <listOfParameters>
  <parameter id="kdeg" value="100"/>
  <parameter id="kon"  value="0.25"/>
  <parameter id="koff" value="0.75"/>
 </listOfParameters>
 <listOfReactions>
  <reaction id="Rdeg" reversible="false">
   <listOfReactants><speciesReference species="M"/></listOfReactants>
   <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>kdeg</ci><ci>M</ci></apply></math></kineticLaw>
  </reaction>
  <reaction id="Ron" reversible="false">
   <listOfReactants><speciesReference species="Soff"/></listOfReactants>
   <listOfProducts><speciesReference species="Son"/></listOfProducts>
   <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>kon</ci><ci>Soff</ci></apply></math></kineticLaw>
  </reaction>
  <reaction id="Roff" reversible="false">
   <listOfReactants><speciesReference species="Son"/></listOfReactants>
   <listOfProducts><speciesReference species="Soff"/></listOfProducts>
   <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>koff</ci><ci>Son</ci></apply></math></kineticLaw>
  </reaction>
 </listOfReactions>
</model></sbml>)";

TEST_F(GillespieTests, NegativeAmountDoesNotFreezeRun) {
    RoadRunner rr(NonIntegerDegraderSBML);
    rr.setIntegrator("gillespie");
    rr.getIntegrator()->setValue("variable_step_size", false);
    rr.setSelections(std::vector<std::string>{"time", "Son"});

    // The toggle is decoupled from M, so its time-averaged on-fraction must
    // approach kon/(kon+koff) = 0.25.  M degrades far faster than the toggle
    // switches, so before the fix M's tiny dip below zero froze every run almost
    // immediately, pinning Son near its initial value of 0.  Average over
    // independent trajectories to keep the assertion away from per-seed
    // stochastic noise.
    double sum = 0.0;
    long rows = 0;
    const int nrep = 20;
    for (int seed = 1; seed <= nrep; ++seed) {
        rr.reset();
        rr.getIntegrator()->setValue("seed", seed);
        const ls::DoubleMatrix* results = rr.simulate(0, 4000, 2001);
        for (unsigned int row = 0; row < results->RSize(); ++row) {
            sum += results->Element(row, 1);
            ++rows;
        }
    }
    double onFraction = sum / static_cast<double>(rows);
    EXPECT_NEAR(onFraction, 0.25, 0.05)
        << "toggle on-fraction " << onFraction
        << " is far from the analytic 0.25; the run likely froze when M dipped "
           "below zero (issue #1317).";
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

