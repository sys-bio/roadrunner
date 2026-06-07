//
// Created by Ciaran on 07/05/2021.
//

#include <rrRoadRunner.h>
#include "gtest/gtest.h"

#include "TestModelFactory.h"

#include "GillespieIntegrator.h"
#include "rrConfig.h"
#include "RoadRunnerTest.h"

#include <cmath>
#include <vector>
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

/**
 * Regression test for https://github.com/sys-bio/roadrunner/issues/1318
 *
 * An immigration-death process whose immigration rate is an explicit function
 * of time: N is produced at rate lam(t) = A*(exp(c t) - exp(d t)) (zeroth order
 * in N, and zero at t = 0) and removed at rate mu*N.  Because the system is
 * affine, the ensemble mean satisfies exactly  m'(t) = lam(t) - mu m,  whose
 * closed-form solution is the oracle below.  The direct method sampled the
 * waiting time from a propensity frozen at the start of each reporting interval:
 * at t = 0 the propensity is zero, so it jumped straight to the next reporting
 * time leaving N pinned at 0 for the whole first interval, and it then trailed
 * the true mean by about one interval for the rest of the run.  Integrating the
 * (time-varying) propensity over the waiting time removes the lag.
 */
static const std::string TimeDependentImmigrationSBML = R"(<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4"><model id="td">
 <listOfCompartments><compartment id="c" size="1"/></listOfCompartments>
 <listOfSpecies>
  <species id="N" compartment="c" initialAmount="0" hasOnlySubstanceUnits="true"/>
 </listOfSpecies>
 <listOfParameters>
  <parameter id="A"  value="10"/>
  <parameter id="cc" value="-0.2156"/>
  <parameter id="dd" value="-0.783"/>
  <parameter id="mu" value="0.5"/>
  <parameter id="lam" value="0" constant="false"/>
 </listOfParameters>
 <listOfRules>
  <assignmentRule variable="lam">
   <math xmlns="http://www.w3.org/1998/Math/MathML">
    <apply><times/><ci>A</ci>
     <apply><minus/>
      <apply><exp/><apply><times/><ci>cc</ci><csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time"> t </csymbol></apply></apply>
      <apply><exp/><apply><times/><ci>dd</ci><csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time"> t </csymbol></apply></apply>
     </apply>
    </apply>
   </math>
  </assignmentRule>
 </listOfRules>
 <listOfReactions>
  <reaction id="Birth" reversible="false">
   <listOfProducts><speciesReference species="N"/></listOfProducts>
   <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><ci>lam</ci></math></kineticLaw>
  </reaction>
  <reaction id="Death" reversible="false">
   <listOfReactants><speciesReference species="N"/></listOfReactants>
   <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>mu</ci><ci>N</ci></apply></math></kineticLaw>
  </reaction>
 </listOfReactions>
</model></sbml>)";

TEST_F(GillespieTests, TimeDependentPropensityMatchesODE) {
    // closed-form mean of m'(t) = A(e^{c t}-e^{d t}) - mu m,  m(0)=0
    const double A = 10.0, c = -0.2156, d = -0.783, mu = 0.5;
    auto odeMean = [&](double t) {
        return A * ((std::exp(c * t) - std::exp(-mu * t)) / (c + mu)
                  - (std::exp(d * t) - std::exp(-mu * t)) / (d + mu));
    };

    RoadRunner rr(TimeDependentImmigrationSBML);
    rr.setIntegrator("gillespie");
    rr.getIntegrator()->setValue("variable_step_size", false);
    rr.setSelections(std::vector<std::string>{"time", "N"});

    const int nrep = 300;
    const int npts = 6;                 // coarse grid: t = 0, 2, 4, 6, 8, 10
    std::vector<double> mean(npts, 0.0);
    for (int seed = 1; seed <= nrep; ++seed) {
        rr.reset();
        rr.getIntegrator()->setValue("seed", seed);
        const ls::DoubleMatrix* results = rr.simulate(0, 10, npts);
        for (int row = 0; row < npts; ++row)
            mean[row] += results->Element(row, 1);
    }
    for (int row = 0; row < npts; ++row)
        mean[row] /= nrep;

    // The SSA mean must track the analytic ODE mean at every reporting time.
    // Before the fix the t = 2 value was pinned at ~0 (frozen first interval)
    // against an analytic ~4.3, far outside this tolerance.
    for (int row = 1; row < npts; ++row) {
        double t = 10.0 * row / (npts - 1);
        EXPECT_NEAR(mean[row], odeMean(t), 0.7)
            << "SSA mean " << mean[row] << " at t=" << t
            << " does not track the ODE mean " << odeMean(t)
            << " (time-dependent propensity not integrated; issue #1318).";
    }
}

