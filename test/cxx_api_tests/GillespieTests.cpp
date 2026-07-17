//
// Created by Ciaran on 07/05/2021.
//

#include <rrRoadRunner.h>
#include "gtest/gtest.h"

#include "TestModelFactory.h"

#include "GillespieIntegrator.h"
#include "rrConfig.h"
#include "RoadRunnerTest.h"
#include "C/rrc_api.h"

#include <cmath>
#include <string>
#include <vector>
using namespace rr;
using namespace rrc;

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
    RoadRunner rr(TimeDependentImmigrationSBML);
    rr.setIntegrator("gillespie");
    rr.getIntegrator()->setValue("variable_step_size", false);
    rr.setSelections(std::vector<std::string>{"time", "N"});

    const double A = rr.getValue("A"), c = rr.getValue("cc");
    const double d = rr.getValue("dd"), mu = rr.getValue("mu");
    // closed-form mean of m'(t) = A(e^{c t}-e^{d t}) - mu m,  m(0)=0
    auto odeMean = [&](double t) {
      return A * ((std::exp(c * t) - std::exp(-mu * t)) / (c + mu)
        - (std::exp(d * t) - std::exp(-mu * t)) / (d + mu));
      };

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

/**
 * Regression tests for https://github.com/sys-bio/roadrunner/issues/1337
 *
 * The C API's Gillespie-averaging entry points - gillespieMeanOnGrid(Ex) and
 * gillespieMeanSDOnGrid(Ex) in wrappers/C/rrc_api.cpp - have two independent
 * bugs, both reproduced below with the S1 -> S2 mass-action model from the
 * issue (k1 = 0.5, S1(0) = 100). Each of these functions runs its own fresh
 * batch of `numberOfSimulations` Gillespie simulations internally (that's the
 * whole point of the numberOfSimulations parameter), so unlike the plain
 * gillespie()/gillespieOnGrid() calls, nothing about them should ever depend
 * on some earlier, unrelated simulation having already run:
 *
 * 1. They used to size their accumulator matrix from
 *    RoadRunner::getSimulationData(), which is a default-constructed (0x0,
 *    null-backed) matrix until *some* simulation has populated it, and
 *    otherwise just holds whatever shape that last simulation happened to
 *    have. If nothing had run yet, the very first write into that 0x0
 *    accumulator dereferenced a null pointer and crashed the process - even
 *    if the "gillespie" integrator had already been selected. If something
 *    unrelated *had* run, but on a different grid, the accumulator was
 *    silently the wrong shape. Both are fixed by sizing directly from the
 *    requested grid and the model's current selection list, which are known
 *    up front and don't require having run anything.
 *
 * 2. In the accumulation loop, every write targeted column 0 unconditionally
 *    (`avg(j, 0)`) instead of looping over all columns. Column 0 is time, so
 *    only the time column was ever averaged; every species column was left
 *    at its zero-initialized value. gillespieMeanSDOnGrid had the same
 *    column-0 bug for its variance accumulator, and on top of that never
 *    finished the Welford calculation (divide by n-1, sqrt) or copied it
 *    into the result - it copied the (broken) mean into Weights instead.
 */
static const std::string S1ToS2MassActionSBML = R"(<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4"><model id="s1_s2">
 <listOfCompartments><compartment id="c" size="1"/></listOfCompartments>
 <listOfSpecies>
  <species id="S1" compartment="c" initialAmount="100" hasOnlySubstanceUnits="true"/>
  <species id="S2" compartment="c" initialAmount="0"   hasOnlySubstanceUnits="true"/>
 </listOfSpecies>
 <listOfParameters><parameter id="k1" value="0.5"/></listOfParameters>
 <listOfReactions>
  <reaction id="R1" reversible="false">
   <listOfReactants><speciesReference species="S1"/></listOfReactants>
   <listOfProducts><speciesReference species="S2"/></listOfProducts>
   <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>k1</ci><ci>S1</ci></apply></math></kineticLaw>
  </reaction>
 </listOfReactions>
</model></sbml>)";

// Finds the column in an RRCData result whose header contains `name` (e.g.
// "S1"), returning -1 if not found. Used instead of a hardcoded index so
// these tests don't depend on the exact default column ordering/formatting.
static int findColumnContaining(RRCDataPtr data, const std::string &name) {
    for (int i = 0; i < data->CSize; i++) {
        if (std::string(data->ColumnHeaders[i]).find(name) != std::string::npos) {
            return i;
        }
    }
    return -1;
}

TEST_F(GillespieTests, MeanOnGridWorksWithoutAnyPriorSimulation) {
    RRHandle rrHandle = createRRInstance();
    ASSERT_TRUE(loadSBML(rrHandle, S1ToS2MassActionSBML.c_str()));

    // No gillespie()/gillespieOnGrid()/gillespieOnGridEx() call has happened
    // yet, and none should be needed: gillespieMeanOnGridEx runs its own
    // batch of simulations. Before the fix this dereferenced a null pointer
    // (issue #1337, bug 1).
    RRCDataPtr meanResult = gillespieMeanOnGridEx(rrHandle, 0, 5, 6, 3);
    ASSERT_NE(meanResult, nullptr);
    EXPECT_EQ(meanResult->RSize, 6);
    EXPECT_GT(meanResult->CSize, 1);
    freeRRCData(meanResult);

    // Same for the mean+SD variant, on a fresh handle so there's still no
    // prior simulation of any kind.
    RRHandle rrHandle2 = createRRInstance();
    ASSERT_TRUE(loadSBML(rrHandle2, S1ToS2MassActionSBML.c_str()));
    RRCDataPtr meanSDResult = gillespieMeanSDOnGridEx(rrHandle2, 0, 5, 6, 3);
    ASSERT_NE(meanSDResult, nullptr);
    EXPECT_EQ(meanSDResult->RSize, 6);
    EXPECT_GT(meanSDResult->CSize, 1);
    freeRRCData(meanSDResult);

    freeRRInstance(rrHandle);
    freeRRInstance(rrHandle2);
}

TEST_F(GillespieTests, MeanOnGridIgnoresUnrelatedPriorSimulationShape) {
    RRHandle rrHandle = createRRInstance();
    ASSERT_TRUE(loadSBML(rrHandle, S1ToS2MassActionSBML.c_str()));

    // Run something unrelated first, on a coarser grid (3 points) than what's
    // requested below (51 points). Under the old code, sizing the accumulator
    // from this leftover result would silently produce a wrong-shaped result
    // or, in the worst case, an out-of-bounds write.
    RRCDataPtr warmup = gillespieOnGridEx(rrHandle, 0, 5, 3);
    ASSERT_NE(warmup, nullptr);
    freeRRCData(warmup);

    RRCDataPtr result = gillespieMeanOnGridEx(rrHandle, 0, 5, 51, 3);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->RSize, 51);

    freeRRCData(result);
    freeRRInstance(rrHandle);
}

TEST_F(GillespieTests, MeanOnGridAveragesAllSpeciesColumns) {
    RRHandle rrHandle = createRRInstance();
    ASSERT_TRUE(loadSBML(rrHandle, S1ToS2MassActionSBML.c_str()));

    RRCDataPtr result = gillespieMeanOnGridEx(rrHandle, 0, 5, 6, 10);
    ASSERT_NE(result, nullptr);

    int s1Col = findColumnContaining(result, "S1");
    int s2Col = findColumnContaining(result, "S2");
    ASSERT_GE(s1Col, 0);
    ASSERT_GE(s2Col, 0);

    // Every replicate starts at the same, exactly known initial condition, so
    // the row-0 average must equal it precisely, regardless of the Gillespie
    // RNG. Before the fix, only column 0 (time) was ever accumulated into, so
    // every species column - including this one - stayed at 0.
    EXPECT_DOUBLE_EQ(result->Data[0 * result->CSize + s1Col], 100.0);
    EXPECT_DOUBLE_EQ(result->Data[0 * result->CSize + s2Col], 0.0);

    // Mass is conserved by the single S1 -> S2 reaction, so S1 + S2 == 100 in
    // every individual trajectory, and therefore in their average too. This
    // fails under the old code, where both columns average out to 0.
    for (int row = 0; row < result->RSize; row++) {
        double s1 = result->Data[row * result->CSize + s1Col];
        double s2 = result->Data[row * result->CSize + s2Col];
        EXPECT_NEAR(s1 + s2, 100.0, 1e-9) << "row " << row;
    }

    freeRRCData(result);
    freeRRInstance(rrHandle);
}

TEST_F(GillespieTests, MeanSDOnGridReturnsMeanAndRealStandardDeviationTogether) {
    RRHandle rrHandle = createRRInstance();
    ASSERT_TRUE(loadSBML(rrHandle, S1ToS2MassActionSBML.c_str()));

    // gillespieMeanSDOnGrid is the "single call" way to get both the mean and
    // the standard deviation from the same batch of runs (they're accumulated
    // together in one pass over `numberOfSimulations` fresh simulations).
    RRCDataPtr result = gillespieMeanSDOnGridEx(rrHandle, 0, 5, 6, 30);
    ASSERT_NE(result, nullptr);
    ASSERT_NE(result->Weights, nullptr);

    int s1Col = findColumnContaining(result, "S1");
    ASSERT_GE(s1Col, 0);

    // The mean (Data) must be correct, same as gillespieMeanOnGrid.
    EXPECT_DOUBLE_EQ(result->Data[0 * result->CSize + s1Col], 100.0);

    // At t=0 every replicate is identical, so S1's standard deviation (Weights)
    // must be exactly zero there.
    EXPECT_DOUBLE_EQ(result->Weights[0 * result->CSize + s1Col], 0.0);

    // By the final time point, 30 independent stochastic trajectories of a
    // decaying species must disagree with each other, i.e. have a strictly
    // positive standard deviation. Before the fix, Weights was just a second
    // copy of the (broken, all-zero) mean, so this was 0 too.
    int lastRow = result->RSize - 1;
    EXPECT_GT(result->Weights[lastRow * result->CSize + s1Col], 0.0);

    freeRRCData(result);
    freeRRInstance(rrHandle);
}

