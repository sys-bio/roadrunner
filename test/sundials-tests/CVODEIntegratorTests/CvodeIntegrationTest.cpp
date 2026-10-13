

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "rrRoadRunner.h"
#include "rrExecutableModel.h"
#include "CVODEIntegrator.h"
#include "Solver.h"
#include "TestModelFactory.h"
#include "Matrix.h"
#include "CvodeIntegrationTest.h"

#include <algorithm>
#include <cmath>
#include <string>


using namespace rr;
using namespace testing;


TEST_F(CVODEIntegratorTests, TestSimpleFluxWithRoadRunner) {
    checkModelSimulatesWithRoadRunner("SimpleFlux");
}

TEST_F(CVODEIntegratorTests, TestModel269WithRoadRunner) {
    checkModelSimulatesWithRoadRunner("Model269");
}

TEST_F(CVODEIntegratorTests, TestModel28WithRoadRunner) {
    checkModelSimulatesWithRoadRunner("Model28");
}

TEST_F(CVODEIntegratorTests, TestFactorialInRateLawWithRoadRunner) {
    checkModelSimulatesWithRoadRunner("FactorialInRateLaw");
}

TEST_F(CVODEIntegratorTests, SimpleFluxWithRoadRunner) {
    checkModelSimulatesWithRoadRunner("SimpleFlux");
}

TEST_F(CVODEIntegratorTests, OpenLinearFluxWithRoadRunner) {
    checkModelSimulatesWithRoadRunner("OpenLinearFlux");
}

TEST_F(CVODEIntegratorTests, OneEvent10TriggersWithRoadRunner) {
    checkModelSimulatesWithRoadRunner("OneEvent10Triggers");
}



/**
 * These tests use the LLVMExecutableModel instead of the RoadRunner interface.
 */

TEST_F(CVODEIntegratorTests, TestSimpleFluxWithModel) {
    SimpleFlux testModel;
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

TEST_F(CVODEIntegratorTests, FactorialInRateLawWithModel) {
    FactorialInRateLaw testModel;
    checkModelIntegrates<CVODEIntegrator>(&testModel);
}

TEST_F(CVODEIntegratorTests, OneEvent10Triggers) {
    OneEvent10Triggers testModel;
    checkModelIntegrates<CVODEIntegrator>(&testModel);
}


TEST_F(CVODEIntegratorTests, CheckAbsToleranceDefaults) {
    TestModel* testModelPtr = TestModelFactory("SimpleFlux");

    // load model
    RoadRunner r(testModelPtr->str());
    std::vector<double> atols = r.getIntegrator()->getAbsoluteToleranceVector();
    ASSERT_EQ(atols.size(), 2);
    EXPECT_NEAR(atols[0], 1e-11, 1e-16);
    EXPECT_NEAR(atols[1], 1e-12, 1e-16);

    r.setValue("init(default_compartment)", 1e-10);
    r.setValue("init([S1])", 10.0);
    r.setValue("init([S2])", 1.0);
    atols = r.getIntegrator()->getAbsoluteToleranceVector();
    ASSERT_EQ(atols.size(), 2);
    EXPECT_NEAR(atols[0], 1e-21, 1e-26);
    EXPECT_NEAR(atols[1], 1e-22, 1e-26);

    r.setValue("default_compartment", 1e-15);
    r.setValue("[S1]", 10.0);
    r.setValue("[S2]", 1.0);
    atols = r.getIntegrator()->getAbsoluteToleranceVector();
    ASSERT_EQ(atols.size(), 2);
    EXPECT_NEAR(atols[0], 1e-26, 1e-31);
    EXPECT_NEAR(atols[1], 1e-27, 1e-31);

    r.setValue("S1", 1e100);
    r.setValue("S2", 1e-100);
    atols = r.getIntegrator()->getAbsoluteToleranceVector();
    ASSERT_EQ(atols.size(), 2);
    EXPECT_NEAR(atols[0], 1e88, 1e80);
    EXPECT_NEAR(atols[1], 1e-112, 1e-120);
}

TEST_F(CVODEIntegratorTests, CheckAbsToleranceNegValues) {
    TestModel* testModelPtr = TestModelFactory("SimpleFlux");

    // load model
    RoadRunner r(testModelPtr->str());

    r.setValue("default_compartment", -1e-15);
    r.setValue("[S1]", -10.0);
    r.setValue("[S2]", -1.0);
    std::vector<double> atols = r.getIntegrator()->getAbsoluteToleranceVector();
    ASSERT_EQ(atols.size(), 2);
    EXPECT_NEAR(atols[0], 1e-26, 1e-31);
    EXPECT_NEAR(atols[1], 1e-27, 1e-31);
}

TEST_F(CVODEIntegratorTests, CheckAbsToleranceSetDouble) {
    TestModel* testModelPtr = TestModelFactory("SimpleFlux");

    // load model
    RoadRunner r(testModelPtr->str());
    r.getIntegrator()->setValue("absolute_tolerance", 1e-15);
    std::vector<double> atols = r.getIntegrator()->getAbsoluteToleranceVector();
    ASSERT_EQ(atols.size(), 2);
    EXPECT_NEAR(atols[0], 1e-14, 1e-19);
    EXPECT_NEAR(atols[1], 1e-15, 1e-19);

    r.setValue("init(default_compartment)", 1e-10);
    r.setValue("init([S1])", 10.0);
    r.setValue("init([S2])", 1.0);
    atols = r.getIntegrator()->getAbsoluteToleranceVector();
    ASSERT_EQ(atols.size(), 2);
    EXPECT_NEAR(atols[0], 1e-24, 1e-29);
    EXPECT_NEAR(atols[1], 1e-25, 1e-29);

    r.setValue("default_compartment", 1e-15);
    r.setValue("[S1]", 10.0);
    r.setValue("[S2]", 1.0);
    atols = r.getIntegrator()->getAbsoluteToleranceVector();
    ASSERT_EQ(atols.size(), 2);
    EXPECT_NEAR(atols[0], 1e-29, 1e-34);
    EXPECT_NEAR(atols[1], 1e-30, 1e-34);
}

TEST_F(CVODEIntegratorTests, CheckAbsToleranceSetVector) {
    TestModel* testModelPtr = TestModelFactory("SimpleFlux");

    // load model
    RoadRunner r(testModelPtr->str());
    std::vector<double> orig = r.getIntegrator()->getAbsoluteToleranceVector();
    r.getIntegrator()->setValue("absolute_tolerance", orig);
    std::vector<double> atols = r.getIntegrator()->getAbsoluteToleranceVector();
    ASSERT_EQ(atols.size(), 2);
    EXPECT_NEAR(atols[0], 1e-11, 1e-16);
    EXPECT_NEAR(atols[1], 1e-12, 1e-16);

    orig[0] = 1e10;
    orig[1] = 1e-10;
    r.getIntegrator()->setValue("absolute_tolerance", orig);
    atols = r.getIntegrator()->getAbsoluteToleranceVector();
    ASSERT_EQ(atols.size(), 2);
    EXPECT_NEAR(atols[0], 1e10, 1.0);
    EXPECT_NEAR(atols[1], 1e-10, 1e-20);

    //Setting the values of S1 and S2 will no longer change the tolerance vector.
    r.setValue("S1", 1e100);
    r.setValue("S2", 1e-100);
    atols = r.getIntegrator()->getAbsoluteToleranceVector();
    ASSERT_EQ(atols.size(), 2);
    EXPECT_NEAR(atols[0], 1e10, 1.0);
    EXPECT_NEAR(atols[1], 1e-10, 1e-20);
}

// S' = -0.5 S, S(0) = 10, exact solution 10 exp(-0.5 t)
static const std::string DecaySBML = R"(<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4"><model id="decay">
 <listOfCompartments><compartment id="c" size="1"/></listOfCompartments>
 <listOfSpecies>
  <species id="S" compartment="c" initialConcentration="10" hasOnlySubstanceUnits="false"/>
 </listOfSpecies>
 <listOfParameters><parameter id="k" value="0.5"/></listOfParameters>
 <listOfReactions>
  <reaction id="J0" reversible="false">
   <listOfReactants><speciesReference species="S"/></listOfReactants>
   <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>k</ci><ci>S</ci></apply></math></kineticLaw>
  </reaction>
 </listOfReactions>
</model></sbml>)";

static double decayRelativeError(double value, double t) {
    const double exact = 10. * std::exp(-0.5 * t);
    return std::fabs(value - exact) / exact;
}

// Largest relative error of S over the rows [first, last) of a (time, S) result.
static double maxDecayRelativeError(const ls::DoubleMatrix &result, unsigned first, unsigned last) {
    double worst = 0;
    for (unsigned i = first; i < last; i++) {
        worst = std::max(worst, decayRelativeError(result(i, 1), result(i, 0)));
    }
    return worst;
}

// variable step simulation from 0 to end, as simulate(0, end) does without a number of points
static ls::DoubleMatrix simulateDecayVariableStep(RoadRunner &rr, double end) {
    rr.getIntegrator()->setValue("variable_step_size", true);
    SimulateOptions &opt = rr.getSimulateOptions();
    opt.setStartTime(0);
    opt.setDuration(end);
    opt.setSteps(0);
    return *rr.simulate();
}

/**
 * Regression test for O-3: the last row of a CVODE variable step run was a linear
 * interpolation between the last two internal steps, whose error scales with the
 * square of the step size (about 1e-4 here whatever the tolerances), while the other
 * rows are within about 1e-8.
 */
TEST_F(CVODEIntegratorTests, VariableStepLastRowIsAsAccurateAsTheOthers) {
    for (double end : {10., 9.95, 5., 1.}) {
        RoadRunner rr(DecaySBML);
        rr.setIntegrator("cvode");
        rr.getIntegrator()->setValue("absolute_tolerance", 1e-12);
        rr.getIntegrator()->setValue("relative_tolerance", 1e-10);
        ls::DoubleMatrix result = simulateDecayVariableStep(rr, end);

        ASSERT_GT(result.numRows(), 2u);
        unsigned last = result.numRows() - 1;
        EXPECT_EQ(end, result(last, 0)) << "end " << end;
        EXPECT_LT(maxDecayRelativeError(result, 0, last), 1e-6) << "end " << end;
        EXPECT_LT(decayRelativeError(result(last, 1), result(last, 0)), 1e-6)
            << "end " << end << ": the last row is less accurate than the others";
    }
}

/**
 * Regression test for sys-bio/roadrunner#1362: a variable step simulate(0, end) left the
 * model at the end of the last internal step, past the requested end time, so the
 * model time and state were wrong and a simulation continued from there started from
 * the wrong state.
 */
TEST_F(CVODEIntegratorTests, VariableStepSimulationEndsAtTheRequestedEndTime) {
    RoadRunner rr(DecaySBML);
    rr.setIntegrator("cvode");
    rr.getIntegrator()->setValue("absolute_tolerance", 1e-12);
    rr.getIntegrator()->setValue("relative_tolerance", 1e-10);

    const double end = 10;
    ls::DoubleMatrix first = simulateDecayVariableStep(rr, end);
    ASSERT_GT(first.numRows(), 2u);
    EXPECT_EQ(end, first(first.numRows() - 1, 0));

    // the model itself is at the end time, not at the end of the last internal step
    EXPECT_EQ(end, rr.getCurrentTime());
    EXPECT_LT(decayRelativeError(rr.getValue("S"), end), 1e-6);

    // continuing from there gives the same solution as an uninterrupted run
    rr.getIntegrator()->setValue("variable_step_size", true);
    SimulateOptions &opt = rr.getSimulateOptions();
    opt.setStartTime(end);
    opt.setDuration(end);
    opt.setSteps(0);
    ls::DoubleMatrix second = *rr.simulate();

    ASSERT_GT(second.numRows(), 2u);
    unsigned last = second.numRows() - 1;
    EXPECT_EQ(end, second(0, 0));
    EXPECT_EQ(2 * end, second(last, 0));
    EXPECT_LT(maxDecayRelativeError(second, 0, last + 1), 1e-6);
    EXPECT_EQ(2 * end, rr.getCurrentTime());
}

// Same decay, with an event at time @T@ that sets S to 50.
static const std::string DecayWithEventSBML = R"(<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4"><model id="decay_with_event">
 <listOfCompartments><compartment id="c" size="1"/></listOfCompartments>
 <listOfSpecies>
  <species id="S" compartment="c" initialConcentration="10" hasOnlySubstanceUnits="false"/>
 </listOfSpecies>
 <listOfParameters><parameter id="k" value="0.5"/></listOfParameters>
 <listOfReactions>
  <reaction id="J0" reversible="false">
   <listOfReactants><speciesReference species="S"/></listOfReactants>
   <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>k</ci><ci>S</ci></apply></math></kineticLaw>
  </reaction>
 </listOfReactions>
 <listOfEvents>
  <event id="e1">
   <trigger><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><gt/><csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time"> time </csymbol><cn>@T@</cn></apply></math></trigger>
   <listOfEventAssignments>
    <eventAssignment variable="S"><math xmlns="http://www.w3.org/1998/Math/MathML"><cn>50</cn></math></eventAssignment>
   </listOfEventAssignments>
  </event>
 </listOfEvents>
</model></sbml>)";

static std::string decayWithEvent(double eventTime) {
    std::string sbml = DecayWithEventSBML;
    sbml.replace(sbml.find("@T@"), 3, std::to_string(eventTime));
    return sbml;
}

// The two rows that a variable step run reports for the event, the state before it at a
// time just before the event and the state after it at a time just after it.
struct EventRows {
    unsigned after = 0;      // index of the row after the event (S has been set to 50)
    double timeBefore = 0;   // event time - time of the row before it
    double timeAfter = 0;    // time of the row after it - event time
    bool sorted = true;      // the times of the whole result increase
};

static EventRows findEventRows(const ls::DoubleMatrix &result, double eventTime) {
    EventRows rows;
    for (unsigned i = 1; i < result.numRows(); i++) {
        if (result(i, 0) <= result(i - 1, 0)) {
            rows.sorted = false;
        }
    }
    for (unsigned i = 1; i < result.numRows(); i++) {
        if (result(i, 1) > 40) {
            rows.after = i;
            rows.timeBefore = eventTime - result(i - 1, 0);
            rows.timeAfter = result(i, 0) - eventTime;
            break;
        }
    }
    return rows;
}

/**
 * Regression test for the root tolerance of CVODEIntegrator::integrate, which was a function
 * level static: the first integration in the process set it (100 * 32 * epsilon * (start + step)
 * of that call), and it was then used to place the two rows of every event of every later
 * integration, so the rows of an event at time 1 were 3.6e-9 from it after an unrelated
 * integration from 0 to 10000, and 7e-12 from it without one.  When it exceeded the first
 * step after the event, the times of the output were not in order.
 *
 * (The old value was whatever the first integration in the process had set, so to see this test
 * fail without the fix, run it alone with --gtest_filter: then its long integration is the first.)
 */
TEST_F(CVODEIntegratorTests, VariableStepEventRowsDoNotDependOnEarlierIntegrations) {
    {
        // an unrelated, long integration first
        RoadRunner longRun(DecaySBML);
        longRun.getIntegrator()->setValue("variable_step_size", false);
        longRun.simulate(0, 10000, 3);
    }

    RoadRunner rr(decayWithEvent(1));
    ls::DoubleMatrix result = simulateDecayVariableStep(rr, 10);
    EventRows rows = findEventRows(result, 1);

    ASSERT_GT(rows.after, 0u) << "no row after the event";
    EXPECT_TRUE(rows.sorted);
    EXPECT_GT(rows.timeBefore, 0);
    EXPECT_LT(rows.timeBefore, 1e-11);
    EXPECT_GT(rows.timeAfter, 0);
    EXPECT_LT(rows.timeAfter, 1e-11);
    // the state just before the event (CVODE's default tolerances give about 1e-5 here), then the state the event sets
    EXPECT_NEAR(10. * std::exp(-0.5), result(rows.after - 1, 1), 1e-4);
    EXPECT_NEAR(50., result(rows.after, 1), 1e-6);
}

/**
 * The two rows of an event are placed at the time at which CVODE found the root, minus and plus
 * its root tolerance, which follows the time scale of the event (100 * 32 * epsilon * (t + step)).
 * CVODE finds the root of an event at 10000 up to about 1.5e-9 after the event, so with the old
 * offset of about 1e-11 the row "before" the event came after it (event time - row = -1.5e-9);
 * and an event at 10000 gets a much larger offset than an event at 1, not the same one.
 */
TEST_F(CVODEIntegratorTests, VariableStepEventRowOffsetFollowsTheTimeOfTheEvent) {
    double offsets[2];
    int n = 0;
    for (double eventTime : {1., 10000.}) {
        RoadRunner rr(decayWithEvent(eventTime));
        ls::DoubleMatrix result = simulateDecayVariableStep(rr, eventTime + 10);
        EventRows rows = findEventRows(result, eventTime);

        ASSERT_GT(rows.after, 0u) << "no row after the event at " << eventTime;
        EXPECT_TRUE(rows.sorted) << "event at " << eventTime;
        EXPECT_GT(rows.timeBefore, 0) << "event at " << eventTime;
        EXPECT_GT(rows.timeAfter, 0) << "event at " << eventTime;
        EXPECT_LT(rows.timeAfter, 1e-6 * eventTime) << "event at " << eventTime;
        offsets[n++] = rows.timeAfter;
    }
    EXPECT_GT(offsets[1], 100 * offsets[0]);
}
