

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "rrRoadRunner.h"
#include "rrExecutableModel.h"
#include "CVODEIntegrator.h"
#include <cvodes/cvodes.h>
#include "Solver.h"
#include "TestModelFactory.h"
#include "Matrix.h"
#include "CvodeIntegrationTest.h"
#include "RoadRunnerTest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>


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
 * CVODE sets its counters to zero when it is reinitialized after an event. The
 * statistics of the integrator are summed over the reinitializations (issue 1411).
 */
TEST_F(CVODEIntegratorTests, StatisticsAreSummedOverReinitializations) {
    RoadRunner r(OneEvent10Triggers().str());
    r.simulate(0, 10, 101);
    auto *cvode = dynamic_cast<CVODEIntegrator *>(r.getIntegrator());
    ASSERT_TRUE(cvode);
    std::unordered_map<std::string, Setting> stats = cvode->getStatistics();

    long stepsSinceLastReinit = 0;
    ASSERT_EQ(CV_SUCCESS, CVodeGetNumSteps(cvode->getCvodeMemory(), &stepsSinceLastReinit));
    ASSERT_GT(stats["num_steps"].getAs<std::int64_t>(), stepsSinceLastReinit);

    std::int64_t numEvents = stats["num_events"].getAs<std::int64_t>();
    ASSERT_GT(numEvents, 1);
    ASSERT_EQ(numEvents, stats["num_reinits"].getAs<std::int64_t>());

    // a new simulation starts from zero
    r.reset();
    r.simulate(0, 10, 101);
    std::unordered_map<std::string, Setting> again = cvode->getStatistics();
    for (const std::string key: {"num_steps", "num_rhs_evals", "num_jac_evals", "num_events", "num_reinits"}) {
        ASSERT_EQ(stats[key].getAs<std::int64_t>(), again[key].getAs<std::int64_t>()) << key;
    }

    cvode->resetStatistics();
    ASSERT_EQ(0, cvode->getStatistics()["num_steps"].getAs<std::int64_t>());
}

namespace {
    // the counters of CVODE that the statistics return
    const std::vector<std::string> cvodeCounters = {
        "num_steps", "num_rhs_evals", "num_lin_solv_setups", "num_err_test_fails",
        "num_nonlin_solv_iters", "num_nonlin_solv_conv_fails", "num_step_solve_fails",
        "num_jac_evals", "num_lin_rhs_evals", "num_lin_iters", "num_lin_conv_fails",
        "num_g_evals", "num_stab_lim_order_reds", "num_reinits", "num_events",
    };

    // the times that the statistics return with the setting collect_timing
    const std::vector<std::string> cvodeTimes = {
        "time_total", "time_rhs_evals", "time_g_evals", "time_lin_solv_setups", "time_lin_solv_solves",
    };

    std::int64_t counterOf(const std::unordered_map<std::string, Setting> &stats, const std::string &key) {
        return stats.at(key).getAs<std::int64_t>();
    }

    double timeOf(const std::unordered_map<std::string, Setting> &stats, const std::string &key) {
        return stats.at(key).getAs<double>();
    }

    void expectSameResults(const ls::DoubleMatrix &expected, const ls::DoubleMatrix &actual, double tolerance) {
        ASSERT_EQ(expected.RSize(), actual.RSize());
        ASSERT_EQ(expected.CSize(), actual.CSize());
        for (unsigned row = 0; row < expected.RSize(); row++) {
            for (unsigned col = 0; col < expected.CSize(); col++) {
                EXPECT_NEAR(expected.Element(row, col), actual.Element(row, col), tolerance)
                    << "row " << row << ", column " << col;
            }
        }
    }
}

TEST_F(CVODEIntegratorTests, StatisticsHaveTheCountersOfCVODE) {
    RoadRunner r(OneEvent10Triggers().str());
    // only CVODE's own difference quotient counts the evaluations in num_lin_rhs_evals
    Setting jacobian = r.getIntegrator()->getValue("jacobian");
    r.getIntegrator()->setValue("jacobian", Setting(std::string("difference_quotient")));
    r.simulate(0, 10, 101);
    std::unordered_map<std::string, Setting> stats = r.getIntegrator()->getStatistics();
    r.getIntegrator()->setValue("jacobian", jacobian);

    for (const std::string &key: cvodeCounters) {
        ASSERT_EQ(1u, stats.count(key)) << key;
        ASSERT_GE(counterOf(stats, key), 0) << key;
    }
    ASSERT_GT(counterOf(stats, "num_steps"), 0);
    ASSERT_GE(counterOf(stats, "num_rhs_evals"), counterOf(stats, "num_steps"));
    ASSERT_GT(counterOf(stats, "num_jac_evals"), 0);
    ASSERT_GT(counterOf(stats, "num_lin_rhs_evals"), 0);
    ASSERT_GT(counterOf(stats, "num_g_evals"), 0);

    for (const std::string key: {"actual_init_step", "last_step", "current_step", "current_time",
                                 "last_order", "current_order"}) {
        ASSERT_EQ(1u, stats.count(key)) << key;
    }
    // the times are only returned when they are collected
    for (const std::string &key: cvodeTimes) {
        ASSERT_EQ(0u, stats.count(key)) << key;
    }
}

TEST_F(CVODEIntegratorTests, StatisticsOfALongerSimulationAreLarger) {
    RoadRunner r(OneEvent10Triggers().str());
    r.simulate(0, 10, 101);
    std::int64_t total = counterOf(r.getIntegrator()->getStatistics(), "num_steps");
    r.simulate(0, 1, 11);
    std::int64_t beforeFirstEvent = counterOf(r.getIntegrator()->getStatistics(), "num_steps");
    ASSERT_GT(total, 5 * beforeFirstEvent);
}

TEST_F(CVODEIntegratorTests, ResetStatisticsSetsAllCountersToZero) {
    RoadRunner r(OneEvent10Triggers().str());
    r.simulate(0, 10, 101);
    r.getIntegrator()->resetStatistics();
    std::unordered_map<std::string, Setting> stats = r.getIntegrator()->getStatistics();
    for (const std::string &key: cvodeCounters) {
        ASSERT_EQ(0, counterOf(stats, key)) << key;
    }
}

TEST_F(CVODEIntegratorTests, NonStiffSolverHasNoLinearSolverCounters) {
    RoadRunner r(OneEvent10Triggers().str());
    r.getIntegrator()->setValue("stiff", false);
    r.simulate(0, 10, 101);
    std::unordered_map<std::string, Setting> stats = r.getIntegrator()->getStatistics();
    ASSERT_GT(counterOf(stats, "num_steps"), 0);
    for (const std::string key: {"num_lin_solv_setups", "num_jac_evals", "num_lin_rhs_evals",
                                 "num_lin_iters", "num_lin_conv_fails"}) {
        ASSERT_EQ(0, counterOf(stats, key)) << key;
    }
}

TEST_F(CVODEIntegratorTests, OneStepStatistics) {
    RoadRunner r(OneEvent10Triggers().str());
    r.oneStep(0, 0.5);
    std::int64_t first = counterOf(r.getIntegrator()->getStatistics(), "num_steps");
    ASSERT_GT(first, 0);
    // without a reset the statistics keep counting
    r.oneStep(0.5, 0.5, false);
    ASSERT_GT(counterOf(r.getIntegrator()->getStatistics(), "num_steps"), first);
}

TEST_F(CVODEIntegratorTests, CollectTimingReportsTheTimes) {
    RoadRunner r(OneEvent10Triggers().str());
    r.getIntegrator()->setValue("collect_timing", true);
    r.simulate(0, 10, 101);
    std::unordered_map<std::string, Setting> stats = r.getIntegrator()->getStatistics();

    for (const std::string &key: cvodeTimes) {
        ASSERT_EQ(1u, stats.count(key)) << key;
        ASSERT_GE(timeOf(stats, key), 0.0) << key;
    }
    ASSERT_GT(timeOf(stats, "time_total"), 0.0);
    ASSERT_GT(timeOf(stats, "time_rhs_evals"), 0.0);
    ASSERT_GT(timeOf(stats, "time_lin_solv_setups"), 0.0);
    ASSERT_GT(timeOf(stats, "time_lin_solv_solves"), 0.0);
    // the parts of the integration are inside of it
    double parts = timeOf(stats, "time_rhs_evals") + timeOf(stats, "time_g_evals")
                   + timeOf(stats, "time_lin_solv_setups") + timeOf(stats, "time_lin_solv_solves");
    ASSERT_LE(parts, timeOf(stats, "time_total"));
}

/**
 * The linear solver is wrapped to time its setups and solves, which must not change the results.
 */
TEST_F(CVODEIntegratorTests, CollectTimingDoesNotChangeTheResults) {
    RoadRunner withoutTiming(OneEvent10Triggers().str());
    RoadRunner withTiming(OneEvent10Triggers().str());
    withTiming.getIntegrator()->setValue("collect_timing", true);

    const ls::DoubleMatrix *expected = withoutTiming.simulate(0, 10, 101);
    const ls::DoubleMatrix *actual = withTiming.simulate(0, 10, 101);
    expectSameResults(*expected, *actual, 0.0);
    ASSERT_EQ(counterOf(withoutTiming.getIntegrator()->getStatistics(), "num_steps"),
              counterOf(withTiming.getIntegrator()->getStatistics(), "num_steps"));
}

/**
 * Changing collect_timing re-creates CVODE, which wraps the linear solver only while the times
 * are collected.
 */
TEST_F(CVODEIntegratorTests, CollectTimingCanBeToggled) {
    RoadRunner r(OneEvent10Triggers().str());
    Integrator *integrator = r.getIntegrator();
    ls::DoubleMatrix expected = *r.simulate(0, 10, 101);

    for (bool timing: {true, false, true}) {
        integrator->setValue("collect_timing", timing);
        r.reset();
        const ls::DoubleMatrix *actual = r.simulate(0, 10, 101);
        expectSameResults(expected, *actual, 1e-10);

        std::unordered_map<std::string, Setting> stats = integrator->getStatistics();
        ASSERT_EQ(timing ? 1u : 0u, stats.count("time_lin_solv_setups")) << "collect_timing " << timing;
        if (timing) {
            ASSERT_GT(timeOf(stats, "time_lin_solv_setups"), 0.0);
        }
    }
}

TEST_F(CVODEIntegratorTests, TogglingCollectTimingInTheMiddleOfARunKeepsTheCounts) {
    RoadRunner r(OneEvent10Triggers().str());
    r.oneStep(0, 0.5);
    std::int64_t first = counterOf(r.getIntegrator()->getStatistics(), "num_steps");
    ASSERT_GT(first, 0);

    r.getIntegrator()->setValue("collect_timing", true);
    ASSERT_GE(counterOf(r.getIntegrator()->getStatistics(), "num_steps"), first);

    r.oneStep(0.5, 0.5, false);
    std::unordered_map<std::string, Setting> stats = r.getIntegrator()->getStatistics();
    ASSERT_GT(counterOf(stats, "num_steps"), first);
    ASSERT_EQ(1u, stats.count("time_total"));
}

TEST_F(CVODEIntegratorTests, CollectTimingWithTheNonStiffSolver) {
    RoadRunner r(OneEvent10Triggers().str());
    r.getIntegrator()->setValue("collect_timing", true);
    r.getIntegrator()->setValue("stiff", false);
    r.simulate(0, 10, 101);
    std::unordered_map<std::string, Setting> stats = r.getIntegrator()->getStatistics();
    ASSERT_GT(timeOf(stats, "time_total"), 0.0);
    ASSERT_GT(timeOf(stats, "time_rhs_evals"), 0.0);
    // there is no linear solver to time
    ASSERT_EQ(0.0, timeOf(stats, "time_lin_solv_setups"));
    ASSERT_EQ(0.0, timeOf(stats, "time_lin_solv_solves"));
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
    SimpleFlux testModel;

    // load model
    RoadRunner r(testModel.str());
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

    // The tolerances follow the initial values, so changing the current values leaves them alone.
    r.setValue("default_compartment", 1e-15);
    r.setValue("[S1]", 10.0);
    r.setValue("[S2]", 1.0);
    atols = r.getIntegrator()->getAbsoluteToleranceVector();
    ASSERT_EQ(atols.size(), 2);
    EXPECT_NEAR(atols[0], 1e-21, 1e-26);
    EXPECT_NEAR(atols[1], 1e-22, 1e-26);

    // An initial value too small to scale by is treated as zero, and scaled by the initial volume
    // (1e-10) instead of the current volume (1e-15).
    r.setValue("init(S1)", 1e100);
    r.setValue("init(S2)", 1e-100);
    atols = r.getIntegrator()->getAbsoluteToleranceVector();
    ASSERT_EQ(atols.size(), 2);
    EXPECT_NEAR(atols[0], 1e88, 1e80);
    EXPECT_NEAR(atols[1], 1e-22, 1e-26);
}

TEST_F(CVODEIntegratorTests, CheckAbsToleranceNegValues) {
    SimpleFlux testModel;

    // load model
    RoadRunner r(testModel.str());

    r.getModel()->setValue("init(S1)", -1e-14);
    r.getModel()->setValue("init(S2)", -1e-15);
    std::vector<double> atols = r.getIntegrator()->getAbsoluteToleranceVector();
    ASSERT_EQ(atols.size(), 2);
    EXPECT_NEAR(atols[0], 1e-26, 1e-31);
    EXPECT_NEAR(atols[1], 1e-27, 1e-31);
}

TEST_F(CVODEIntegratorTests, CheckAbsToleranceSetDouble) {
    SimpleFlux testModel;

    // load model
    RoadRunner r(testModel.str());
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

    // Changing the current values leaves the tolerances alone.
    r.setValue("default_compartment", 1e-15);
    r.setValue("[S1]", 10.0);
    r.setValue("[S2]", 1.0);
    atols = r.getIntegrator()->getAbsoluteToleranceVector();
    ASSERT_EQ(atols.size(), 2);
    EXPECT_NEAR(atols[0], 1e-24, 1e-29);
    EXPECT_NEAR(atols[1], 1e-25, 1e-29);
}

TEST_F(CVODEIntegratorTests, CheckAbsToleranceSetVector) {
    SimpleFlux testModel;

    // load model
    RoadRunner r(testModel.str());
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

/**
 * The absolute tolerance is scaled by the initial values, so it does not change as the
 * state decays, however the simulation is divided into calls to simulate.
 */
TEST_F(CVODEIntegratorTests, AbsToleranceDoesNotFollowTheCurrentState) {
    RoadRunner rr(DecaySBML);
    rr.setIntegrator("cvode");

    std::vector<double> before = rr.getIntegrator()->getAbsoluteToleranceVector();
    ASSERT_EQ(before.size(), 1u);
    EXPECT_NEAR(before[0], 1e-11, 1e-16);

    SimulateOptions &opt = rr.getSimulateOptions();
    opt.setSteps(10);
    opt.setStartTime(0);
    opt.setDuration(10);
    rr.simulate();
    opt.setStartTime(10);
    rr.simulate();

    ASSERT_LT(rr.getValue("S"), 1e-3);
    std::vector<double> after = rr.getIntegrator()->getAbsoluteToleranceVector();
    ASSERT_EQ(after.size(), 1u);
    EXPECT_EQ(before[0], after[0]);
}

/**
 * A species that decays to a value too small to represent used to give a zero
 * tolerance, which made CVODE fail.
 */
TEST_F(CVODEIntegratorTests, StackedSimulationsOfADecayToZero) {
    RoadRunner rr(DecaySBML);
    rr.setIntegrator("cvode");

    SimulateOptions &opt = rr.getSimulateOptions();
    opt.setSteps(10);
    opt.setDuration(750);
    for (double start : {0., 750., 1500., 2250.}) {
        opt.setStartTime(start);
        EXPECT_NO_THROW(rr.simulate()) << "start " << start;
    }
    EXPECT_LT(rr.getValue("S"), 1e-300);
}

// S' = -k S, and a parameter p with the rate rule p' = 1, so the state vector is [p, S].
static const std::string DecayWithRateRuleSBML = R"(<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4"><model id="decay_with_rate_rule">
 <listOfCompartments><compartment id="c" size="1"/></listOfCompartments>
 <listOfSpecies>
  <species id="S" compartment="c" initialConcentration="10" hasOnlySubstanceUnits="false"/>
 </listOfSpecies>
 <listOfParameters><parameter id="k" value="0.5"/><parameter id="p" value="2" constant="false"/></listOfParameters>
 <listOfRules>
  <rateRule variable="p"><math xmlns="http://www.w3.org/1998/Math/MathML"><cn>1</cn></math></rateRule>
 </listOfRules>
 <listOfReactions>
  <reaction id="J0" reversible="false">
   <listOfReactants><speciesReference species="S"/></listOfReactants>
   <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>k</ci><ci>S</ci></apply></math></kineticLaw>
  </reaction>
 </listOfReactions>
</model></sbml>)";

/**
 * The tolerance vector holds the rate rule variables followed by the independent floating
 * species.  Each value has to stay with its own variable when the model is regenerated.
 */
TEST_F(CVODEIntegratorTests, IndividualTolerancesSurviveRegeneratingTheModel) {
    RoadRunner rr(DecayWithRateRuleSBML);
    rr.setIntegrator("cvode");
    rr.getIntegrator()->setIndividualTolerance("p", 0.5);
    rr.getIntegrator()->setIndividualTolerance("S", 0.25);

    // changing an initial value regenerates the model
    rr.setValue("init(k)", 0.7);

    std::vector<double> atols = rr.getIntegrator()->getAbsoluteToleranceVector();
    ASSERT_EQ(atols.size(), 2u);
    EXPECT_EQ(atols[0], 0.5);
    EXPECT_EQ(atols[1], 0.25);
}

/**
 * A tolerance set for an element (as a vector, or with setIndividualTolerance) is used as it
 * is, so changing that element's initial value leaves it alone.  A scalar tolerance is scaled
 * by the initial values, so it follows them.  The state vector is [p, S].
 */
TEST_F(CVODEIntegratorTests, AbsToleranceVectorIgnoresInitialValuesButScalarFollowsThem) {
    // a vector
    {
        RoadRunner rr(DecayWithRateRuleSBML);
        rr.setIntegrator("cvode");
        rr.getIntegrator()->setValue("absolute_tolerance", std::vector<double>{0.5, 0.25});
        rr.setValue("init(p)", 4.0);
        rr.setValue("init([S])", 20.0);
        std::vector<double> atols = rr.getIntegrator()->getAbsoluteToleranceVector();
        ASSERT_EQ(atols.size(), 2u);
        EXPECT_EQ(atols[0], 0.5);
        EXPECT_EQ(atols[1], 0.25);
    }

    // individual tolerances
    {
        RoadRunner rr(DecayWithRateRuleSBML);
        rr.setIntegrator("cvode");
        rr.getIntegrator()->setIndividualTolerance("p", 0.5);
        rr.getIntegrator()->setIndividualTolerance("S", 0.25);
        rr.setValue("init(p)", 4.0);
        rr.setValue("init([S])", 20.0);
        std::vector<double> atols = rr.getIntegrator()->getAbsoluteToleranceVector();
        ASSERT_EQ(atols.size(), 2u);
        EXPECT_EQ(atols[0], 0.5);
        EXPECT_EQ(atols[1], 0.25);
    }

    // a scalar
    {
        RoadRunner rr(DecayWithRateRuleSBML);
        rr.setIntegrator("cvode");
        rr.getIntegrator()->setValue("absolute_tolerance", 1e-6);
        std::vector<double> atols = rr.getIntegrator()->getAbsoluteToleranceVector();
        ASSERT_EQ(atols.size(), 2u);
        EXPECT_NEAR(atols[0], 2e-6, 1e-12);
        EXPECT_NEAR(atols[1], 10e-6, 1e-12);

        rr.setValue("init(p)", 4.0);
        rr.setValue("init([S])", 20.0);
        atols = rr.getIntegrator()->getAbsoluteToleranceVector();
        ASSERT_EQ(atols.size(), 2u);
        EXPECT_NEAR(atols[0], 4e-6, 1e-12);
        EXPECT_NEAR(atols[1], 20e-6, 1e-12);
    }
}

// Fixture for tests that load a model file from test/models.
class CVODEConservedMoietyTests : public RoadRunnerTest {
};

/**
 * With conserved moieties the dependent species are not in the tolerance vector, which is
 * then shorter than the number of floating species.  Regenerating the model has to keep
 * each tolerance with its species and leave the dependent species out.
 */
TEST_F(CVODEConservedMoietyTests, IndividualTolerancesSurviveRegeneratingTheModel) {
    RoadRunner rr((rrTestModelsDir_ / "ModelAnalysis" / "two_cycles.xml").string());
    rr.setConservedMoietyAnalysis(true);

    ExecutableModel *model = rr.getModel();
    const int numInd = model->getNumIndFloatingSpecies();
    ASSERT_GT(numInd, 0);
    ASSERT_LT(numInd, model->getNumFloatingSpecies());

    std::unordered_map<std::string, double> expected;
    for (int i = 0; i < numInd; i++) {
        std::string id = model->getFloatingSpeciesId(i);
        expected[id] = 0.1 * (i + 1);
        rr.getIntegrator()->setIndividualTolerance(id, expected[id]);
    }

    // changing an initial value regenerates the model
    const std::string init = "init([" + model->getFloatingSpeciesId(0) + "])";
    rr.setValue(init, rr.getValue(init));

    model = rr.getModel();
    ASSERT_EQ(model->getNumIndFloatingSpecies(), numInd);
    std::vector<double> atols = rr.getIntegrator()->getAbsoluteToleranceVector();
    ASSERT_EQ(atols.size(), static_cast<size_t>(numInd));
    for (int i = 0; i < numInd; i++) {
        const std::string id = model->getFloatingSpeciesId(i);
        EXPECT_EQ(atols[i], expected.at(id)) << "species " << id;
    }
}

// Like test/models/SBMLFeatures/stoich_rr.xml, but the stoichiometry N starts at 5.
// N is governed by a rate rule, dN/dt = 1, and dX/dt = N*0.5.  The state vector is [N, X].
static const std::string RateRuleStoichSBML = R"(<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model metaid="__main" id="__main">
    <listOfCompartments>
      <compartment sboTerm="SBO:0000410" id="default_compartment" spatialDimensions="3" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="X" compartment="default_compartment" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false" initialConcentration="0"/>
    </listOfSpecies>
    <listOfRules>
      <rateRule variable="N">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <cn type="integer"> 1 </cn>
        </math>
      </rateRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="_J0" reversible="true">
        <listOfProducts>
          <speciesReference id="N" species="X" stoichiometry="5" constant="false"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <cn> 0.5 </cn>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>)";

/**
 * A species reference governed by a rate rule is part of the state vector.  Its absolute
 * tolerance is scaled by its initial value, not by the value it has reached.
 */
TEST_F(CVODEIntegratorTests, AbsToleranceOfARateRuleStoichiometryUsesItsInitialValue) {
    RoadRunner rr(RateRuleStoichSBML);
    rr.setIntegrator("cvode");
    rr.getIntegrator()->setValue("absolute_tolerance", 1e-6);

    // N grows from 5 to 7, so X reaches 2.5*2 + 0.25*2^2 = 6
    const ls::DoubleMatrix result = *rr.simulate(0.0, 2.0, 11);
    ASSERT_EQ(result.numRows(), 11u);
    ASSERT_NEAR(result(10, 1), 6.0, 1e-3);

    std::vector<double> atols = rr.getIntegrator()->getAbsoluteToleranceVector();
    ASSERT_EQ(atols.size(), 2u);
    // N has initial value 5.  X starts at zero, so it is scaled by the compartment volume, 1.
    EXPECT_NEAR(atols[0], 1e-6 * 5.0, 1e-12);
    EXPECT_NEAR(atols[1], 1e-6 * 1.0, 1e-12);
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

/**
 * Tests for the warnings that CVODE issues itself ("t + h = t on the next step" and so on).
 * SUNDIALS writes them to the SUNLogger of its context, which roadrunner connects to rr::Logger
 * (see SundialsLogger.h). They therefore appear in the Logger's output when the Logger level is
 * LOG_WARNING or more verbose, and do not appear on stdout.
 *
 * The model decays quickly (k = 1000) from t = 613190, where the spacing of doubles (1.2e-10)
 * is of the order of the steps CVODE takes once A has decayed away, so that it warns about
 * t + h = t at the tolerances of 1e-12 and 1e-10 used below.
 */
static const std::string StiffDecayAtLargeTimeSBML = R"(<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model metaid="m" id="m">
    <listOfParameters>
      <parameter id="A" value="1" constant="false"/>
      <parameter id="B" value="0" constant="false"/>
      <parameter id="k" value="1000" constant="true"/>
    </listOfParameters>
    <listOfRules>
      <rateRule variable="A">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><apply><minus/><ci> k </ci></apply><ci> A </ci></apply>
        </math>
      </rateRule>
      <rateRule variable="B">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><minus/><apply><times/><ci> k </ci><ci> A </ci></apply><apply><times/><cn> 0.5 </cn><ci> B </ci></apply></apply>
        </math>
      </rateRule>
    </listOfRules>
  </model>
</sbml>)";

// Collects what the Logger writes to its console stream, and puts the stream and the level back afterwards.
class ScopedLoggerCapture {
public:
    ScopedLoggerCapture() : previousLevel(Logger::getLevel()) { Logger::setConsoleStream(&stream); }
    ~ScopedLoggerCapture() {
        Logger::setConsoleStream(&std::clog);
        Logger::setLevel(previousLevel);
    }
    std::string str() const { return stream.str(); }
private:
    std::ostringstream stream;
    int previousLevel;
};

struct RunOutput {
    std::string logger; // what the Logger wrote
    std::string out;    // what was written to stdout
};

static int countOccurrences(const std::string &text, const std::string &what) {
    int count = 0;
    for (size_t pos = text.find(what); pos != std::string::npos; pos = text.find(what, pos + 1)) {
        count++;
    }
    return count;
}

// One run of the model above at the given Logger level
static RunOutput runStiffDecayAtLogLevel(RoadRunner &rr, Logger::Level level) {
    ScopedLoggerCapture loggerCapture;
    Logger::setLevel(level);
    rr.reset(SelectionRecord::ALL);
    rr.getIntegrator()->setValue("variable_step_size", false);
    rr.getIntegrator()->setValue("absolute_tolerance", 1e-12);
    rr.getIntegrator()->setValue("relative_tolerance", 1e-10);

    RunOutput result;
    testing::internal::CaptureStdout();
    rr.simulate(613190, 20, 5); // t = 613190 to 613210, 5 points (the second argument is a duration)
    fflush(stdout);             // SUNDIALS used to write through C stdio, which is buffered
    result.out = testing::internal::GetCapturedStdout();
    result.logger = loggerCapture.str();
    return result;
}

TEST_F(CVODEIntegratorTests, CvodeWarningsAreNotLoggedBelowLogLevelWarning) {
    for (Logger::Level level : {Logger::LOG_FATAL, Logger::LOG_CRITICAL, Logger::LOG_ERROR}) {
        RoadRunner rr(StiffDecayAtLargeTimeSBML);
        RunOutput output = runStiffDecayAtLogLevel(rr, level);
        EXPECT_EQ(0, countOccurrences(output.logger, "t + h = t")) << "Logger level " << level;
        EXPECT_EQ(0, countOccurrences(output.out, "t + h = t")) << "Logger level " << level;
    }
}

TEST_F(CVODEIntegratorTests, CvodeWarningsAreLoggedFromLogLevelWarning) {
    for (Logger::Level level : {Logger::LOG_WARNING, Logger::LOG_INFORMATION}) {
        RoadRunner rr(StiffDecayAtLargeTimeSBML);
        RunOutput output = runStiffDecayAtLogLevel(rr, level);
        EXPECT_GT(countOccurrences(output.logger, "t + h = t"), 0) << "Logger level " << level;
        // the message says which SUNDIALS function issued it
        EXPECT_GT(countOccurrences(output.logger, "CVode ("), 0) << "Logger level " << level;
    }
}

// they used to be printed to stdout by SUNDIALS, whatever roadrunner's Logger was set to
TEST_F(CVODEIntegratorTests, CvodeWarningsAreNotWrittenToStdout) {
    RoadRunner rr(StiffDecayAtLargeTimeSBML);
    RunOutput output = runStiffDecayAtLogLevel(rr, Logger::LOG_WARNING);
    EXPECT_GT(countOccurrences(output.logger, "t + h = t"), 0);
    EXPECT_EQ(0, countOccurrences(output.out, "t + h = t"));
    EXPECT_EQ(0, countOccurrences(output.out, "[WARNING]"));
}

// the Logger level is checked as each message is logged, so changing it between runs of one RoadRunner takes effect
TEST_F(CVODEIntegratorTests, CvodeWarningsFollowTheLogLevelBetweenRuns) {
    RoadRunner rr(StiffDecayAtLargeTimeSBML);
    EXPECT_EQ(0, countOccurrences(runStiffDecayAtLogLevel(rr, Logger::LOG_ERROR).logger, "t + h = t"));
    EXPECT_GT(countOccurrences(runStiffDecayAtLogLevel(rr, Logger::LOG_WARNING).logger, "t + h = t"), 0);
    EXPECT_EQ(0, countOccurrences(runStiffDecayAtLogLevel(rr, Logger::LOG_ERROR).logger, "t + h = t"));
    EXPECT_GT(countOccurrences(runStiffDecayAtLogLevel(rr, Logger::LOG_WARNING).logger, "t + h = t"), 0);
}

/**
 * Two events with the same trigger time and no priority: SBML leaves their order arbitrary, and the
 * last one applied decides x. The order is chosen with the model's random number generator, so it
 * is random by default and RoadRunner::setSeed makes it repeatable (it used to come from std::rand,
 * which setSeed does not reach and which every new model reseeded from the clock).
 */
static const std::string SimultaneousEventsSBML = R"(<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="m">
    <listOfParameters>
      <parameter id="x" value="0" constant="false"/>
    </listOfParameters>
    <listOfEvents>
      <event id="e1" useValuesFromTriggerTime="true">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML"><apply><gt/><csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time"> time </csymbol><cn> 1 </cn></apply></math>
        </trigger>
        <listOfEventAssignments>
          <eventAssignment variable="x">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn type="integer"> 1 </cn></math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
      <event id="e2" useValuesFromTriggerTime="true">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML"><apply><gt/><csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time"> time </csymbol><cn> 1 </cn></apply></math>
        </trigger>
        <listOfEventAssignments>
          <eventAssignment variable="x">
            <math xmlns="http://www.w3.org/1998/Math/MathML"><cn type="integer"> 2 </cn></math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>)";

// x at the end of a run in which two events fire at the same time, after setSeed(seed)
static double lastEventApplied(RoadRunner &rr, int64_t seed) {
    rr.setSeed(seed);
    rr.getIntegrator()->setValue("variable_step_size", false);
    rr.simulate(0, 2, 3);
    return rr.getValue("x");
}

TEST_F(CVODEIntegratorTests, SimultaneousEventsAreOrderedRandomly) {
    RoadRunner rr(SimultaneousEventsSBML);
    int ones = 0, twos = 0;
    for (int64_t seed = 1; seed <= 100; seed++) {
        double x = lastEventApplied(rr, seed);
        ASSERT_TRUE(x == 1 || x == 2) << "x = " << x << " with seed " << seed;
        (x == 1 ? ones : twos)++;
    }
    // both orders happen, and neither dominates (the chance of fewer than 20 in 100 fair draws is below 1e-9)
    EXPECT_GT(ones, 20);
    EXPECT_GT(twos, 20);
}

TEST_F(CVODEIntegratorTests, SimultaneousEventsAreOrderedRepeatablyWithASeed) {
    RoadRunner rr(SimultaneousEventsSBML);
    RoadRunner other(SimultaneousEventsSBML);
    for (int64_t seed = 1; seed <= 20; seed++) {
        double first = lastEventApplied(rr, seed);
        // the same seed gives the same order again, also on a second RoadRunner
        EXPECT_EQ(first, lastEventApplied(rr, seed)) << "seed " << seed;
        EXPECT_EQ(first, lastEventApplied(other, seed)) << "seed " << seed;
    }
}

// setSeed(seed, false) only reseeds the random number generators: it does not reset the model
TEST_F(CVODEIntegratorTests, SetSeedWithoutResetKeepsTheCurrentAndInitialValuesAndTheTime) {
    RoadRunner rr(SimultaneousEventsSBML);
    // setting an initial value regenerates and resets the model, so it has to come first
    rr.setValue("init(x)", 7);
    rr.getIntegrator()->setValue("variable_step_size", false);
    rr.simulate(0, 0.5, 3);  // before the events at t = 1
    rr.setValue("x", 5);
    ASSERT_EQ(5, rr.getValue("x"));
    ASSERT_EQ(0.5, rr.getCurrentTime());

    rr.setSeed(42, false);

    EXPECT_EQ(5, rr.getValue("x"));
    EXPECT_EQ(7, rr.getValue("init(x)"));
    EXPECT_EQ(0.5, rr.getCurrentTime());
    EXPECT_EQ(42, rr.getSeed());
}

// by default setSeed also resets the model (to its original state, like reset(SelectionRecord::ALL))
TEST_F(CVODEIntegratorTests, SetSeedResetsTheModelByDefault) {
    RoadRunner rr(SimultaneousEventsSBML);
    rr.getIntegrator()->setValue("variable_step_size", false);
    rr.simulate(0, 0.5, 3);
    rr.setValue("x", 5);
    ASSERT_EQ(0.5, rr.getCurrentTime());

    rr.setSeed(42);

    EXPECT_EQ(0, rr.getValue("x"));
    EXPECT_EQ(0, rr.getCurrentTime());
    EXPECT_EQ(42, rr.getSeed());
}

// model pulses
//   Q = 0
//   last = 0
//   E: at 0.005 after (time - last >= 0.005), persistent=false, fromTrigger=false: last = time, Q = Q + 0.01
// end
static const std::string PulsesSBML = R"(<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model metaid="pulses" id="pulses">
    <listOfParameters>
      <parameter id="Q" value="0" constant="false"/>
      <parameter id="last" value="0" constant="false"/>
    </listOfParameters>
    <listOfEvents>
      <event id="E" useValuesFromTriggerTime="false">
        <trigger initialValue="true" persistent="false">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply>
              <geq/>
              <apply>
                <minus/>
                <csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time"> time </csymbol>
                <ci> last </ci>
              </apply>
              <cn> 0.005 </cn>
            </apply>
          </math>
        </trigger>
        <delay>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <cn> 0.005 </cn>
          </math>
        </delay>
        <listOfEventAssignments>
          <eventAssignment variable="last">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time"> time </csymbol>
            </math>
          </eventAssignment>
          <eventAssignment variable="Q">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply>
                <plus/>
                <ci> Q </ci>
                <cn> 0.01 </cn>
              </apply>
            </math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>)";

// variable step simulation of the given duration from start
static ls::DoubleMatrix simulateVariableStep(RoadRunner &rr, double start, double duration) {
    rr.getIntegrator()->setValue("variable_step_size", true);
    SimulateOptions &opt = rr.getSimulateOptions();
    opt.setStartTime(start);
    opt.setDuration(duration);
    opt.setSteps(0);
    return *rr.simulate();
}

// what resetAll does in the python API: all values back to their current initial values
static void resetAllValues(RoadRunner &rr) {
    rr.reset(SelectionRecord::TIME | SelectionRecord::RATE | SelectionRecord::FLOATING |
             SelectionRecord::BOUNDARY | SelectionRecord::COMPARTMENT | SelectionRecord::GLOBAL_PARAMETER |
             SelectionRecord::STOICHIOMETRY);
}

/**
 * Regression test for sys-bio/roadrunner#1381: when the last step of CVODE of a variable step
 * simulation found the root of an event after the end, the integrator kept the event pending
 * through resetAll.  The first call of the integrator of the next simulation applied it with the
 * trigger states of the end of the previous simulation, and the event of this model, whose
 * trigger is not persistent, fired one period late.
 */
TEST_F(CVODEIntegratorTests, EventFoundAfterTheEndOfAVariableStepSimulationIsNotPendingAfterAReset) {
    RoadRunner fresh(PulsesSBML);
    fresh.setSelections(std::vector<std::string>{"time", "Q"});
    fresh.getIntegrator()->setValue("variable_step_size", false);
    const ls::DoubleMatrix expected = *fresh.simulate(0, 1, 11);

    RoadRunner rr(PulsesSBML);
    rr.setSelections(std::vector<std::string>{"time", "Q"});
    // the event of the model has its root just after the end, at 10
    simulateVariableStep(rr, 0, 9.9999999);
    resetAllValues(rr);
    rr.getIntegrator()->setValue("variable_step_size", false);
    const ls::DoubleMatrix observed = *rr.simulate(0, 1, 11);

    ASSERT_EQ(expected.numRows(), observed.numRows());
    for (unsigned i = 0; i < expected.numRows(); i++) {
        EXPECT_NEAR(expected(i, 1), observed(i, 1), 1e-9) << "row " << i;
    }
}

/**
 * The symptom of the same flag when a simulation continues after the variable step simulation
 * (without a reset): the first row after the start was at start + the root tolerance, which is
 * 7e-12 here, with the values of the start, and the simulation then did not reach the same
 * state as one without the break.
 */
TEST_F(CVODEIntegratorTests, VariableStepSimulationContinuesWithoutASpuriousRowAfterAnEventFoundAfterItsEnd) {
    const double end = 9.9999999;

    RoadRunner whole(PulsesSBML);
    whole.setSelections(std::vector<std::string>{"time", "Q"});
    const ls::DoubleMatrix uninterrupted = simulateVariableStep(whole, 0, end + 0.1);

    RoadRunner rr(PulsesSBML);
    rr.setSelections(std::vector<std::string>{"time", "Q"});
    simulateVariableStep(rr, 0, end);
    const ls::DoubleMatrix second = simulateVariableStep(rr, end, 0.1);

    ASSERT_GT(second.numRows(), 2u);
    EXPECT_EQ(end, second(0, 0));
    EXPECT_GT(second(1, 0) - second(0, 0), 1e-11);
    for (unsigned i = 1; i < second.numRows(); i++) {
        EXPECT_GT(second(i, 0), second(i - 1, 0)) << "row " << i;
    }
    EXPECT_EQ(uninterrupted(uninterrupted.numRows() - 1, 0), second(second.numRows() - 1, 0));
    EXPECT_NEAR(uninterrupted(uninterrupted.numRows() - 1, 1), second(second.numRows() - 1, 1), 1e-9);
}

// model delays
//   var X = 0
//   var Y = 0
//   X' = 1
//   E: at (X >= 1): X = 0
//   D: at 0.3 after (time >= 0.5): Y = Y + 100
//   L: at 0.2 after (time >= 0.25), fromTrigger=false: Y = Y + 1
// end
static const std::string DelaysSBML = R"(<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model metaid="delays" id="delays">
    <listOfParameters>
      <parameter id="X" value="0" constant="false"/>
      <parameter id="Y" value="0" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <rateRule variable="X">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <cn type="integer"> 1 </cn>
        </math>
      </rateRule>
    </listOfRules>
    <listOfEvents>
      <event id="E" useValuesFromTriggerTime="true">
        <trigger initialValue="true" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply>
              <geq/>
              <ci> X </ci>
              <cn type="integer"> 1 </cn>
            </apply>
          </math>
        </trigger>
        <listOfEventAssignments>
          <eventAssignment variable="X">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <cn type="integer"> 0 </cn>
            </math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
      <event id="D" useValuesFromTriggerTime="true">
        <trigger initialValue="true" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply>
              <geq/>
              <csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time"> time </csymbol>
              <cn> 0.5 </cn>
            </apply>
          </math>
        </trigger>
        <delay>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <cn> 0.3 </cn>
          </math>
        </delay>
        <listOfEventAssignments>
          <eventAssignment variable="Y">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply>
                <plus/>
                <ci> Y </ci>
                <cn type="integer"> 100 </cn>
              </apply>
            </math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
      <event id="L" useValuesFromTriggerTime="false">
        <trigger initialValue="true" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply>
              <geq/>
              <csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time"> time </csymbol>
              <cn> 0.25 </cn>
            </apply>
          </math>
        </trigger>
        <delay>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <cn> 0.2 </cn>
          </math>
        </delay>
        <listOfEventAssignments>
          <eventAssignment variable="Y">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply>
                <plus/>
                <ci> Y </ci>
                <cn type="integer"> 1 </cn>
              </apply>
            </math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>)";

/**
 * Regression test for sys-bio/roadrunner#1382: CVODE steps in CV_ONE_STEP mode, which ignores
 * the output time, so with variable_step_size an event with a delay fired at the end of the
 * next step of CVODE, i.e. at the next root of another event, instead of at its time.  Here
 * L is due at 0.45 (trigger at 0.25, delay 0.2) and D at 0.8 (trigger at 0.5, delay 0.3); L
 * fired at the trigger of D, at 0.5, and D saw Y = 0 and assigned 100 instead of 101.
 */
TEST_F(CVODEIntegratorTests, DelayedEventsFireAtTheirTimeInAVariableStepSimulation) {
    RoadRunner rr(DelaysSBML);
    rr.setSelections(std::vector<std::string>{"time", "X", "Y"});
    const ls::DoubleMatrix result = simulateVariableStep(rr, 0, 1.5);
    ASSERT_GT(result.numRows(), 2u);

    double timeOfL = -1;
    double timeOfD = -1;
    for (unsigned i = 1; i < result.numRows(); i++) {
        EXPECT_GT(result(i, 0), result(i - 1, 0)) << "row " << i;
        if (timeOfL < 0 && result(i, 2) > 0.5) {
            timeOfL = result(i, 0);
        }
        if (timeOfD < 0 && result(i, 2) > 50) {
            timeOfD = result(i, 0);
            // D has the value of Y after L, 1, at the time of its trigger
            EXPECT_EQ(101, result(i, 2));
        }
    }
    EXPECT_NEAR(0.45, timeOfL, 1e-6);
    EXPECT_NEAR(0.8, timeOfD, 1e-6);
    EXPECT_EQ(101, result(result.numRows() - 1, 2));
}

// model piecewise_switch
//   compartment C = 1
//   species A in C = 10
//   k := piecewise(2, time < 3.3, 0.5)
//   J: A -> ; k*A
// end
static const std::string PiecewiseSwitchSBML = R"(<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model metaid="piecewise_switch" id="piecewise_switch">
    <listOfCompartments>
      <compartment id="C" spatialDimensions="3" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="C" initialConcentration="10" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="k">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <piecewise>
            <piece>
              <cn type="integer"> 2 </cn>
              <apply>
                <lt/>
                <csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time"> time </csymbol>
                <cn> 3.3 </cn>
              </apply>
            </piece>
            <otherwise>
              <cn> 0.5 </cn>
            </otherwise>
          </piecewise>
        </math>
      </assignmentRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="J" reversible="true">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply>
              <times/>
              <ci> k </ci>
              <ci> A </ci>
            </apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>)";

/**
 * The same as above for a piecewise trigger, which is a root of the integrator like the trigger of
 * an event (sys-bio/roadrunner#1381).  The switch of the model is at 3.3, just after the end of the
 * variable step simulation.
 */
TEST_F(CVODEIntegratorTests, PiecewiseSwitchFoundAfterTheEndOfAVariableStepSimulationIsNotPendingAfterAReset) {
    RoadRunner fresh(PiecewiseSwitchSBML);
    fresh.setSelections(std::vector<std::string>{"time", "A"});
    fresh.getIntegrator()->setValue("variable_step_size", false);
    const ls::DoubleMatrix expected = *fresh.simulate(0, 10, 11);

    RoadRunner rr(PiecewiseSwitchSBML);
    rr.setSelections(std::vector<std::string>{"time", "A"});
    simulateVariableStep(rr, 0, 3.2999999);
    resetAllValues(rr);
    rr.getIntegrator()->setValue("variable_step_size", false);
    const ls::DoubleMatrix observed = *rr.simulate(0, 10, 11);

    ASSERT_EQ(expected.numRows(), observed.numRows());
    for (unsigned i = 0; i < expected.numRows(); i++) {
        EXPECT_NEAR(expected(i, 1), observed(i, 1), 1e-9) << "row " << i;
    }
}

/**
 * Regression test for sys-bio/roadrunner#1383: a model taken from the model cache, as it is
 * while another instance of the same SBML is alive, was created with no piecewise triggers, so
 * CVODE did not stop at the switch of a piecewise and the instance integrated differently from
 * a compiled one (5e-8 absolute here).
 */
TEST_F(CVODEIntegratorTests, ModelFromTheModelCacheHasThePiecewiseTriggersOfACompiledModel) {
    RoadRunner first(PiecewiseSwitchSBML);   // compiled
    RoadRunner cached(PiecewiseSwitchSBML);  // from the cache, as long as first is alive
    ASSERT_GT(first.getModel()->getNumPiecewiseTriggers(), 0);
    EXPECT_EQ(first.getModel()->getNumPiecewiseTriggers(), cached.getModel()->getNumPiecewiseTriggers());

    first.setSelections(std::vector<std::string>{"time", "A"});
    cached.setSelections(std::vector<std::string>{"time", "A"});
    const ls::DoubleMatrix a = *first.simulate(0, 10, 11);
    const ls::DoubleMatrix b = *cached.simulate(0, 10, 11);
    ASSERT_EQ(a.numRows(), b.numRows());
    for (unsigned i = 0; i < a.numRows(); i++) {
        EXPECT_NEAR(a(i, 1), b(i, 1), 1e-14) << "row " << i;
    }
    // the analytic solution at t = 4 is 10 exp(-2 * 3.3 - 0.5 * 0.7), up to the default tolerances
    EXPECT_NEAR(10 * std::exp(-2 * 3.3 - 0.5 * 0.7), b(4, 1), 1e-6);
}

// model sawtooth
//   var X = 0
//   var Y = 0
//   X' = 1
//   E: at (X >= 1): X = 0, Y = Y + 1
// end
static const std::string SawtoothSBML = R"(<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model metaid="sawtooth" id="sawtooth">
    <listOfParameters>
      <parameter id="X" value="0" constant="false"/>
      <parameter id="Y" value="0" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <rateRule variable="X">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <cn type="integer"> 1 </cn>
        </math>
      </rateRule>
    </listOfRules>
    <listOfEvents>
      <event id="E" useValuesFromTriggerTime="true">
        <trigger initialValue="true" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply>
              <geq/>
              <ci> X </ci>
              <cn type="integer"> 1 </cn>
            </apply>
          </math>
        </trigger>
        <listOfEventAssignments>
          <eventAssignment variable="X">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <cn type="integer"> 0 </cn>
            </math>
          </eventAssignment>
          <eventAssignment variable="Y">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply>
                <plus/>
                <ci> Y </ci>
                <cn type="integer"> 1 </cn>
              </apply>
            </math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>)";

/**
 * Regression test for sys-bio/roadrunner#1384, a remaining case of #1362: when the end of a
 * variable step simulation was within the root tolerance of an event (about 1e-12 here, around
 * the event at t = 1), the last row was a linear interpolation between the state before the
 * event and the state after it, and the model was left at the time of the event, in the state
 * after it, instead of at the end.
 *
 * Every row is the state before the event (Y = 0, X = t) or after it (Y = 1, X = t - 1).  An
 * end well before the event does not apply it, an end well after it does.  Between, within the
 * tolerance, either is fine as long as the last row is at the end and the model is in its state.
 */
TEST_F(CVODEIntegratorTests, VariableStepSimulationEndingWithinTheRootToleranceOfAnEvent) {
    for (double end : {1.0 - 1e-12, 1.0 - 1e-13, 1.0, 1.0 + 1e-13, 1.0 + 1e-12, 1.0 + 1e-9}) {
        std::ostringstream what;
        what.precision(17);
        what << "end " << end;

        RoadRunner rr(SawtoothSBML);
        rr.setSelections(std::vector<std::string>{"time", "X", "Y"});
        const ls::DoubleMatrix result = simulateVariableStep(rr, 0, end);
        ASSERT_GT(result.numRows(), 2u) << what.str();
        const unsigned last = result.numRows() - 1;

        EXPECT_EQ(end, result(last, 0)) << what.str();
        for (unsigned i = 0; i <= last; i++) {
            if (i > 0) {
                EXPECT_GT(result(i, 0), result(i - 1, 0)) << what.str() << ", row " << i;
            }
            const double t = result(i, 0);
            const double X = result(i, 1);
            const double Y = result(i, 2);
            if (Y == 0) {
                EXPECT_NEAR(t, X, 1e-9) << what.str() << ", row " << i << " (before the event)";
            } else {
                EXPECT_EQ(1, Y) << what.str() << ", row " << i;
                EXPECT_NEAR(t - 1, X, 1e-9) << what.str() << ", row " << i << " (after the event)";
            }
        }

        // the model is at the end, in the state of the last row
        EXPECT_EQ(end, rr.getCurrentTime()) << what.str();
        EXPECT_DOUBLE_EQ(result(last, 1), rr.getValue("X")) << what.str();
        EXPECT_DOUBLE_EQ(result(last, 2), rr.getValue("Y")) << what.str();

        if (end <= 1.0 - 1e-12) {
            EXPECT_EQ(0, result(last, 2)) << what.str();
        }
        if (end >= 1.0 + 1e-12) {
            EXPECT_EQ(1, result(last, 2)) << what.str();
        }
    }
}

// model end_trigger
//   var X = 1
//   var n = 0
//   X' = -0.1 * X
//   E: at (time <op> T), after @DELAY@: n = n + 1   (the trigger is false at the start)
static const std::string EndTriggerSBML = R"(<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model metaid="end_trigger" id="end_trigger">
    <listOfParameters>
      <parameter id="X" value="1" constant="false"/>
      <parameter id="n" value="0" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <rateRule variable="X">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply>
            <times/>
            <cn> -0.1 </cn>
            <ci> X </ci>
          </apply>
        </math>
      </rateRule>
    </listOfRules>
    <listOfEvents>
      <event id="E" useValuesFromTriggerTime="true">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply>
              <@OP@/>
              <csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time"> time </csymbol>
              <cn> @T@ </cn>
            </apply>
          </math>
        </trigger>@DELAY@
        <listOfEventAssignments>
          <eventAssignment variable="n">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply>
                <plus/>
                <ci> n </ci>
                <cn type="integer"> 1 </cn>
              </apply>
            </math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>)";

// the model with the comparison of the trigger (geq or gt), its time and the delay of the event (none if 0)
static std::string endTriggerModel(const std::string &op, double triggerTime, double delay) {
    std::string delayElement;
    if (delay > 0) {
        delayElement = "\n        <delay>\n          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
                       "            <cn> " + std::to_string(delay) + " </cn>\n          </math>\n        </delay>";
    }
    std::string sbml = EndTriggerSBML;
    const std::pair<std::string, std::string> substitutions[] = {
            {"@OP@", op}, {"@T@", std::to_string(triggerTime)}, {"@DELAY@", delayElement}};
    for (const auto &substitution : substitutions) {
        sbml.replace(sbml.find(substitution.first), substitution.first.size(), substitution.second);
    }
    return sbml;
}

struct EndTriggerCase {
    const char *op;
    double triggerTime;
    double end;
    int nAtEnd;  // n at the end of the first simulation
};

/**
 * Regression test for sys-bio/roadrunner#1445 (a regression of #1390): an event whose trigger was
 * true at the end of a variable step simulation never fired.  The root of the event was not
 * found after the end, because the trigger was already true there (time >= 1 with the end at 1),
 * or a few ulps before it (time >= 2 with the end a few ulps after 2) and reported by CVODE after
 * it.  The event fires at the end, with a row before it, and once: a simulation that continues
 * from the end does not fire it again.  An event with a trigger that is not true at the end
 * (time > 1 with the end at 1) fires after it, once.
 */
TEST_F(CVODEIntegratorTests, EventWhoseTriggerIsTrueAtTheEndOfAVariableStepSimulationFiresAtTheEnd) {
    const double stop = 5;
    const EndTriggerCase cases[] = {
            {"geq", 1.0, 1.0, 1},
            {"geq", 2.0, 2.0 + 1e-15, 1},
            {"gt", 1.0, 1.0, 0},
    };

    for (const EndTriggerCase &c : cases) {
        std::ostringstream what;
        what.precision(17);
        what << "time " << c.op << " " << c.triggerTime << ", end " << c.end;
        SCOPED_TRACE(what.str());

        RoadRunner rr(endTriggerModel(c.op, c.triggerTime, 0));
        rr.setSelections(std::vector<std::string>{"time", "X", "n"});

        const ls::DoubleMatrix first = simulateVariableStep(rr, 0, c.end);
        ASSERT_GT(first.numRows(), 2u);
        const unsigned last = first.numRows() - 1;
        for (unsigned i = 1; i <= last; i++) {
            EXPECT_GT(first(i, 0), first(i - 1, 0)) << "row " << i;
        }
        EXPECT_EQ(c.end, first(last, 0));
        EXPECT_EQ(c.nAtEnd, first(last, 2));
        if (c.nAtEnd > 0) {
            // the row before the event
            EXPECT_EQ(0, first(last - 1, 2));
        }
        EXPECT_NEAR(std::exp(-0.1 * c.end), first(last, 1), 1e-5);
        EXPECT_EQ(c.end, rr.getCurrentTime());
        EXPECT_EQ(c.nAtEnd, rr.getValue("n"));

        const ls::DoubleMatrix second = simulateVariableStep(rr, c.end, stop - c.end);
        ASSERT_GT(second.numRows(), 2u);
        const unsigned lastSecond = second.numRows() - 1;
        for (unsigned i = 1; i <= lastSecond; i++) {
            EXPECT_GT(second(i, 0), second(i - 1, 0)) << "row " << i;
        }
        EXPECT_EQ(c.end, second(0, 0));
        EXPECT_EQ(c.nAtEnd, second(0, 2));
        EXPECT_EQ(stop, second(lastSecond, 0));
        EXPECT_EQ(1, second(lastSecond, 2));
        EXPECT_NEAR(std::exp(-0.1 * stop), second(lastSecond, 1), 1e-5);
    }
}

/**
 * An event with a delay whose trigger becomes true at the end of a variable step simulation is
 * not applied at the end, it is due after it.  It stays pending when the simulation ends, and
 * fires once, at its time, in a simulation that continues from the end.
 */
TEST_F(CVODEIntegratorTests, DelayedEventTriggeredAtTheEndOfAVariableStepSimulationFiresAfterItsDelay) {
    const double triggerTime = 1.0;
    const double delay = 0.5;
    const double stop = 5;

    RoadRunner rr(endTriggerModel("geq", triggerTime, delay));
    rr.setSelections(std::vector<std::string>{"time", "X", "n"});

    const ls::DoubleMatrix first = simulateVariableStep(rr, 0, triggerTime);
    ASSERT_GT(first.numRows(), 2u);
    EXPECT_EQ(triggerTime, first(first.numRows() - 1, 0));
    for (unsigned i = 0; i < first.numRows(); i++) {
        EXPECT_EQ(0, first(i, 2)) << "row " << i;
    }
    EXPECT_EQ(triggerTime, rr.getCurrentTime());
    EXPECT_EQ(0, rr.getValue("n"));

    const ls::DoubleMatrix second = simulateVariableStep(rr, triggerTime, stop - triggerTime);
    ASSERT_GT(second.numRows(), 2u);
    const unsigned last = second.numRows() - 1;
    EXPECT_EQ(triggerTime, second(0, 0));
    EXPECT_EQ(0, second(0, 2));

    bool fired = false;
    unsigned firedRow = 0;
    for (unsigned i = 1; i <= last; i++) {
        EXPECT_GT(second(i, 0), second(i - 1, 0)) << "row " << i;
        if (!fired && second(i, 2) > 0.5) {
            fired = true;
            firedRow = i;
        }
    }
    ASSERT_TRUE(fired);
    EXPECT_NEAR(triggerTime + delay, second(firedRow, 0), 1e-6);
    for (unsigned i = firedRow; i <= last; i++) {
        EXPECT_EQ(1, second(i, 2)) << "row " << i;
    }
    EXPECT_EQ(stop, second(last, 0));
    EXPECT_NEAR(std::exp(-0.1 * stop), second(last, 1), 1e-5);
}

// model delayed
//   X = 0; X' = 1
//   Y = 0
//   D: at 0.2 after (time >= 0.25): Y = Y + 1
//   E: at (time >= 0.6): Y = Y + 10
static const std::string DelayedEventSBML = R"(<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model metaid="delayed" id="delayed">
    <listOfParameters>
      <parameter id="X" value="0" constant="false"/>
      <parameter id="Y" value="0" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <rateRule variable="X">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <cn> 1 </cn>
        </math>
      </rateRule>
    </listOfRules>
    <listOfEvents>
      <event id="D" useValuesFromTriggerTime="true">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply>
              <geq/>
              <csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time"> time </csymbol>
              <cn> 0.25 </cn>
            </apply>
          </math>
        </trigger>
        <delay>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <cn> 0.2 </cn>
          </math>
        </delay>
        <listOfEventAssignments>
          <eventAssignment variable="Y">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply>
                <plus/>
                <ci> Y </ci>
                <cn> 1 </cn>
              </apply>
            </math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
      <event id="E" useValuesFromTriggerTime="true">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply>
              <geq/>
              <csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time"> time </csymbol>
              <cn> 0.6 </cn>
            </apply>
          </math>
        </trigger>
        <listOfEventAssignments>
          <eventAssignment variable="Y">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply>
                <plus/>
                <ci> Y </ci>
                <cn> 10 </cn>
              </apply>
            </math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>)";

/**
 * Regression test for sys-bio/roadrunner#1446 (a regression of #1390): a variable step simulation
 * of an event with a delay had no row of the state before the event.  CVODE is stopped at the
 * time of the event, and the integrator applied the event to the model there, before the row at
 * the stop was reported.  Y is not in the state vector of CVODE, so the state written back for the
 * row before the event did not undo it, and both rows at the time of the event showed Y after it.
 *
 * Event E has no delay and is the reference: its two rows are Y before it and Y after it.  The
 * delayed event D must give the same, with the row before it at its time (0.25 + 0.2), not at
 * the CVODE step before it.  The trigger of D, at 0.25, ends a CVODE step (the root), so it has one
 * row.  Nothing is assigned there, so no row after it is added.
 */
TEST_F(CVODEIntegratorTests, DelayedEventInAVariableStepSimulationHasARowOfTheStateBeforeIt) {
    struct EventRows {
        const char *name;
        double time;
        double yBefore;
        double yAfter;
    };
    const EventRows events[] = {
            {"E", 0.6, 1, 11},
            {"D", 0.45, 0, 1},
    };

    RoadRunner rr(DelayedEventSBML);
    rr.setSelections(std::vector<std::string>{"time", "X", "Y"});
    const ls::DoubleMatrix result = simulateVariableStep(rr, 0, 1);
    ASSERT_GT(result.numRows(), 4u);
    const unsigned last = result.numRows() - 1;
    for (unsigned i = 1; i <= last; i++) {
        EXPECT_GT(result(i, 0), result(i - 1, 0)) << "row " << i;
    }
    EXPECT_EQ(1.0, result(last, 0));
    EXPECT_EQ(11, result(last, 2));

    const double timeTolerance = 1e-6;
    for (const EventRows &event : events) {
        SCOPED_TRACE(std::string("event ") + event.name);

        unsigned nearEvent = 0;
        unsigned first = 0;
        for (unsigned i = 0; i <= last; i++) {
            if (fabs(result(i, 0) - event.time) <= timeTolerance) {
                if (nearEvent == 0) {
                    first = i;
                }
                nearEvent++;
            }
        }
        // the row before the event and the row after it
        ASSERT_EQ(2u, nearEvent);
        EXPECT_EQ(event.yBefore, result(first, 2)) << "row " << first << " (before the event)";
        EXPECT_EQ(event.yAfter, result(first + 1, 2)) << "row " << first + 1 << " (after the event)";
        EXPECT_NEAR(event.time, result(first, 1), timeTolerance);
        EXPECT_NEAR(event.time, result(first + 1, 1), timeTolerance);
    }

    unsigned nearTrigger = 0;
    for (unsigned i = 0; i <= last; i++) {
        if (fabs(result(i, 0) - 0.25) <= timeTolerance) {
            EXPECT_EQ(0, result(i, 2)) << "row " << i;
            nearTrigger++;
        }
    }
    EXPECT_EQ(1u, nearTrigger) << "rows at the time of the trigger of D";
}

// an event with a delay that adds an increment to a variable when time reaches the trigger time
static std::string delayedIncrementEvent(const std::string &id, const std::string &variable, double increment,
                                         double triggerTime, double delay) {
    return "<event id=\"" + id + "\" useValuesFromTriggerTime=\"true\">"
           "<trigger initialValue=\"false\" persistent=\"true\">"
           "<math xmlns=\"http://www.w3.org/1998/Math/MathML\"><apply><geq/>"
           "<csymbol encoding=\"text\" definitionURL=\"http://www.sbml.org/sbml/symbols/time\"> time </csymbol>"
           "<cn> " + std::to_string(triggerTime) + " </cn></apply></math></trigger>"
           "<delay><math xmlns=\"http://www.w3.org/1998/Math/MathML\"><cn> " + std::to_string(delay) +
           " </cn></math></delay>"
           "<listOfEventAssignments><eventAssignment variable=\"" + variable + "\">"
           "<math xmlns=\"http://www.w3.org/1998/Math/MathML\"><apply><plus/><ci> " + variable + " </ci>"
           "<cn> " + std::to_string(increment) + " </cn></apply></math></eventAssignment>"
           "</listOfEventAssignments></event>";
}

// model two_delayed
//   X = 0; X' = 1
//   Y = 0
//   Z = 0
//   D1: at 0.2 after (time >= 0.25): Y = Y + 1
//   D2: at 0.2 after (time >= 0.25): Z = Z + 10
static std::string twoDelayedEventsModel() {
    return R"(<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model metaid="two_delayed" id="two_delayed">
    <listOfParameters>
      <parameter id="X" value="0" constant="false"/>
      <parameter id="Y" value="0" constant="false"/>
      <parameter id="Z" value="0" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <rateRule variable="X">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <cn> 1 </cn>
        </math>
      </rateRule>
    </listOfRules>
    <listOfEvents>)" + delayedIncrementEvent("D1", "Y", 1, 0.25, 0.2) +
           delayedIncrementEvent("D2", "Z", 10, 0.25, 0.2) + "</listOfEvents></model></sbml>";
}

/**
 * Two events with a delay that are assigned at the same time have one row before them and one row
 * after both of them, not a row between.  They are simultaneous, so there is no state of the model
 * between the two assignments, whatever order the priorities or a tie-break give.  Both events are
 * triggered at 0.25 and have a delay of 0.2, so both are due at 0.45.  The trigger, which ends a
 * CVODE step, has one row.
 */
TEST_F(CVODEIntegratorTests, TwoDelayedEventsDueAtTheSameTimeHaveOneRowBeforeBothAndOneAfterBoth) {
    RoadRunner rr(twoDelayedEventsModel());
    rr.setSelections(std::vector<std::string>{"time", "X", "Y", "Z"});
    const ls::DoubleMatrix result = simulateVariableStep(rr, 0, 1);
    ASSERT_GT(result.numRows(), 4u);
    const unsigned last = result.numRows() - 1;

    for (unsigned i = 0; i <= last; i++) {
        if (i > 0) {
            EXPECT_GT(result(i, 0), result(i - 1, 0)) << "row " << i;
        }
        // neither event is assigned, or both are: there is no row with only one of them assigned
        const bool before = result(i, 2) == 0 && result(i, 3) == 0;
        const bool after = result(i, 2) == 1 && result(i, 3) == 10;
        EXPECT_TRUE(before || after) << "row " << i << ": Y = " << result(i, 2) << ", Z = " << result(i, 3);
    }
    EXPECT_EQ(1.0, result(last, 0));
    EXPECT_EQ(1, result(last, 2));
    EXPECT_EQ(10, result(last, 3));

    const double timeTolerance = 1e-6;

    unsigned nearTrigger = 0;
    for (unsigned i = 0; i <= last; i++) {
        if (fabs(result(i, 0) - 0.25) <= timeTolerance) {
            nearTrigger++;
        }
    }
    EXPECT_EQ(1u, nearTrigger) << "rows at the time of the triggers";

    unsigned nearEvents = 0;
    unsigned first = 0;
    for (unsigned i = 0; i <= last; i++) {
        if (fabs(result(i, 0) - 0.45) <= timeTolerance) {
            if (nearEvents == 0) {
                first = i;
            }
            nearEvents++;
        }
    }
    // the row before both events and the row after both
    ASSERT_EQ(2u, nearEvents);
    EXPECT_EQ(0, result(first, 2)) << "row " << first << " (before the events)";
    EXPECT_EQ(0, result(first, 3)) << "row " << first << " (before the events)";
    EXPECT_EQ(1, result(first + 1, 2)) << "row " << first + 1 << " (after the events)";
    EXPECT_EQ(10, result(first + 1, 3)) << "row " << first + 1 << " (after the events)";
    EXPECT_NEAR(0.45, result(first, 1), timeTolerance);
    EXPECT_NEAR(0.45, result(first + 1, 1), timeTolerance);
}

// model dose
//   A = 0; k = 0.1; A' = -k * A
//   Y = 0
//   E: at (time >= 1.6): A = A + 10, Y = Y + 1
static const std::string DoseSBML = R"(<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model metaid="dose" id="dose">
    <listOfParameters>
      <parameter id="A" value="0" constant="false"/>
      <parameter id="Y" value="0" constant="false"/>
      <parameter id="k" value="0.1" constant="true"/>
    </listOfParameters>
    <listOfRules>
      <rateRule variable="A">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply>
            <times/>
            <apply>
              <minus/>
              <ci> k </ci>
            </apply>
            <ci> A </ci>
          </apply>
        </math>
      </rateRule>
    </listOfRules>
    <listOfEvents>
      <event id="E" useValuesFromTriggerTime="true">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply>
              <geq/>
              <csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time"> time </csymbol>
              <cn> 1.6 </cn>
            </apply>
          </math>
        </trigger>
        <listOfEventAssignments>
          <eventAssignment variable="A">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply>
                <plus/>
                <ci> A </ci>
                <cn> 10 </cn>
              </apply>
            </math>
          </eventAssignment>
          <eventAssignment variable="Y">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply>
                <plus/>
                <ci> Y </ci>
                <cn> 1 </cn>
              </apply>
            </math>
          </eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>)";

/**
 * Regression test for the follow-up to sys-bio/roadrunner#1445 (reported by sbmlsim): an event whose
 * trigger is false at the end of a variable step simulation fired there.  A segment of a simulation
 * ends one unit in the last place before a change, so that an event at the time of the change fires
 * after it, and the event fired at the end of the segment and again in the simulation that continued.
 *
 * The caller of the integrator gives it the length of the step, and the integrator adds it to the
 * start of the step to find the end.  For some starts of the last step that sum is not the requested
 * end but the time of the event, one unit in the last place after it, so the root of the event was
 * taken to be at the end.  Ends from 1.6 down to eight units in the last place below it are tried, so
 * that the test does not depend on the steps that CVODE takes.  Only an end at or after 1.6 has the
 * event fired, and the event fires once in all of the simulations that continue to a later time.
 */
TEST_F(CVODEIntegratorTests, EventWhoseTriggerIsFalseAtTheEndOfAVariableStepSimulationDoesNotFire) {
    const double eventTime = 1.6;
    const double stop = 5;

    RoadRunner rr(DoseSBML);
    rr.setSelections(std::vector<std::string>{"time", "A", "Y"});

    double end = eventTime;
    for (int ulps = 0; ulps <= 8; ulps++, end = std::nextafter(end, 0.0)) {
        std::ostringstream what;
        what.precision(17);
        what << "end " << end << " (" << ulps << " units in the last place below " << eventTime << ")";
        SCOPED_TRACE(what.str());

        resetAllValues(rr);
        const double fired = ulps == 0 ? 1 : 0;

        const ls::DoubleMatrix first = simulateVariableStep(rr, 0, end);
        ASSERT_GT(first.numRows(), 2u);
        const unsigned last = first.numRows() - 1;
        for (unsigned i = 1; i <= last; i++) {
            EXPECT_GT(first(i, 0), first(i - 1, 0)) << "row " << i;
        }
        EXPECT_EQ(end, first(last, 0));
        EXPECT_EQ(10 * fired, first(last, 1));
        EXPECT_EQ(fired, first(last, 2));
        EXPECT_EQ(end, rr.getCurrentTime());
        EXPECT_EQ(fired, rr.getValue("Y"));

        const ls::DoubleMatrix second = simulateVariableStep(rr, end, stop - end);
        ASSERT_GT(second.numRows(), 2u);
        const unsigned lastSecond = second.numRows() - 1;
        for (unsigned i = 1; i <= lastSecond; i++) {
            EXPECT_GT(second(i, 0), second(i - 1, 0)) << "row " << i;
        }
        EXPECT_EQ(end, second(0, 0));
        EXPECT_EQ(fired, second(0, 2));
        EXPECT_EQ(stop, second(lastSecond, 0));
        EXPECT_EQ(1, second(lastSecond, 2));
        EXPECT_NEAR(10 * std::exp(-0.1 * (stop - eventTime)), second(lastSecond, 1), 1e-4);
    }
}

/**
 * The fixed step version of the test above: the output times are given, and the integrator is given
 * the length of the step from one output time to the next, and adds it to the previous output time to
 * find the end of the step.  For the previous output time here, that sum for an end one unit in the
 * last place below the time of the event is the time of the event, so the row at the end had the event
 * fired.  Only an end at or after 1.6 has the event fired, and it fires once in the simulation that
 * continues to a later time.
 */
TEST_F(CVODEIntegratorTests, EventWhoseTriggerIsFalseAtAnOutputTimeOfAFixedStepSimulationDoesNotFire) {
    const double eventTime = 1.6;
    const double previousOutputTime = 0.5492163665262385;
    const double stop = 5;

    RoadRunner rr(DoseSBML);
    rr.setSelections(std::vector<std::string>{"time", "A", "Y"});
    rr.getIntegrator()->setValue("variable_step_size", false);

    double end = eventTime;
    for (int ulps = 0; ulps <= 8; ulps++, end = std::nextafter(end, 0.0)) {
        std::ostringstream what;
        what.precision(17);
        what << "end " << end << " (" << ulps << " units in the last place below " << eventTime << ")";
        SCOPED_TRACE(what.str());

        resetAllValues(rr);
        const double fired = ulps == 0 ? 1 : 0;

        const ls::DoubleMatrix first = *rr.simulate(std::vector<double>{0.0, previousOutputTime, end});
        ASSERT_EQ(3u, first.numRows());
        EXPECT_EQ(previousOutputTime, first(1, 0));
        EXPECT_EQ(0, first(1, 1));
        EXPECT_EQ(0, first(1, 2));
        EXPECT_EQ(end, first(2, 0));
        EXPECT_EQ(10 * fired, first(2, 1));
        EXPECT_EQ(fired, first(2, 2));

        const ls::DoubleMatrix second = *rr.simulate(std::vector<double>{end, stop});
        ASSERT_EQ(2u, second.numRows());
        EXPECT_EQ(fired, second(0, 2));
        EXPECT_EQ(1, second(1, 2));
        EXPECT_NEAR(10 * std::exp(-0.1 * (stop - eventTime)), second(1, 1), 1e-4);
    }
}

/**
 * A reset returns to time 0, where an event whose trigger is false at that time must not fire.  The
 * reset tested the events at the start time of the last simulation, which was the time of the event
 * here, so the event fired during the reset and the next simulation began with its assignments.
 */
TEST_F(CVODEIntegratorTests, ResetAfterASimulationStartingAtTheTimeOfAnEventDoesNotFireTheEvent) {
    RoadRunner rr(DoseSBML);
    rr.setSelections(std::vector<std::string>{"time", "A", "Y"});
    rr.getIntegrator()->setValue("variable_step_size", false);

    rr.simulate(std::vector<double>{0.0, 1.6});
    const ls::DoubleMatrix later = *rr.simulate(std::vector<double>{1.6, 5.0});
    EXPECT_EQ(1, later(1, 2));

    resetAllValues(rr);
    EXPECT_EQ(0, rr.getValue("A"));
    EXPECT_EQ(0, rr.getValue("Y"));

    const ls::DoubleMatrix again = *rr.simulate(std::vector<double>{0.0, 1.0});
    ASSERT_EQ(2u, again.numRows());
    for (unsigned i = 0; i < again.numRows(); i++) {
        EXPECT_EQ(0, again(i, 1)) << "row " << i;
        EXPECT_EQ(0, again(i, 2)) << "row " << i;
    }
}

// ---- The roots of the relational leaves of event triggers ----
// The models below have X' = 1, three counters n, m and p, and events that add one to a counter.

static const std::string LeafTimeMath = R"(<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time"> time </csymbol>)";

// MathML for a comparison of a symbol, X or time, with a constant.  The op is gt, lt, geq, ...
static std::string leafRelation(const std::string &op, const std::string &symbol, const std::string &value) {
    const std::string left = symbol == "time" ? LeafTimeMath : "<ci> " + symbol + " </ci>";
    return "<apply><" + op + "/>" + left + "<cn> " + value + " </cn></apply>";
}

// MathML for and, or, not, ... of the operands
static std::string leafLogical(const std::string &op, const std::vector<std::string> &operands) {
    std::string result = "<apply><" + op + "/>";
    for (const std::string &operand : operands) {
        result += operand;
    }
    return result + "</apply>";
}

// the trigger of a window: true for low < symbol < high, which is shorter than the steps of CVODE
static std::string leafWindow(const std::string &low, const std::string &high, const std::string &symbol = "X") {
    return leafLogical("and", {leafRelation("gt", symbol, low), leafRelation("lt", symbol, high)});
}

// an event that adds one to the variable when its trigger becomes true, after the delay if it has one
static std::string leafEvent(const std::string &id, const std::string &trigger, const std::string &variable = "n",
                             const std::string &delay = "", bool initialValue = false, bool persistent = true) {
    std::string xml = "      <event id=\"" + id + "\" useValuesFromTriggerTime=\"true\">\n";
    xml += std::string("        <trigger initialValue=\"") + (initialValue ? "true" : "false") +
           "\" persistent=\"" + (persistent ? "true" : "false") + "\">\n";
    xml += "          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">" + trigger + "</math>\n";
    xml += "        </trigger>\n";
    if (!delay.empty()) {
        xml += "        <delay><math xmlns=\"http://www.w3.org/1998/Math/MathML\"><cn> " + delay + " </cn></math></delay>\n";
    }
    xml += "        <listOfEventAssignments>\n";
    xml += "          <eventAssignment variable=\"" + variable + "\">\n";
    xml += "            <math xmlns=\"http://www.w3.org/1998/Math/MathML\"><apply><plus/><ci> " + variable +
           " </ci><cn type=\"integer\"> 1 </cn></apply></math>\n";
    xml += "          </eventAssignment>\n";
    xml += "        </listOfEventAssignments>\n";
    xml += "      </event>\n";
    return xml;
}

static std::string leafModelSBML(const std::string &events) {
    return std::string(R"(<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model metaid="leaves" id="leaves">
    <listOfParameters>
      <parameter id="X" value="0" constant="false"/>
      <parameter id="n" value="0" constant="false"/>
      <parameter id="m" value="0" constant="false"/>
      <parameter id="p" value="0" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <rateRule variable="X">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <cn type="integer"> 1 </cn>
        </math>
      </rateRule>
    </listOfRules>
    <listOfEvents>
)") + events + R"(    </listOfEvents>
  </model>
</sbml>)";
}

// The counters n, m and p at the end of simulations to time 10 have the expected values, whatever the
// output times of a fixed step simulation and the maximum time step of a variable step simulation are.
static void checkLeafEventCounts(RoadRunner &rr, double expectedCountN, double expectedCountM = 0,
                                 double expectedCountP = 0) {
    rr.setSelections(std::vector<std::string>{"time", "n", "m", "p"});

    rr.getIntegrator()->setValue("variable_step_size", false);
    for (int points : {11, 21, 51, 101}) {
        SCOPED_TRACE("fixed step with " + std::to_string(points) + " output points");
        resetAllValues(rr);
        std::vector<double> times;
        for (int i = 0; i < points; i++) {
            times.push_back(10.0 * i / (points - 1));
        }
        const ls::DoubleMatrix result = *rr.simulate(times);
        ASSERT_EQ(static_cast<unsigned>(points), result.numRows());
        EXPECT_EQ(expectedCountN, result(points - 1, 1));
        EXPECT_EQ(expectedCountM, result(points - 1, 2));
        EXPECT_EQ(expectedCountP, result(points - 1, 3));
    }

    for (double maximumTimeStep : {0.0, 1.0, 0.1}) {
        SCOPED_TRACE("variable step with a maximum time step of " + std::to_string(maximumTimeStep));
        resetAllValues(rr);
        rr.getIntegrator()->setValue("maximum_time_step", maximumTimeStep);
        const ls::DoubleMatrix result = simulateVariableStep(rr, 0, 10);
        ASSERT_GT(result.numRows(), 1u);
        EXPECT_EQ(expectedCountN, result(result.numRows() - 1, 1));
        EXPECT_EQ(expectedCountM, result(result.numRows() - 1, 2));
        EXPECT_EQ(expectedCountP, result(result.numRows() - 1, 3));
    }
}

static std::unordered_map<std::string, Setting> leafStatistics(RoadRunner &rr) {
    auto *cvode = dynamic_cast<CVODEIntegrator *>(rr.getIntegrator());
    EXPECT_TRUE(cvode);
    return cvode->getStatistics();
}

TEST_F(CVODEIntegratorTests, EventWhoseAndTriggerIsTrueOnlyWithinOneStepFires) {
    RoadRunner rr(leafModelSBML(leafEvent("E", leafWindow("0.5", "0.8"))));
    EXPECT_EQ(2, rr.getModel()->getNumBooleanLeaves());
    checkLeafEventCounts(rr, 1);
}

TEST_F(CVODEIntegratorTests, EventWhoseNotOrTriggerIsTrueOnlyWithinOneStepFires) {
    const std::string trigger = leafLogical("not", {leafLogical("or", {leafRelation("lt", "X", "0.5"),
                                                                       leafRelation("gt", "X", "0.8")})});
    RoadRunner rr(leafModelSBML(leafEvent("E", trigger)));
    EXPECT_EQ(2, rr.getModel()->getNumBooleanLeaves());
    checkLeafEventCounts(rr, 1);
}

TEST_F(CVODEIntegratorTests, EventWhoseTriggerIsAWindowOfTimeFires) {
    RoadRunner rr(leafModelSBML(leafEvent("E", leafWindow("0.5", "0.8", "time"))));
    EXPECT_EQ(2, rr.getModel()->getNumBooleanLeaves());
    checkLeafEventCounts(rr, 1);
}

// The event fires where the window opens, not before it, and its assignment is there from then on.
TEST_F(CVODEIntegratorTests, EventWithAWindowTriggerFiresWhereTheWindowOpens) {
    RoadRunner rr(leafModelSBML(leafEvent("E", leafWindow("0.5", "0.8"))));
    rr.setSelections(std::vector<std::string>{"time", "n", "m"});

    rr.getIntegrator()->setValue("variable_step_size", false);
    const ls::DoubleMatrix fixed = *rr.simulate(std::vector<double>{0.0, 0.4, 1.0, 2.0});
    ASSERT_EQ(4u, fixed.numRows());
    EXPECT_EQ(0, fixed(1, 1));
    EXPECT_EQ(1, fixed(2, 1));
    EXPECT_EQ(1, fixed(3, 1));

    resetAllValues(rr);
    const ls::DoubleMatrix variable = simulateVariableStep(rr, 0, 2);
    unsigned first = 0;
    for (unsigned i = 0; i < variable.numRows() && first == 0; i++) {
        if (variable(i, 1) > 0.5) {
            first = i;
        }
    }
    ASSERT_GT(first, 0u);
    EXPECT_NEAR(0.5, variable(first, 0), 1e-6);
    EXPECT_EQ(0, variable(first - 1, 1));
    EXPECT_LE(variable(first - 1, 0), 0.5 + 1e-9);
    EXPECT_EQ(1, variable(variable.numRows() - 1, 1));
}

TEST_F(CVODEIntegratorTests, EventWhoseTriggerHasTwoWindowsFiresOncePerWindow) {
    const std::string trigger = leafLogical("or", {leafWindow("0.5", "0.8"), leafWindow("2", "2.3")});
    RoadRunner rr(leafModelSBML(leafEvent("E", trigger)));
    EXPECT_EQ(4, rr.getModel()->getNumBooleanLeaves());
    checkLeafEventCounts(rr, 2);
}

// The comparison X > 0.5 is a root once, and both events fire.
TEST_F(CVODEIntegratorTests, TwoEventsThatShareAComparisonBothFire) {
    RoadRunner rr(leafModelSBML(leafEvent("E1", leafWindow("0.5", "0.8"), "n") +
                                leafEvent("E2", leafWindow("0.5", "2"), "m")));
    EXPECT_EQ(3, rr.getModel()->getNumBooleanLeaves());
    checkLeafEventCounts(rr, 1, 1);
}

// A trigger with one comparison, a comparison under a not, or the same comparison twice, has no leaf
// roots: the root of the event is enough.
TEST_F(CVODEIntegratorTests, TriggersWithOneDistinctComparisonHaveNoLeafRoots) {
    const std::string comparison = leafRelation("gt", "X", "0.5");
    RoadRunner rr(leafModelSBML(
            leafEvent("E1", comparison, "n") +
            leafEvent("E2", leafLogical("not", {comparison}), "m", "", true) +
            leafEvent("E3", leafLogical("and", {comparison, comparison}), "p")));
    EXPECT_EQ(0, rr.getModel()->getNumBooleanLeaves());
    checkLeafEventCounts(rr, 1, 0, 1);
}

TEST_F(CVODEIntegratorTests, NestedTriggerHasTheLeavesOfAllItsComparisons) {
    const std::string trigger = leafLogical("not", {leafLogical("or", {leafWindow("0.5", "0.8"),
                                                                       leafRelation("gt", "X", "3")})});
    RoadRunner rr(leafModelSBML(leafEvent("E", trigger, "n", "", true)));
    EXPECT_EQ(3, rr.getModel()->getNumBooleanLeaves());
    // true at the start, false from 0.5 to 0.8 and after 3, true from 0.8 to 3: fires once, at 0.8
    checkLeafEventCounts(rr, 1);
}

// A window that is open at the start, with an initial trigger value of true, does not fire.
TEST_F(CVODEIntegratorTests, EventWhoseWindowIsOpenAtTheStartWithAnInitialValueOfTrueDoesNotFire) {
    RoadRunner rr(leafModelSBML(leafEvent("E", leafWindow("-1", "0.8"), "n", "", true)));
    EXPECT_EQ(2, rr.getModel()->getNumBooleanLeaves());
    checkLeafEventCounts(rr, 0);
}

TEST_F(CVODEIntegratorTests, EventWithANonPersistentWindowTriggerFires) {
    RoadRunner rr(leafModelSBML(leafEvent("E", leafWindow("0.5", "0.8"), "n", "", false, false)));
    checkLeafEventCounts(rr, 1);
}

// The trigger of a persistent event with a delay is false again before the delay is over, and the
// event fires after its delay.
TEST_F(CVODEIntegratorTests, PersistentEventWithADelayFiresAfterItsWindowIsClosed) {
    RoadRunner rr(leafModelSBML(leafEvent("E", leafWindow("0.5", "0.8"), "n", "2", false, true)));
    EXPECT_EQ(2, rr.getModel()->getNumBooleanLeaves());
    checkLeafEventCounts(rr, 1);

    rr.setSelections(std::vector<std::string>{"time", "n", "m"});
    rr.getIntegrator()->setValue("variable_step_size", false);
    resetAllValues(rr);
    const ls::DoubleMatrix result = *rr.simulate(std::vector<double>{0.0, 2.0, 3.0});
    ASSERT_EQ(3u, result.numRows());
    EXPECT_EQ(0, result(1, 1));
    EXPECT_EQ(1, result(2, 1));
}

// The trigger of a non-persistent event with a delay goes false before the delay is over, which
// removes the event.  This is a change at a root of a comparison, with an event queued.
TEST_F(CVODEIntegratorTests, NonPersistentEventWithADelayIsRemovedWhenItsWindowIsClosed) {
    RoadRunner rr(leafModelSBML(leafEvent("E", leafWindow("0.5", "0.8"), "n", "2", false, false)));
    EXPECT_EQ(2, rr.getModel()->getNumBooleanLeaves());
    checkLeafEventCounts(rr, 0);
}

// The roots of comparisons at which no event is triggered change nothing: the rows of a variable step
// simulation are those of the same simulation without them, and CVODE is not reinitialized.
TEST_F(CVODEIntegratorTests, LeafRootsThatChangeNothingAddNoRowsAndNoReinitializations) {
    // the trigger is never true, though X > 0.5 and X < 0.3 each change
    const std::string never = leafLogical("and", {leafRelation("gt", "X", "0.5"), leafRelation("lt", "X", "0.3")});
    RoadRunner withLeaves(leafModelSBML(leafEvent("E", never)));
    RoadRunner without(leafModelSBML(leafEvent("E", leafRelation("gt", "X", "1000"))));
    ASSERT_EQ(2, withLeaves.getModel()->getNumBooleanLeaves());
    ASSERT_EQ(0, without.getModel()->getNumBooleanLeaves());
    withLeaves.setSelections(std::vector<std::string>{"time", "X"});
    without.setSelections(std::vector<std::string>{"time", "X"});

    const ls::DoubleMatrix a = simulateVariableStep(withLeaves, 0, 10);
    const ls::DoubleMatrix b = simulateVariableStep(without, 0, 10);
    ASSERT_EQ(b.numRows(), a.numRows());
    for (unsigned i = 0; i < a.numRows(); i++) {
        EXPECT_DOUBLE_EQ(b(i, 0), a(i, 0)) << "row " << i;
        EXPECT_DOUBLE_EQ(b(i, 1), a(i, 1)) << "row " << i;
    }
    EXPECT_EQ(0, leafStatistics(withLeaves)["num_reinits"].getAs<std::int64_t>());
    EXPECT_EQ(0, leafStatistics(without)["num_reinits"].getAs<std::int64_t>());
}

// The event fires at the root of X > 0.5, which reinitializes CVODE.  The root of X < 0.8, where the
// trigger goes false, changes nothing, so CVODE continues from it.
TEST_F(CVODEIntegratorTests, RootOfAnBooleanLeafThatChangesNothingDoesNotReinitializeCVODE) {
    RoadRunner rr(leafModelSBML(leafEvent("E", leafWindow("0.5", "0.8"))));
    rr.setSelections(std::vector<std::string>{"time", "n"});
    rr.getIntegrator()->setValue("variable_step_size", false);

    const ls::DoubleMatrix result = *rr.simulate(std::vector<double>{0.0, 5.0, 10.0});
    ASSERT_EQ(3u, result.numRows());
    EXPECT_EQ(1, result(2, 1));

    std::unordered_map<std::string, Setting> stats = leafStatistics(rr);
    EXPECT_EQ(1, stats["num_events"].getAs<std::int64_t>());
    EXPECT_EQ(1, stats["num_reinits"].getAs<std::int64_t>());
}

// An event with a simple trigger reinitializes CVODE once each time it fires, as before.
TEST_F(CVODEIntegratorTests, EventWithASimpleTriggerReinitializesCVODEOncePerFiring) {
    RoadRunner rr(SawtoothSBML);
    ASSERT_EQ(0, rr.getModel()->getNumBooleanLeaves());
    rr.simulate(0, 10, 11);
    std::unordered_map<std::string, Setting> stats = leafStatistics(rr);
    ASSERT_GT(stats["num_events"].getAs<std::int64_t>(), 1);
    EXPECT_EQ(stats["num_events"].getAs<std::int64_t>(), stats["num_reinits"].getAs<std::int64_t>());
}

// A model taken from the model cache, as it is while another instance of the same SBML is alive, has
// the boolean leaves of a compiled one.
TEST_F(CVODEIntegratorTests, ModelFromTheModelCacheHasTheBooleanLeavesOfACompiledModel) {
    const std::string sbml = leafModelSBML(leafEvent("E", leafWindow("0.5", "0.8")));
    RoadRunner first(sbml);   // compiled
    RoadRunner cached(sbml);  // from the cache, as long as first is alive
    ASSERT_EQ(2, first.getModel()->getNumBooleanLeaves());
    EXPECT_EQ(2, cached.getModel()->getNumBooleanLeaves());
    checkLeafEventCounts(cached, 1);
}

TEST_F(CVODEIntegratorTests, SavedStateHasTheBooleanLeaves) {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "roadrunner-tests" / "BooleanLeaves";
    std::filesystem::create_directories(dir);
    const std::string file = (dir / "window.rr").string();

    RoadRunner rr(leafModelSBML(leafEvent("E", leafWindow("0.5", "0.8"))));
    rr.saveState(file);
    RoadRunner loaded;
    loaded.loadState(file);
    std::filesystem::remove(file);

    EXPECT_EQ(2, loaded.getModel()->getNumBooleanLeaves());
    checkLeafEventCounts(loaded, 1);
}

// The trigger of this event goes false at X = 1, which changes nothing: the rows of a variable step
// simulation are those of the same simulation with an event that never changes, and CVODE is not
// reinitialized.
TEST_F(CVODEIntegratorTests, RootWhereATriggerGoesFalseAddsNoRowAndNoReinitialization) {
    RoadRunner goesFalse(leafModelSBML(leafEvent("E", leafRelation("lt", "X", "1"), "n", "", true)));
    RoadRunner unchanged(leafModelSBML(leafEvent("E", leafRelation("gt", "X", "1000"))));
    ASSERT_EQ(0, goesFalse.getModel()->getNumBooleanLeaves());
    goesFalse.setSelections(std::vector<std::string>{"time", "X", "n"});
    unchanged.setSelections(std::vector<std::string>{"time", "X", "n"});

    const ls::DoubleMatrix a = simulateVariableStep(goesFalse, 0, 10);
    const ls::DoubleMatrix b = simulateVariableStep(unchanged, 0, 10);
    ASSERT_EQ(b.numRows(), a.numRows());
    for (unsigned i = 0; i < a.numRows(); i++) {
        EXPECT_DOUBLE_EQ(b(i, 0), a(i, 0)) << "row " << i;
        EXPECT_DOUBLE_EQ(b(i, 1), a(i, 1)) << "row " << i;
        EXPECT_EQ(0, a(i, 2)) << "row " << i;
    }
    EXPECT_EQ(0, leafStatistics(goesFalse)["num_reinits"].getAs<std::int64_t>());
    EXPECT_EQ(0, leafStatistics(unchanged)["num_reinits"].getAs<std::int64_t>());
}

// The trigger becomes true at X = 0.3, and X > 0.5 is true from 0.5 on, which changes nothing: the
// root of the comparison reinitializes CVODE no more than the event does.
TEST_F(CVODEIntegratorTests, RootOfAComparisonMaskedByAnotherComparisonDoesNotReinitializeCVODE) {
    const std::string trigger = leafLogical("or", {leafRelation("gt", "X", "0.3"), leafRelation("gt", "X", "0.5")});
    RoadRunner rr(leafModelSBML(leafEvent("E", trigger)));
    ASSERT_EQ(2, rr.getModel()->getNumBooleanLeaves());
    rr.setSelections(std::vector<std::string>{"time", "n"});
    rr.getIntegrator()->setValue("variable_step_size", false);

    const ls::DoubleMatrix result = *rr.simulate(std::vector<double>{0.0, 5.0, 10.0});
    ASSERT_EQ(3u, result.numRows());
    EXPECT_EQ(1, result(2, 1));

    std::unordered_map<std::string, Setting> stats = leafStatistics(rr);
    EXPECT_EQ(1, stats["num_events"].getAs<std::int64_t>());
    EXPECT_EQ(1, stats["num_reinits"].getAs<std::int64_t>());
}

// ---- Piecewise functions in kinetic laws with local parameters ----

// -> S with the rate piecewise(k, time < tsw, 0): S' = 2 until 3.3 and 0 after, so S(10) = 6.6.  The
// parameters k and tsw are local to the kinetic law or global.
static std::string piecewiseKineticLawSBML(bool kIsLocal, bool tswIsLocal) {
    const std::string kGlobal = R"(<parameter id="k" value="2" constant="true"/>)";
    const std::string tswGlobal = R"(<parameter id="tsw" value="3.3" constant="true"/>)";
    const std::string kLocal = R"(<localParameter id="k" value="2"/>)";
    const std::string tswLocal = R"(<localParameter id="tsw" value="3.3"/>)";
    const std::string globalList = (kIsLocal ? "" : kGlobal) + (tswIsLocal ? "" : tswGlobal);
    const std::string localList = (kIsLocal ? kLocal : "") + (tswIsLocal ? tswLocal : "");
    return std::string(R"(<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model metaid="piecewiseKineticLaw" id="piecewiseKineticLaw">
    <listOfCompartments>
      <compartment id="C" spatialDimensions="3" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="C" initialAmount="0" hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
)") + (globalList.empty() ? std::string() : "    <listOfParameters>" + globalList + "</listOfParameters>\n") + R"(    <listOfReactions>
      <reaction id="R" reversible="false" fast="false">
        <listOfProducts>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <piecewise>
              <piece>
                <ci> k </ci>
                <apply><lt/>)" + LeafTimeMath + R"(<ci> tsw </ci></apply>
              </piece>
              <otherwise><cn> 0 </cn></otherwise>
            </piecewise>
          </math>
)" + (localList.empty() ? std::string() : "          <listOfLocalParameters>" + localList + "</listOfLocalParameters>\n") + R"(        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>)";
}

// The piecewise of a kinetic law is a root whether its parameters are local or global.
TEST_F(CVODEIntegratorTests, PiecewiseInAKineticLawWithLocalParametersHasPiecewiseTriggers) {
    RoadRunner reference(piecewiseKineticLawSBML(false, false));
    ASSERT_GT(reference.getModel()->getNumPiecewiseTriggers(), 0);

    for (const auto &locals : {std::make_pair(true, true), std::make_pair(true, false), std::make_pair(false, true)}) {
        SCOPED_TRACE(std::string("k is ") + (locals.first ? "local" : "global") +
                     ", tsw is " + (locals.second ? "local" : "global"));
        RoadRunner rr(piecewiseKineticLawSBML(locals.first, locals.second));
        EXPECT_EQ(reference.getModel()->getNumPiecewiseTriggers(), rr.getModel()->getNumPiecewiseTriggers());
    }
}

// CVODE is reinitialized at the switch of the piecewise at 3.3 whether the parameters are local or
// global, and the integration is the same.
TEST_F(CVODEIntegratorTests, PiecewiseInAKineticLawWithLocalParametersIsReinitializedAtItsSwitch) {
    RoadRunner reference(piecewiseKineticLawSBML(false, false));
    RoadRunner local(piecewiseKineticLawSBML(true, true));
    reference.setSelections(std::vector<std::string>{"time", "S"});
    local.setSelections(std::vector<std::string>{"time", "S"});

    const ls::DoubleMatrix a = *reference.simulate(0, 10, 11);
    const ls::DoubleMatrix b = *local.simulate(0, 10, 11);
    ASSERT_EQ(a.numRows(), b.numRows());
    for (unsigned i = 0; i < a.numRows(); i++) {
        EXPECT_NEAR(a(i, 1), b(i, 1), 1e-14) << "row " << i;
    }

    const std::int64_t reinits = leafStatistics(reference)["num_reinits"].getAs<std::int64_t>();
    EXPECT_GT(reinits, 0);
    EXPECT_EQ(reinits, leafStatistics(local)["num_reinits"].getAs<std::int64_t>());
}

// ---- Piecewise functions inside function definitions ----

// Same model as piecewiseKineticLawSBML, but the piecewise is in the function definition f(a, b, c) =
// piecewise(a, b < c, 0) and the kinetic law is f(k, time, tsw).  With nested set, the kinetic law calls
// g(k, time, tsw) where g(a, b, c) = f(a, b, c).
static std::string piecewiseFunctionDefinitionSBML(bool nested, bool paramsAreLocal) {
    const std::string globals = R"(<parameter id="k" value="2" constant="true"/><parameter id="tsw" value="3.3" constant="true"/>)";
    const std::string locals = R"(<localParameter id="k" value="2"/><localParameter id="tsw" value="3.3"/>)";
    const std::string lambdaHead = R"(<lambda><bvar><ci> a </ci></bvar><bvar><ci> b </ci></bvar><bvar><ci> c </ci></bvar>)";
    const std::string gDefinition = nested ? R"(<functionDefinition id="g"><math xmlns="http://www.w3.org/1998/Math/MathML">)" + lambdaHead +
        R"(<apply><ci> f </ci><ci> a </ci><ci> b </ci><ci> c </ci></apply></lambda></math></functionDefinition>)" : std::string();
    return std::string(R"(<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model metaid="piecewiseFunctionDefinition" id="piecewiseFunctionDefinition">
    <listOfFunctionDefinitions>
      <functionDefinition id="f">
        <math xmlns="http://www.w3.org/1998/Math/MathML">)") + lambdaHead + R"(
          <piecewise>
            <piece><ci> a </ci><apply><lt/><ci> b </ci><ci> c </ci></apply></piece>
            <otherwise><cn> 0 </cn></otherwise>
          </piecewise></lambda>
        </math>
      </functionDefinition>)" + gDefinition + R"(
    </listOfFunctionDefinitions>
    <listOfCompartments>
      <compartment id="C" spatialDimensions="3" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="C" initialAmount="0" hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
)" + (paramsAreLocal ? std::string() : "    <listOfParameters>" + globals + "</listOfParameters>\n") + R"(    <listOfReactions>
      <reaction id="R" reversible="false" fast="false">
        <listOfProducts>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><ci> )" + (nested ? "g" : "f") + R"( </ci><ci> k </ci>)" + LeafTimeMath + R"(<ci> tsw </ci></apply>
          </math>
)" + (paramsAreLocal ? "          <listOfLocalParameters>" + locals + "</listOfLocalParameters>\n" : std::string()) + R"(        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>)";
}

// A piecewise inside a function definition (called directly or through another function, with global or
// local parameters as arguments) has the same roots and the same integration as the inlined piecewise.
TEST_F(CVODEIntegratorTests, PiecewiseInAFunctionDefinitionIsTheSameAsTheInlinedPiecewise) {
    RoadRunner reference(piecewiseKineticLawSBML(false, false));
    reference.setSelections(std::vector<std::string>{"time", "S"});
    const ls::DoubleMatrix a = *reference.simulate(0, 10, 11);
    const std::int64_t referenceReinits = leafStatistics(reference)["num_reinits"].getAs<std::int64_t>();
    ASSERT_GT(referenceReinits, 0);

    for (bool nested : {false, true}) {
        for (bool paramsAreLocal : {false, true}) {
            SCOPED_TRACE(std::string(nested ? "nested" : "direct") + " call, " + (paramsAreLocal ? "local" : "global") + " parameters");
            RoadRunner rr(piecewiseFunctionDefinitionSBML(nested, paramsAreLocal));
            EXPECT_EQ(reference.getModel()->getNumPiecewiseTriggers(), rr.getModel()->getNumPiecewiseTriggers());

            rr.setSelections(std::vector<std::string>{"time", "S"});
            const ls::DoubleMatrix b = *rr.simulate(0, 10, 11);
            ASSERT_EQ(a.numRows(), b.numRows());
            for (unsigned i = 0; i < a.numRows(); i++) {
                EXPECT_NEAR(a(i, 1), b(i, 1), 1e-14) << "row " << i;
            }
            EXPECT_EQ(referenceReinits, leafStatistics(rr)["num_reinits"].getAs<std::int64_t>());
        }
    }
}

// ---- Editing a model that has piecewise functions ----

// The reaction of piecewiseKineticLawSBML(true, true) as a stand-alone element for addReaction.
static const std::string PiecewiseReactionXML = std::string(R"(<reaction xmlns="http://www.sbml.org/sbml/level3/version2/core" id="R" reversible="false" fast="false">
  <listOfProducts>
    <speciesReference species="S" stoichiometry="1" constant="true"/>
  </listOfProducts>
  <kineticLaw>
    <math xmlns="http://www.w3.org/1998/Math/MathML">
      <piecewise>
        <piece>
          <ci> k </ci>
          <apply><lt/>)") + LeafTimeMath + R"(<ci> tsw </ci></apply>
        </piece>
        <otherwise><cn> 0 </cn></otherwise>
      </piecewise>
    </math>
    <listOfLocalParameters>
      <localParameter id="k" value="2"/>
      <localParameter id="tsw" value="3.3"/>
    </listOfLocalParameters>
  </kineticLaw>
</reaction>)";

// The S column of a simulation from 0 to 10 with 11 points.
static std::vector<double> simulateSpeciesS(RoadRunner &rr) {
    rr.setSelections(std::vector<std::string>{"time", "S"});
    const ls::DoubleMatrix result = *rr.simulate(0, 10, 11);
    std::vector<double> s;
    for (unsigned i = 0; i < result.numRows(); i++) {
        s.push_back(result(i, 1));
    }
    return s;
}

static void expectSameSimulation(const std::vector<double> &expected, const std::vector<double> &actual) {
    ASSERT_EQ(expected.size(), actual.size());
    for (size_t i = 0; i < expected.size(); i++) {
        EXPECT_NEAR(expected[i], actual[i], 1e-14) << "row " << i;
    }
}

// Removing the reaction without regenerating leaves the compiled model, with its piecewise triggers,
// untouched, and it still integrates as before.
TEST_F(CVODEIntegratorTests, ReactionRemovedWithoutRegeneratingKeepsTheCompiledPiecewiseTriggers) {
    RoadRunner reference(piecewiseKineticLawSBML(true, true));
    RoadRunner edited(piecewiseKineticLawSBML(true, true));
    ASSERT_GT(reference.getModel()->getNumPiecewiseTriggers(), 0);

    edited.removeReaction("R", false, false);

    EXPECT_EQ(reference.getModel()->getNumPiecewiseTriggers(), edited.getModel()->getNumPiecewiseTriggers());
    expectSameSimulation(simulateSpeciesS(reference), simulateSpeciesS(edited));
    EXPECT_EQ(leafStatistics(reference)["num_reinits"].getAs<std::int64_t>(),
              leafStatistics(edited)["num_reinits"].getAs<std::int64_t>());
}

// Regenerating after the removal drops the piecewise triggers and nothing flows into S.
TEST_F(CVODEIntegratorTests, ReactionRemovedWithoutRegeneratingHasNoPiecewiseTriggersOnceRegenerated) {
    RoadRunner rr(piecewiseKineticLawSBML(true, true));
    rr.removeReaction("R", false, false);
    simulateSpeciesS(rr);

    rr.regenerateModel();
    rr.reset();

    EXPECT_EQ(0, rr.getModel()->getNumPiecewiseTriggers());
    for (double s : simulateSpeciesS(rr)) {
        EXPECT_EQ(0.0, s);
    }
}

TEST_F(CVODEIntegratorTests, ReactionRemovedWithRegeneratingHasNoPiecewiseTriggers) {
    RoadRunner rr(piecewiseKineticLawSBML(true, true));
    rr.removeReaction("R");

    EXPECT_EQ(0, rr.getModel()->getNumPiecewiseTriggers());
    for (double s : simulateSpeciesS(rr)) {
        EXPECT_EQ(0.0, s);
    }
}

// A reaction removed and added back, with its local parameters, is the same model as the original.
TEST_F(CVODEIntegratorTests, ReactionAddedBackWithLocalParametersHasTheSamePiecewiseTriggers) {
    RoadRunner reference(piecewiseKineticLawSBML(true, true));
    const std::vector<double> expected = simulateSpeciesS(reference);
    const std::int64_t expectedReinits = leafStatistics(reference)["num_reinits"].getAs<std::int64_t>();

    for (bool regenerateWithTheAdd : {true, false}) {
        SCOPED_TRACE(regenerateWithTheAdd ? "regenerated by addReaction" : "regenerated afterwards");
        RoadRunner rr(piecewiseKineticLawSBML(true, true));
        rr.removeReaction("R");
        rr.addReaction(PiecewiseReactionXML, regenerateWithTheAdd);
        if (!regenerateWithTheAdd) {
            rr.regenerateModel();
        }

        EXPECT_EQ(reference.getModel()->getNumPiecewiseTriggers(), rr.getModel()->getNumPiecewiseTriggers());
        expectSameSimulation(expected, simulateSpeciesS(rr));
        EXPECT_EQ(expectedReinits, leafStatistics(rr)["num_reinits"].getAs<std::int64_t>());
    }
}

// The same removals when the piecewise is in a function definition.
TEST_F(CVODEIntegratorTests, ReactionWithAPiecewiseFunctionDefinitionRemovedWithoutRegenerating) {
    RoadRunner reference(piecewiseFunctionDefinitionSBML(true, true));
    RoadRunner edited(piecewiseFunctionDefinitionSBML(true, true));
    ASSERT_GT(reference.getModel()->getNumPiecewiseTriggers(), 0);

    edited.removeReaction("R", false, false);

    EXPECT_EQ(reference.getModel()->getNumPiecewiseTriggers(), edited.getModel()->getNumPiecewiseTriggers());
    expectSameSimulation(simulateSpeciesS(reference), simulateSpeciesS(edited));

    edited.regenerateModel();

    EXPECT_EQ(0, edited.getModel()->getNumPiecewiseTriggers());
}

// ---- Piecewise functions whose branch changes within one step, or does not change ----

// MathML of piecewise(v1, c1, v2, c2, ..., otherwise) from (value, condition) pairs of MathML
static std::string branchPiecewise(const std::vector<std::pair<std::string, std::string>> &pieces, const std::string &otherwise) {
    std::string math = "<piecewise>";
    for (const auto &piece : pieces) {
        math += "<piece>" + piece.first + piece.second + "</piece>";
    }
    return math + "<otherwise>" + otherwise + "</otherwise></piecewise>";
}

// A model with the rate rules S' = rate and X' = 1, the extra parameters, the events and the extra rules.
static std::string branchRateRuleModel(const std::string &rate, const std::string &extraParameters = "",
                                       const std::string &events = "", const std::string &extraRules = "") {
    return std::string(R"(<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model metaid="branches" id="branches">
    <listOfParameters>
      <parameter id="S" value="0" constant="false"/>
      <parameter id="X" value="0" constant="false"/>)") + extraParameters + R"(
    </listOfParameters>
    <listOfRules>
      <rateRule variable="S"><math xmlns="http://www.w3.org/1998/Math/MathML">)" + rate + R"(</math></rateRule>
      <rateRule variable="X"><math xmlns="http://www.w3.org/1998/Math/MathML"><cn type="integer"> 1 </cn></math></rateRule>
)" + extraRules + R"(    </listOfRules>
)" + (events.empty() ? std::string() : "    <listOfEvents>\n" + events + "    </listOfEvents>\n") + R"(  </model>
</sbml>)";
}

// A model with a reaction -> S whose rate is the kinetic law, with the local and global parameters.
static std::string branchReactionModel(const std::string &rate, const std::string &localParameters,
                                       const std::string &globalParameters) {
    return std::string(R"(<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model metaid="branchReaction" id="branchReaction">
    <listOfCompartments>
      <compartment id="C" spatialDimensions="3" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="C" initialAmount="0" hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
)") + (globalParameters.empty() ? std::string() : "    <listOfParameters>" + globalParameters + "</listOfParameters>\n") + R"(    <listOfReactions>
      <reaction id="R" reversible="false" fast="false">
        <listOfProducts>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">)" + rate + R"(</math>
)" + (localParameters.empty() ? std::string() : "          <listOfLocalParameters>" + localParameters + "</listOfLocalParameters>\n") + R"(        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>)";
}

// The rate is 1 only for 3 < time < 3.01, a window shorter than the steps of CVODE, so S(10) = 0.01.
// CVODE is reinitialized where the window opens and where it closes.
TEST_F(CVODEIntegratorTests, PiecewiseWhoseConditionIsTrueOnlyWithinOneStepIsIntegrated) {
    RoadRunner rr(branchRateRuleModel(branchPiecewise({{"<cn> 1 </cn>", leafWindow("3", "3.01", "time")}}, "<cn> 0 </cn>")));
    const std::vector<double> s = simulateSpeciesS(rr);
    EXPECT_NEAR(0.01, s.back(), 1e-6);
    EXPECT_EQ(2, leafStatistics(rr)["num_reinits"].getAs<std::int64_t>());
}

// The same window in a variable of the model and not in time.
TEST_F(CVODEIntegratorTests, PiecewiseWhoseConditionIsTrueOnlyWithinOneStepOfAVariableIsIntegrated) {
    RoadRunner rr(branchRateRuleModel(branchPiecewise({{"<cn> 1 </cn>", leafWindow("3", "3.01", "X")}}, "<cn> 0 </cn>")));
    const std::vector<double> s = simulateSpeciesS(rr);
    EXPECT_NEAR(0.01, s.back(), 1e-6);
    EXPECT_EQ(2, leafStatistics(rr)["num_reinits"].getAs<std::int64_t>());
}

// The same window in a kinetic law whose parameters are local or global.
TEST_F(CVODEIntegratorTests, PiecewiseInAKineticLawWhoseConditionIsTrueOnlyWithinOneStepIsIntegrated) {
    const std::string window = "<apply><and/><apply><gt/>" + LeafTimeMath + "<ci> a </ci></apply><apply><lt/>" +
                               LeafTimeMath + "<ci> b </ci></apply></apply>";
    const std::string rate = branchPiecewise({{"<ci> k </ci>", window}}, "<cn> 0 </cn>");
    const std::string parameters[] = {R"(<localParameter id="a" value="3"/><localParameter id="b" value="3.01"/><localParameter id="k" value="1"/>)",
                                      R"(<parameter id="a" value="3" constant="true"/><parameter id="b" value="3.01" constant="true"/><parameter id="k" value="1" constant="true"/>)"};
    for (bool local : {true, false}) {
        SCOPED_TRACE(local ? "local parameters" : "global parameters");
        RoadRunner rr(branchReactionModel(rate, local ? parameters[0] : "", local ? "" : parameters[1]));
        const std::vector<double> s = simulateSpeciesS(rr);
        EXPECT_NEAR(0.01, s.back(), 1e-6);
        EXPECT_EQ(2, leafStatistics(rr)["num_reinits"].getAs<std::int64_t>());
    }
}

// CVODE is reinitialized once at each threshold of a piecewise with sequential thresholds.
TEST_F(CVODEIntegratorTests, PiecewiseWithSequentialThresholdsReinitializesCVODEOncePerThreshold) {
    RoadRunner rr(branchRateRuleModel(branchPiecewise({{"<cn> 1 </cn>", leafRelation("lt", "time", "1")},
                                                       {"<cn> 2 </cn>", leafRelation("lt", "time", "2")},
                                                       {"<cn> 3 </cn>", leafRelation("lt", "time", "3")}}, "<cn> 0 </cn>")));
    const std::vector<double> s = simulateSpeciesS(rr);
    EXPECT_NEAR(6.0, s.back(), 1e-5);
    EXPECT_EQ(3, leafStatistics(rr)["num_reinits"].getAs<std::int64_t>());
}

// The condition time < 3 is true only while time < 5 is, so its change at 3 does not change the branch
// that is taken.  CVODE is not reinitialized there, and the rows of a variable step simulation are those
// of the same model without that condition.
TEST_F(CVODEIntegratorTests, PiecewiseConditionThatChangesWhileAnEarlierOneHoldsAddsNoRowAndNoReinitialization) {
    RoadRunner masked(branchRateRuleModel(branchPiecewise({{"<cn> 1 </cn>", leafRelation("lt", "time", "5")},
                                                           {"<cn> 2 </cn>", leafRelation("lt", "time", "3")}}, "<cn> 0 </cn>")));
    RoadRunner without(branchRateRuleModel(branchPiecewise({{"<cn> 1 </cn>", leafRelation("lt", "time", "5")}}, "<cn> 0 </cn>")));
    masked.setSelections(std::vector<std::string>{"time", "S"});
    without.setSelections(std::vector<std::string>{"time", "S"});

    const ls::DoubleMatrix a = simulateVariableStep(masked, 0, 10);
    const ls::DoubleMatrix b = simulateVariableStep(without, 0, 10);
    ASSERT_EQ(b.numRows(), a.numRows());
    for (unsigned i = 0; i < a.numRows(); i++) {
        EXPECT_DOUBLE_EQ(b(i, 0), a(i, 0)) << "row " << i;
        EXPECT_DOUBLE_EQ(b(i, 1), a(i, 1)) << "row " << i;
    }
    EXPECT_NEAR(5.0, a(a.numRows() - 1, 1), 1e-5);
    EXPECT_EQ(1, leafStatistics(masked)["num_reinits"].getAs<std::int64_t>());
    EXPECT_EQ(1, leafStatistics(without)["num_reinits"].getAs<std::int64_t>());
}

// An event changes the branch of the piecewise at 2.5 by an assignment, and the branch goes back at 4.5
// where time < 4.5 changes.  CVODE is reinitialized at both.
TEST_F(CVODEIntegratorTests, PiecewiseBranchChangedByAnEventAssignmentIsReinitializedWhereItChangesBack) {
    const std::string condition = leafLogical("and", {leafRelation("gt", "g", "0"), leafRelation("lt", "time", "4.5")});
    RoadRunner rr(branchRateRuleModel(branchPiecewise({{"<cn> 1 </cn>", condition}}, "<cn> 0 </cn>"),
                                      R"(<parameter id="g" value="0" constant="false"/>)",
                                      leafEvent("E", leafRelation("gt", "time", "2.5"), "g")));
    const std::vector<double> s = simulateSpeciesS(rr);
    EXPECT_NEAR(2.0, s.back(), 1e-5);
    EXPECT_EQ(2, leafStatistics(rr)["num_reinits"].getAs<std::int64_t>());
}

// ---- The rows of variable step simulations at roots ----

// the number of rows whose time is within the tolerance of t
static int rowsAtTime(const ls::DoubleMatrix &result, double t, double tolerance = 1e-9) {
    int rows = 0;
    for (unsigned i = 0; i < result.numRows(); i++) {
        if (fabs(result(i, 0) - t) < tolerance) {
            rows++;
        }
    }
    return rows;
}

// the times of the rows within 1e-4 of t, for the message of a failed expectation
static std::string rowTimesNear(const ls::DoubleMatrix &result, double t) {
    std::ostringstream times;
    times.precision(17);
    for (unsigned i = 0; i < result.numRows(); i++) {
        if (fabs(result(i, 0) - t) < 1e-4) {
            times << " " << result(i, 0);
        }
    }
    return times.str();
}

static void expectTimesIncrease(const ls::DoubleMatrix &result) {
    for (unsigned i = 1; i < result.numRows(); i++) {
        EXPECT_GT(result(i, 0), result(i - 1, 0)) << "row " << i;
    }
}

// The window of a condition has a row where it opens and a row where it closes, where CVODE is reinitialized.
// The first steps of CVODE after a reinitialization with S = 0 are tiny, so the rows after the root of the
// opening are within 1e-9 of it too; the row of the root itself is within 1e-12.
TEST_F(CVODEIntegratorTests, VariableStepRowsOfAPiecewiseWindowAreAtItsEdges) {
    RoadRunner rr(branchRateRuleModel(branchPiecewise({{"<cn> 1 </cn>", leafWindow("3", "3.01", "time")}}, "<cn> 0 </cn>")));
    rr.setSelections(std::vector<std::string>{"time", "S"});
    const ls::DoubleMatrix result = simulateVariableStep(rr, 0, 10);

    EXPECT_GE(rowsAtTime(result, 3, 1e-12), 1) << "rows near 3:" << rowTimesNear(result, 3);
    EXPECT_GE(rowsAtTime(result, 3.01, 1e-12), 1) << "rows near 3.01:" << rowTimesNear(result, 3.01);
    expectTimesIncrease(result);
    EXPECT_NEAR(0.01, result(result.numRows() - 1, 1), 1e-6);
}

// A piecewise with sequential thresholds has one row at each of them.
TEST_F(CVODEIntegratorTests, VariableStepRowsOfAPiecewiseWithSequentialThresholdsAreAtTheThresholds) {
    RoadRunner rr(branchRateRuleModel(branchPiecewise({{"<cn> 1 </cn>", leafRelation("lt", "time", "1")},
                                                       {"<cn> 2 </cn>", leafRelation("lt", "time", "2")},
                                                       {"<cn> 3 </cn>", leafRelation("lt", "time", "3")}}, "<cn> 0 </cn>")));
    rr.setSelections(std::vector<std::string>{"time", "S"});
    const ls::DoubleMatrix result = simulateVariableStep(rr, 0, 10);

    for (double threshold : {1.0, 2.0, 3.0}) {
        EXPECT_EQ(1, rowsAtTime(result, threshold)) << "threshold " << threshold << ", rows near it:" << rowTimesNear(result, threshold);
    }
    expectTimesIncrease(result);
}

// X < 2.5 changes at 2.5, after time < 1 has made the condition false, so it changes nothing: the rows
// are those of the same model without that comparison.
TEST_F(CVODEIntegratorTests, VariableStepRowsOfACompoundPiecewiseConditionExcludeTheMaskedComparison) {
    const std::string compound = leafLogical("and", {leafRelation("lt", "time", "1"), leafRelation("lt", "X", "2.5")});
    RoadRunner masked(branchRateRuleModel(branchPiecewise({{"<cn> 1 </cn>", compound}}, "<cn> 0 </cn>")));
    RoadRunner without(branchRateRuleModel(branchPiecewise({{"<cn> 1 </cn>", leafRelation("lt", "time", "1")}}, "<cn> 0 </cn>")));
    masked.setSelections(std::vector<std::string>{"time", "X", "S"});
    without.setSelections(std::vector<std::string>{"time", "X", "S"});

    const ls::DoubleMatrix a = simulateVariableStep(masked, 0, 10);
    const ls::DoubleMatrix b = simulateVariableStep(without, 0, 10);
    ASSERT_EQ(b.numRows(), a.numRows());
    for (unsigned i = 0; i < a.numRows(); i++) {
        for (unsigned j = 0; j < 3; j++) {
            EXPECT_DOUBLE_EQ(b(i, j), a(i, j)) << "row " << i << ", column " << j;
        }
    }
    EXPECT_EQ(1, leafStatistics(masked)["num_reinits"].getAs<std::int64_t>());
    EXPECT_EQ(1, leafStatistics(without)["num_reinits"].getAs<std::int64_t>());
}

// X > 0.5 becomes true after the trigger X > 0.3 or X > 0.5 has, so it changes nothing: the rows are
// those of the event with the trigger X > 0.3.
TEST_F(CVODEIntegratorTests, VariableStepRowsOfAnOrTriggerExcludeTheMaskedComparison) {
    const std::string orTrigger = leafLogical("or", {leafRelation("gt", "X", "0.3"), leafRelation("gt", "X", "0.5")});
    RoadRunner masked(leafModelSBML(leafEvent("E", orTrigger)));
    RoadRunner without(leafModelSBML(leafEvent("E", leafRelation("gt", "X", "0.3"))));
    ASSERT_EQ(2, masked.getModel()->getNumBooleanLeaves());
    masked.setSelections(std::vector<std::string>{"time", "X", "n"});
    without.setSelections(std::vector<std::string>{"time", "X", "n"});

    const ls::DoubleMatrix a = simulateVariableStep(masked, 0, 10);
    const ls::DoubleMatrix b = simulateVariableStep(without, 0, 10);
    ASSERT_EQ(b.numRows(), a.numRows());
    for (unsigned i = 0; i < a.numRows(); i++) {
        for (unsigned j = 0; j < 3; j++) {
            EXPECT_DOUBLE_EQ(b(i, j), a(i, j)) << "row " << i << ", column " << j;
        }
    }
    EXPECT_EQ(1, leafStatistics(masked)["num_reinits"].getAs<std::int64_t>());
    EXPECT_EQ(1, leafStatistics(without)["num_reinits"].getAs<std::int64_t>());
}

// ---- A leaf shared by several events and piecewise functions ----

static const std::string CounterParameters = R"(<parameter id="n" value="0" constant="false"/><parameter id="m" value="0" constant="false"/>)";

// the final row of a simulation to time 10 with the selections
static std::vector<double> sharedLeafFinalRow(RoadRunner &rr, const std::vector<std::string> &selections) {
    rr.setSelections(selections);
    const ls::DoubleMatrix result = *rr.simulate(0, 10, 11);
    std::vector<double> row;
    for (unsigned j = 0; j < result.numCols(); j++) {
        row.push_back(result(result.numRows() - 1, j));
    }
    return row;
}

// X > 3 is a leaf of both triggers.  Both events fire at its root, which reinitializes CVODE once.  The
// triggers go false later at roots that change nothing.
TEST_F(CVODEIntegratorTests, LeafSharedByTwoEventsIsOneRootAndOneReinitialization) {
    const std::string first = leafEvent("E1", leafWindow("3", "3.01"), "n");
    const std::string second = leafEvent("E2", leafWindow("3", "5"), "m");
    RoadRunner rr(branchRateRuleModel("<cn> 0 </cn>", CounterParameters, first + second));
    EXPECT_EQ(3, rr.getModel()->getNumBooleanLeaves());

    const std::vector<double> row = sharedLeafFinalRow(rr, {"time", "S", "n", "m"});
    EXPECT_EQ(1, row[2]);
    EXPECT_EQ(1, row[3]);
    EXPECT_EQ(1, leafStatistics(rr)["num_reinits"].getAs<std::int64_t>());
}

// time > 3 is a leaf of both piecewise functions.  Their branches change at it together, with one
// reinitialization; each of the other leaves changes the branch of one of them.
TEST_F(CVODEIntegratorTests, LeafSharedByTwoPiecewiseFunctionsIsOneRootAndOneReinitializationWhereTheyChange) {
    const std::string tRule = R"(      <rateRule variable="T"><math xmlns="http://www.w3.org/1998/Math/MathML">)" +
                              branchPiecewise({{"<cn> 2 </cn>", leafWindow("3", "5", "time")}}, "<cn> 0 </cn>") +
                              "</math></rateRule>\n";
    RoadRunner rr(branchRateRuleModel(branchPiecewise({{"<cn> 1 </cn>", leafWindow("3", "3.01", "time")}}, "<cn> 0 </cn>"),
                                      R"(<parameter id="T" value="0" constant="false"/>)", "", tRule));
    EXPECT_EQ(3, rr.getModel()->getNumBooleanLeaves());

    const std::vector<double> row = sharedLeafFinalRow(rr, {"time", "S", "T"});
    EXPECT_NEAR(0.01, row[1], 1e-6);
    EXPECT_NEAR(4.0, row[2], 1e-5);
    EXPECT_EQ(3, leafStatistics(rr)["num_reinits"].getAs<std::int64_t>());
}

// X > 3 is a leaf of the trigger and of the condition.  The event fires and the branch changes at the same
// root, which reinitializes CVODE once.  The trigger goes false at a root that changes nothing.
TEST_F(CVODEIntegratorTests, LeafSharedByAnEventAndAPiecewiseFunctionReinitializesCVODEOnce) {
    RoadRunner rr(branchRateRuleModel(branchPiecewise({{"<cn> 1 </cn>", leafWindow("3", "5")}}, "<cn> 0 </cn>"),
                                      CounterParameters, leafEvent("E", leafWindow("3", "3.01"), "n")));
    EXPECT_EQ(3, rr.getModel()->getNumBooleanLeaves());

    const std::vector<double> row = sharedLeafFinalRow(rr, {"time", "S", "n"});
    EXPECT_NEAR(2.0, row[1], 1e-5);
    EXPECT_EQ(1, row[2]);
    EXPECT_EQ(2, leafStatistics(rr)["num_reinits"].getAs<std::int64_t>());
}

// X < 3 is a leaf of both.  Where it changes, the event trigger goes false, which changes nothing, and
// the piecewise function takes another branch, which needs CVODE to be reinitialized.
TEST_F(CVODEIntegratorTests, RootWhereAnEventNeedsNothingButAPiecewiseFunctionDoesReinitializesCVODE) {
    RoadRunner rr(branchRateRuleModel(branchPiecewise({{"<cn> 1 </cn>", leafRelation("lt", "X", "3")}}, "<cn> 0 </cn>"),
                                      CounterParameters, leafEvent("E", leafWindow("1", "3"), "n")));
    EXPECT_EQ(2, rr.getModel()->getNumBooleanLeaves());

    const std::vector<double> row = sharedLeafFinalRow(rr, {"time", "S", "n"});
    EXPECT_NEAR(3.0, row[1], 1e-5);
    EXPECT_EQ(1, row[2]);
    EXPECT_EQ(2, leafStatistics(rr)["num_reinits"].getAs<std::int64_t>());
}

// X > 3 is a leaf of the trigger and of a condition that X > 2 has made true already.  Where it changes,
// the branch does not, and the event fires, which needs CVODE to be reinitialized.
TEST_F(CVODEIntegratorTests, RootWhereAPiecewiseFunctionNeedsNothingButAnEventDoesReinitializesCVODE) {
    const std::string condition = leafLogical("or", {leafRelation("gt", "X", "2"), leafRelation("gt", "X", "3")});
    RoadRunner rr(branchRateRuleModel(branchPiecewise({{"<cn> 1 </cn>", condition}}, "<cn> 0 </cn>"),
                                      CounterParameters, leafEvent("E", leafWindow("3", "3.01"), "n")));
    EXPECT_EQ(3, rr.getModel()->getNumBooleanLeaves());

    const std::vector<double> row = sharedLeafFinalRow(rr, {"time", "S", "n"});
    EXPECT_NEAR(8.0, row[1], 1e-5);
    EXPECT_EQ(1, row[2]);
    EXPECT_EQ(2, leafStatistics(rr)["num_reinits"].getAs<std::int64_t>());
}

// ---- Triggers that are true at a single instant ----
// The event adds one to n when the trigger of time == 3 or of time <= 3 && time >= 3 becomes true.  The
// trigger is true for an instant only, so no step of CVODE ends where it is true unless it ends at 3.

// Counts of n at time 10 of simulations whose output times do not include 3, and of a variable step simulation.
static void checkEventsOfSingleInstantTrigger(const std::string &trigger, double expectedCount) {
    RoadRunner rr(leafModelSBML(leafEvent("E", trigger)));
    rr.setSelections(std::vector<std::string>{"time", "n"});
    std::cout << "boolean leaves: " << rr.getModel()->getNumBooleanLeaves() << std::endl;

    rr.getIntegrator()->setValue("variable_step_size", false);
    for (int points : {8, 14, 1000}) {
        SCOPED_TRACE("fixed step with " + std::to_string(points) + " output points");
        resetAllValues(rr);
        std::vector<double> times;
        for (int i = 0; i < points; i++) {
            times.push_back(10.0 * i / (points - 1));
        }
        const ls::DoubleMatrix result = *rr.simulate(times);
        std::cout << points << " output points: n = " << result(points - 1, 1) << std::endl;
        EXPECT_EQ(expectedCount, result(points - 1, 1));
    }

    for (double maximumTimeStep : {0.0, 1.0, 0.1}) {
        SCOPED_TRACE("variable step with a maximum time step of " + std::to_string(maximumTimeStep));
        resetAllValues(rr);
        rr.getIntegrator()->setValue("maximum_time_step", maximumTimeStep);
        const ls::DoubleMatrix result = simulateVariableStep(rr, 0, 10);
        std::cout << "maximum time step " << maximumTimeStep << ": n = " << result(result.numRows() - 1, 1)
                  << ", rows = " << result.numRows()
                  << ", reinits = " << leafStatistics(rr)["num_reinits"].getAs<std::int64_t>() << std::endl;
        EXPECT_EQ(expectedCount, result(result.numRows() - 1, 1));
    }
}

TEST_F(CVODEIntegratorTests, EventWhoseTriggerIsTimeEqualToAConstantFiresOnce) {
    checkEventsOfSingleInstantTrigger(leafRelation("eq", "time", "3"), 1);
}

TEST_F(CVODEIntegratorTests, EventWhoseTriggerIsTimeAtMostAndAtLeastAConstantFiresOnce) {
    checkEventsOfSingleInstantTrigger(
        leafLogical("and", {leafRelation("leq", "time", "3"), leafRelation("geq", "time", "3")}), 1);
}
