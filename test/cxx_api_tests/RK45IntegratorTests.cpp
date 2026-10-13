#include "gtest/gtest.h"
#include "rrRoadRunner.h"
#include "rrExecutableModel.h"
#include "RoadRunnerTest.h"

#include <algorithm>
#include <cmath>
#include <string>

using namespace rr;

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

static double exactS(double t) { return 10. * std::exp(-0.5 * t); }

static double relativeError(double value, double t) {
    return std::fabs(value - exactS(t)) / exactS(t);
}

// Rows of the result are (time, S). Largest relative error of S over the rows
// [first, last).
static double maxRelativeError(const ls::DoubleMatrix &result, unsigned first, unsigned last) {
    double worst = 0;
    for (unsigned i = first; i < last; i++) {
        worst = std::max(worst, relativeError(result(i, 1), result(i, 0)));
    }
    return worst;
}

class RK45IntegratorTests : public RoadRunnerTest {
public:
    RK45IntegratorTests() = default;

    // variable step simulation from 0 to end, as simulate(0, end) does without a number of points
    static ls::DoubleMatrix simulateVariableStep(RoadRunner &rr, double end) {
        rr.getIntegrator()->setValue("variable_step_size", true);
        SimulateOptions &opt = rr.getSimulateOptions();
        opt.setStartTime(0);
        opt.setDuration(end);
        opt.setSteps(0);
        return *rr.simulate();
    }
};

/**
 * Regression test for O-3: RK45Integrator::integrate took a single adaptive step and
 * returned, so on a fixed output grid the model only advanced by that small step
 * (about 0.006 here) per output interval, whatever the interval was, while the time
 * column was right.
 */
TEST_F(RK45IntegratorTests, FixedOutputGridIntegratesToTheOutputTimes) {
    for (int points : {2, 6, 11, 101}) {
        RoadRunner rr(DecaySBML);
        rr.setIntegrator("rk45");
        rr.getIntegrator()->setValue("variable_step_size", false);

        const ls::DoubleMatrix *result = rr.simulate(0, 10, points);
        ASSERT_EQ((unsigned) points, result->numRows());

        for (unsigned i = 0; i < result->numRows(); i++) {
            EXPECT_NEAR(10. * i / (points - 1), (*result)(i, 0), 1e-12) << points << " points, row " << i;
        }
        EXPECT_EQ(10., (*result)(0, 1));
        EXPECT_LT(maxRelativeError(*result, 0, result->numRows()), 1e-8)
            << points << " points: S does not follow 10 exp(-0.5 t)";
    }
}

TEST_F(RK45IntegratorTests, FixedOutputGridAgreesWithCVODE) {
    RoadRunner rk45(DecaySBML);
    rk45.setIntegrator("rk45");
    rk45.getIntegrator()->setValue("variable_step_size", false);
    ls::DoubleMatrix a = *rk45.simulate(0, 10, 21);

    // CVODE with tight tolerances: with its defaults (relative tolerance 1e-6) it is
    // itself only accurate to about 1e-5 here, much less than rk45.
    RoadRunner cvode(DecaySBML);
    cvode.setIntegrator("cvode");
    cvode.getIntegrator()->setValue("variable_step_size", false);
    cvode.getIntegrator()->setValue("absolute_tolerance", 1e-12);
    cvode.getIntegrator()->setValue("relative_tolerance", 1e-10);
    ls::DoubleMatrix b = *cvode.simulate(0, 10, 21);

    ASSERT_EQ(a.numRows(), b.numRows());
    for (unsigned i = 0; i < a.numRows(); i++) {
        EXPECT_NEAR(b(i, 1), a(i, 1), 1e-6 * std::fabs(b(i, 1))) << "row " << i;
    }
}

TEST_F(RK45IntegratorTests, IntegrateReturnsEndOfIntervalWithoutVariableStepSize) {
    RoadRunner rr(DecaySBML);
    rr.setIntegrator("rk45");
    rr.getIntegrator()->setValue("variable_step_size", false);
    Integrator *integrator = rr.getIntegrator();

    integrator->restart(0);
    double t = integrator->integrate(0, 2);
    EXPECT_EQ(2., t);
    EXPECT_EQ(2., rr.getModel()->getTime());
    EXPECT_NEAR(exactS(2), rr.getValue("S"), 1e-8);

    t = integrator->integrate(t, 3);
    EXPECT_EQ(5., t);
    EXPECT_NEAR(exactS(5), rr.getValue("S"), 1e-8);
}

/**
 * Variable step: every row, and in particular the last one, is the result of an error
 * controlled step. Before the fix the last row was a linear interpolation, about 1e-4
 * off with the default tolerance of this integrator.
 */
TEST_F(RK45IntegratorTests, VariableStepRunLandsOnEndTimeAndIsAccurate) {
    for (double end : {10., 9.95, 5., 1.}) {
        RoadRunner rr(DecaySBML);
        rr.setIntegrator("rk45");
        ls::DoubleMatrix result = simulateVariableStep(rr, end);

        ASSERT_GT(result.numRows(), 2u);
        unsigned last = result.numRows() - 1;
        EXPECT_EQ(end, result(last, 0)) << "end " << end;
        for (unsigned i = 1; i <= last; i++) {
            EXPECT_GT(result(i, 0), result(i - 1, 0)) << "end " << end << ", row " << i;
        }
        EXPECT_LT(maxRelativeError(result, 0, last), 1e-8) << "end " << end;
        EXPECT_LT(relativeError(result(last, 1), result(last, 0)), 1e-8)
            << "end " << end << ": the last row is less accurate than the others";
        // the model is left at the end time
        EXPECT_EQ(end, rr.getModel()->getTime()) << "end " << end;
    }
}

/**
 * The step size limits are read when integrating, not when the integrator is created.
 */
TEST_F(RK45IntegratorTests, MaximumTimeStepSetAfterCreationIsUsed) {
    RoadRunner rr(DecaySBML);
    rr.setIntegrator("rk45");
    rr.getIntegrator()->setValue("maximum_time_step", 0.01);
    ls::DoubleMatrix result = simulateVariableStep(rr, 1);

    // at least 100 steps of at most 0.01
    EXPECT_GE(result.numRows(), 101u);
    for (unsigned i = 1; i < result.numRows(); i++) {
        EXPECT_LE(result(i, 0) - result(i - 1, 0), 0.01 + 1e-12) << "row " << i;
    }
    EXPECT_LT(maxRelativeError(result, 0, result.numRows()), 1e-8);
}

TEST_F(RK45IntegratorTests, EpsilonControlsTheError) {
    double errors[2];
    unsigned rows[2];
    int n = 0;
    for (double epsilon : {1e-6, 1e-12}) {
        RoadRunner rr(DecaySBML);
        rr.setIntegrator("rk45");
        rr.getIntegrator()->setValue("epsilon", epsilon);
        ls::DoubleMatrix result = simulateVariableStep(rr, 10);
        errors[n] = maxRelativeError(result, 0, result.numRows());
        rows[n] = result.numRows();
        n++;
    }
    EXPECT_LT(errors[0], 1e-3);
    EXPECT_LT(errors[1], 1e-8);
    EXPECT_LT(errors[1], errors[0]);
    EXPECT_GT(rows[1], rows[0]);
}
