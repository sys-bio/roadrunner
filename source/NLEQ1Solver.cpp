#pragma hdrstop
#include "NLEQ1Solver.h"
#include "rrExecutableModel.h"
#include "rrStringUtils.h"
#include "rrUtils.h"
#include "nleq1/nleq1.h"
#include "rrLogger.h"
#include "rrUtils.h"
#include "rrException.h"
#include "rrConfig.h"
#include "rrNLEQ1Interface.h"

#include <Poco/Mutex.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include "tr1proxy/rr_unordered_map.h"

namespace rr
{

NLEQ1Solver::NLEQ1Solver(ExecutableModel *_model) :
    model(_model)
{
    resetSettings();
}

NLEQ1Solver::~NLEQ1Solver()
{
}

void NLEQ1Solver::syncWithModel(ExecutableModel* m)
{
    model = m;
}

void NLEQ1Solver::loadConfigSettings()
{
    SteadyStateSolver::loadConfigSettings();
    // Load settings specific to solver integrator

    NLEQ1Solver::setValue("allow_presimulation", Config::getBool(Config::STEADYSTATE_PRESIMULATION));
    NLEQ1Solver::setValue("presimulation_maximum_steps", Config::getInt(Config::STEADYSTATE_PRESIMULATION_MAX_STEPS));
    NLEQ1Solver::setValue("presimulation_time", Config::getDouble(Config::STEADYSTATE_PRESIMULATION_TIME));
    NLEQ1Solver::setValue("allow_approx", Config::getBool(Config::STEADYSTATE_APPROX));
    NLEQ1Solver::setValue("approx_tolerance", Config::getDouble(Config::STEADYSTATE_APPROX_TOL));
    NLEQ1Solver::setValue("approx_maximum_steps", Config::getInt(Config::STEADYSTATE_APPROX_MAX_STEPS));
    NLEQ1Solver::setValue("approx_time", Config::getDouble(Config::STEADYSTATE_APPROX_TIME));
    NLEQ1Solver::setValue("relative_tolerance", Config::getDouble(Config::STEADYSTATE_RELATIVE));
    NLEQ1Solver::setValue("maximum_iterations", Config::getInt(Config::STEADYSTATE_MAXIMUM_NUM_STEPS));
    NLEQ1Solver::setValue("minimum_damping", Config::getDouble(Config::STEADYSTATE_MINIMUM_DAMPING));
    NLEQ1Solver::setValue("broyden_method", Config::getInt(Config::STEADYSTATE_BROYDEN));
    NLEQ1Solver::setValue("linearity", Config::getInt(Config::STEADYSTATE_LINEARITY));
}

void NLEQ1Solver::resetSettings()
{
    Solver::resetSettings();

    // Set default settings.
    addSetting("allow_presimulation", false, "Allow Presimulation", "Flag for starting steady state analysis with simulation (bool).", "(bool) This flag does not affect the usage of approximation routine when the default steaty state solver fails");
    addSetting("presimulation_maximum_steps", 100, "Presimulation Maximum Steps", "Maximum number of steps that can be taken for presimulation before steady state analysis (int).", "(int) Takes priority over presimulation_time. Only used when allow_presimulation is True");
    addSetting("presimulation_time", 100, "Presimulation Time", "End time for presimulation steady state analysis (double).", "(double) presimulation_maximum_steps takes priority. Only used when allow_presimulation is True");
    addSetting("allow_approx", false, "Allow Approximiation", "Flag for using steady state approximation routine when steady state solver fails (bool).", "(bool) Approximation routine will run only when the default solver fails to fine a solution. This flag does not affect usage of approximation routine for pre-simulation");
    addSetting("approx_tolerance", 1e-6, "Approximation Tolerance", "Tolerance for steady state approximation routine (double).", "(double) Absolute tolerance used by steady state approximation routine. Only used when steady state approximation routine is used");
    addSetting("approx_maximum_steps", 10000, "Approximation Maximum Steps", "Maximum number of steps that can be taken for steady state approximation routine (int).", "(int) Takes priority over approx_time. Only used when steady state approximation routine is used");
    addSetting("approx_time", 10000, "Approximation Time", "End time for steady state approximation routine (double).", "(double) approx_maximum_steps takes priority. Only used when steady state approximation routine is used");
    addSetting("relative_tolerance", 1e-12, "Relative Tolerance", "Specifies the relative tolerance (double).", "(double) Relative tolerance used by the solver");
    addSetting("maximum_iterations", 100, "Maximum Iterations", "The maximum number of iterations the solver is allowed to use (int)", "(int) Iteration caps off at the maximum, regardless of whether a solution has been reached");
    addSetting("minimum_damping", 1e-20, "Minimum Damping", "The minimum damping factor (double).", "(double) Minumum damping factor used by the algorithm");
    addSetting("broyden_method", 0, "Broyden Method", "Switches on Broyden method (int)", "(int) Broyden method is a quasi-Newton approximation for rank-1 updates");
    addSetting("linearity", 3, "Problem Linearity", "Specifies linearity of the problem (int).", "(int) 1 is for linear problem and 4 is for extremly nonlinear problem");
    NLEQ1Solver::loadConfigSettings();
}

std::string NLEQ1Solver::getName() const {
	return NLEQ1Solver::getNLEQName();
}

std::string NLEQ1Solver::getNLEQName() {
	return "nleq1";
}

std::string NLEQ1Solver::getDescription() const {
	return NLEQ1Solver::getNLEQDescription();
}

std::string NLEQ1Solver::getNLEQDescription() {
	return "NLEQ1 is a non-linear equation solver which uses a global Newton "
     "method with adaptive damping strategies (see http://elib.zib.de/pub/elib/codelib/NewtonLib/)";
}

std::string NLEQ1Solver::getHint() const {
	return NLEQ1Solver::getNLEQHint();
}

std::string NLEQ1Solver::getNLEQHint() {
	return "Steady-state nonlinear systems of equations solver";
}

double NLEQ1Solver::solve()
{
    Log(Logger::LOG_DEBUG) << "NLEQ1Solver::solve";

    NLEQ1Interface* nleq1 = new NLEQ1Interface(model);

    nleq1->allowPreSim = getValue("allow_presimulation");
    nleq1->preSimMaximumSteps = getValue("presimulation_maximum_steps");
    nleq1->preSimTime = getValue("presimulation_time");
    nleq1->allowApprox = getValue("allow_approx");
    nleq1->approxTolerance = getValue("approx_tolerance");
    nleq1->approxMaximumSteps = getValue("approx_maximum_steps");
    nleq1->approxTime = getValue("approx_time");
    nleq1->relativeTolerance = getValue("relative_tolerance");
    nleq1->maxIterations = getValue("maximum_iterations");
    nleq1->minDamping = getValue("minimum_damping");
    nleq1->broyden = getValue("broyden_method");
    nleq1->linearity = getValue("linearity");

    double result = nleq1->solve();
    delete nleq1;
    return result;
}

}    //end of namespace


