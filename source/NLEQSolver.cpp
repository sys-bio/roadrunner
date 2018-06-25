#pragma hdrstop
#include "NLEQSolver.h"
#include "rrExecutableModel.h"
#include "rrStringUtils.h"
#include "rrUtils.h"
#include "nleq/nleq2.h"
#include "rrLogger.h"
#include "rrUtils.h"
#include "rrException.h"
#include "rrConfig.h"
#include "rrNLEQInterface.h"

#include <Poco/Mutex.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include "tr1proxy/rr_unordered_map.h"

namespace rr
{

NLEQSolver::NLEQSolver(ExecutableModel *_model) :
    model(_model)
{
    resetSettings();
}

NLEQSolver::~NLEQSolver()
{
}

void NLEQSolver::syncWithModel(ExecutableModel* m)
{
    model = m;
}

void NLEQSolver::loadConfigSettings()
{
    SteadyStateSolver::loadConfigSettings();
    // Load settings specific to solver integrator

    NLEQSolver::setValue("maximum_iterations", Config::getInt(Config::STEADYSTATE_MAXIMUM_NUM_STEPS));
    NLEQSolver::setValue("relative_tolerance", Config::getDouble(Config::STEADYSTATE_RELATIVE));
    NLEQSolver::setValue("minimum_damping", Config::getDouble(Config::STEADYSTATE_MINIMUM_DAMPING));
    NLEQSolver::setValue("broyden_method", Config::getDouble(Config::STEADYSTATE_BROYDEN));
    NLEQSolver::setValue("linearity", Config::getDouble(Config::STEADYSTATE_LINEARITY));
}

void NLEQSolver::resetSettings()
{
    Solver::resetSettings();

    // Set default settings.
    addSetting("maximum_iterations", 100, "Maximum Iterations", "The maximum number of iterations the solver is allowed to use (int)", "(int) Iteration caps off at the maximum, regardless of whether a solution has been reached");
    addSetting("minimum_damping", 1e-4, "Minimum Damping", "The minimum damping factor (double).", "(double) Minumum damping factor used by the algorithm");
    addSetting("relative_tolerance", 1e-16, "Relative Tolerance", "Specifies the relative tolerance (double).", "(double) Relative tolerance used by the solver");
    addSetting("broyden_method", 0, "Broyden Method", "Switches on Broyden method (int)", "(int) Broyden method is a quasi-Newton approximation for rank-1 updates");
    addSetting("linearity", 3, "Problem Linearity", "Specifies linearity of the problem (int).", "(int) 1 is for linear problem and 4 is for extremly nonlinear problem");
    NLEQSolver::loadConfigSettings();
}

std::string NLEQSolver::getName() const {
	return NLEQSolver::getNLEQName();
}

std::string NLEQSolver::getNLEQName() {
	return "nleq2";
}

std::string NLEQSolver::getDescription() const {
	return NLEQSolver::getNLEQDescription();
}

std::string NLEQSolver::getNLEQDescription() {
	return "NLEQ2 is a non-linear equation solver which uses a global Newton "
     "method with adaptive damping strategies (see http://www.zib.de/weimann/NewtonLib/index.html)";
}

std::string NLEQSolver::getHint() const {
	return NLEQSolver::getNLEQHint();
}

std::string NLEQSolver::getNLEQHint() {
	return "Steady-state nonlinear systems of equations solver";
}

double NLEQSolver::solve()
{
    Log(Logger::LOG_DEBUG) << "NLEQSolver::solve";

    NLEQInterface* nleq = new NLEQInterface(model);

//     nleq->maxIterations = getValue("maximum_iterations");
//     nleq->relativeTolerance = getValue("relative_tolerance");
//     nleq->minDamping = getValue("minimum_damping");

    double result = nleq->solve();
    delete nleq;
    return result;
}

}    //end of namespace


