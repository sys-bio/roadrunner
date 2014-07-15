#pragma hdrstop
#include "CVODEIntegrator.h"
#include "rrExecutableModel.h"
#include "rrException.h"
#include "rrLogger.h"
#include "rrStringUtils.h"
#include "rrException.h"
#include "rrUtils.h"

#include <cvode/cvode.h>
#include <cvode/cvode_dense.h>
#include <nvector/nvector_serial.h>
#include <cstring>
#include <iomanip>
#include <math.h>
#include <map>
#include <algorithm>
#include <assert.h>
#include <Poco/Logger.h>


using namespace std;
namespace rr
{

int cvodeDyDtFcn(realtype t, N_Vector cv_y, N_Vector cv_ydot, void *userData);
int cvodeRootFcn (realtype t, N_Vector y, realtype *gout, void *userData);

// Sets the value of an element in a N_Vector object
inline void SetVector (N_Vector v, int Index, double Value)
{
    double *data = NV_DATA_S(v);
    data[Index] = Value;
}

inline double GetVector (N_Vector v, int Index)
{
    double *data = NV_DATA_S(v);
    return data[Index];
}

const int CVODEIntegrator::mDefaultMaxNumSteps = 10000;
const int CVODEIntegrator::mDefaultMaxAdamsOrder = 12;
const int CVODEIntegrator::mDefaultMaxBDFOrder = 5;

/**
 * Purpose
 * This function processes error and warning messages from CVODE and its
 * sub-modules.
 *
 * Arguments:
 * error_code: is the error code.
 * module: is the name of the CVODE module reporting the error.
 * function: is the name of the function in which the error occurred.
 * msg: is the error message.
 * eh_data: is a pointer to user data, the same as the eh_data parameter
 * passed to CVodeSetErrHandlerFn.
 * Return value: A CVErrHandlerFn function has no return value.
 *
 * Notes:
 * error_code is negative for errors and positive (CV_WARNING) for warnings.
 * If a function that returns a pointer to memory encounters an error,
 * it sets error_code to 0.
 */
static void cvodeErrHandler(int error_code, const char *module, const char *function,
        char *msg, void *eh_data);


/**
 * decode the cvode error code to a string
 */
static std::string cvodeDecodeError(int cvodeError, bool exInfo = true);

/**
 * macro to throw a (hopefully) usefull error message
 */
#define handleCVODEError(errCode) \
        { std::string _err_what = std::string("CVODE Error: ") + \
          cvodeDecodeError(errCode); \
          throw IntegratorException(_err_what, std::string(__FUNC__)); }

void CVODEIntegrator::setListener(IntegratorListenerPtr p)
{
    listener = p;
}

IntegratorListenerPtr CVODEIntegrator::getListener()
{
    return listener;
}

CVODEIntegrator::CVODEIntegrator(ExecutableModel *aModel, const SimulateOptions* options)
:
mStateVector(NULL),
mCVODE_Memory(NULL),
mLastTimeValue(0),
mLastEvent(0),
mOneStepCount(0),
mFollowEvents(true),
mMaxAdamsOrder(mDefaultMaxAdamsOrder),
mMaxBDFOrder(mDefaultMaxBDFOrder),
mModel(aModel),
stateVectorVariables(false),
options(*options)
{
    Log(Logger::LOG_INFORMATION) << "creating CVODEIntegrator";

    if(aModel)
    {
        createCVode();
    }

    // if we pass a 0, we re-load all the values from our options.
    // saves from copying it twice.
    setSimulateOptions(0);
}

CVODEIntegrator::~CVODEIntegrator()
{
    freeCVode();
}

void CVODEIntegrator::setSimulateOptions(const SimulateOptions* o)
{
    if (o && (o->integratorFlags & SimulateOptions::STIFF) !=
            (options.integratorFlags & SimulateOptions::STIFF))
    {
        // if the integrator is changed from stiff to standard, this
        // requires re-creating the CVode objects.
        Log(Logger::LOG_INFORMATION) << "re-creating CVode, interator stiffness has changed";
        options = *o;
        freeCVode();
        createCVode();
    }
    if(o)
    {
        // don't need to re-create, can just set params.
        options = *o;
    }

    if (mCVODE_Memory == 0)
    {
        return;
    }

    if (options.initialTimeStep > 0)
    {
        CVodeSetInitStep(mCVODE_Memory, options.initialTimeStep);
    }

    if (options.minimumTimeStep > 0)
    {
        CVodeSetMinStep(mCVODE_Memory, options.minimumTimeStep);
    }

    if (options.maximumTimeStep > 0)
    {
        CVodeSetMaxStep(mCVODE_Memory, options.maximumTimeStep);
    }

    if (options.maximumNumSteps > 0)
    {
        CVodeSetMaxNumSteps(mCVODE_Memory, options.maximumNumSteps);
    }
    else
    {
        CVodeSetMaxNumSteps(mCVODE_Memory, mDefaultMaxNumSteps);
    }

    setCVODETolerances();
}

void CVODEIntegrator::reInit(double t0)
{
    // if we have no state vector variables and no events, we never
    // create an integrator.
    if (mCVODE_Memory == 0)
    {
        return;
    }

    int result = CVodeReInit(mCVODE_Memory,  t0, mStateVector);

    if (result != CV_SUCCESS)
    {
        handleCVODEError(result);
    }

    setCVODETolerances();
}

double CVODEIntegrator::integrate(double timeStart, double hstep)
{
    Log(lDebug3)<<"---------------------------------------------------";
    Log(lDebug3)<<"--- O N E     S T E P      ( "<<mOneStepCount<< " ) ";
    Log(lDebug3)<<"---------------------------------------------------";

    mOneStepCount++;

    double timeEnd = 0.0;
    double tout = timeStart + hstep;
    int strikes = 3;

    const int itask = ((options.integratorFlags & SimulateOptions::MULTI_STEP)
            || (options.integratorFlags & SimulateOptions::VARIABLE_STEP))
            ? CV_ONE_STEP : CV_NORMAL;

    // get the original event status
    vector<unsigned char> eventStatus(mModel->getEventTriggers(0, 0, 0), false);


    // here we stop for a too small timestep ... this seems troublesome to me ...
    while (tout - timeEnd >= 1E-16)
    {
        // here we bail in case we have no ODEs set up with CVODE ... though we should
        // still at least evaluate the model function
        if (!haveVariables() && mModel->getNumEvents() == 0)
        {
            mModel->convertToAmounts();
            mModel->getStateVectorRate(tout, 0, 0);
            return tout;
        }

        if (mLastTimeValue > timeStart)
        {
            restart(timeStart);
        }

        double nextTargetEndTime = tout;

        if (mModel->getPendingEventSize() > 0 &&
                mModel->getNextPendingEventTime(false) < nextTargetEndTime)
        {
            nextTargetEndTime = mModel->getNextPendingEventTime(true);
        }

        // event status before time step
        mModel->getEventTriggers(eventStatus.size(), 0, (eventStatus.size()==0 ? NULL : &eventStatus[0]));

        // time step
        int nResult = CVode(mCVODE_Memory, nextTargetEndTime,  mStateVector, &timeEnd, itask);

        if (nResult == CV_ROOT_RETURN && mFollowEvents)
        {
            Log(Logger::LOG_DEBUG) << ("---------------------------------------------------");
            Log(Logger::LOG_DEBUG) << "--- E V E N T   ( " << mOneStepCount << ", time: " << timeEnd << " ) ";
            Log(Logger::LOG_DEBUG) << ("---------------------------------------------------");


            bool tooCloseToStart = fabs(timeEnd - mLastEvent) > options.relative;

            if(tooCloseToStart)
            {
                strikes =  3;
            }
            else
            {
                strikes--;
            }

            if (tooCloseToStart || strikes > 0)
            {
                // evaluate events
                handleRootsForTime(timeEnd, eventStatus);
                restart(timeEnd);
                mLastEvent = timeEnd;

                if (listener)
                {
                    listener->onEvent(this, mModel, timeEnd);
                }
            }
        }
        else if (nResult == CV_SUCCESS || !mFollowEvents)
        {
            mModel->setTime(timeEnd);
            assignResultsToModel();

            if (listener)
            {
                listener->onTimeStep(this, mModel, timeEnd);
            }
        }
        else
        {
            handleCVODEError(nResult);
        }

        mLastTimeValue = timeEnd;

        try
        {
            mModel->testConstraints();
        }
        catch (const std::exception& e)
        {
            Log(Logger::LOG_WARNING) << "Constraint Violated at time = " << timeEnd << ": " << e.what();
        }

        assignPendingEvents(timeEnd, tout);

        if (tout - timeEnd > 1E-16)
        {
            timeStart = timeEnd;
        }
        Log(Logger::LOG_TRACE) << "time step, tout: " << tout << ", timeEnd: " << timeEnd;

        if (options.integratorFlags & SimulateOptions::VARIABLE_STEP)
        {
            return timeEnd;
        }
    }
    return timeEnd;

}

bool CVODEIntegrator::haveVariables()
{
    return stateVectorVariables;
}

void CVODEIntegrator::createCVode()
{
    if(!mModel)
    {
        return;
    }

    assert(mStateVector == 0 && mCVODE_Memory == 0 &&
            "calling cvodeCreate, but cvode objects already exist");

    // still need cvode state vector size if we have no vars, but have
    // events, needed so root finder works.
    int allocStateVectorSize = 0;
    int realStateVectorSize = mModel->getStateVector(0);

    // cvode return code
    int err;

    if(realStateVectorSize > 0)
    {
        stateVectorVariables = true;
        allocStateVectorSize = realStateVectorSize;
    }
    else if (mModel->getNumEvents() > 0)
    {
        allocStateVectorSize = 1;
        stateVectorVariables = false;
    }
    else
    {
        stateVectorVariables = false;
        return;
    }

    // allocate and init the cvode arrays
    mStateVector = N_VNew_Serial(allocStateVectorSize);
    for (int i = 0; i < allocStateVectorSize; i++)
    {
        SetVector(mStateVector, i, 0.);
    }

    if (options.integratorFlags & SimulateOptions::STIFF)
    {
        Log(Logger::LOG_INFORMATION) << "using stiff integrator";
        mCVODE_Memory = (void*) CVodeCreate(CV_BDF, CV_NEWTON);
    }
    else
    {
        Log(Logger::LOG_INFORMATION) << "using non-stiff integrator";
        mCVODE_Memory = (void*) CVodeCreate(CV_ADAMS, CV_FUNCTIONAL);
    }

    assert(mCVODE_Memory && "could not create Cvode, CVodeCreate failed");

    if ((err = CVodeSetErrHandlerFn(mCVODE_Memory, cvodeErrHandler, NULL)) != CV_SUCCESS)
    {
        handleCVODEError(err);
    }

    // use non default CVODE value here, default is too short
    // for some sbml tests.
    CVodeSetMaxNumSteps(mCVODE_Memory, mDefaultMaxNumSteps);

    double t0 = 0.0;

    if ((err = CVodeSetUserData(mCVODE_Memory, (void*) this)) != CV_SUCCESS)
    {
        handleCVODEError(err);
    }

    if ((err = CVodeInit(mCVODE_Memory, cvodeDyDtFcn, t0, mStateVector)) != CV_SUCCESS)
    {
        handleCVODEError(err);
    }

    if (mModel->getNumEvents() > 0)
    {
        if ((err = CVodeRootInit(mCVODE_Memory, mModel->getNumEvents(),
                cvodeRootFcn)) != CV_SUCCESS)
        {
            handleCVODEError(err);
        }
        Log(Logger::LOG_TRACE) << "CVRootInit executed.....";
    }

    // only allocate this if we are using stiff solver.
    // otherwise, CVode will NOT free it if using standard solver.
    if (options.integratorFlags & SimulateOptions::STIFF)
    {
        if ((err = CVDense(mCVODE_Memory, allocStateVectorSize)) != CV_SUCCESS)
        {
            handleCVODEError(err);
        }
    }

    setCVODETolerances();

    mModel->resetEvents();
}

void CVODEIntegrator::assignPendingEvents(double timeEnd, double tout)
{
    double *stateVector = mStateVector ? NV_DATA_S(mStateVector) : 0;
    int handled = mModel->applyPendingEvents(stateVector, timeEnd, tout);
    if (handled > 0)
    {
        restart(timeEnd);
    }
}

void CVODEIntegrator::testRootsAtInitialTime()
{
    vector<unsigned char> initialEventStatus(mModel->getEventTriggers(0, 0, 0), false);
    mModel->getEventTriggers(initialEventStatus.size(), 0, (initialEventStatus.size()==0 ? NULL : &initialEventStatus[0]));
    handleRootsForTime(0, initialEventStatus);
}


void CVODEIntegrator::handleRootsForTime(double timeEnd, vector<unsigned char> &previousEventStatus)
{
    double *stateVector = mStateVector ? NV_DATA_S(mStateVector) : 0;
    mModel->applyEvents(timeEnd, (previousEventStatus.size()==0 ? NULL : &previousEventStatus[0]), stateVector, stateVector);
    reInit(timeEnd);
}

void CVODEIntegrator::assignResultsToModel()
{
    if (mStateVector)
    {
        mModel->setStateVector(NV_DATA_S(mStateVector));
    }
}

void CVODEIntegrator::setCVODETolerances()
{
    if (mStateVector == 0 || mModel == 0)
    {
        return;
    }

    // If we have a model with only events, cvode still needs a state vector
    // of length 1 to integrate.
    if (!haveVariables() && mModel->getNumEvents() > 0)
    {
        SetVector(mStateVector, 0, 1.0);
    }

    int err;
    if ((err = CVodeSStolerances(mCVODE_Memory, options.relative, options.absolute)) != CV_SUCCESS)
    {
        handleCVODEError(err);
    }

    Log(Logger::LOG_INFORMATION) << "Set tolerance to abs: " << setprecision(16)
            << options.absolute << ", rel: " << options.relative;

}


void CVODEIntegrator::restart(double time)
{
    if (!mModel) {
        return;
    }

    // apply any events that trigger before or at time 0.
    // important NOT to set model time before we check get
    // the initial event state, initially time is < 0.
    if (time <= 0.0) {

        // copy state vector into cvode memory, need to do this before evaluating
        // roots because the applyEvents method copies the cvode state vector
        // into the model
        if (mStateVector)
        {
            mModel->getStateVector(NV_DATA_S(mStateVector));
        }

        testRootsAtInitialTime();
    }

    mModel->setTime(time);

    // copy state vector into cvode memory
    if (mStateVector && mCVODE_Memory)
    {
        mModel->getStateVector(NV_DATA_S(mStateVector));
    }

    // set tolerances and so forth.
    if(mCVODE_Memory)
    {
        reInit(time);
    }
}





// Cvode calls this to compute the dy/dts. This routine in turn calls the
// model function which is located in the host application.
int cvodeDyDtFcn(realtype time, N_Vector cv_y, N_Vector cv_ydot, void *userData)
{
    double* y = NV_DATA_S (cv_y);
    double* ydot = NV_DATA_S(cv_ydot);
    CVODEIntegrator* cvInstance = (CVODEIntegrator*) userData;

    assert(cvInstance && "userData pointer is NULL in cvode dydt callback");

    ExecutableModel *model = cvInstance->mModel;

    model->getStateVectorRate(time, y, ydot);

    if (!cvInstance->stateVectorVariables && cvInstance->mStateVector &&
            NV_LENGTH_S(cvInstance->mStateVector) == 1)
    {
        ydot[0] = 0.0;
    }

    Log(Logger::LOG_TRACE) << __FUNC__ << ", model: " << model;

    return CV_SUCCESS;
}

void CVODEIntegrator::freeCVode()
{
    // cvode does not check for null values.
    if(mCVODE_Memory)
    {
        CVodeFree( &mCVODE_Memory);
    }

    if(mStateVector)
    {
        N_VDestroy_Serial(mStateVector);
    }

    mCVODE_Memory = 0;
    mStateVector = 0;
}

// int (*CVRootFn)(realtype t, N_Vector y, realtype *gout, void *user_data)
// Cvode calls this to check for event changes
int cvodeRootFcn (realtype time, N_Vector y_vector, realtype *gout, void *user_data)
{
    CVODEIntegrator* cvInstance = (CVODEIntegrator*) user_data;

    assert(cvInstance && "user data pointer is NULL on CVODE root callback");

    ExecutableModel *model = cvInstance->mModel;

    double* y = NV_DATA_S (y_vector);

    model->getEventRoots(time, y, gout);

    return CV_SUCCESS;
}



_xmlNode* CVODEIntegrator::createConfigNode()
{
    _xmlNode *cap = Configurable::createCapabilityNode("Integration", "CVODE",
            "CVODE Integrator");

    Configurable::addChild(cap, Configurable::createParameterNode(
            "BDFOrder", "Maximum order for BDF Method", mMaxBDFOrder));
    Configurable::addChild(cap, Configurable::createParameterNode(
            "AdamsOrder", "Maximum order for Adams Method", mMaxAdamsOrder));
    Configurable::addChild(cap, Configurable::createParameterNode(
            "rtol", "Relative Tolerance", options.relative));
    Configurable::addChild(cap, Configurable::createParameterNode(
            "atol", "Absolute Tolerance", options.absolute));
    Configurable::addChild(cap, Configurable::createParameterNode(
            "maxsteps", "Maximum number of internal stepsc", options.maximumNumSteps));
    Configurable::addChild(cap, Configurable::createParameterNode(
            "initstep", "the initial step size", options.initialTimeStep));
    Configurable::addChild(cap, Configurable::createParameterNode(
            "minstep", "specifies a lower bound on the magnitude of the step size.", options.minimumTimeStep));
    Configurable::addChild(cap, Configurable::createParameterNode(
            "maxstep", "specifies an upper bound on the magnitude of the step size.", options.maximumTimeStep));

    return cap;
}

void CVODEIntegrator::loadConfig(const _xmlDoc* doc)
{
    mMaxBDFOrder = Configurable::getParameterIntValue(doc, "Integration", "BDFOrder");
    mMaxAdamsOrder = Configurable::getParameterIntValue(doc, "Integration", "AdamsOrder");
    options.relative = Configurable::getParameterDoubleValue(doc, "Integration", "rtol");
    options.absolute = Configurable::getParameterDoubleValue(doc, "Integration", "atol");
    options.maximumNumSteps = Configurable::getParameterIntValue(doc, "Integration", "maxsteps");
    options.initialTimeStep = Configurable::getParameterDoubleValue(doc, "Integration", "initstep");
    options.minimumTimeStep = Configurable::getParameterDoubleValue(doc, "Integration", "minstep");
    options.maximumTimeStep = Configurable::getParameterDoubleValue(doc, "Integration", "maxstep");
}


void CVODEIntegrator::setValue(const std::string& key, const rr::Variant& value)
{
}

Variant CVODEIntegrator::getValue(const std::string& key) const
{
    return Variant();
}

bool CVODEIntegrator::hasKey(const std::string& key) const
{
    return false;
}

int CVODEIntegrator::deleteValue(const std::string& key)
{
    return -1;
}

std::vector<std::string> CVODEIntegrator::getKeys() const
{
    return std::vector<std::string>();
}


static std::string cvodeDecodeError(int cvodeError, bool exInfo)
{
    std::string result;

    switch (cvodeError)
    {
    case CV_TOO_MUCH_WORK:
        result = "CV_TOO_MUCH_WORK";
        if (exInfo)
        {
            result += ": The solver took mxstep internal steps but "
                    "could not reach tout. The default value for "
                    "mxstep is MXSTEP_DEFAULT = 500.";
        }
        break;
    case CV_TOO_MUCH_ACC:
        result = "CV_TOO_MUCH_ACC";
        if (exInfo)
        {
            result += ": The solver could not satisfy the accuracy "
                    "demanded by the user for some internal step.";
        }
        break;
    case CV_ERR_FAILURE:
        result = "CV_ERR_FAILURE";
        if (exInfo)
        {
            result += ": Error test failures occurred too many times "
                    "(= MXNEF = 7) during one internal time step or"
                    "occurred with |h| = hmin.";
        }
        break;
    case CV_CONV_FAILURE:
        result = "CV_CONV_FAILURE";
        if (exInfo)
        {
            result += ": Convergence test failures occurred too many "
                    "times (= MXNCF = 10) during one internal time"
                    "step or occurred with |h| = hmin.";
        }
        break;
    case CV_LINIT_FAIL:
        result = "CV_LINIT_FAIL";
        if (exInfo)
        {
            result += ": The linear solver's initialization function "
                    "failed.";
        }
        break;
    case CV_LSETUP_FAIL:
        result = "CV_LSETUP_FAIL";
        if (exInfo)
        {
            result += ": The linear solver's setup routine failed in an "
                    "unrecoverable manner.";
        }
        break;
    case CV_LSOLVE_FAIL:
        result = "CV_LSOLVE_FAIL";
        if (exInfo)
        {
            result += ": The linear solver's solve routine failed in an "
                    "unrecoverable manner.";
        }
        break;
    case CV_RHSFUNC_FAIL:
        result = "CV_RHSFUNC_FAIL";
        break;
    case CV_FIRST_RHSFUNC_ERR:
        result = "CV_FIRST_RHSFUNC_ERR";
        break;
    case CV_REPTD_RHSFUNC_ERR:
        result = "CV_REPTD_RHSFUNC_ERR";
        break;
    case CV_UNREC_RHSFUNC_ERR:
        result = "CV_UNREC_RHSFUNC_ERR";
        break;
    case CV_RTFUNC_FAIL:
        result = "CV_RTFUNC_FAIL";
        break;
    case CV_MEM_FAIL:
        result = "CV_MEM_FAIL";
        break;
    case CV_MEM_NULL:
        result = "CV_MEM_NULL";
        if (exInfo)
        {
            result += ": The cvode_mem argument was NULL.";
        }
        break;
    case CV_ILL_INPUT:
        result = "CV_ILL_INPUT";
        if (exInfo)
        {
            result += ": One of the inputs to CVode is illegal. This "
                    "includes the situation when a component of the "
                    "error weight vectors becomes < 0 during "
                    "internal time-stepping.  It also includes the "
                    "situation where a root of one of the root "
                    "functions was found both at t0 and very near t0. "
                    "The ILL_INPUT flag will also be returned if the "
                    "linear solver routine CV--- (called by the user "
                    "after calling CVodeCreate) failed to set one of "
                    "the linear solver-related fields in cvode_mem or "
                    "if the linear solver's init routine failed. In "
                    "any case, the user should see the printed "
                    "error message for more details.";
        }
        break;
    case CV_NO_MALLOC:
        result = "CV_NO_MALLOC";
        if (exInfo)
        {
            result += ": indicating that cvode_mem has not been "
                    "allocated (i.e., CVodeInit has not been "
                    "called).";
        }
        break;
    case CV_BAD_K:
        result = "CV_BAD_K";
        if (exInfo)
        {
            result += ": k is not in the range 0, 1, ..., qu.";
        }
        break;
    case CV_BAD_T:
        result = "CV_BAD_T";
        if (exInfo)
        {
            result += ": t is not in the interval [tn-hu,tn].";
        }
        break;
    case CV_BAD_DKY:
        result = "CV_BAD_DKY";
        if (exInfo)
        {
            result += ": The dky argument was NULL.";
        }
        break;
    case CV_TOO_CLOSE:
        result = "CV_TOO_CLOSE:";
        break;
    default:
        result = "UNKNOWN_CODE";
        break;
    }
    return result;
}



/**
 * Purpose
 * This function processes error and warning messages from CVODE and its
 * sub-modules.
 *
 * Arguments:
 * error_code: is the error code.
 * module: is the name of the CVODE module reporting the error.
 * function: is the name of the function in which the error occurred.
 * msg: is the error message.
 * eh_data: is a pointer to user data, the same as the eh_data parameter
 * passed to CVodeSetErrHandlerFn.
 * Return value: A CVErrHandlerFn function has no return value.
 *
 * Notes:
 * error_code is negative for errors and positive (CV_WARNING) for warnings.
 * If a function that returns a pointer to memory encounters an error,
 * it sets error_code to 0.
 */
void cvodeErrHandler(int error_code, const char *module, const char *function,
        char *msg, void *eh_data)
{
    if (error_code < 0) {
        Log(Logger::LOG_ERROR) << "CVODE Error: " << cvodeDecodeError(error_code, false)
                               << ", Module: " << module << ", Function: " << function
                               << ", Message: " << msg;

    }
    else if(error_code == CV_WARNING) {
        Log(Logger::LOG_WARNING) << "CVODE Warning: "
                                 << ", Module: " << module << ", Function: " << function
                                 << ", Message: " << msg;
    }
}

}

