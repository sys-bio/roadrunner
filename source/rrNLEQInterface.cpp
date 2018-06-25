#pragma hdrstop
#include "rrNLEQInterface.h"
#include "rrExecutableModel.h"
#include "rrStringUtils.h"
#include "rrUtils.h"
#include "nleq/nleq2.h"
#include "rrLogger.h"
#include "rrUtils.h"
#include "rrException.h"
#include "rrConfig.h"

#include <Poco/Mutex.h>
#include <assert.h>
#include <math.h>

namespace rr
{

// NLEQ is an ancient Fortran77 routine that assumes that there is only
// one program which has a hard coded function in it.
// So, there is no concept of a user suplied data block, have to store
// the model in this static location -- only a single thread at a time
// may use the nleq steady state.
static ExecutableModel* callbackModel = NULL;

// mutex to ensure only one thead
using Poco::Mutex;
static Mutex mutex;

// the NLEQ callback, we use same data types as f2c here.
static void ModelFunction(int* nx, double* y, double* fval, int* pErr);

static string ErrorForStatus(int error);

static bool isError(int e)
{
    return e != 4 && e != 5;
}

static bool isWarning(int e)
{
    //case 4:     return ("Warning: Superlinear or quadratic convergence slowed down near the solution");
    //case 5:     return ("Warning: Error Tolerance reached but solution is suspect");

    return e == 4 || e == 5;
}

NLEQInterface::NLEQInterface(ExecutableModel *_model) :
    IWK(0),
    LIWK(0),
    LWRK(0),
    RWK(0),
    XScal(0),
    ierr(0),
    iopt(0),
    model(0),
    nOpts(50),
    relativeTolerance(Config::getDouble(Config::STEADYSTATE_RELATIVE)),
    maxIterations(Config::getInt(Config::STEADYSTATE_MAXIMUM_NUM_STEPS)),
    minDamping(Config::getDouble(Config::STEADYSTATE_MINIMUM_DAMPING)),
    broyden(Config::getDouble(Config::STEADYSTATE_BROYDEN)),
    linearity(Config::getDouble(Config::STEADYSTATE_LINEARITY))
{
    model = _model;

    if(model)
    {
        setup();
    }
}

NLEQInterface::~NLEQInterface()
{
    delete[] IWK;
    delete[] RWK;
    delete[] XScal;
    delete[] iopt;
}

void NLEQInterface::setup()
{
    // size of state vector
    n = model->getStateVector(0);
    Log(Logger::LOG_DEBUG) << "NLEQInterface: size of state vector = " << n;

    // Allocate space, see NLEQ docs for details
    LWRK = (n + 2 + 15)*n + 61;
    LIWK = n + 52;

    XScal = new double[n];
    // Set up default scaling factors
    for (int i = 0; i < n; i++)
    {
        XScal[i] = 1.0;
    }

    iopt = new long[nOpts];
    for (int i = 0; i < nOpts; i++)
    {
        iopt[i] = 0;
    }

    // Set for Highly nonlinear problem
    iopt[31 - 1] = linearity;

    // Set for Broyden method
    iopt[32 - 1] = broyden;

    // Initialise all array elements to 0.0
    IWK = new long[LIWK];
    for (int i = 0; i < LIWK; i++)
    {
        IWK[i] = 0;
    }
    IWK[31 - 1] = maxIterations; // Max iterations


    RWK = new double[LWRK];
    for (int i = 0; i < LWRK; i++)
    {
        RWK[i] = 0.0;
    }
    RWK[22 - 1] = minDamping; // Minimal allowed damping factor
}

bool NLEQInterface::isAvailable()
{
    NLEQInterface *temp= new NLEQInterface(NULL);
    if(temp)
    {
        bool val = temp->getNumberOfModelEvaluations() ? true : false;
        delete temp;
        return val;
    }

    return false;
}

double NLEQInterface::solve()
{
    // lock so only one thread can be here.
    Mutex::ScopedLock lock(mutex);

    Log(Logger::LOG_DEBUG) << "NLEQInterface::solve";

    // Set up a dummy Jacobian, actual Jacobian is computed
    // by NLEQ using finite differences
    //    double* Jacobian = new double[1];

    ierr = 0;

    // Set up default scaling factors
    for (int i = 0; i < n; i++)
    {
        XScal[i] = 1.0;
    }

    for (int i = 0; i < nOpts; i++)
    {
        iopt[i] = 0;
    }

    iopt[31 - 1] = linearity; // Set for Highly nonlinear problem

    iopt[32 - 1] = broyden; // Set for Broyden method

    // Initialise all array elements to 0.0
    for (int i = 0; i < LIWK; i++)
    {
        IWK[i] = 0;
    }

    IWK[31 - 1] = maxIterations; // Max iterations

    for (int i = 0; i < LWRK; i++)
    {
        RWK[i] = 0.0;
    }

    RWK[22 - 1] = minDamping; // Minimal allowed damping factor

    // For some reason NLEQ modifies the tolerance value, use a local copy instead
    double tmpTol = relativeTolerance;

    // set up the thread local variables, only this thread
    // access them.
    if (callbackModel)
    {
        throw Exception("global callbackModel is set, this should never happen!");
    }

    // should be exception safe, just make sure

    try
    {
        callbackModel = model;
        vector<double> stateVector(n);
        model->getStateVector(&stateVector[0]);

        NLEQ2(  &n,
                &ModelFunction,
                NULL,
                &stateVector[0],
                XScal,
                &tmpTol,
                iopt,
                &ierr,
                &LIWK,
                IWK,
                &LWRK,
                RWK);

        // done, clear it.
        callbackModel = NULL;
    }
    catch(...)
    {
        // clear the global model and re-throw the exception.
        callbackModel = NULL;
        throw;
    }

    if(ierr > 0 )
    {
        if (isWarning(ierr)) {
            Log(Logger::LOG_WARNING) << ErrorForStatus(ierr);
        } else {
            string err = ErrorForStatus(ierr);
            Log(Logger::LOG_ERROR)<<"Error :"<<err;
            throw NLEQException(err);
        }
    }

    return computeSumsOfSquares();
}



/*     FCN(N,X,F,IFAIL) Ext    Function subroutine */
/*       N              Int    Number of vector components (input) */
/*       X(N)           Dble   Vector of unknowns (input) */
/*       F(N)           Dble   Vector of function values (output) */
/*       IFAIL          Int    FCN evaluation-failure indicator. (output) */
/*                             On input:  Has always value 0 (zero). */
/*                             On output: Indicates failure of FCN eval- */
/*                                uation, if having a value <= 2. */
/*                             If <0: NLEQ1 will be terminated with */
/*                                    error code = 82, and IFAIL stored */
/*                                    to IWK(23). */
/*                             If =1: A new trial Newton iterate will */
/*                                    computed, with the damping factor */
/*                                    reduced to it's half. */
/*                             If =2: A new trial Newton iterate will */
/*                                    computed, with the damping factor */
/*                                    reduced by a reduct. factor, which */
/*                                    must be output through F(1) by FCN, */
/*                                    and it's value must be >0 and < 1. */
/*                             Note, that if IFAIL = 1 or 2, additional */
/*                             conditions concerning the damping factor, */
/*                             e.g. the minimum damping factor or the */
/*                             bounded damping strategy may also influ- */
/*                             ence the value of the reduced damping */
/*                             factor. */

void ModelFunction(int* nx, double* y, double* fval, int* pErr)
{
    ExecutableModel* model = callbackModel;
    assert(model && "model is NULL");

    assert(*nx == model->getStateVector(0) && "incorrect state vector size");

    // sets the state vector
    model->setStateVector(y);

    model->getStateVectorRate(0, y, fval);

    if (rr::Logger::getLevel() >= Logger::LOG_DEBUG)
    {
        std::stringstream ss;

        ss << "NLEQ ModelFunction" << std::endl;

        ss << "y: [";
        for (int i = 0; i < *nx; ++i)
        {
            ss << y[i];
            if (i + 1 < *nx)
            {
                ss << ", ";
            }
        }
        ss << "]" << std::endl;

        ss << "dydt: [";
        for (int i = 0; i < *nx; ++i)
        {
            ss << fval[i];
            if (i + 1 < *nx)
            {
                ss << ", ";
            }
        }
        ss << "]" << std::endl;

        Log(Logger::LOG_DEBUG) << ss.str();
    }

    *pErr = 0;
}

void NLEQInterface::setScalingFactors(const vector<double>& sx)
{
    for (int i = 0; i < n; i++)
    {
        XScal[i] = sx[i];
    }
}

int NLEQInterface::getNumberOfNewtonIterations()
{
    return IWK[0];
}

int NLEQInterface::getNumberOfCorrectorSteps()
{
    return IWK[2];
}

int NLEQInterface::getNumberOfModelEvaluations()
{
    return IWK[3];
}

int NLEQInterface::getNumberOfJacobianEvaluations()
{
    return IWK[4];
}

int NLEQInterface::getNumberOfModelEvaluationsForJacobian()
{
    return IWK[7];
}

string ErrorForStatus(int error)
{
    switch (error)
    {
    case 1:     return ("Jacobian matrix singular in NLEQ. Failed to converge to steady state. Check if Jacobian matrix is non-invertible or steady state solution does not exist.");
    case 2:     return ("Maximum iterations exceeded");
    case 3:     return ("Damping factor has became to small to continue");
    case 4:     return ("Warning: Superlinear or quadratic convergence slowed down near the solution");
    case 5:     return ("Warning: Error Tolerance reached but solution is suspect");
    case 10:    return ("Integer or real workspace too small in NLEQ");
    case 20:    return ("Bad input to size of model parameter");
    case 21:    return ("Nonpositive value for RTOL supplied to NLEQ");
    case 22:    return ("Negative scaling value via vector XSCAL supplied");
    case 30:    return ("One or more fields specified in IOPT are invalid (NLEQ)");
    case 80:    return ("Error signalled by linear solver routine N1FACT, in NLEQ");
    case 81:    return ("Error signalled by linear solver routine N1SOLV, in NLEQ");
    case 82:    return ("Possible negative concentrations in solution (NLEQ)");
    case 83:    return ("Error signalled by user routine JAC in NLEQ");
    default:    return (format("Unknown error in NLEQ, errCode = {0}", error));
    }
}



// steady state solver options
static const char* keys[] =
{
        "relativeTolerance"
        "maxIterations",
        "minDamping"
        "broyden"
        "linearity"

        "relativeTolerance.description"
        "maxIterations.description",
        "minDamping.description"
        "broyden.description"
        "linearity.description"

        "relativeTolerance.hint"
        "maxIterations.hint",
        "minDamping.hint"
        "broyden.hint"
        "linearity.hint"
};

void NLEQInterface::setItem(const std::string& key, const rr::Variant& value)
{
}

Variant NLEQInterface::getItem(const std::string& key) const
{
    return 0;
}

bool NLEQInterface::hasKey(const std::string& key) const
{
    return false;
}

int NLEQInterface::deleteItem(const std::string& key)
{
    return 0;
}

std::vector<std::string> NLEQInterface::getKeys() const
{
    return std::vector<std::string>(&keys[0], &keys[sizeof(keys)/sizeof(char*)]);
}

const Dictionary* NLEQInterface::getSteadyStateOptions()
{
    static BasicDictionary dict;

    dict.setItem("steadyState", "NLEQ");
    dict.setItem("steadyState.hint", "NLEQ hint");
    dict.setItem("steadyState.description", "NLEQ description");

    dict.setItem("relativeTolerance", Config::getDouble(Config::STEADYSTATE_RELATIVE));
    dict.setItem("maxIterations", Config::getInt(Config::STEADYSTATE_MAXIMUM_NUM_STEPS));
    dict.setItem("minDamping", Config::getDouble(Config::STEADYSTATE_MINIMUM_DAMPING));
    dict.setItem("broyden", Config::getDouble(Config::STEADYSTATE_BROYDEN));
    dict.setItem("linearity", Config::getDouble(Config::STEADYSTATE_LINEARITY));

    dict.setItem("relativeTolerance.description", "relativeTolerance.description");
    dict.setItem("maxIterations.description", "maxIterations.description");
    dict.setItem("minDamping.description", "minDamping.description");
    dict.setItem("broyden.description", "broyden.description");
    dict.setItem("linearity.description", "linearity.description");

    dict.setItem("relativeTolerance.hint", "relativeTolerance.hint");
    dict.setItem("maxIterations.hint", "maxIterations.hint");
    dict.setItem("minDamping.hint", "minDamping.hint");
    dict.setItem("broyden.hint", "broyden.hint");
    dict.setItem("linearity.hint", "linearity.hint");

    return &dict;
}

double NLEQInterface::computeSumsOfSquares()
{
    double sum = 0;
    vector<double> rates(model->getStateVector(0));
    model->getStateVectorRate(0, 0, &rates[0]);

    for (int i = 0; i < n; i++)
    {
        sum = sum + pow(rates[i], 2.0);
    }
    return sqrt(sum);
}



}    //end of namespace


