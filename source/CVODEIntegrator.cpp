#pragma hdrstop

#include "CVODEIntegrator.h"
#include "rrExecutableModel.h"
#include "rrException.h"
#include "rrLogger.h"
#include "rrStringUtils.h"
#include "rrException.h"
#include "rrConfig.h"
#include "rrUtils.h"
#include "SundialsLogger.h"


#include <cvodes/cvodes.h>                          /* prototypes for CVODE fcts., consts.          */
#include <nvector/nvector_serial.h>               /* access to serial N_Vector                    */
#include <sunnonlinsol/sunnonlinsol_fixedpoint.h> /* access to the fixed point SUNNonlinearSolver */
#include <sunnonlinsol/sunnonlinsol_newton.h> /* access to the fixed point SUNNonlinearSolver */
#include <sunlinsol/sunlinsol_dense.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunmatrix/sunmatrix_sparse.h>
#ifdef RR_USE_KLU
#include <sunlinsol/sunlinsol_klu.h>
#endif

#include <cstring>
#include <iomanip>
#include <cmath>
#include <map>
#include <algorithm>
#include <limits>
#include <cassert>
#include <chrono>
#include <Poco/Logger.h>
#include <iostream>

#define CVODE_INT_TYPECODE 0x7799ff00

//

namespace rr {
    const int CVODEIntegrator::mDefaultMaxNumSteps = 20000;
    const int CVODEIntegrator::mDefaultMaxAdamsOrder = 12;
    const int CVODEIntegrator::mDefaultMaxBDFOrder = 5;

    int cvodeDyDtFcn(sunrealtype t, N_Vector cv_y, N_Vector cv_ydot, void *userData);

    int cvodeEventAndPiecewiseRootFcn(sunrealtype t, N_Vector y, sunrealtype *gout, void *userData);

    int cvodeJacFcn(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J, void *userData,
                    N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

    // Sets the value of an element in a N_Vector object
    inline void SetVector(N_Vector v, int Index, double Value) {
        double *data = NV_DATA_S(v);
        data[Index] = Value;
    }

    inline double GetVector(N_Vector v, int Index) {
        double *data = NV_DATA_S(v);
        return data[Index];
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
    static void cvodeErrHandler(int line, const char* function, const char* file,
        const char* msg, SUNErrCode error_code,
        void* err_user_data, SUNContext sunctx);
    /**
    * macro to throw a (hopefully) useful error message
    */
#define handleCVODEError(errCode) \
    { std::string _err_what = std::string("CVODE Error: ") + \
    decodeSundialsError(this, errCode, true); \
    throw IntegratorException(_err_what, std::string(__FUNC__)); }

    namespace {
        /**
         * @brief Adds the time from its construction to its destruction to a time,
         * if it is enabled.
         */
        class ScopedTimer {
        public:
            ScopedTimer(bool enabled, double &time)
                : time(enabled ? &time : nullptr) {
                if (this->time) {
                    start = std::chrono::steady_clock::now();
                }
            }

            ~ScopedTimer() {
                if (time) {
                    *time += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                }
            }

            ScopedTimer(const ScopedTimer &) = delete;

            ScopedTimer &operator=(const ScopedTimer &) = delete;

        private:
            double *time;
            std::chrono::steady_clock::time_point start;
        };
    }

    /**
     * @brief A linear solver that times the setups and solves of the linear solver
     * it wraps, for the statistics of CVODEIntegrator (setting collect_timing).
     * @details It owns the wrapped linear solver. All other operations are forwarded.
     */
    struct TimedLinearSolver {
        SUNLinearSolver inner;
        CVODEIntegrator *integrator;

        static SUNLinearSolver innerOf(SUNLinearSolver S) {
            return static_cast<TimedLinearSolver *>(S->content)->inner;
        }

        static CVODEIntegrator *integratorOf(SUNLinearSolver S) {
            return static_cast<TimedLinearSolver *>(S->content)->integrator;
        }

        static SUNLinearSolver_Type getType(SUNLinearSolver S) {
            return SUNLinSolGetType(innerOf(S));
        }

        static SUNLinearSolver_ID getID(SUNLinearSolver S) {
            return SUNLinSolGetID(innerOf(S));
        }

        static SUNErrCode setATimes(SUNLinearSolver S, void *data, SUNATimesFn fn) {
            return SUNLinSolSetATimes(innerOf(S), data, fn);
        }

        static SUNErrCode setPreconditioner(SUNLinearSolver S, void *data, SUNPSetupFn setup, SUNPSolveFn solve) {
            return SUNLinSolSetPreconditioner(innerOf(S), data, setup, solve);
        }

        static SUNErrCode setScalingVectors(SUNLinearSolver S, N_Vector s1, N_Vector s2) {
            return SUNLinSolSetScalingVectors(innerOf(S), s1, s2);
        }

        static SUNErrCode setZeroGuess(SUNLinearSolver S, sunbooleantype onoff) {
            return SUNLinSolSetZeroGuess(innerOf(S), onoff);
        }

        static SUNErrCode initialize(SUNLinearSolver S) {
            return SUNLinSolInitialize(innerOf(S));
        }

        static int setup(SUNLinearSolver S, SUNMatrix A) {
            CVODEIntegrator *integrator = integratorOf(S);
            ScopedTimer timer(integrator->collectTiming, integrator->times.linSolvSetups);
            return SUNLinSolSetup(innerOf(S), A);
        }

        static int solve(SUNLinearSolver S, SUNMatrix A, N_Vector x, N_Vector b, sunrealtype tol) {
            CVODEIntegrator *integrator = integratorOf(S);
            ScopedTimer timer(integrator->collectTiming, integrator->times.linSolvSolves);
            return SUNLinSolSolve(innerOf(S), A, x, b, tol);
        }

        static int numIters(SUNLinearSolver S) {
            return SUNLinSolNumIters(innerOf(S));
        }

        static sunrealtype resNorm(SUNLinearSolver S) {
            return SUNLinSolResNorm(innerOf(S));
        }

        static sunindextype lastFlag(SUNLinearSolver S) {
            return SUNLinSolLastFlag(innerOf(S));
        }

        static N_Vector resid(SUNLinearSolver S) {
            return SUNLinSolResid(innerOf(S));
        }

        static SUNErrCode free(SUNLinearSolver S) {
            if (S == nullptr) {
                return SUN_SUCCESS;
            }
            if (S->content) {
                SUNLinSolFree(innerOf(S));
                delete static_cast<TimedLinearSolver *>(S->content);
                S->content = nullptr;
            }
            SUNLinSolFreeEmpty(S);
            return SUN_SUCCESS;
        }

        /**
         * @brief a TimedLinearSolver that wraps @param inner, or nullptr if it cannot be
         * created, in which case @param inner is freed.
         * @details CVODE checks which operations a linear solver has, so the wrapper has
         * the same ones as @param inner.
         */
        static SUNLinearSolver create(SUNLinearSolver inner, CVODEIntegrator *integrator) {
            SUNLinearSolver S = SUNLinSolNewEmpty(inner->sunctx);
            if (S == nullptr) {
                SUNLinSolFree(inner);
                return nullptr;
            }
            S->content = new TimedLinearSolver{inner, integrator};
            const SUNLinearSolver_Ops ops = inner->ops;
            S->ops->gettype = ops->gettype ? TimedLinearSolver::getType : nullptr;
            S->ops->getid = ops->getid ? TimedLinearSolver::getID : nullptr;
            S->ops->setatimes = ops->setatimes ? TimedLinearSolver::setATimes : nullptr;
            S->ops->setpreconditioner = ops->setpreconditioner ? TimedLinearSolver::setPreconditioner : nullptr;
            S->ops->setscalingvectors = ops->setscalingvectors ? TimedLinearSolver::setScalingVectors : nullptr;
            S->ops->setzeroguess = ops->setzeroguess ? TimedLinearSolver::setZeroGuess : nullptr;
            S->ops->initialize = ops->initialize ? TimedLinearSolver::initialize : nullptr;
            S->ops->setup = ops->setup ? TimedLinearSolver::setup : nullptr;
            S->ops->solve = ops->solve ? TimedLinearSolver::solve : nullptr;
            S->ops->numiters = ops->numiters ? TimedLinearSolver::numIters : nullptr;
            S->ops->resnorm = ops->resnorm ? TimedLinearSolver::resNorm : nullptr;
            S->ops->lastflag = ops->lastflag ? TimedLinearSolver::lastFlag : nullptr;
            S->ops->resid = ops->resid ? TimedLinearSolver::resid : nullptr;
            S->ops->free = TimedLinearSolver::free;
            return S;
        }
    };


    CVODEIntegrator::CVODEIntegrator(ExecutableModel *aModel)
            :
            Integrator(aModel),
            mStateVector(nullptr),
            mCVODE_Memory(nullptr),
            mLastEventTime(0),
            stateVectorVariables(false),
            variableStepPendingEvent(false),
            variableStepTimeEndEvent(false),
            variableStepEventTolerance(0),
            typecode_(CVODE_INT_TYPECODE) {
        rrLog(Logger::LOG_INFORMATION) << "creating CVODEIntegrator";

        CVODEIntegrator::resetSettings();

        if (aModel) {
            create();

            // allocate space for the event status array
            eventStatus = std::vector<unsigned char>(mModel->getEventTriggers(0, nullptr, nullptr), false);
        }

        // Update parameter settings within CVODE.
        updateCVODE();
    }

    CVODEIntegrator::~CVODEIntegrator() {
        if (mCVODE_Memory && mModel) {
            freeSundialsMemory();
        }
        if (mSunContext) {
            SUNContext_Free(&mSunContext);
        }
    }

    void CVODEIntegrator::setListener(IntegratorListenerPtr p) {
        listener = p;
    }

    IntegratorListenerPtr CVODEIntegrator::getListener() {
        return listener;
    }

    void CVODEIntegrator::checkType() const {
        if (typecode_ != CVODE_INT_TYPECODE)
            throw std::runtime_error("CVODEIntegrator::checkType failed, memory bug");
    }

    void CVODEIntegrator::checkVectorSize(int expected, size_t real) const {
        if (expected != real)
            throw std::runtime_error(
                    "CVODEIntegrator::checkVectorSize failed, real size is " + CVODEIntegrator::ToString(real) +
                    ", whereas expected size is " + CVODEIntegrator::ToString(expected));
    }

    void CVODEIntegrator::checkIndex(int index, int size) const {
        if (index < 0 || index >= size)
            throw std::out_of_range("CVODEIntegrator::checkIndex failed, index " + CVODEIntegrator::ToString(index) +
                                    " out of range from " + CVODEIntegrator::ToString(0) + " to " +
                                    CVODEIntegrator::ToString(size - 1));
    }

    std::string CVODEIntegrator::ToString(int val) const {
        std::stringstream stream;
        stream << val;
        return stream.str();
    }

    std::string CVODEIntegrator::ToString(size_t val) const {
        std::stringstream stream;
        stream << val;
        return stream.str();
    }


    void CVODEIntegrator::resetSettings() {
        Solver::resetSettings();

        // Set default integrator settings.
        addSetting("relative_tolerance", Setting(Config::getDouble(Config::CVODE_MIN_RELATIVE)), "Relative Tolerance",
                   "Specifies the scalar relative tolerance (double).",
                   "(double) CVODE calculates a std::vector of error weights which is used in all error and convergence tests. The weighted RMS norm for the relative tolerance should not become smaller than this value.");
        addSetting("absolute_tolerance", Setting(Config::getDouble(Config::CVODE_MIN_ABSOLUTE)), "Absolute Tolerance",
                   "Specifies the scalar or std::vector absolute tolerance (double or double std::vector).",
                   "(double or double std::vector) CVODE calculates a std::vector of error weights which is used in all error and convergence tests. If a single value is set, that vector is the single value multiplied by the value of the elements of the initial state vector (floating species, and elements set by rate rules).  If the value of the element is zero or close to zero (within 1e-30), the absolute tolerance value is either multiplied by the species' compartment volume, or, if the element is not a species, the absolute tolerance value is used directly.  Note that the state vector uses species amounts, so if the vector is being set, the values need to be scaled to the amounts of the species, not their concentrations.  (When setting a single value, this will be taken care of automatically.)  The vector being used by the solver can be obtained by calling 'getAbsoluteToleranceVector' on the CVODE integrator.  The weighted RMS norm for the absolute tolerance should not become smaller than this value.");
        addSetting("stiff", Setting(true), "Stiff",
                   "Specifies whether the integrator attempts to solve stiff equations. (bool)",
                   "(bool) Specifies whether the integrator attempts to solve stiff equations. Ensure the integrator can solver stiff differential equations by setting this value to true.");
        addSetting("maximum_bdf_order", Setting(mDefaultMaxBDFOrder), "Maximum BDF Order",
                   "Specifies the maximum order for Backward Differentiation Formula integration. (int)",
                   "(int) Specifies the maximum order for Backward Differentiation Formula integration. This integration method is used for stiff problems. Default value is 5.");
        addSetting("maximum_adams_order", Setting(mDefaultMaxAdamsOrder), "Maximum Adams Order",
                   "Specifies the maximum order for Adams-Moulton intergration. (int)",
                   "(int) Specifies the maximum order for Adams-Moulton intergration. This integration method is used for non-stiff problems. Default value is 12.");
        addSetting("maximum_num_steps", Setting(mDefaultMaxNumSteps), "Maximum Number of Steps",
                   "Specifies the maximum number of steps to be taken by the CVODE solver in its attempt to reach tout. (int)",
                   "(int) Maximum number of steps to be taken by the CVODE solver in its attempt to reach tout.");
        addSetting("maximum_time_step", Setting(0.0), "Maximum Time Step",
                   "Specifies the maximum absolute value of step size allowed. (double)",
                   "(double) The maximum absolute value of step size allowed.");
        addSetting("minimum_time_step", Setting(0.0), "Minimum Time Step",
                   "Specifies the minimum absolute value of step size allowed. (double)",
                   "(double) The minimum absolute value of step size allowed.");
        addSetting("initial_time_step", Setting(0.0), "Initial Time Step",
                   "Specifies the initial time step size. (double)",
                   "(double) Specifies the initial time step size. If inappropriate, CVODE will attempt to estimate a better initial time step.");
        addSetting("multiple_steps", Setting(false), "Multiple Steps",
                   "Perform a multiple time step simulation. (bool)",
                   "(bool) Perform a multiple time step simulation.");
        addSetting("variable_step_size", Setting(false), "Variable Step Size",
                   "Perform a variable time step simulation. (bool)",
                   "(bool) Enabling this setting will allow the integrator to adapt the size of each time step. This will result in a non-uniform time column.  The number of steps or points will be ignored, and the max number of output rows will be used instead.");
        addSetting("max_output_rows", Setting(Config::getInt(Config::MAX_OUTPUT_ROWS)), "Maximum Output Rows",
                   "For variable step size simulations, the maximum number of output rows produced (int).",
                   "(int) This will set the maximum number of output rows for variable step size integration.  This may truncate some simulations that may not reach the desired end time, but prevents infinite or massive output for simulations where the variable step size ends up decreasing too much.");
        addSetting("collect_timing", Setting(false), "Collect Timing",
                   "Measure the time of the integration and of its parts for getStatistics. (bool)",
                   "(bool) Enabling this setting measures the time of the integration, of the evaluations of the model rates and root functions, and of the setups and solves of the linear solver. getStatistics returns these times (time_total, time_rhs_evals, time_g_evals, time_lin_solv_setups and time_lin_solv_solves). The measurement adds a little time to every evaluation of the model, so it is off by default. Changing it re-creates CVODE when the stiff solver is used, because the linear solver is wrapped to time it.");
        addSetting("jacobian", Setting(std::string("sparse_difference_quotient")), "Jacobian",
                   "How CVODE gets the Jacobian of the stiff solver: \"difference_quotient\" or \"sparse_difference_quotient\" (string).",
                   "(string) \"difference_quotient\" lets CVODE approximate the Jacobian by forward differences, one evaluation of the model per state. \"sparse_difference_quotient\" (default) computes the same forward differences with the sparsity pattern of the Jacobian, one evaluation of the reaction rates per color of its columns. A model without a sparsity pattern (e.g. with conserved moiety analysis) falls back to \"difference_quotient\".");
        addSetting("linear_solver", Setting(std::string("klu")), "Linear Solver",
                   "The linear solver of the stiff solver: \"dense\" or \"klu\" (string).",
                   "(string) \"dense\" is a dense direct solver. \"klu\" (default) is the sparse direct solver KLU; it needs jacobian = \"sparse_difference_quotient\" and a roadrunner built with RR_USE_KLU, otherwise CVODE falls back to the dense solver.");
        CVODEIntegrator::loadConfigSettings();
    }


    void CVODEIntegrator::syncWithModel(ExecutableModel *m) {
        if (mCVODE_Memory) {
            freeSundialsMemory();
        }

        mModel = m;

        mLastEventTime = 0;
        stateVectorVariables = false;
        variableStepPendingEvent = false;
        variableStepTimeEndEvent = false;
        variableStepEventTolerance = 0;

        if (m) {
            create();

            // allocate space for the event status array
            eventStatus = std::vector<unsigned char>(mModel->getEventTriggers(0, nullptr, nullptr), false);
        }

        // Update parameter settings within CVODE.
        updateCVODE();
    }

    void CVODEIntegrator::loadConfigSettings() {
        //  VARIABLE STEP SIZE
        bool bVal = false;
        if (getIntegrationMethod() == Integrator::Deterministic) {
            bVal = Config::getBool(Config::SIMULATEOPTIONS_DETERMINISTIC_VARIABLE_STEP);
            Integrator::setValue("variable_step_size", Setting(bVal));
        }
        else if (getIntegrationMethod() == Integrator::Stochastic) {
            bVal = Config::getBool(Config::SIMULATEOPTIONS_STOCHASTIC_VARIABLE_STEP);
            Integrator::setValue("variable_step_size", Setting(bVal));
        }
        Integrator::setValue("max_output_rows", Setting(Config::getInt(Config::MAX_OUTPUT_ROWS)));

        // STIFFNESS
        bVal = Config::getBool(Config::SIMULATEOPTIONS_STIFF);
        Integrator::setValue("stiff", Setting(bVal));

        // MULTIPLE STEPS
        bVal = Config::getBool(Config::SIMULATEOPTIONS_MULTI_STEP);
        Integrator::setValue("multiple_steps", Setting(bVal));
    }

    void CVODEIntegrator::loadSBMLSettings(std::string const &filename) {
        if (filename.empty()) {
            rrLog(Logger::LOG_ERROR) << "Empty file name for settings file";
        } else {
            std::map<std::string, std::string> options;
            std::map<std::string, std::string>::iterator it;
            //Read each line in the settings file
            std::vector<std::string> lines = getLinesInFile(filename);
            for (int i = 0; i < lines.size(); i++) {
                std::vector<std::string> line = splitString(lines[i], ":");
                if (line.size() == 2) {
                    options.insert(std::pair<std::string, std::string>(line[0], line[1]));
                } else {
                    rrLog(lDebug2) << "Empty line in settings file: " << lines[i];
                }
            }

            rrLog(lDebug3) << "Settings File =============";
            for (it = options.begin(); it != options.end(); it++) {
                rrLog(lDebug) << (*it).first << " => " << (*it).second;
            }
            rrLog(lDebug) << "===========================";

            //Assign values
            it = options.find("absolute");
            if (it != options.end()) {
                if ((*it).second.find("[") == std::string::npos) {
                    // scalar absolute tolerance
                    CVODEIntegrator::setValue("absolute_tolerance", Setting(std::abs(toDouble((*it).second))));
                } else {
                    // std::vector absolute tolerance

                    std::vector<double> v = toDoubleVector((*it).second);
                    // take absolute value of each element
                    for (unsigned int i = 0; i < v.size(); i++)
                        v[i] = std::abs(v[i]);
                    CVODEIntegrator::setValue("absolute_tolerance", Setting(v));
                }
            }

            it = options.find("relative");
            if (it != options.end()) {
                CVODEIntegrator::setValue("relative_tolerance", Setting(std::abs(toDouble((*it).second))));
            }
        }
    }

    std::string CVODEIntegrator::getName() const {
        return "cvode";
    }

    std::string CVODEIntegrator::getDescription() const {
        return "CVODE is a deterministic ODE solver from the SUNDIALS suite "
               "of timecourse integrators. It implements an Adams-Moulton solver "
               "for non-stiff problems and a backward differentiation formula "
               "(BDF) solver for stiff problems.";
    }

    std::string CVODEIntegrator::getHint() const {
        return "Deterministic ODE solver";
    }

    Solver *CVODEIntegrator::construct(ExecutableModel *executableModel) const {
        return new CVODEIntegrator(executableModel);
    }


    Integrator::IntegrationMethod CVODEIntegrator::getIntegrationMethod() const {
        return Integrator::Deterministic;
    }

    void CVODEIntegrator::setIndividualTolerance(std::string sid, double value) {

        // the tolerance std::vector that will be stored
        // Match ExecutableModel::getStateVector(): rate rules first, followed by
        // independent floating species.
        std::vector<double> v = getAbsoluteToleranceVector();

        int speciesIndex = mModel->getFloatingSpeciesIndex(sid);
        std::ptrdiff_t index;
        if (speciesIndex > -1 && speciesIndex < mModel->getNumIndFloatingSpecies()) {
            // sid is an independent floating species
            v[mModel->getNumRateRules() + speciesIndex] = value;
        } else {
            // sid might has a rate rule
            std::vector<std::string> symbols = mModel->getRateRuleSymbols();
            std::vector<std::string>::iterator it = std::find(symbols.begin(), symbols.end(), sid);
            if (it != symbols.end()) {
                // found it
                index = std::distance(symbols.begin(), it);
                v[index] = value;
            } else {
                throw std::invalid_argument("CVODEIntegrator::setIndividualTolerance failed, given sid " + sid +
                                            " is neither a floating species nor has a rate rule.");
            }
        }

        // update the tolerance
        CVODEIntegrator::setValue("absolute_tolerance", Setting(v));
    }



    void CVODEIntegrator::setMaxOrder(int newValue) {
        auto oldOrderValue = getValue("maximum_adams_order");
        if (getValue("stiff").getAs<bool>()) {
            oldOrderValue = getValue("maximum_bdf_order");
        }
        if (newValue <= oldOrderValue.getAs<int>()) {
            CVodeSetMaxOrd(mCVODE_Memory, newValue);
        }
    }

    void CVODEIntegrator::setValue(const std::string &key, Setting val) {
        if (key == "jacobian" || key == "linear_solver") {
            std::string v = val.getAs<std::string>();
            bool valid = key == "jacobian" ? (v == "difference_quotient" || v == "sparse_difference_quotient")
                                           : (v == "dense" || v == "klu");
            if (!valid) {
                throw std::invalid_argument("CVODEIntegrator: invalid value '" + v + "' of setting '" + key + "'");
            }
            Setting previous = getValue(key);
            Integrator::setValue(key, val);
            if (mModel) {
                // the linear solver and the Jacobian are part of the CVODE memory
                freeSundialsMemory();
                try {
                    create();
                } catch (...) {
                    freeSundialsMemory();
                    Integrator::setValue(key, previous);
                    create();
                    throw;
                }
            }
            return;
        }
        // if std::vector tolerance is set, the size of std::vector must be equal to
        // the number of floating species
        if (key == "absolute_tolerance" && val.type() == Setting::DOUBLEVECTOR)
            checkVectorSize(mModel->getNumIndFloatingSpecies() + mModel->getNumRateRules(),
                            val.get<std::vector<double> >().size());

        Integrator::setValue(key, val);

        /// Values and keys are stored in the settings std::map, which is updated
        /// in every call to @ref setValue. In addition, changing CVODE-specific
        /// parameters requires a call into the CVODE library to synchronize
        /// CVODE's internal memory with the settings std::map.
        if (mCVODE_Memory) {

            /**
            * The default value is ADAMS Q MAX = 12 for the Adams-Moulton method and BDF Q MAX
            * = 5 for the BDF method. Since maxord affects the memory requirements for the internal
            * cvodes memory block, its value cannot be increased past its previous value.
            */
            if (key == "maximum_bdf_order") {
//                CVodeSetMaxOrd(mCVODE_Memory, getValue("maximum_bdf_order"));
                setMaxOrder(getValue("maximum_bdf_order").getAs<int>());
            } else if (key == "maximum_adams_order") {
//                CVodeSetMaxOrd(mCVODE_Memory, getValue("maximum_adams_order"));
                setMaxOrder(getValue("maximum_adams_order").getAs<int>());
            } else if (key == "initial_time_step") {
                CVodeSetInitStep(mCVODE_Memory, (double) getValue("initial_time_step"));
            } else if (key == "minimum_time_step") {
                CVodeSetMinStep(mCVODE_Memory, (double) getValue("minimum_time_step"));
            } else if (key == "maximum_time_step") {
                CVodeSetMaxStep(mCVODE_Memory, (double) getValue("maximum_time_step"));
            } else if (key == "maximum_num_steps") {
                CVodeSetMaxNumSteps(mCVODE_Memory, (int) getValue("maximum_num_steps"));
            } else if (key == "absolute_tolerance" || key == "relative_tolerance") {
                setCVODETolerances();

            }
        }
        if (key == "collect_timing") {
            const bool timing = (bool) getValue("collect_timing");
            const bool changed = timing != collectTiming;
            collectTiming = timing;
            // the linear solver is wrapped for timing only while the timing is collected
            if (changed && mCVODE_Memory && (bool) getValue("stiff")) {
                freeSundialsMemory();
                create();
            }
        }
        if (key == "stiff") {
            // If the integrator is changed from stiff to standard, we must re-create CVode.
            rrLog(Logger::LOG_INFORMATION) << "Integrator stiffness has been changed. Re-creating CVode.";
            freeSundialsMemory();
            create();
        }
    }

    void CVODEIntegrator::updateCVODE() {
        if (mCVODE_Memory == nullptr) {
            return;
        }

        CVodeSetInitStep(mCVODE_Memory, (double) getValue("initial_time_step"));
        CVodeSetMinStep(mCVODE_Memory, (double) getValue("minimum_time_step"));
        CVodeSetMaxStep(mCVODE_Memory, (double) getValue("maximum_time_step"));
        CVodeSetMaxNumSteps(mCVODE_Memory, (int) getValue("maximum_num_steps") > 0 ? (int) getValue("maximum_num_steps")
                                                                                   : mDefaultMaxNumSteps);
        if ((bool) getValue("stiff"))
            CVodeSetMaxOrd(mCVODE_Memory, (int) getValue("maximum_bdf_order"));
        else
            CVodeSetMaxOrd(mCVODE_Memory, (int) getValue("maximum_adams_order"));
        setCVODETolerances();
    }

    void CVODEIntegrator::reInit(double t0) {
        // if we have no state std::vector variables and no events, we never
        // created an integrator.
        if (mCVODE_Memory == nullptr) {
            return;
        }

        // CVODE sets its counters to zero
        saveCVODECounters();
        if (integrating) {
            numReInits++;
        }

        int result = CVodeReInit(mCVODE_Memory, t0, mStateVector);

        if (result != CV_SUCCESS) {
            handleCVODEError(result);
        }

        setCVODETolerances();
    }

    double CVODEIntegrator::integrate(double timeStart, double hstep) {
        static const double epsilon = std::numeric_limits<double>::epsilon();

        rrLog(Logger::LOG_DEBUG) << "CVODEIntegrator::integrate("
                                 << timeStart << ", " << hstep << ")";

        ScopedTimer timer(collectTiming, times.total);

        // counts the reinitializations (after events) during the integration
        struct IntegratingFlag {
            bool &flag;

            explicit IntegratingFlag(bool &flag) : flag(flag) { flag = true; }

            ~IntegratingFlag() { flag = false; }
        } integratingFlag(integrating);

        if (variableStepPendingEvent || variableStepTimeEndEvent) {
            return applyVariableStepPendingEvents() + variableStepEventTolerance;
        }

        double timeEnd = timeStart-1; //Just to set it at a minimum; anything lower than timeStart should work.
        double tout = timeStart + hstep;
        int strikes = 3;

        // Set itask based on step size settings.
        int itask = CV_NORMAL;
        bool varstep = (bool) getValue("variable_step_size");
        double relTol = (double) getValue("relative_tolerance");

        if ((bool) getValue("multiple_steps") || (bool) getValue("variable_step_size")) {
            itask = CV_ONE_STEP;
        }

        // A variable step integration to an end time.  CVODE ignores that time in CV_ONE_STEP mode,
        // its steps go past it, and the caller interpolates the last row back to it.  The events
        // after that time are not part of this integration.
        const bool varstepToEnd = varstep && hstep > 0;

        // loop until machine epislon
        while (tout - timeEnd >= epsilon) {
            // here we bail in case we have no ODEs set up with CVODE ... though we should
            // still at least evaluate the model function
            if (!haveVariables() && mModel->getNumEvents() == 0) {
                mModel->getStateVectorRate(tout, nullptr, nullptr);
                return tout;
            }

            double nextTargetEndTime = tout;

            if (mModel->getPendingEventSize() > 0 &&
                mModel->getNextPendingEventTime(false) < nextTargetEndTime) {
                nextTargetEndTime = mModel->getNextPendingEventTime(true);
            }

            // CVODE ignores the output time in CV_ONE_STEP mode and steps past it, so a pending
            // event (an event with a delay) would be applied at the end of that step.  It is told
            // to stop at the time of the event instead.
            if (mCVODE_Memory) {
                double currentTime = 0;
                if (itask == CV_ONE_STEP && nextTargetEndTime < tout &&
                    CVodeGetCurrentTime(mCVODE_Memory, &currentTime) == CV_SUCCESS &&
                    nextTargetEndTime > currentTime) {
                    CVodeSetStopTime(mCVODE_Memory, nextTargetEndTime);
                } else {
                    CVodeClearStopTime(mCVODE_Memory);
                }
            }

            // event status before time step
            mModel->getEventTriggers(eventStatus.size(), nullptr, eventStatus.empty() ? nullptr : &eventStatus[0]);

            // time step
            int nResult = CVode(mCVODE_Memory, nextTargetEndTime, mStateVector, &timeEnd, itask);
            linearSolverCountersCurrent = true;

            if (nResult == CV_ROOT_RETURN) {
                rrLog(Logger::LOG_DEBUG) << "Event detected at time " << timeEnd;

                if (varstepToEnd && timeEnd > tout) {
                    // The root of an event is after the requested end.  The event is not applied or left
                    // pending, and the caller interpolates the state back to the end.  A simulation
                    // that continues from the end finds the event again.
                    assignResultsToModel();
                    mModel->setTime(timeEnd);
                    return timeEnd;
                }

                bool tooCloseToStart = fabs(timeEnd - mLastEventTime) > relTol;

                if (tooCloseToStart) {
                    strikes = 3;
                } else {
                    strikes--;
                }

                // the condition that we are to evaluate and apply events
                if (tooCloseToStart || strikes > 0) {
                    mLastEventTime = timeEnd;

                    if (varstep
                        && (timeEnd - timeStart > 2. * epsilon)) {
                        variableStepPendingEvent = true;
                        // the row before the event must stay after the previous row
                        variableStepEventTolerance = std::min(getRootTolerance(timeEnd), 0.5 * (timeEnd - timeStart));
                        assignResultsToModel();
                        mModel->setTime(timeEnd - epsilon);
                        if (listener) {
                            listener->onTimeStep(this, mModel, timeEnd);
                        }
                        return timeEnd - variableStepEventTolerance;
                    }

                    // apply events, copy post event status into integrator state std::vector.
                    applyEvents(timeEnd, eventStatus);

                    if (listener) {
                        listener->onEvent(this, mModel, timeEnd);
                    }
                }
            } else if (nResult == CV_SUCCESS || nResult == CV_TSTOP_RETURN) {
                // copy integrator state std::vector into model
                assignResultsToModel();


                // need to check if an event occured at the exact time step,
                // if so, add an extra point if we're doing variable step
                if (varstepToEnd && timeEnd > tout) {
                    // The step ended after the requested end, so an event at the end of the step is
                    // after the end too and is not applied.  The caller interpolates the state back to
                    // the requested end.
                    mModel->setTime(timeEnd);
                } else if (varstep
                    && (timeEnd - timeStart > 2. * epsilon)) {
                    // event status before time step

                    mModel->getEventTriggers(eventStatus.size(), 0, eventStatus.empty() ? nullptr : &eventStatus[0]);
                    // apply events and write state to variableStepPostEventState
                    // model state is updated by events.
                    int handled = mModel->applyEvents(timeEnd, eventStatus.empty() ? nullptr : &eventStatus[0], NULL, variableStepPostEventState.empty() ? nullptr : &variableStepPostEventState[0]);
                    workCounters.numEvents += handled;
                    if (handled > 0) {
                        // write original state back to model
                        mModel->setTime(timeEnd - epsilon);
                        assignResultsToModel();
                        variableStepTimeEndEvent = true;
                        variableStepEventTolerance = getRootTolerance(timeEnd);
                        mLastEventTime = timeEnd;
                        return timeEnd - epsilon;
                    }
                } else {


                    mModel->setTime(timeEnd);

                    // only needs to be called after a reg time step completes, the applyEvents
                    // called when a event root is found clears out all pending events and applies
                    // them.
                    applyPendingEvents(timeEnd);
                }

                if (listener) {
                    listener->onTimeStep(this, mModel, timeEnd);
                }
            } else {
                handleCVODEError(nResult);
            }

            try {
                mModel->testConstraints();
            }
            catch (const std::exception &e) {
                rrLog(Logger::LOG_WARNING) << "Constraint Violated at time = " << timeEnd << ": " << e.what();
            }

            if (varstep && (timeEnd - timeStart > 2. * epsilon)) {
                return timeEnd;
            }


            if (tout - timeEnd > epsilon) {
                timeStart = timeEnd;
            }
            rrLog(Logger::LOG_TRACE) << "time step, tout: " << tout << ", timeEnd: " << timeEnd;
        }
        return timeEnd;
    }


    void CVODEIntegrator::tweakTolerances() {
        double minAbs = Config::getDouble(Config::CVODE_MIN_ABSOLUTE);
        double minRel = Config::getDouble(Config::CVODE_MIN_RELATIVE);

        switch (getType("absolute_tolerance")) {

            // all cases below could be convert to a double type
            case Setting::INT32:
            case Setting::INT64:
            case Setting::UINT32:
            case Setting::UINT64:
            case Setting::FLOAT:
            case Setting::DOUBLE:
                // scalar tolerance
                CVODEIntegrator::setValue("absolute_tolerance",
                                          Setting((std::min)((double) CVODEIntegrator::getValue("absolute_tolerance"),
                                                             minAbs)));
                break;


            case Setting::DOUBLEVECTOR: {
                // std::vector tolerance
                std::vector<double> v = CVODEIntegrator::getValue("absolute_tolerance");
                for (int i = 0; i < v.size(); i++)
                    v[i] = (std::min)(v[i], minAbs);
                CVODEIntegrator::setValue("absolute_tolerance", Setting(v));

                break;
            }

            default:
                break;
        }

        CVODEIntegrator::setValue("relative_tolerance",
                                  Setting((std::min)((double) CVODEIntegrator::getValue("relative_tolerance"),
                                                     minRel)));

        rrLog(Logger::LOG_INFORMATION) << "tweaking CVODE tolerances to abs="
                                       << (double) CVODEIntegrator::getValue("absolute_tolerance") << ", rel="
                                       << (double) CVODEIntegrator::getValue("relative_tolerance");
    }


    bool CVODEIntegrator::haveVariables() const {
        return stateVectorVariables;
    }

    void CVODEIntegrator::create() {
        if (!mModel) {
            return;
        }

        assert(mStateVector == nullptr && mCVODE_Memory == nullptr &&
               "calling cvodeCreate, but cvode objects already exist");

        // still need cvode state std::vector size if we have no vars, but have
        // events, needed so root finder works.
        int allocStateVectorSize = 0;
        // when argument is null, returns size of state std::vector (see rrExecutableModel::getStateVector)
        int realStateVectorSize = mModel->getStateVector(nullptr);

        if (mSunContext) {
          SUNContext_Free(&mSunContext);
        }
        SUNContext_Create(SUN_COMM_NULL, &mSunContext);
        routeSundialsLoggingToRoadRunnerLogger(mSunContext);

        // cvode return code
        int err;

        if (realStateVectorSize > 0) {
            stateVectorVariables = true;
            allocStateVectorSize = realStateVectorSize;
        } else if (mModel->getNumEvents() > 0) {
            allocStateVectorSize = 1;
            stateVectorVariables = false;
        } else {
            stateVectorVariables = false;
            return;
        }

        // allocate and init the cvode arrays
        mStateVector = N_VNew_Serial(allocStateVectorSize, mSunContext);
        variableStepPostEventState.resize(allocStateVectorSize);

        auto states = new double[allocStateVectorSize];
        mModel->getStateVector(states);

        for (int i = 0; i < allocStateVectorSize; i++) {
            mStateVector->ops->nvgetarraypointer(mStateVector)[i] = states[i];
        }
        delete[] states;

        if ((bool) getValue("stiff")) {
            rrLog(Logger::LOG_INFORMATION) << "using stiff integrator";
            mCVODE_Memory = (void *) CVodeCreate(CV_BDF, mSunContext);
        } else {
            rrLog(Logger::LOG_INFORMATION) << "using non-stiff integrator";
            mCVODE_Memory = (void *) CVodeCreate(CV_ADAMS, mSunContext);
        }

        assert(mCVODE_Memory && "could not create Cvode, CVodeCreate failed");

        if ((err = SUNContext_PushErrHandler(mSunContext, cvodeErrHandler, this)) != SUN_SUCCESS) {
            handleCVODEError(err);
        }

        // use non default CVODE value here, default is too short
        // for some sbml tests.
        CVodeSetMaxNumSteps(mCVODE_Memory, mDefaultMaxNumSteps);

        double t0 = 0.0;

        if ((err = CVodeSetUserData(mCVODE_Memory, (void *) this)) != CV_SUCCESS) {
            handleCVODEError(err);
        }

        if ((err = CVodeInit(mCVODE_Memory, cvodeDyDtFcn, t0, mStateVector)) != CV_SUCCESS) {
            handleCVODEError(err);
        }

        if (mModel->getNumEvents() + mModel->getNumPiecewiseTriggers() > 0) {
            if ((err = CVodeRootInit(mCVODE_Memory, mModel->getNumEvents() + mModel->getNumPiecewiseTriggers(),
                                     cvodeEventAndPiecewiseRootFcn)) != CV_SUCCESS) {
                handleCVODEError(err);
            }
            rrLog(Logger::LOG_TRACE) << "CVRootInit executed for events.....";
        }

        /**
         * note the cvDirectDemo.c example from sundials is useful
         * to see the different cvode options in action
         */

        /* create fixed point nonlinear solver object */
        if ((bool) getValue("stiff")) {
            // as per the cvode docs (look closely at docs for CVodeCreate)
            // we use the default Newton iteration for stiff

            nonLinSolver = SUNNonlinSol_Newton(mStateVector, mSunContext);

            if (nonLinSolver == nullptr) {
                throw std::runtime_error("CVODEIntegrator::createCVODE: nonLinearSolver_ is nullptr\n");
            }

            if ((err = CVodeSetNonlinearSolver(mCVODE_Memory, nonLinSolver)) != CV_SUCCESS) {
                handleCVODEError(err);
            }

            // Never throw for a setting the model cannot serve: create() also runs when a model
            // is loaded, regenerated or copied. CVODE falls back to the dense difference quotient.
            std::string jacobianSetting = getValue("jacobian").getAs<std::string>();
            std::string linearSolverSetting = getValue("linear_solver").getAs<std::string>();
            mSparseDQJacobian = false;
            if (jacobianSetting == "sparse_difference_quotient" && realStateVectorSize > 0) {
                if (mModel->getStateVectorJacobianPattern()) {
                    mSparseDQJacobian = true;
                } else {
                    rrLogWarn << "no sparsity pattern of the Jacobian, CVODE uses the dense difference quotient: "
                              << mModel->getStateVectorJacobianUnavailableReason();
                }
            }
            mSparseLinearSolver = false;
            if (linearSolverSetting == "klu") {
#ifdef RR_USE_KLU
                if (mSparseDQJacobian) {
                    mSparseLinearSolver = true;
                } else {
                    rrLogWarn << "linear_solver = \"klu\" needs jacobian = \"sparse_difference_quotient\" and a "
                                 "model with a sparsity pattern of the Jacobian; CVODE uses the dense solver";
                }
#else
                rrLogWarn << "linear_solver = \"klu\" needs a roadrunner built with RR_USE_KLU; CVODE uses the "
                             "dense solver";
#endif
            }
            mJacobianValues.clear();
            mJacobianIncrements.clear();
            if (mSparseDQJacobian) {
                mJacobianValues.assign(mModel->getStateVectorJacobianPattern()->nnz(), 0.0);
                mJacobianIncrements.assign(realStateVectorSize, 0.0);
            }

            // the newton method requires use of a linear solver, which we set up here.
            if (mSparseLinearSolver) {
#ifdef RR_USE_KLU
                const CscPattern &p = *mModel->getStateVectorJacobianPattern();
                jac = SUNSparseMatrix(allocStateVectorSize, allocStateVectorSize, p.nnz(), CSC_MAT, mSunContext);
                std::copy(p.colPtr.begin(), p.colPtr.end(), SUNSparseMatrix_IndexPointers(jac));
                std::copy(p.rowIdx.begin(), p.rowIdx.end(), SUNSparseMatrix_IndexValues(jac));
                linSolver = SUNLinSol_KLU(mStateVector, jac, mSunContext);
#endif
            } else {
                jac = SUNDenseMatrix(allocStateVectorSize, allocStateVectorSize, mSunContext);
                linSolver = SUNLinSol_Dense(mStateVector, jac, mSunContext);
            }
            if (linSolver == nullptr) {
                throw std::runtime_error("CVODEIntegrator::createCVODE: call to SunLinSol_Dense returned nullptr. "
                                         "The size of the sundials matrix (created with SUNDenseMatrix) used for the jacobian "
                                         "(" + std::to_string(allocStateVectorSize) + "x" +
                                         std::to_string(allocStateVectorSize)
                                         + ") is inappropriate for this model\n");
            }
            if (collectTiming) {
                // times the setups and solves for the statistics
                linSolver = TimedLinearSolver::create(linSolver, this);
                if (linSolver == nullptr) {
                    throw std::runtime_error("CVODEIntegrator::createCVODE: could not create the linear solver");
                }
            }
            if ((err = CVodeSetLinearSolver(mCVODE_Memory, linSolver, jac)) != CV_SUCCESS) {
                handleCVODEError(err);
            };

            // the sparse difference quotient, or CVODE's dense difference quotient (nullptr)
            if ((err = CVodeSetJacFn(mCVODE_Memory, mSparseDQJacobian ? cvodeJacFcn : nullptr)) != CV_SUCCESS) {
                handleCVODEError(err);
            }
        } else {
            // and fixed point solver (which used to be called functional iteration)
            // for nonstiff problems
            nonLinSolver = SUNNonlinSol_FixedPoint(mStateVector, 0, mSunContext);
            if ((err = CVodeSetNonlinearSolver(mCVODE_Memory, nonLinSolver)) != CV_SUCCESS) {
                handleCVODEError(err);
            }
        }

        setCVODETolerances();
        mModel->resetEvents();
        linearSolverCountersCurrent = false;
    }

    void CVODEIntegrator::testRootsAtInitialTime() {
        std::vector<unsigned char> initialEventStatus(mModel->getEventTriggers(0, nullptr, nullptr), false);
        mModel->getEventTriggers(initialEventStatus.size(), nullptr,
                                 initialEventStatus.empty() ? nullptr : &initialEventStatus[0]);
        applyEvents(mIntegrationStartTime, initialEventStatus);
    }

    void CVODEIntegrator::applyEvents(double timeEnd, std::vector<unsigned char> &previousEventStatus) {
        double *stateVector = mStateVector ? NV_DATA_S(mStateVector) : nullptr;
        workCounters.numEvents += mModel->applyEvents(timeEnd, previousEventStatus.empty() ? nullptr : &previousEventStatus[0], stateVector, stateVector);

        if (timeEnd > mIntegrationStartTime) {
            mModel->setTime(timeEnd);

            // copy state std::vector into cvode memory
            if (mStateVector) {
                mModel->getStateVector(NV_DATA_S(mStateVector));
            }

            // set tolerances and so forth.
            reInit(timeEnd);
        }
    }


    void CVODEIntegrator::applyPendingEvents(double timeEnd) {
        if (mModel) {
            // get current event triggered state
            mModel->getEventTriggers(eventStatus.size(), 0, eventStatus.size() == 0 ? NULL : &eventStatus[0]);
            int handled = mModel->applyEvents(timeEnd, eventStatus.size() == 0 ? NULL : &eventStatus[0], NULL, NULL);
            workCounters.numEvents += handled;
            if (handled > 0) {
                rrLog(Logger::LOG_DEBUG) << __FUNC__;
                restart(timeEnd);
            }
        }
    }

    bool CVODEIntegrator::interpolateToTime(double t) {
        if (!mStateVector || !mCVODE_Memory || !mModel) {
            return false;
        }

        // CVodeGetDky is valid for t within the last internal step, [tn - hu, tn]
        double tn = 0, hu = 0;
        if (CVodeGetCurrentTime(mCVODE_Memory, &tn) != CV_SUCCESS ||
            CVodeGetLastStep(mCVODE_Memory, &hu) != CV_SUCCESS) {
            return false;
        }
        const double fuzz = 100. * std::numeric_limits<double>::epsilon() * (fabs(tn) + fabs(hu));

        // CVODE was reinitialized at the time of the event just applied, so it has no previous step
        // to interpolate in.  When the end is within the root tolerance after that event, the state
        // after the event is used as the state at the end.
        if (fabs(tn - mLastEventTime) <= fuzz && t >= tn && t <= tn + variableStepEventTolerance + fuzz) {
            assignResultsToModel();
            mModel->setTime(t);
            return true;
        }

        if (t < tn - fabs(hu) - fuzz || t > tn + fuzz) {
            return false;
        }
        t = std::min(t, tn);

        if (CVodeGetDky(mCVODE_Memory, t, 0, mStateVector) != CV_SUCCESS) {
            return false;
        }
        assignResultsToModel();
        mModel->setTime(t);
        return true;
    }

    void CVODEIntegrator::assignResultsToModel() const {
        if (mStateVector) {
            mModel->setStateVector(mStateVector->ops->nvgetarraypointer(mStateVector));
        }
    }

    vector<double> CVODEIntegrator::getAbsoluteToleranceVector()
    { 
        int species = mModel->getNumIndFloatingSpecies();
        int rrs = mModel->getNumRateRules();
        vector<double> amount_tolerances;

        switch (getType("absolute_tolerance")) {

            // all cases below could be convert to a double type
            case Setting::INT32:
            case Setting::INT64:
            case Setting::UINT32:
            case Setting::UINT64:
            case Setting::FLOAT:
            case Setting::DOUBLE:
            {
                // scalar tolerance
                double abs = (double)getValue("absolute_tolerance");
                for (int n = 0; n < species + rrs; n++) {
                    amount_tolerances.push_back(abs);
                }
                break;
            }
            case Setting::DOUBLEVECTOR: {
                // std::vector tolerance
                amount_tolerances = getValue("absolute_tolerance").get<std::vector<double>>();
                if (amount_tolerances.size() == species + rrs) {
                    return amount_tolerances;
                }
                //Otherwise, something might have happened, and the amount tolerances vector is the wrong size.  This can happen when the model is being resized, among other situations.
                amount_tolerances.clear();
                for (int n = 0; n < species + rrs; n++) {
                    amount_tolerances.push_back(Config::getDouble(Config::CVODE_MIN_ABSOLUTE));
                }
                break;
            }

            default:
                handleCVODEError(CV_ILL_INPUT);
        }
        // Scale by the model's initial values rather than its current state, so the
        // tolerances do not depend on how the simulation is divided into calls.
        // Magnitudes below zeroThreshold are treated as zero.
        const double zeroThreshold = 1e-30;

        uint ncomp = mModel->getNumCompartments();
        std::vector<double> initVolumes(ncomp, 0.0);
        if (ncomp > 0) {
            mModel->getCompartmentInitVolumes(ncomp, 0, initVolumes.data());
        }

        std::vector<double> initSpecies(species, 0.0);
        if (species > 0) {
            mModel->getFloatingSpeciesInitAmounts(species, 0, initSpecies.data());
        }

        for (int s = 0; s < species; s++) {
            double scale = std::abs(initSpecies[s]);
            if (scale < zeroThreshold) {
                scale = 1.0;
                int comp = mModel->getCompartmentIndexForFloatingSpecies(s);
                if (comp >= 0 && comp < static_cast<int>(ncomp) && initVolumes[comp] != 0.0) {
                    scale = std::abs(initVolumes[comp]);
                }
            }
            amount_tolerances[rrs + s] = amount_tolerances[rrs + s] * scale;
        }

        // init(id) resolves the initial value of any variable set by a rate rule.
        std::vector<double> currentRRs(rrs, 0.0);
        if (rrs > 0) {
            mModel->getRateRuleValues(currentRRs.data());
        }

        std::vector<std::string> symbols = mModel->getRateRuleSymbols();
        for (int rr = 0; rr < rrs; rr++) {
            // the current value is used only if init(id) does not resolve
            double initValue = currentRRs[rr];
            try {
                initValue = mModel->getValue("init(" + symbols[rr] + ")");
            }
            catch (const std::exception &) {
                // keep the current value
            }

            double scale = std::abs(initValue);
            if (scale < zeroThreshold) {
                scale = 1.0;
                int speciesIndex = mModel->getFloatingSpeciesIndex(symbols[rr]);
                if (speciesIndex > -1) {
                    // the symbol defined by the rate rule is a species
                    int comp = mModel->getCompartmentIndexForFloatingSpecies(speciesIndex);
                    if (comp >= 0 && comp < static_cast<int>(ncomp) && initVolumes[comp] != 0.0) {
                        scale = std::abs(initVolumes[comp]);
                    }
                }
            }
            amount_tolerances[rr] = amount_tolerances[rr] * scale;
        }
        return amount_tolerances;
    }

    void CVODEIntegrator::setCVODETolerances() {
        if (mStateVector == nullptr || mModel == nullptr) {
            return;
        }

        vector<double> amount_tolerances = getAbsoluteToleranceVector();
        if (amount_tolerances.size() == 0) {
            amount_tolerances.push_back(1.0);
        }

        // If we have a model with only events, cvode still needs a state std::vector of length 1 to integrate.
        if (!haveVariables() && mModel->getNumEvents() > 0) {
            SetVector(mStateVector, 0, 1.0);
        }



        int err;
        N_Vector nv = N_VMake_Serial(static_cast<long>(amount_tolerances.size()), amount_tolerances.data(), mSunContext);
        err = CVodeSVtolerances(mCVODE_Memory, (double)getValue("relative_tolerance"), nv);
        rrLog(Logger::LOG_INFORMATION) << "Tolerances used: abs=[" << std::setprecision(16);
        for (int i = 0; i < amount_tolerances.size(); i++) {
            if (i != 0) {
                rrLog(Logger::LOG_INFORMATION) << ", ";
            }
            rrLog(Logger::LOG_INFORMATION) << amount_tolerances[i];
        }
        rrLog(Logger::LOG_INFORMATION) << "]; rel=" << (double)getValue("relative_tolerance") << std::endl;
        N_VDestroy_Serial(nv);

        if (err != CV_SUCCESS) {
            handleCVODEError(err);
        }

    }

    void CVODEIntegrator::restart(double time) {
        if (!mModel) {
            return;
        }

        mLastEventTime = time;

        // an event found by a variable step integration, but not yet applied, belongs to the
        // integration that is being left
        variableStepPendingEvent = false;
        variableStepTimeEndEvent = false;
        variableStepEventTolerance = 0;

        // apply any events that trigger before or at t0.
        // important NOT to set model time before we check get
        // the initial event state, initially time is < 0.

        // copy state std::vector into cvode memory, need to do this before evaluating
        // roots because the applyEvents method copies the cvode state std::vector
        // into the model
        if (mStateVector) {
            mModel->getStateVector(NV_DATA_S(mStateVector));
        }

        testRootsAtInitialTime();

        mModel->setTime(time);

        // copy state std::vector into cvode memory
        if (mStateVector && mCVODE_Memory) {
            mModel->getStateVector(NV_DATA_S(mStateVector));
        }

        // set tolerances and so forth.
        if (mCVODE_Memory) {
            reInit(time);
        }
    }

    // Cvode calls this to compute the dy/dts. This routine in turn calls the
    // model function which is located in the host application.
    int cvodeDyDtFcn(sunrealtype time, N_Vector cv_y, N_Vector cv_ydot, void *userData) {
        double *y = NV_DATA_S(cv_y);
        double *ydot = NV_DATA_S(cv_ydot);
        CVODEIntegrator *cvInstance = (CVODEIntegrator *) userData;

        assert(cvInstance && "userData pointer is NULL in cvode dydt callback");

        ScopedTimer timer(cvInstance->collectTiming, cvInstance->times.rhsEvals);

        ExecutableModel *model = cvInstance->mModel;

        model->getStateVectorRate(time, y, ydot);

        if (!cvInstance->stateVectorVariables && cvInstance->mStateVector &&
            NV_LENGTH_S(cvInstance->mStateVector) == 1) {
            ydot[0] = 0.0;
        }

        rrLog(Logger::LOG_TRACE) << __FUNC__ << ", model: " << model;

        return CV_SUCCESS;
    }

    void CVODEIntegrator::freeSundialsMemory() {
        // cvode does not check for null values.
        if (mStateVector) {
            N_VDestroy_Serial(mStateVector);
            mStateVector = nullptr;
        }

        if (mCVODE_Memory) {
            saveCVODECounters();
            CVodeFree(&mCVODE_Memory);
            mCVODE_Memory = nullptr;
        }

        if (nonLinSolver) {
            SUNNonlinSolFree(nonLinSolver);
            nonLinSolver = nullptr;
        }

        if (linSolver) {
            SUNLinSolFree(linSolver);
            linSolver = nullptr;
        }

        if (jac) {
            SUNMatDestroy(jac);
            jac = nullptr;
        }

        mSparseDQJacobian = false;
        mSparseLinearSolver = false;

    }

    double CVODEIntegrator::getRootTolerance(double t) const {
        static const double epsilon = std::numeric_limits<double>::epsilon();

        double hu = 0;
        if (!mCVODE_Memory || CVodeGetLastStep(mCVODE_Memory, &hu) != CV_SUCCESS) {
            hu = 0;
        }
        // CVODE root tolerance, used for backing up when an event fires (see CVODE User Doc pp. 13).
        // It is computed for each event from the time and step of CVODE at that event.  (It used to be
        // a function-level static, set by the first integration in the process.)
        return 100. * (32. * epsilon) * (fabs(t) + fabs(hu));
    }

    double CVODEIntegrator::applyVariableStepPendingEvents() {
        if (variableStepTimeEndEvent) {
            // post event state allready calcuated.
            mModel->setStateVector(variableStepPostEventState.size() == 0 ? NULL : &variableStepPostEventState[0]);
            // copy state std::vector into cvode memory
            if (mStateVector) {
                mModel->getStateVector(NV_DATA_S(mStateVector));
            }
            mModel->setTime(mLastEventTime);
            reInit(mLastEventTime);
        } else {
            // apply events, copy post event status into integrator state std::vector.
            applyEvents(mLastEventTime, eventStatus);
        }

        if (listener) {
            listener->onEvent(this, mModel, mLastEventTime);
        }

        variableStepPendingEvent = false;
        variableStepTimeEndEvent = false;
        return mLastEventTime;
    }

    // int (*CVRootFn)(sunrealtype t, N_Vector y, sunrealtype *gout, void *user_data)
    // Cvode calls this to check for event changes
    int cvodeEventAndPiecewiseRootFcn(sunrealtype time, N_Vector y_vector, sunrealtype* gout, void* user_data) {
        CVODEIntegrator* cvInstance = (CVODEIntegrator*)user_data;

        assert(cvInstance && "user data pointer is NULL on CVODE root callback");

        ScopedTimer timer(cvInstance->collectTiming, cvInstance->times.gEvals);

        ExecutableModel* model = cvInstance->mModel;

        double* y = NV_DATA_S(y_vector);

        if (model->getNumEvents() > 0) {
            model->getEventRoots(time, y, gout);
        }

        if (model->getNumPiecewiseTriggers() > 0) {
            model->getPiecewiseTriggerRoots(time, y, gout + model->getNumEvents());
        }

        return CV_SUCCESS;
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
    void cvodeErrHandler(int line, const char* function, const char* file,
                         const char* msg, SUNErrCode error_code,
                         void* err_user_data, SUNContext sunctx) {
        CVODEIntegrator *i = (CVODEIntegrator *)err_user_data;
        i->checkType();

        if (error_code < 0) {
            rrLog(Logger::LOG_ERROR) << "CVODE Error: " << decodeSundialsError(i, error_code, false)
                                     << ", Function: " << function
                                     << ", Message: " << msg;
        }
        // CVODE's warnings do not come through here: SUNDIALS sends them to the SUNLogger of the
        // SUNContext, which routeSundialsLoggingToRoadRunnerLogger connects to the roadrunner Logger.
    }

    CVODEIntegrator::CVODECounters &CVODEIntegrator::CVODECounters::operator+=(const CVODECounters &other) {
        numSteps += other.numSteps;
        numRhsEvals += other.numRhsEvals;
        numLinSolvSetups += other.numLinSolvSetups;
        numErrTestFails += other.numErrTestFails;
        numNonlinSolvIters += other.numNonlinSolvIters;
        numNonlinSolvConvFails += other.numNonlinSolvConvFails;
        numStepSolveFails += other.numStepSolveFails;
        numGEvals += other.numGEvals;
        numStabLimOrderReds += other.numStabLimOrderReds;
        numJacEvals += other.numJacEvals;
        numLinRhsEvals += other.numLinRhsEvals;
        numLinIters += other.numLinIters;
        numLinConvFails += other.numLinConvFails;
        return *this;
    }

    CVODEIntegrator::CVODECounters &CVODEIntegrator::CVODECounters::operator-=(const CVODECounters &other) {
        numSteps -= other.numSteps;
        numRhsEvals -= other.numRhsEvals;
        numLinSolvSetups -= other.numLinSolvSetups;
        numErrTestFails -= other.numErrTestFails;
        numNonlinSolvIters -= other.numNonlinSolvIters;
        numNonlinSolvConvFails -= other.numNonlinSolvConvFails;
        numStepSolveFails -= other.numStepSolveFails;
        numGEvals -= other.numGEvals;
        numStabLimOrderReds -= other.numStabLimOrderReds;
        numJacEvals -= other.numJacEvals;
        numLinRhsEvals -= other.numLinRhsEvals;
        numLinIters -= other.numLinIters;
        numLinConvFails -= other.numLinConvFails;
        return *this;
    }

    CVODEIntegrator::CVODECounters CVODEIntegrator::readCVODECounters() const {
        CVODECounters c;
        if (!mCVODE_Memory) {
            return c;
        }
        // A counter that CVODE does not have (e.g. of the linear solver of the
        // non-stiff solver, which has none) stays 0.
        CVodeGetNumSteps(mCVODE_Memory, &c.numSteps);
        CVodeGetNumRhsEvals(mCVODE_Memory, &c.numRhsEvals);
        CVodeGetNumLinSolvSetups(mCVODE_Memory, &c.numLinSolvSetups);
        CVodeGetNumErrTestFails(mCVODE_Memory, &c.numErrTestFails);
        CVodeGetNumNonlinSolvIters(mCVODE_Memory, &c.numNonlinSolvIters);
        CVodeGetNumNonlinSolvConvFails(mCVODE_Memory, &c.numNonlinSolvConvFails);
        CVodeGetNumStepSolveFails(mCVODE_Memory, &c.numStepSolveFails);
        CVodeGetNumGEvals(mCVODE_Memory, &c.numGEvals);
        CVodeGetNumStabLimOrderReds(mCVODE_Memory, &c.numStabLimOrderReds);
        if (linearSolverCountersCurrent && linSolver) {
            CVodeGetNumJacEvals(mCVODE_Memory, &c.numJacEvals);
            CVodeGetNumLinRhsEvals(mCVODE_Memory, &c.numLinRhsEvals);
            CVodeGetNumLinIters(mCVODE_Memory, &c.numLinIters);
            CVodeGetNumLinConvFails(mCVODE_Memory, &c.numLinConvFails);
        }
        return c;
    }

    void CVODEIntegrator::saveCVODECounters() {
        savedCounters += readCVODECounters();
        linearSolverCountersCurrent = false;
    }

    void CVODEIntegrator::resetStatistics() {
        Integrator::resetStatistics();
        savedCounters = CVODECounters();
        savedCounters -= readCVODECounters();
        numReInits = 0;
        numJacobianModelEvals = 0;
        times = CVODETimes();
    }

    std::unordered_map<std::string, Setting> CVODEIntegrator::getStatistics() {
        CVODECounters c = savedCounters;
        c += readCVODECounters();

        std::unordered_map<std::string, Setting> stats = {
            {"num_steps", Setting(std::int64_t(c.numSteps))},
            {"num_rhs_evals", Setting(std::int64_t(c.numRhsEvals))},
            {"num_lin_solv_setups", Setting(std::int64_t(c.numLinSolvSetups))},
            {"num_err_test_fails", Setting(std::int64_t(c.numErrTestFails))},
            {"num_nonlin_solv_iters", Setting(std::int64_t(c.numNonlinSolvIters))},
            {"num_nonlin_solv_conv_fails", Setting(std::int64_t(c.numNonlinSolvConvFails))},
            {"num_step_solve_fails", Setting(std::int64_t(c.numStepSolveFails))},
            {"num_jac_evals", Setting(std::int64_t(c.numJacEvals))},
            {"num_lin_rhs_evals", Setting(std::int64_t(c.numLinRhsEvals))},
            {"num_lin_iters", Setting(std::int64_t(c.numLinIters))},
            {"num_lin_conv_fails", Setting(std::int64_t(c.numLinConvFails))},
            {"num_g_evals", Setting(std::int64_t(c.numGEvals))},
            {"num_stab_lim_order_reds", Setting(std::int64_t(c.numStabLimOrderReds))},
            {"num_reinits", Setting(numReInits)},
            {"num_jac_model_evals", Setting(numJacobianModelEvals)},
            {"num_jac_colors", Setting(mSparseDQJacobian ? mModel->getStateVectorJacobianNumColors() : 0)},
            {"sparse_linear_solver", Setting(mSparseLinearSolver)},
            {"num_events", Setting(workCounters.numEvents)},
        };

        if (mCVODE_Memory) {
            double value = 0;
            int order = 0;
            if (CVodeGetActualInitStep(mCVODE_Memory, &value) == CV_SUCCESS) {
                stats["actual_init_step"] = Setting(value);
            }
            if (CVodeGetLastStep(mCVODE_Memory, &value) == CV_SUCCESS) {
                stats["last_step"] = Setting(value);
            }
            if (CVodeGetCurrentStep(mCVODE_Memory, &value) == CV_SUCCESS) {
                stats["current_step"] = Setting(value);
            }
            if (CVodeGetCurrentTime(mCVODE_Memory, &value) == CV_SUCCESS) {
                stats["current_time"] = Setting(value);
            }
            if (CVodeGetLastOrder(mCVODE_Memory, &order) == CV_SUCCESS) {
                stats["last_order"] = Setting(order);
            }
            if (CVodeGetCurrentOrder(mCVODE_Memory, &order) == CV_SUCCESS) {
                stats["current_order"] = Setting(order);
            }
        }

        if (collectTiming) {
            stats["time_total"] = Setting(times.total);
            stats["time_rhs_evals"] = Setting(times.rhsEvals);
            stats["time_g_evals"] = Setting(times.gEvals);
            stats["time_lin_solv_setups"] = Setting(times.linSolvSetups);
            stats["time_lin_solv_solves"] = Setting(times.linSolvSolves);
            stats["time_jac_evals"] = Setting(times.jacEvals);
        }
        return stats;
    }

    /**
     * The sparse difference quotient Jacobian for CVODE (setting jacobian =
     * "sparse_difference_quotient"), written into the dense matrix or, with KLU, into the
     * sparse matrix whose structure is the model's pattern. The increments are those of
     * CVODE's dense difference quotient (cvLsDenseDQJac), so every entry is the forward
     * difference that CVODE would compute.
     */
    int cvodeJacFcn(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J, void *userData,
                    N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
        auto *cv = static_cast<CVODEIntegrator *>(userData);
        ScopedTimer timer(cv->collectTiming, cv->times.jacEvals);
        try {
            const CscPattern &p = *cv->mModel->getStateVectorJacobianPattern();
            const sunindextype n = NV_LENGTH_S(y);
            const double *yData = NV_DATA_S(y);

            // the increments of cvLsDenseDQJac
            N_Vector ewt = tmp1;
            if (CVodeGetErrWeights(cv->mCVODE_Memory, ewt) != CV_SUCCESS) {
                return -1;
            }
            double h = 0;
            CVodeGetCurrentStep(cv->mCVODE_Memory, &h);
            const double uround = std::numeric_limits<double>::epsilon();
            const double srur = std::sqrt(uround);
            const double fnorm = N_VWrmsNorm(fy, ewt);
            const double minInc = fnorm != 0.0 ? 1000.0 * std::fabs(h) * uround * (double) n * fnorm : 1.0;
            const double *ewtData = NV_DATA_S(ewt);
            for (sunindextype j = 0; j < n; ++j) {
                cv->mJacobianIncrements[j] = std::max(srur * std::fabs(yData[j]), minInc / ewtData[j]);
            }

            cv->mModel->getStateVectorJacobianDQ(t, yData, cv->mJacobianIncrements.data(),
                                                 cv->mJacobianValues.data());
            cv->numJacobianModelEvals += cv->mModel->getStateVectorJacobianNumColors() + 1;

            const double *values = cv->mJacobianValues.data();
            if (SUNMatGetID(J) == SUNMATRIX_DENSE) {
                SUNMatZero(J);
                for (int j = 0; j < p.nCols; ++j) {
                    double *col = SUNDenseMatrix_Column(J, j);
                    for (int k = p.colPtr[j]; k < p.colPtr[j + 1]; ++k) {
                        col[p.rowIdx[k]] = values[k];
                    }
                }
            } else {
                std::copy(p.colPtr.begin(), p.colPtr.end(), SUNSparseMatrix_IndexPointers(J));
                std::copy(p.rowIdx.begin(), p.rowIdx.end(), SUNSparseMatrix_IndexValues(J));
                std::copy(values, values + p.nnz(), SUNSparseMatrix_Data(J));
            }
        } catch (const std::exception &e) {
            rrLogErr << "sparse difference quotient Jacobian failed: " << e.what();
            return -1;
        }
        return 0;
    }

    void *CVODEIntegrator::getCvodeMemory() const {
        return mCVODE_Memory;
    }

    N_Vector CVODEIntegrator::getStateVector() const {
        return mStateVector;
    }

    SUNNonlinearSolver CVODEIntegrator::getSolver() const {
        return nonLinSolver;
    }


}

