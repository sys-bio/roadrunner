// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

// == FILEDOC =================================================

/** @file CVODEIntegrator.h
* @author WBC, ETS, MTK, JKM
* @copyright Apache License, Version 2.0
* @brief Contains the RoadRunner CVODE integrator interface
**/

//todo replace the include guards to match the filename
#ifndef rrCvodeInterfaceH
#define rrCvodeInterfaceH

#include "Integrator.h"
#include "rrRoadRunnerOptions.h"

#include <string>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <sundials/sundials_linearsolver.h>
#include <sundials/sundials_nonlinearsolver.h>
#include <cvodes/cvodes.h>
#include <sundials/sundials_nvector.h>
#include "ForwardSensitivitySolver.h"


namespace rr {
    using std::string;

    class ExecutableModel;

    class RoadRunner;

    class ForwardSensitivitySolver;

    /**
     * @author WBC, ETS, MTK
     * @brief A RoadRunner integrator based on CVODE; serves as RoadRunner's main integrator for ODEs
     * @details Provides a wrapper around CVODE from the SUNDIALS suite.
     * CVODE has two main solvers: an Adams-Moulton solver for non-stiff problems,
     * and a backward differentiation formula (BDF) solver for stiff problems.
     * See: https://computation.llnl.gov/casc/sundials/documentation/toms_sundials.pdf
     */
    class CVODEIntegrator : public Integrator {

    public:

        friend class ForwardSensitivitySolver;

        using Integrator::Integrator;

        /**
         * @author WBC, ETS, MTK
         * @brief Constructor: takes an executable model, does not own the pointer
         */
        explicit CVODEIntegrator(ExecutableModel *oModel);

        /**
         * @author WBC, ETS, MTK
         * @brief Destructor
         */
        ~CVODEIntegrator() override;

        /**
        * @author JKM
        * @brief Called whenever a new model is loaded to allow integrator
        * to reset internal state
        */
        void syncWithModel(ExecutableModel *m) override;

        // ** Loading Settings *************************************************

        /**
         * It looks like this method only get used
         * inside resetSettings.
         */
        void loadConfigSettings() override;

        /**
         * @author WBC
         * @brief Load an SBML settings file and apply the configuration options
         * @note Can assign relative and absolute tolerances
         */
        void loadSBMLSettings(const std::string &filename) override;

        // ** Meta Info ********************************************************

        /**
         * @author WBC
         * @brief Get the name for this integrator
         * @note Delegates to @ref getName
         */
        std::string getName() const override;

        /**
         * @author WBC
         * @brief Get the description for this integrator
         * @note Delegates to @ref getDescription
         */
        std::string getDescription() const override;

        /**
         * @author WBC
         * @brief Get the hint for this integrator
         * @note Delegates to @ref getHint
         */
        std::string getHint() const override;

        /**
         * @brief construct an instance of type CVODEIntegrator.
         * @details implements the Registrar interface. Used in
         * factory creation of Integrators.
         */
        Solver *construct(ExecutableModel *executableModel) const override;

        // ** Getters / Setters ************************************************

        /**
         * @author WBC, ETS, MTK
         * @brief Always deterministic for CVODE
         */
        IntegrationMethod getIntegrationMethod() const override;

        /**
         * @author WBC, ETS, MTK
         * @brief Sets the value of an integrator setting (e.g. absolute_tolerance)
         */
        void setValue(const std::string &setting, Setting value) override;

        /**
         * @brief sets the value of maximum order, which defaults to 12 for Adams (non-stiff)
         * and 5 for BDF (Stiff).
         * @details Once a CVODEIntegrator has been created, increasing the
         * order value beyond the original value is not possible (because the underlying
         * sundials objects require more memory). Attempts to do so will simply be ignored.
         */
        void setMaxOrder(int newValue);

        /**
         * @author FY
         * @brief Sets tolerance for individual species or rate rule-controlled element.
         * @details The absolute tolerance for the given sid is set to @ref value.  This
         * fixes the absolute tolerance vector moving forward, even if other initial values
         * are changed.
         */
        void setIndividualTolerance(std::string sid, double value) override;

        /**
        * @author JKM
        * @brief Reset all integrator settings to their respective default values
        */
        void resetSettings() override;


        /**
         * @author JKM, WBC, ETS, MTK
         * @brief Fix tolerances for SBML tests
         * @details In order to ensure that the results of the SBML test suite
         * remain valid, this method enforces a lower bound on tolerance values.
         * Sets minimum absolute and relative tolerances to
         * Config::CVODE_MIN_ABSOLUTE and Config::CVODE_MIN_RELATIVE resp.
         */
        void tweakTolerances() override;


        // ** Integration Routines *********************************************

        /**
         * @author WBC, ETS, MTK
         * @brief Main integration routine
         */
        double integrate(double t0, double hstep) override;

        /**
         * @author WBC, ETS, MTK
         * @brief Reset time to zero and reinitialize model
         * @details Applies events which occur before time zero.
         * Reinitializes CVODE and the executable model.
         */
        void restart(double timeStart) override;

        /**
         * @brief Sets the model to the state at t, from the interpolating polynomial
         * (Nordsieck history) of the last internal step of CVODE.
         * @details Its error is controlled by the tolerances like that of a step.
         * Returns false if t is not within the last step.
         */
        bool interpolateToTime(double t) override;

        // ** Listeners ********************************************************

        /**
         * @author WBC, ETS
         * @brief Gets the integrator listener
         */
        IntegratorListenerPtr getListener() override;

        /**
         * @author WBC, ETS
         * @brief Sets the integrator listener
         */
        void setListener(IntegratorListenerPtr) override;

        /**
         * @author JKM
         * @brief Does a RT type check which throws if it fails, EVEN IF RTTI IS DISABLED
         */
        void checkType() const;


        /**
        * @author FY
        * @brief Does a size check which throws if it fails
        */
        void checkVectorSize(int expected, size_t real) const;

        /**
        * @author FY
        * @brief Does a index check which throws if it is out of bound
        */
        void checkIndex(int index, int size) const;

        /**
        * @author FY
        * @brief Converts integer to std::string for error print
        */
        std::string ToString(int val) const;

        /**
        * @author LPS
        * @brief Converts size_t to std::string for error print
        */
        std::string ToString(size_t val) const;

        /**
         * @brief getter for the internal state std::vector
         */
        N_Vector getStateVector() const;

        /**
         * @brief getter for the internal Sundials linear solver object
         */
        SUNNonlinearSolver getSolver() const;

        /**
         * @brief getter for the internal CVode memory buffer
         */
        void *getCvodeMemory() const;

        /**
         * @brief Get the absolute tolerance vector for the solver.
         * 
         * The absolute tolerance vector is either set directly by the user as a vector, or is generated from the single absolute tolerance value (either default or set by the user) multiplied by the initial value of every element in the state vector (independent floating species, and elements controlled by rate rules).  If that initial value is zero, the corresponding element of the absolute tolerance vector is the single value multiplied by the compartment volume for species, or by one for all other values.
         */
        std::vector<double> getAbsoluteToleranceVector() override;

        /**
         * @brief The counters of CVODE since the last call of resetStatistics.
         * @details CVODE sets its counters to zero when it is reinitialized, which
         * this integrator does after an event and when a simulation starts. The
         * counters returned here are summed over these reinitializations. Besides
         * the keys of Integrator::getStatistics they are (with the CVODE function
         * that returns the counter of one reinitialization):
         *  - num_lin_solv_setups: the setups of the linear solver (CVodeGetNumLinSolvSetups);
         *  - num_nonlin_solv_iters: the iterations of the nonlinear solver (CVodeGetNumNonlinSolvIters);
         *  - num_nonlin_solv_conv_fails: the convergence failures of the nonlinear
         *    solver (CVodeGetNumNonlinSolvConvFails);
         *  - num_step_solve_fails: the steps that failed because the nonlinear solver
         *    did not converge (CVodeGetNumStepSolveFails);
         *  - num_jac_evals: the evaluations of the Jacobian (CVodeGetNumJacEvals);
         *  - num_lin_rhs_evals: the evaluations of the rates of the model for the
         *    finite difference Jacobian (CVodeGetNumLinRhsEvals), which are not
         *    part of num_rhs_evals;
         *  - num_lin_iters, num_lin_conv_fails: the iterations and convergence
         *    failures of an iterative linear solver (CVodeGetNumLinIters,
         *    CVodeGetNumLinConvFails);
         *  - num_g_evals: the evaluations of the root functions, i.e. of the event
         *    triggers and piecewise conditions (CVodeGetNumGEvals);
         *  - num_stab_lim_order_reds: the order reductions of the BDF stability limit
         *    detection (CVodeGetNumStabLimOrderReds);
         *  - num_reinits: the reinitializations of CVODE during the integration, i.e.
         *    after events.
         * The non-stiff solver has no linear solver, so its num_lin_solv_setups,
         * num_jac_evals, num_lin_rhs_evals, num_lin_iters and num_lin_conv_fails are 0.
         *
         * The values of the last step since the last reinitialization, if CVODE exists:
         * actual_init_step, last_step, current_step (the step size to try next),
         * last_order, current_order and current_time (CVodeGetActualInitStep,
         * CVodeGetLastStep, CVodeGetCurrentStep, CVodeGetLastOrder,
         * CVodeGetCurrentOrder, CVodeGetCurrentTime).
         *
         * With the setting collect_timing, the times in seconds of
         *  - time_total: the integration, i.e. the calls of integrate;
         *  - time_rhs_evals: the evaluations of the rates of the model, including
         *    those for the finite difference Jacobian;
         *  - time_g_evals: the evaluations of the root functions;
         *  - time_lin_solv_setups, time_lin_solv_solves: the setups (the LU
         *    factorization) and solves of the linear solver.
         */
        std::unordered_map<std::string, Setting> getStatistics() override;

        void resetStatistics() override;

    private:
        /**
         * @brief The counters that CVODE sets to zero when it is reinitialized.
         */
        struct CVODECounters {
            long numSteps = 0;
            long numRhsEvals = 0;
            long numLinSolvSetups = 0;
            long numErrTestFails = 0;
            long numNonlinSolvIters = 0;
            long numNonlinSolvConvFails = 0;
            long numStepSolveFails = 0;
            long numGEvals = 0;
            long numStabLimOrderReds = 0;
            // the counters of the linear solver interface
            long numJacEvals = 0;
            long numLinRhsEvals = 0;
            long numLinIters = 0;
            long numLinConvFails = 0;

            CVODECounters &operator+=(const CVODECounters &other);

            CVODECounters &operator-=(const CVODECounters &other);
        };

        /**
         * @brief The times of the work of CVODE, see getStatistics.
         */
        struct CVODETimes {
            double total = 0;
            double rhsEvals = 0;
            double gEvals = 0;
            double linSolvSetups = 0;
            double linSolvSolves = 0;
        };

        /**
         * @brief the counters of CVODE since it was last (re)initialized.
         */
        CVODECounters readCVODECounters() const;

        /**
         * @brief adds the counters of CVODE to the statistics, before CVODE is
         * reinitialized or freed and sets them to zero.
         */
        void saveCVODECounters();

        /**
         * @brief The statistics are these plus the counters of CVODE (readCVODECounters).
         * @details resetStatistics sets them to minus the counters of CVODE, and
         * saveCVODECounters adds the counters of CVODE to them.
         */
        CVODECounters savedCounters;

        /**
         * @brief whether CVODE has integrated since it was last (re)initialized.
         * @details CVODE sets the counters of the linear solver interface to zero
         * at the first step after a (re)initialization, not at the (re)initialization.
         * Until that step they still hold the counts that saveCVODECounters added.
         */
        bool linearSolverCountersCurrent = false;

        std::int64_t numReInits = 0;

        // whether integrate is running, which is when a reinitialization is counted
        bool integrating = false;

        CVODETimes times;

        // the setting collect_timing
        bool collectTiming = false;

        // defaults directly from CVODE docs
        static const int mDefaultMaxNumSteps;
        static const int mDefaultMaxAdamsOrder;
        static const int mDefaultMaxBDFOrder;

        // cvode components
        void *mCVODE_Memory = nullptr;
        N_Vector mStateVector = nullptr;
        SUNMatrix jac = nullptr;
        SUNNonlinearSolver nonLinSolver = nullptr;
        SUNLinearSolver linSolver = nullptr;
        SUNContext mSunContext = nullptr;

        IntegratorListenerPtr listener;
        double mLastEventTime;
        bool variableStepPendingEvent;
        bool variableStepTimeEndEvent;

        /**
         * Offset of the two rows that a variable step run reports for an event: the
         * state before it at (event time - offset), the state after it at (event time + offset).
         * Set when the event is found, from the time CVODE had reached and its last step.
         */
        double variableStepEventTolerance;
        std::vector<double> variableStepPostEventState;
        std::vector<unsigned char> eventStatus;

        void testRootsAtInitialTime();

        bool haveVariables() const;

        void assignResultsToModel() const;


        /**
         * @author WBC, ETS, JKM
         * @brief Propagates changes in the "absolute_tolerance" and
         * "relative_tolerance" settings to the CVODE library.
         */
        void setCVODETolerances();

        /**
         * @brief Reinitialize sundials objects
         */
        void reInit(double t0);

        /**
         * @author WBC, ETS, JKM
         * @brief Propagates changes in the following settings to CVODE:\n
         * *  initial_time_step \n
         * *  minimum_time_step \n
         * *  maximum_time_step \n
         * *  maximum_num_steps \n
         * *  absolute_tolerance (via @ref setCVODETolerances) \n
         * *  relative_tolerance (via @ref setCVODETolerances) \n
         */
        void updateCVODE();

        void applyPendingEvents(double timeEnd);

        void applyEvents(double timeEnd, std::vector<unsigned char> &previousEventStatus);

        double applyVariableStepPendingEvents();

        /**
         * The tolerance with which CVODE locates a root found at time t:
         * 100 * unit roundoff * (|t| + |h|), h being the last step of CVODE, with a
         * safety factor of 32 (see the CVODE User Doc).  It depends only on the
         * time scale of the event itself, and not on any other integration.
         */
        double getRootTolerance(double t) const;

        void create();

        void freeSundialsMemory();

        bool stateVectorVariables;

        unsigned long typecode_;

        friend int cvodeDyDtFcn(double t, N_Vector cv_y, N_Vector cv_ydot, void *f_data);

        friend int cvodeEventAndPiecewiseRootFcn(double t, N_Vector y, double *gout, void *g_data);

        friend SUNLinearSolver newTimedLinearSolver(SUNLinearSolver inner, CVODEIntegrator *integrator);

        friend struct TimedLinearSolver;

    };

    template <class SundialsType = CVODEIntegrator>
    std::string decodeSundialsError(SundialsType* solver, int cvodeError, bool exInfo) {
        std::string result;
        std::stringstream ss;
        ss << (int) solver->getValue("maximum_num_steps");
        std::string max_steps = ss.str();

        switch (cvodeError) {
            case CV_TOO_MUCH_WORK:
                result = "CV_TOO_MUCH_WORK";
                if (exInfo) {
                    result += ": The solver took mxstep (" + max_steps + ") internal steps but " +
                              "could not reach tout.";
                }
                break;
            case CV_TOO_MUCH_ACC:
                result = "CV_TOO_MUCH_ACC";
                if (exInfo) {
                    result += ": The solver could not satisfy the accuracy "
                              "demanded by the user for some internal step.";
                }
                break;
            case CV_ERR_FAILURE:
                result = "CV_ERR_FAILURE";
                if (exInfo) {
                    result += ": Error test failures occurred too many times "
                              "(= MXNEF = 7) during one internal time step or"
                              "occurred with |h| = hmin.";
                }
                break;
            case CV_CONV_FAILURE:
                result = "CV_CONV_FAILURE";
                if (exInfo) {
                    result += ": Convergence test failures occurred too many "
                              "times (= MXNCF = 10) during one internal time"
                              "step or occurred with |h| = hmin.";
                }
                break;
            case CV_LINIT_FAIL:
                result = "CV_LINIT_FAIL";
                if (exInfo) {
                    result += ": The linear solver's initialization function "
                              "failed.";
                }
                break;
            case CV_LSETUP_FAIL:
                result = "CV_LSETUP_FAIL";
                if (exInfo) {
                    result += ": The linear solver's setup routine failed in an "
                              "unrecoverable manner.";
                }
                break;
            case CV_LSOLVE_FAIL:
                result = "CV_LSOLVE_FAIL";
                if (exInfo) {
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
                if (exInfo) {
                    result += ": The cvode_mem argument was NULL.";
                }
                break;
            case CV_ILL_INPUT:
                result = "CV_ILL_INPUT";
                if (exInfo) {
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
                if (exInfo) {
                    result += ": indicating that cvode_mem has not been "
                              "allocated (i.e., CVodeInit has not been "
                              "called).";
                }
                break;
            case CV_BAD_K:
                result = "CV_BAD_K";
                if (exInfo) {
                    result += ": k is not in the range 0, 1, ..., qu.";
                }
                break;
            case CV_BAD_T:
                result = "CV_BAD_T";
                if (exInfo) {
                    result += ": t is not in the interval [tn-hu,tn].";
                }
                break;
            case CV_BAD_DKY:
                result = "CV_BAD_DKY";
                if (exInfo) {
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

}

#endif
