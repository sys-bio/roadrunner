//
// Created by Ciaran on 26/02/2021.
//

#include "KinsolSteadyStateSolver.h"
#include <cassert>
#include "KinsolErrHandler.h"
#include "rrConfig.h"
#include "CVODEIntegrator.h"

namespace rr {

    KinsolSteadyStateSolver::KinsolSteadyStateSolver(ExecutableModel *executableModel)
            : SteadyStateSolver(executableModel) {
        KinsolSteadyStateSolver::resetSettings();
    }

    void KinsolSteadyStateSolver::syncWithModel(ExecutableModel *m) {
        freeKinsol();
        mModel = m;
        if (m) {
            createKinsol();
        }
    }

    void KinsolSteadyStateSolver::createKinsol() {
        if (!mModel) {
            return;
        }

        assert(mStateVector == nullptr && mKinsol_Memory == nullptr &&
               "calling createKinsol, but kinsol objects already exist");

        SUNContext_Create(NULL, &mSunContext);

        // when argument is null, returns size of state std::vector (see rrExecutableModel::getStateVector)
        int stateVectorSize = mModel->getStateVector(nullptr);

        // create our N_Vector
        mStateVector = N_VNew_Serial(stateVectorSize, mSunContext);
        assert(mStateVector && "Sundials failed to create N_Vector for state variables");

        fscale = N_VNew_Serial(stateVectorSize, mSunContext);
        assert(fscale && "Sundials failed to create N_Vector for fscale");
        N_VConst(1, fscale); // no scaling. Implement if wanted.

        uscale = N_VNew_Serial(stateVectorSize, mSunContext);
        assert(uscale && "Sundials failed to create N_Vector for fscale");
        N_VConst(1, uscale); // no scaling. Implement if wanted.

        // initialise to model values
        mModel->getStateVector(mStateVector->ops->nvgetarraypointer(mStateVector));

        int err;

        // allocate the main kinsol memory block
        mKinsol_Memory = KINCreate(mSunContext);

        assert(mKinsol_Memory && "Could not create kinsol memory block, Kinsol failed");

        // make non negative
        constraints = N_VNew_Serial(stateVectorSize, mSunContext);
        assert(constraints && "Sundials failed to create N_Vector for fscale");
        // constraints. If,
        //  0 -> No constraints
        //  1 -> >= 0
        // -1 -> <= 0
        //  2  -> >0
        // -2  -> <0
        N_VConst(2, constraints);

        if (!(bool)getValue("allow_negative")) {
            KINSetConstraints(mKinsol_Memory, constraints);
        }

        // set our own error handler. This should be the first thing called after creating kinsol memory block
        // This is the only function where we need to collect the error code and decode it, since
        // the purpose of using this function is to enable automatic error handling.
        if ((err = KINSetErrHandlerFn(
                mKinsol_Memory,
                reinterpret_cast<KINErrHandlerFn>(kinsolErrHandler),
                this)
            ) != KIN_SUCCESS) {
            decodeKinsolError(err);
        }

        // Set the kinsol "user data".
        // Kinsol uses callback functions for computing the rate of change.
        // Kinsol passes a void* memory buffer to `FixedPointIteration::kinsolDyDtFcn` where we
        // cast back to "FixedPointIteration".
        KINSetUserData(mKinsol_Memory, (void *) this);
    }

    void KinsolSteadyStateSolver::freeKinsol() {
        if (mKinsol_Memory) {
            KINFree(&mKinsol_Memory);
        }

        if (mStateVector) {
            N_VDestroy_Serial(mStateVector);
        }
        if (fscale) {
            N_VDestroy_Serial(fscale);
        }
        if (uscale) {
            N_VDestroy_Serial(uscale);
        }
        if (constraints) {
            N_VDestroy_Serial(constraints);
        }

        mKinsol_Memory = nullptr;
        mStateVector = nullptr;
        fscale = nullptr;
        uscale = nullptr;
        constraints = nullptr;
    }

    void KinsolSteadyStateSolver::resetSettings() {
        SteadyStateSolver::resetSettings();

        std::string desc = "Max. number of iterations the nonlinear solver is allowed to use. ";
        addSetting("num_max_iters", Setting(200), "Maximum Nonlinear Iterations", desc, desc);

        addSetting("allow_negative", Setting(false), "Allow negative values",
                   "Ensures non-negative results",
                   "(bool)Constrains the problem such that all values are non-negative at all times");

        desc = "Kinsol logger level. Default=0, no additional output. Max=3.";
        addSetting("print_level",Setting( 0), "Kinsol Print Level", desc, desc);

        desc = "Form of nu coefficient. One of eta_choice1, eta_choice2 or eta_constant";
        addSetting("eta_form", Setting("eta_choice1"), "ETA Form", desc, desc);

        desc = "No initial matrix setup";
        addSetting("no_init_setup", Setting(false), "No Init Setup", desc, desc);

        desc = "No residual monitoring";
        addSetting("no_res_monitoring", Setting(false), "No Residual Monitoring", desc, desc);

        desc = "Max. iterations without matrix setup";
        addSetting("max_setup_calls", Setting(10), "Max Setup Calls", desc, desc);

        desc = "Max. iterations without residual check";
        addSetting("max_subsetup_calls", Setting(5), "Max Sub Setup Calls", desc, desc);

        desc = "Constant value of nu";
        addSetting("eta_constant_value", Setting(0.1), "ETA Constant Value", desc, desc);

        desc = "Value of gamma where 0 << gamma << 1.0. Use 0 to indidate default value of 0.9.";
        addSetting("eta_param_gamma", Setting(0), "ETA Gamma", desc, desc);

        desc = "Value of alpha where 1.0 < alpha < 2.0. Use 0 to indicate default value of 2.0. ";
        addSetting("eta_param_alpha", Setting(0), "ETA alpha", desc, desc);

        desc = "Value of omega_min - lower bound residual monitoring";
        addSetting("res_mon_min", Setting(0.00001), "Residual Monitoring Param Minimum", desc, desc);

        desc = "Value of omega_max - upper bound residual monitoring";
        addSetting("res_mon_max", Setting(0.9), "Residual Monitoring Param Minimum", desc, desc);

        desc = "Constant value of omega";
        addSetting("res_mon_constant_value", Setting(0.9), "Residual Monitoring Constant Value", desc, desc);

        desc = "Lower bound on epsilon";
        addSetting("no_min_eps", Setting(false), "No Minimum Epsilon", desc, desc);

        desc = "Max. scaled length of Newton step. If 0 use default value which is 1000*||D_u*u_0||2.";
        addSetting("max_newton_step", Setting(0), "Max Newton Step size", desc, desc);

        desc = "Max. number of beta-condition failures";
        addSetting("max_beta_fails", Setting(10), "Max Beta Fails", desc, desc);

        desc = "Function-norm stopping tolerance. If 0 use default of uround^1/3.";
        addSetting("func_norm_tol", Setting(0), "Func Norm Tol", desc, desc);

        desc = "Scaled-step stopping tolerance. If 0 use default of uround^2/3";
        addSetting("scaled_step_tol", Setting(0), "Scaled Step Tol", desc, desc);

        desc = "The function KINSetRelErrFunc speciffies the relative error in computing F(u), which "
               "is used in the difference quotient approximation to the Jacobian matrix. "
               "Set to 0 for default which equals U = unit roundoff.";
        addSetting("rel_err_func", Setting(0), "Relative Error Function", desc, desc);

    }

    void KinsolSteadyStateSolver::getSolverStatsFromKinsol() {
        KINGetNumFuncEvals(mKinsol_Memory, &numFuncEvals);
        KINGetNumNonlinSolvIters(mKinsol_Memory, &numNolinSolvIters);
        KINGetNumBetaCondFails(mKinsol_Memory, &numBetaCondFails);
        KINGetNumBacktrackOps(mKinsol_Memory, &numBacktrackOps);
        KINGetFuncNorm(mKinsol_Memory, &funcNorm);
        KINGetStepLength(mKinsol_Memory, &stepLength);
        KINGetNumNonlinSolvIters(mKinsol_Memory, &numNonlinSolvIters);
    }

    void KinsolSteadyStateSolver::setFScale(double value) {
        N_VConst(value, fscale);
    }

    void KinsolSteadyStateSolver::setFScale(const std::vector<double> &value) {
        int stateSize = mStateVector->ops->nvgetlength(mStateVector);
        if (value.size() != stateSize) {
            std::ostringstream err;
            err << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__
                << ": size of std::vector to set the fscale variable does not "
                   "equal the number of states in the model (" << stateSize << "!=" << value.size() << ")" << std::endl;
            throw std::runtime_error(err.str());
        }
        double *dptr = mStateVector->ops->nvgetarraypointer(fscale);
        *dptr = *value.data();
    }

    void KinsolSteadyStateSolver::setUScale(double value) {
        N_VConst(value, uscale);
    }

    void KinsolSteadyStateSolver::setUScale(std::vector<double> value) {
        int stateSize = mStateVector->ops->nvgetlength(mStateVector);
        if (value.size() != stateSize) {
            std::ostringstream err;
            err << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__
                << ": size of std::vector to set the uscale variable does not "
                   "equal the number of states in the model (" << stateSize << "!=" << value.size() << ")" << std::endl;
            throw std::runtime_error(err.str());
        }
        double *dptr = mStateVector->ops->nvgetarraypointer(uscale);
        *dptr = *value.data();
    }

    void *KinsolSteadyStateSolver::getKinsolMemory() const {
        return mKinsol_Memory;
    }

    /**
     * The settings configuration system in roadrunner make this particular
     * aspect of interacting with sundials libraries a little akward.
     * When a user updates a setting with setValue, they are only
     * updating the value in roadrunner. The change doesn't occur in sundials
     * until we call the corresponding sundials function with the new value.
     * We cannot use regular "setters" because they won't get called when a user
     * changes a value. Instead, here we update all kinsol options
     * at once, and this method is called before we call KIN_Solve.
     */
    void KinsolSteadyStateSolver::updateKinsol() {
        // throw if invalid option chosen.
        std::vector<std::string> validEtaForms({"eta_choice1", "eta_choice2", "eta_constant"});
        std::string etaChoice = getValue("eta_form").get<std::string>(); //
        if (std::find(validEtaForms.begin(), validEtaForms.end(), etaChoice) == validEtaForms.end()) {
            std::ostringstream err;
            err << "\"" << etaChoice << "\". Valid options are ";
            for (auto &x: validEtaForms) {
                err << "\"" << x << "\", ";
            }
            throw InvalidKeyException(err.str());
        }
        if (etaChoice == "eta_choice1") {
            KINSetEtaForm(mKinsol_Memory, KIN_ETACHOICE1);
        } else if (etaChoice == "eta_choice2") {
            KINSetEtaForm(mKinsol_Memory, KIN_ETACHOICE2);
        } else if (etaChoice == "eta_constant") {
            KINSetEtaForm(mKinsol_Memory, KIN_ETACONSTANT);
        }
        KINSetNumMaxIters(mKinsol_Memory, (int)getValue("num_max_iters"));
        KINSetPrintLevel(mKinsol_Memory, (int)getValue("print_level"));
        KINSetNoInitSetup(mKinsol_Memory, (bool)getValue("no_init_setup"));
        KINSetNoResMon(mKinsol_Memory, (bool)getValue("no_res_monitoring"));
        KINSetMaxSetupCalls(mKinsol_Memory, (int)getValue("max_setup_calls"));
        KINSetMaxSubSetupCalls(mKinsol_Memory, (int)getValue("max_subsetup_calls"));
        KINSetEtaConstValue(mKinsol_Memory, (double)getValue("eta_constant_value"));
        KINSetEtaParams(mKinsol_Memory, (double)getValue("eta_param_gamma"), (double)getValue("eta_param_alpha"));
        KINSetResMonParams(mKinsol_Memory, (double)getValue("res_mon_min"), (double)getValue("res_mon_max"));
        KINSetResMonConstValue(mKinsol_Memory, (bool)getValue("res_mon_constant_value"));
        KINSetNoMinEps(mKinsol_Memory, (bool)getValue("no_min_eps"));
        KINSetMaxNewtonStep(mKinsol_Memory, (int)getValue("max_newton_step"));
        KINSetMaxBetaFails(mKinsol_Memory, (int)getValue("max_beta_fails"));
        KINSetFuncNormTol(mKinsol_Memory, (double)getValue("func_norm_tol"));
        KINSetScaledStepTol(mKinsol_Memory, (double)getValue("scaled_step_tol"));
        KINSetRelErrFunc(mKinsol_Memory, (double)getValue("rel_err_func"));
    }

    std::unordered_map<std::string, Setting>& KinsolSteadyStateSolver::getSolverStats() {
//        std::unordered_map<std::string, Setting> map;
        solverStats["numFuncEvals"] = Setting(numFuncEvals);
        solverStats["numNolinSolvIters"] = Setting(numNolinSolvIters);
        solverStats["numBetaCondFails"] = Setting(numBetaCondFails);
        solverStats["numBacktrackOps"] = Setting(numBacktrackOps);
        solverStats["funcNorm"] = Setting(funcNorm);
        solverStats["stepLength"] = Setting(stepLength);
        solverStats["numJacEvals"] = Setting(numJacEvals);
        solverStats["numJtimesEvals"] = Setting(numJtimesEvals);
        solverStats["numLinConvFails"] = Setting(numLinConvFails);
        solverStats["numLinFuncEvals"] = Setting(numLinFuncEvals);
        solverStats["numLinIters"] = Setting(numLinIters);
        solverStats["numNonlinSolvIters"] = Setting(numNonlinSolvIters);
        solverStats["numPrecEvals"] = Setting(numPrecEvals);
        solverStats["numPrecSolves"] = Setting(numPrecSolves);
        return solverStats;
    }

    void KinsolSteadyStateSolver::printSolverStats() {
        std::cout << "Solver Stats: " << std::endl;
        for (auto &it: getSolverStats()) {
            std::cout << "\t" << it.first << " = " << it.second.get<std::string>() << std::endl;
        }
    }

    double KinsolSteadyStateSolver::solveForSteadyState(KinsolSteadyStateSolver *solverInstance, int kinsolStrategy) {
        if (kinsolStrategy < 0 || kinsolStrategy > 4) {
            throw std::invalid_argument("kinsolStrategy should be 0, 1, 2, or 3 for "
                                        "KIN_NONE, KIN_LINESEARCH, KIN_PICARD, KIN_FP respectively");
        }

        assert(mKinsol_Memory && "Kinsol memory block is nullptr");
        assert(mStateVector && "Solvers state std::vector is nullptr");

        // ensures options have been correctly propagated to kinsol
        // before solving
        solverInstance->updateKinsol();

        int flag = KINSol(
                mKinsol_Memory,   // kinsol memory block
                mStateVector,     // initial guess and solution std::vector
                // global strategy, options defined in kinsol.h
                kinsolStrategy,
                uscale,      //scaling std::vector for the variable cc
                fscale      //scaling std::vector for the variable fval
        );

        char *flagName = KINGetReturnFlagName(flag);

        // errors are handled automatically by the error handler for kinsol.
        // here we handle warnings and success flags
        switch (flag) {
            case KIN_SUCCESS: {
                rrLog(Logger::LOG_INFORMATION) << "Steady state found";
                break;
            }
            case KIN_INITIAL_GUESS_OK: {
                rrLog(Logger::LOG_INFORMATION) << "Steady state found. The guess u = u0 satisifed the "
                                                  "system F(u) = 0 within the tolerances specified (the"
                                                  "scaled norm of F(u0) is less than 0.01*fnormtol)." << std::endl;
                break;
            }
            case KIN_STEP_LT_STPTOL: {
                rrLog(Logger::LOG_WARNING)
                    << "kinsol stopped based on scaled step length. This means that the current iterate may"
                       "be an approximate solution of the given nonlinear system, but it is also quite possible"
                       "that the algorithm is \"stalled\" (making insufficient progress) near an invalid solution,"
                       "or that the scalar scsteptol is too large (see ScaledStepTol to"
                       "change ScaledStepTol from its default value)." << std::endl;
                break;
            }
            default: {
                std::string errMsg = decodeKinsolError(flag);
                throw std::runtime_error("Kinsol Error: " + errMsg);
            };
        }
        free(flagName);

        getSolverStatsFromKinsol();

        // update the model's state values
        mModel->setStateVector(mStateVector->ops->nvgetarraypointer(mStateVector));

        return funcNorm;


    }

}