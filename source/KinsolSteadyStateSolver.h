//
// Created by Ciaran on 26/02/2021.
//

#ifndef ROADRUNNER_KINSOLSTEADYSTATESOLVER_H
#define ROADRUNNER_KINSOLSTEADYSTATESOLVER_H

#include <nvector/nvector_serial.h>  /* access to serial N_Vector       */
#include <sundials/sundials_types.h> /* defs. of realtype, sunindextype */
#include "kinsol/kinsol.h"
#include "SteadyStateSolver.h"
#include "rrExecutableModel.h"
#include "rrException.h"
#include "KinsolErrHandler.h"

namespace rr {
    class ExecutableModel;

    class RoadRunner;

    /**
     * Utility macro for checking for a variable for nullptr
     * @param x the pointer that is being checked for null
     * @param fcn the name of the function call that produced the null
     * @param the return type for @param fcn
     */
#define KinsolCheckForNull(x, fcn, type) \
    if (x == nullptr){                   \
        std::ostringstream err_;          \
        err_ << "The value of object with type: \"" << type << "\" " \
    "which is the output of sundials function " << fcn << " is nullptr" << std::endl;\
        throw  NullPointerException(err_.str());                                  \
    };                               \


    /**
     * @brief base class to steady state solvers from the
     * Sundials package.
     * @details The steady state solver in Sundials
     * is called Kinsol. They have multiple different strategies
     * some which require model integration, and some that do not.
     *
     */
    class KinsolSteadyStateSolver : public SteadyStateSolver {
    public:
        using SteadyStateSolver::SteadyStateSolver;

        /**
         * @brief constructor for creating KinsolSteadyStateSolver
         * from an executable model only. This constructor
         * should be inherited by subclasses
         */
        explicit KinsolSteadyStateSolver(ExecutableModel *executableModel);

        /**
         * @brief The default constructor for constructing without a model.
         * @details useful only for testing.
         */
        KinsolSteadyStateSolver() = default;

        /**
         * @brief virtual destructor
         */
        ~KinsolSteadyStateSolver() override = default;

        /**
         * @brief implementation of Solver::syncWithModel.
         * @details called by the main RoadRunner class to
         * syncronize the model with its solvers (to be sure
         * were not simulating an out of date model).
         */
        void syncWithModel(ExecutableModel *m) override;

        /**
         * @brief set all elements of the fscale variable to @param value.
         */
        void setFScale(double value);

        /**
         * @brief set elements of the fscale variable to @param value.
         * @param value should have the same size as the state std::vector
         */
        void setFScale(const std::vector<double> &value);

        /**
         * @brief set all elements of the uscale variable to @param value.
         */
        void setUScale(double value);

        /**
         * @brief set elements of the uscale variable to @param value.
         * @param value should have the same size as the state std::vector
         */
        void setUScale(std::vector<double> value);


        /**
         * @brief kinsol callback function that implements system equations
         * @details This is the function that gets called by the
         * kinsol `KINSol` command for solving the system.
         * @note this member must be static in order to be used as a
         * function pointer by KINInit.
         */
        template<class KinsolSteadyStateSolverType>
        static int kinsolDyDtFcn(N_Vector stateVecIn, N_Vector stateVecOut, void *userData) {
            // we need to use the state std::vector y (chosen by the solver)
            // to compute the rates dydt and then update the state std::vector
            // using the rates.
            double *y = NV_DATA_S(stateVecIn);

            // initialize the out NVector with numerical limits for double
            N_VConst(std::numeric_limits<double>::max(), stateVecOut);
            double *dydt = NV_DATA_S(stateVecOut);

            // cast user data back into our solver type
            auto solver = (KinsolSteadyStateSolverType *) userData;

            assert(solver && "userData pointer is nullptr in callback kinsolDyDtFcn");

            ExecutableModel *model = solver->mModel;

            assert(model && "model is nullptr");

            // collect the number of states in this model
            int numStates = stateVecIn->ops->nvgetlength(stateVecIn);

            model->getStateVectorRate(model->getTime(), y, dydt);

            for (int i = 0; i < numStates; i++) {
                if (dydt[i] == std::numeric_limits<double>::max()) {
                    std::ostringstream err;
                    err << __FILE__":" << __LINE__ << ":" << "kinsolDyDtFcn";
                    err << ": steady state solver \"" << solver->getName()
                        << "\". The rate for variable \"" << model->getStateVectorId(i)
                        << "\" is equal to the numerical ";
                    err << "limits for a double." << std::endl;
                    throw KinsolException(err.str());
                }
            }
            rrLog(Logger::LOG_TRACE) << __FUNCTION__ << ", model: " << model;

            return KIN_SUCCESS;
        };

        /**
         * @brief Generic solver method. When this method is called
         * sundials steady state algorithms solver for steady state.
         * Subclasses use specialized versions of this template
         * using their desired kinsol strategy.
         * @details this method cannot be called `solve()` since that
         * is already taken with different signature requirements.
         * @param solverInstance pointer (non owning) to instance of subclass of
         *  KinsolSteadyStateSolver.
         * @param kinsolStrategy one of:
         *  - KIN_NONE: basic newton iteration
         *  - KIN_LINESEARCH: newton iteration with linesearch globalization
         *  - KIN_FP: Fixed point iteration, with anderson acceleration (no linear solver is used)
         *  - KIN_PICCARD: Piccad iteration, with anderson aceleration (with linear solver)
         * These options are macro's defined by sundials. They correspond to :
         *      KIN_NONE       0
         *      KIN_LINESEARCH 1
         *      KIN_PICARD     2
         *      KIN_FP         3
         *
         */
        double solveForSteadyState(KinsolSteadyStateSolver* solverInstance, int kinsolStrategy);

        /**
         * @brief getter for the underlying memory block
         * that is used by kinsol.
         * @details mostly useful in testing.
         */
        void *getKinsolMemory() const;

        /**
         * @brief return a mapping containing solver statistics.
         * @details only filled once solve has been called.
         * @see getSolverStatsFromKinsol which actually collects the data
         * from kinsol. Whereas this method returns the data in a unordered_map
         */
         std::unordered_map<std::string, Setting>& getSolverStats();

         /**
          * @brief display solver stats to console
          */
          void printSolverStats();

    protected:

        /**
         * @brief Destory kinsol objects
         * @details Destroys the kinsol objects that
         * are shared among KinsolSteadyStateSolvers
         */
        virtual void freeKinsol();

        /**
         * @brief Calls methods for updating options in kinsol
         * @details User passes in arguments like uScale and they
         * are stored at the class level. This method passed them
         * on to kinsol.
         */
        virtual void updateKinsol();

        /**
         * @brief Initialize kinsol objects.
         * @details All subclasses require some of the same
         * initial calls (such as N_VNew_Serial and KINCreate).
         */
        virtual void createKinsol();

        /**
         * @brief add settings to the Solver::settings
         * std::map when called for the first time or
         * reset the settings std::map back to default values
         * when called subsequent times.
         */
        void resetSettings() override;


        /**
        * @brief Pointer to the kinsol memory block
        */
        void *mKinsol_Memory = nullptr;

        /**
         * @brief a sundials N_Vector for storing system state variables
         */
        N_Vector mStateVector = nullptr;

        /**
         * @brief std::vector containing diagonal elements of scaling matrix DF for F(u)
         * chosen so that the components of D_F*F(u) (as a matrix multiplication) all
         * have roughly the same magnitude when u is not too near a root of F(u). In
         * the case of a fixed-point iteration, consider F(u) = G(u) - u.
         * @note here u is mStateVector, F is the std::vector of rate functions and D_u an internal
         * variable inside kinsol
         *
         */
        N_Vector fscale = nullptr;

        /**
         * @brief std::vector containing diagonal elements of scaling matrix Du for std::vector
         * u chosen so that the components of D_u*u (as a matrix multiplication) all
         * have roughly the same magnitude when u is close to a root of F(u).
         * @note here u is mStateVector, F is the std::vector of rate functions and D_u an internal
         * variable inside kinsol
         */
        N_Vector uscale = nullptr;

        /**
         * @brief vector of constraints used to ensure
         * positive solution.
         * @details only in effect when allow_negative option set to false
         */
        N_Vector constraints = nullptr;


        /**
         * @brief after solve() has been called, this method
         * collects some details such as number of function evals
         *
         */
        virtual void getSolverStatsFromKinsol();

        /**
         * @brief kinsol output variables.
         * @details fileld when
         */
        long int numFuncEvals;
        long int numNolinSolvIters;
        long int numBetaCondFails;
        long int numBacktrackOps;
        double funcNorm;
        double stepLength;
        long int numJacEvals;
        long int numJtimesEvals;
        long int numLinConvFails;
        long int numLinFuncEvals;
        long int numLinIters;
        long int numNonlinSolvIters;
        long int numPrecEvals;
        long int numPrecSolves;

        /**
         * @brief stores the number of iterations required
         * by the solver to converge.
         * @details empty until after solve() has been called.
         * This method uses kinsol's KINGetNumNonlinSolvIters function
         */
        long int nSolverIterations = 0;

        std::unordered_map<std::string, Setting> solverStats;


    };


}


#endif //ROADRUNNER_KINSOLSTEADYSTATESOLVER_H
