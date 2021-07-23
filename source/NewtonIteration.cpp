//
// Created by Ciaran on 26/02/2021.
//

#include "NewtonIteration.h"
#include "rrExecutableModel.h"
#include <cassert>
#include "KinsolErrHandler.h"

#include "sunmatrix/sunmatrix_dense.h"
#include "sunlinsol/sunlinsol_dense.h"

namespace rr {

    NewtonIteration::NewtonIteration(ExecutableModel *executableModel)
    :  KinsolSteadyStateSolver(executableModel) {
        // note: we deliberately use the NewtonIteration namespace here
        // because calling virtual methods from constructors is dangerous.
        // We must ensure we call the right version of createKinsol
        NewtonIteration::resetSettings(); // note: must be called before createKinsol
        NewtonIteration::createKinsol();
    }

    NewtonIteration::~NewtonIteration() {
        NewtonIteration::freeKinsol();
    }

    void NewtonIteration::createKinsol() {
        int err;

        KinsolSteadyStateSolver::createKinsol();

        // kinsolDyDtFcn has the signature of "KINSysFn"
        if ((err = KINInit(mKinsol_Memory, kinsolDyDtFcn < NewtonIteration > , mStateVector)) != KIN_SUCCESS) {
            decodeKinsolError(err);
        }

        /* Create dense SUNMatrix for jacobian */
        int numEq = mStateVector->ops->nvgetlength(mStateVector);
        jac = SUNDenseMatrix(numEq, numEq);
        KinsolCheckForNull(jac, "SUNDenseMatrix", "SUNMatrix")

        /* Create dense SUNLinearSolver object */
        linearSolver = SUNLinSol_Dense(mStateVector, jac);
        KinsolCheckForNull(jac, "SUNLinSol_Dense", "SUNLinearSolver")

        /* Attach the matrix and linear solver to KINSOL */
        // memory leak somewhere???
        if ((err = KINSetLinearSolver(mKinsol_Memory, linearSolver, jac))) {
            decodeKinsolError(err);
        }
    }

    void NewtonIteration::freeKinsol() {
        KinsolSteadyStateSolver::freeKinsol();

        if (linearSolver) {
            SUNLinSolFree(linearSolver);
        }

        if (jac) {
            SUNMatDestroy(jac);
        }

        linearSolver = nullptr;
        jac = nullptr;
    }

    double NewtonIteration::solveNewtonIteration(const std::string &strategy) {
        return solveForSteadyState(this, getValue("strategy").get<std::string>() == "basic" ? KIN_NONE : KIN_LINESEARCH);
    }

    void NewtonIteration::updateKinsol() {
        KinsolSteadyStateSolver::updateKinsol();
    }

    void NewtonIteration::resetSettings() {
        KinsolSteadyStateSolver::resetSettings();
    }

    void NewtonIteration::getSolverStatsFromKinsol() {
        KinsolSteadyStateSolver::getSolverStatsFromKinsol();
        KINGetNumLinConvFails(mKinsol_Memory, &numLinConvFails);
        KINGetNumJacEvals(mKinsol_Memory, &numJacEvals);
        KINGetNumJtimesEvals(mKinsol_Memory, &numJtimesEvals);
        KINGetNumLinFuncEvals(mKinsol_Memory, &numLinFuncEvals);
        KINGetNumLinIters(mKinsol_Memory, &numLinIters);
        KINGetNumPrecEvals(mKinsol_Memory, &numPrecEvals);
        KINGetNumPrecSolves(mKinsol_Memory, &numPrecSolves);
    }

    std::string NewtonIteration::getName() const {
        return "NewtonIteration";
    }

    std::string NewtonIteration::getDescription() const {
        return "Uses the sundials implementation of newton "
               "iteration";
    }

    std::string NewtonIteration::getHint() const {
        return "Newton Iteration";
    }


}
