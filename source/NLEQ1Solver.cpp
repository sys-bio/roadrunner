#pragma hdrstop

#include "NLEQ1Solver.h"
#include "rrNLEQ1Interface.h"


namespace rr {

    NLEQ1Solver::NLEQ1Solver(ExecutableModel *_model) : NLEQSolver(_model) {
        NLEQ1Solver::resetSettings();

    }

    std::string NLEQ1Solver::getName() const {
        return "nleq1";
    }

    std::string NLEQ1Solver::getDescription() const {
        return "NLEQ1 is a non-linear equation solver which uses a global Newton "
               "method with adaptive damping strategies (see http://elib.zib.de/pub/elib/codelib/NewtonLib/)";
    }

    std::string NLEQ1Solver::getHint() const {
        return "Steady-state nonlinear systems of equations solver";
    }

    double NLEQ1Solver::solve() {
        return solveNLEQ<NLEQ1Interface>();
    }

    Solver *NLEQ1Solver::construct(ExecutableModel *executableModel) const {
        return new NLEQ1Solver(executableModel);
    }

}    //end of namespace


