//
// Created by Ciaran on 26/03/2021.
//

#include "LinesearchNewtonIteration.h"

namespace rr {
    
    LinesearchNewtonIteration::LinesearchNewtonIteration(ExecutableModel *executableModel) 
    : NewtonIteration(executableModel) {}

    std::string LinesearchNewtonIteration::getName() const {
        return "newton_linesearch";
    }

    std::string LinesearchNewtonIteration::getDescription() const {
        return "Sundials implementation of newton iteration algorithm "
               "with linesearch globalization for solving steady state problems.";
    }

    std::string LinesearchNewtonIteration::getHint() const {
        return "Newton iteration with linesearch strategy";
    }

    double LinesearchNewtonIteration::solve() {
        return solveForSteadyState(this, KIN_LINESEARCH);
    }

    Solver *LinesearchNewtonIteration::construct(ExecutableModel *model) const {
        return new LinesearchNewtonIteration(model);
    }

}