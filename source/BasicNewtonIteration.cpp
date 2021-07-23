//
// Created by Ciaran on 26/03/2021.
//

#include "BasicNewtonIteration.h"

namespace rr {
    
    BasicNewtonIteration::BasicNewtonIteration(ExecutableModel *executableModel)
    : NewtonIteration(executableModel) {}

    std::string BasicNewtonIteration::getName() const {
        return "newton";
    }

    std::string BasicNewtonIteration::getDescription() const {
        return "Sundials implementation of newton iteration algorithm "
               "for solving steady state problems.";
    }

    std::string BasicNewtonIteration::getHint() const {
        return "Newton iteration";
    }

    double BasicNewtonIteration::solve() {
        return solveForSteadyState(this, KIN_NONE);
    }

    Solver *BasicNewtonIteration::construct(ExecutableModel *model) const {
        return new BasicNewtonIteration(model);
    }

}