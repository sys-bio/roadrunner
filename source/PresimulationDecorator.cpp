//
// Created by Ciaran on 12/03/2021.
//

#include "PresimulationDecorator.h"
#include "SteadyStateSolverDecorator.h"
#include "CVODEIntegrator.h"
#include "rrException.h"
#include "rrExecutableModel.h"

namespace rr {


    PresimulationDecorator::PresimulationDecorator(SteadyStateSolver *solver)
            : SteadyStateSolverDecorator(solver) {}

    double PresimulationDecorator::solve() {
        if (!solver_->getModel()) {
            throw NullPointerException("PresimulationDecorator::solve(): mModel instance in solver object is nullptr");
        }
        try {
            // if we can solve without presimulation, we do so
            return SteadyStateSolverDecorator::solve();

          // std::runtime_error does not work here because NLEQ throws std::exception
          // and its best not to touch it as we might break something else.
        } catch (std::exception &e) {
            CVODEIntegrator integrator(solver_->getModel());
            // integrate one interval between 0 and presimulation_time.
            integrator.integrate(0, (double)solver_->getValue("presimulation_time"));
            solver_->syncWithModel(solver_->getModel());
            return solver_->solve();
        }
    }

    std::string PresimulationDecorator::decoratorName() const {
        return "Presimulation";
    }

    Solver *PresimulationDecorator::construct(ExecutableModel *executableModel) const {
        return new PresimulationDecorator(executableModel);
    }
}