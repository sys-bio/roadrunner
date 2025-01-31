//
// Created by Ciaran on 12/03/2021.
//

#include "PresimulationProgramDecorator.h"
#include "SteadyStateSolverDecorator.h"
#include "CVODEIntegrator.h"
#include "rrException.h"
#include "rrExecutableModel.h"

namespace rr {


    PresimulationProgramDecorator::PresimulationProgramDecorator(SteadyStateSolver *solver)
            : SteadyStateSolverDecorator(solver) {}

    double PresimulationProgramDecorator::solve() {
        if (!solver_->getModel()) {
            throw NullPointerException("PresimulationProgramDecorator::solve(): mModel instance in solver object is nullptr");
        }
        try {
            // try no presimulation first
            return SteadyStateSolverDecorator::solve();
        } catch  (std::exception& e) {
            rr::Setting presimulationTimesVariant = getValue("presimulation_times");
            std::vector<double> times = presimulationTimesVariant.get<std::vector<double>>();
            for (const auto &timePoint: times) {
                solver_->getModel()->reset();
                CVODEIntegrator integrator(solver_->getModel());
                // integrate one interval between 0 and presimulation_time.
                integrator.integrate(0, timePoint);
                solver_->syncWithModel(solver_->getModel());

                try {
                    return solver_->solve();
                } catch (std::exception &err) {
                    if (timePoint == times.back())
                        throw CoreException("SteadyStateSolver failed to solve presimulation program: ", err.what());
                }
            }
        }
        return 0.0; // if we get here all presimulation failed, and we may try to approximate the results
    }

    std::string PresimulationProgramDecorator::decoratorName() const {
        return "PresimulationProgram";
    }

    Solver *PresimulationProgramDecorator::construct(ExecutableModel *executableModel) const {
        return new PresimulationProgramDecorator(executableModel);
    }
}