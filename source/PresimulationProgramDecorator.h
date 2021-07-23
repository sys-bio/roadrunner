//
// Created by Ciaran on 12/03/2021.
//

#ifndef ROADRUNNER_PresimulationProgramDecorator_H
#define ROADRUNNER_PresimulationProgramDecorator_H

#include "SteadyStateSolverDecorator.h"

namespace rr {

    /**
     * @brief wrapper class to change the solve() method of SteadyStateSolver types.
     * @details Pass a solver to the PresimulationProgramDecorator. The resulting
     * object can be used as a SteadyStateSolver; the solve() method of which
     * will apply presimulation to a steadyState calculation.
     */
    class PresimulationProgramDecorator : public SteadyStateSolverDecorator {
    public:
        using SteadyStateSolverDecorator::SteadyStateSolverDecorator;

        ~PresimulationProgramDecorator() override = default;

        /**
         * @brief construct a PresimulationProgramDecorator from a solver. Use the
         * presimulation_time solver option to set the time point to integrate to
         * before computing steady state.
         * @param solver pointer to instantiated SteadyStateSolver type.
         * @details options already present in the solver or the model
         * associated with the solver are respected.
         */
        explicit PresimulationProgramDecorator(SteadyStateSolver *solver);

        double solve() override;

        Solver* construct(ExecutableModel* executableModel) const override;

    private:
        std::string decoratorName() const override;
    };

}


#endif //ROADRUNNER_PresimulationProgramDecorator_H
