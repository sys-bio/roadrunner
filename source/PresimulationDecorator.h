//
// Created by Ciaran on 12/03/2021.
//

#ifndef ROADRUNNER_PRESIMULATIONDECORATOR_H
#define ROADRUNNER_PRESIMULATIONDECORATOR_H

#include "SteadyStateSolverDecorator.h"

namespace rr {

    /**
     * @brief wrapper class to change the solve() method of SteadyStateSolver types.
     * @details Pass a solver to the PresimulationDecorator. The resulting
     * object can be used as a SteadyStateSolver; the solve() method of which
     * will apply presimulation to a steadyState calculation.
     */
    class PresimulationDecorator : public SteadyStateSolverDecorator {
    public:
        using SteadyStateSolverDecorator::SteadyStateSolverDecorator;

        ~PresimulationDecorator() override = default;

        /**
         * @brief construct a PresimulationDecorator from a solver. Use the
         * presimulation_time solver option to set the time point to integrate to
         * before computing steady state.
         * @param solver pointer to instantiated SteadyStateSolver type.
         * @details options already present in the solver or the model
         * associated with the solver are respected.
         */
        explicit PresimulationDecorator(SteadyStateSolver *solver);

        double solve() override;

        Solver * construct(ExecutableModel* executableModel) const override;

    private:
        std::string decoratorName() const override;
    };

}


#endif //ROADRUNNER_PRESIMULATIONDECORATOR_H
