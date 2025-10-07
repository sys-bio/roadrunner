//
// Created by Ciaran on 14/03/2021.
//

#ifndef ROADRUNNER_APPROXSTEADYSTATEDECORATOR_H
#define ROADRUNNER_APPROXSTEADYSTATEDECORATOR_H

#include "SteadyStateSolverDecorator.h"

namespace rr {
    /** @cond PRIVATE */

    class ApproxSteadyStateDecorator : public SteadyStateSolverDecorator {
    public:
        using SteadyStateSolverDecorator::SteadyStateSolverDecorator;

        ~ApproxSteadyStateDecorator() override = default;

        explicit ApproxSteadyStateDecorator(SteadyStateSolver* solver) ;

        double solve() override;

        Solver* construct(ExecutableModel* executableModel) const override;

        virtual bool hasApproxSetup() override;

    private:
        std::string decoratorName() const override;

    };
    /** @endcond PRIVATE */
}


#endif //ROADRUNNER_APPROXSTEADYSTATEDECORATOR_H
