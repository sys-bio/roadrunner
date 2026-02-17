//
// Created by Ciaran on 12/03/2021.
//

#include "SteadyStateSolverDecorator.h"

namespace rr {

    void SteadyStateSolverDecorator::syncWithModel(ExecutableModel *m) {
        return solver_->syncWithModel(m);
    }

    double SteadyStateSolverDecorator::solve() {
        return solver_->solve();
    }

    SteadyStateSolverDecorator::SteadyStateSolverDecorator(SteadyStateSolver *solver)
        : SteadyStateSolver(solver->getModel()), solver_(solver)
    {
        settings = solver_->getSettingsMap();
        mModel = nullptr;
    }

    std::string SteadyStateSolverDecorator::getName() const {
        return decoratorName() + "(" + solver_->getName() + ")";
    }

    std::string SteadyStateSolverDecorator::getDescription() const {
        return decoratorName() + "(" + solver_->getDescription() + ")";
    }

    std::string SteadyStateSolverDecorator::getHint() const {
        return decoratorName() + "(" + solver_->getHint() + ")";
    }

    void SteadyStateSolverDecorator::resetSettings()  {
        return solver_->resetSettings();
    }

    ExecutableModel* SteadyStateSolverDecorator::getModel() const
    {
        return solver_->getModel();
    }

    bool SteadyStateSolverDecorator::hasPresimSetup()
    {
      return solver_->hasPresimSetup();
    }

    bool SteadyStateSolverDecorator::hasApproxSetup()
    {
      return solver_->hasApproxSetup();
    }

    std::string SteadyStateSolverDecorator::decoratorName() const {
        return "SteadyStateSolverDecorator";
    }

}