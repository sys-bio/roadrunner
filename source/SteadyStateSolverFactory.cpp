//
// Created by Ciaran on 15/07/2021.
//

#include "SteadyStateSolverFactory.h"
#include <mutex>
#include "NLEQ1Solver.h"
#include "NLEQ2Solver.h"
#include "BasicNewtonIteration.h"
#include "LinesearchNewtonIteration.h"


namespace rr {
    static std::mutex steadyStateSolverFactoryMutex;
    static std::mutex steadyStateSolverRegistrationMutex;


    SteadyStateSolverFactory &SteadyStateSolverFactory::getInstance() {
        return RegistrationFactory::getInstance<SteadyStateSolverFactory>(steadyStateSolverFactoryMutex);
    }

    void SteadyStateSolverFactory::Register() {
        static bool flag = false;
        if (!flag) {
            std::lock_guard <std::mutex> mtx(steadyStateSolverRegistrationMutex);
            flag = true;
            // instantiate instance of Solver* without a model instance and it has access
            // to Registrar* methods.
            SteadyStateSolverFactory::getInstance().registerSolver(new NLEQ1Solver);
            SteadyStateSolverFactory::getInstance().registerSolver(new NLEQ2Solver);
            SteadyStateSolverFactory::getInstance().registerSolver(new BasicNewtonIteration);
            SteadyStateSolverFactory::getInstance().registerSolver(new LinesearchNewtonIteration);
        }
    }

}