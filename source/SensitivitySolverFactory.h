//
// Created by Ciaran on 15/07/2021.
//

#ifndef ROADRUNNER_SENSITIVITYSOLVERFACTORY_H
#define ROADRUNNER_SENSITIVITYSOLVERFACTORY_H

#include "RegistrationFactory.h"
#include "ForwardSensitivitySolver.h"

namespace rr {

    /**
     * @author JKM, WBC
     * @brief Constructs new integrators
     * @details Implements the factory and singleton patterns.
     * Constructs a new integrator given the name (e.g. cvode, gillespie)
     * and returns a base pointer to @ref rr::SensitivitySolver.
     */
    class RR_DECLSPEC SensitivitySolverFactory
            : public RegistrationFactory {
    public:

        static SensitivitySolverFactory &getInstance();

        static void Register();
    };
}

#endif //ROADRUNNER_SENSITIVITYSOLVERFACTORY_H
