//
// Created by Ciaran on 15/07/2021.
//

#ifndef ROADRUNNER_STEADYSTATESOLVERFACTORY_H
#define ROADRUNNER_STEADYSTATESOLVERFACTORY_H

#include "rrExporter.h"
#include "RegistrationFactory.h"

namespace rr {


    /**
     * @author JKM, WBC
     * @brief Constructs new integrators
     * @details Implements the factory and singleton patterns.
     * Constructs a new integrator given the name (e.g. cvode, gillespie)
     * and returns a base pointer to @ref rr::SteadyStateSolver.
     */
    class RR_DECLSPEC SteadyStateSolverFactory
            : public RegistrationFactory {
    public:

    /**
     * @author JKM, WBC
     * @brief Returns the singleton instance of the solver factory
     * @details If one exists return is otherwise create
     * one. This method implements the sigleton pattern
     * and is thread safe due to use of std::mutex.
     */
    static SteadyStateSolverFactory &getInstance();

    static void Register();
};

}

#endif //ROADRUNNER_STEADYSTATESOLVERFACTORY_H
