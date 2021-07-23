//
// Created by Ciaran Welsh on 14/06/2021.
//

#ifndef ROADRUNNER_REGISTRATIONFACTORY_H
#define ROADRUNNER_REGISTRATIONFACTORY_H

#include "rrExporter.h"
#include <iostream>
#include <mutex>
#include "Registrable.h"
#include "Solver.h"

namespace rr {

    // fwd
    class ExecutableModel;
    class Solver;

    typedef std::vector<Registrable *> RegistrableVector;

    /**
     * @author JKM, WBC
     * @brief Constructs new integrators
     * @details Implements the factory and singleton patterns.
     * Constructs a new integrator given the name (e.g. cvode, gillespie)
     * and returns a base pointer to @ref rr::Integrator.
     */
    class RR_DECLSPEC RegistrationFactory {
    public:
        virtual ~RegistrationFactory();

        /**
         * @author JKM, WBC
         * @brief Constructs a new integrator given the name
         * (e.g. cvode, gillespie)
         */
        Solver *New(const std::string& name, ExecutableModel *m) const;

        /**
         * @author JKM, WBC
         * @brief Registers a new integrator with the factory
         * so that it can be constructed
         * @details Should be called at startup for new integrators.
         */
        virtual void registerSolver(Registrable *i);

        /**
         * @author JKM, WBC
         * @brief Returns the singleton instance of the integrator factory
         */
        template<class FactoryType>
        static FactoryType &getInstance(std::mutex& mutex){
            std::lock_guard<std::mutex> lockGuard(mutex);
            static FactoryType factory;
            return factory;
        }

        virtual std::size_t size() const;

        virtual std::string name(std::size_t n) const;

        virtual std::string hint(std::size_t n) const;

        virtual std::string description(std::size_t n) const;

    protected:
        /**
         * @author JKM, WBC
         * @brief Prevents external instantiation
         */
        RegistrationFactory() = default;

        RegistrableVector registrars;

    };


}

#endif //ROADRUNNER_REGISTRATIONFACTORY_H
