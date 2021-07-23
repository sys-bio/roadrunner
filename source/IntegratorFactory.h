//
// Created by Ciaran on 15/07/2021.
//

#ifndef ROADRUNNER_INTEGRATORFACTORY_H
#define ROADRUNNER_INTEGRATORFACTORY_H

#include <mutex>
#include "rrExporter.h"
#include "RegistrationFactory.h"

namespace rr{

    /**
     * @author JKM, WBC
     * @brief Constructs new integrators
     * @details This is a singleton class. All methods except for
     * getInstance are fully determined by superclass FactoryWithRegistration.
     */
    class RR_DECLSPEC IntegratorFactory : public RegistrationFactory {
    public:
        /**
         * @brief get an instance of this IntegratorFactory.
         * @details If one exists
         * return is otherwise create one. This method implements the
         * sigleton pattern and is thread safe due to use of std::mutex.
         */
        static IntegratorFactory &getInstance();

        /**
         * @author JKM
         * @brief Registers all integrators at startup
         * @details Is run at first instantiation of @ref RoadRunner.
         * Subsequent calls have no effect.
         */
        static void Register();

    };

}

#endif //ROADRUNNER_INTEGRATORFACTORY_H
