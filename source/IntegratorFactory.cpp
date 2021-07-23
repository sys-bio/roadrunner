//
// Created by Ciaran on 15/07/2021.
//

#include "IntegratorFactory.h"
#include "CVODEIntegrator.h"
#include "GillespieIntegrator.h"
#include "RK4Integrator.h"
#include "RK45Integrator.h"
#include "EulerIntegrator.h"

namespace rr{

    static std::mutex integratorFactoryMutex;
    static std::mutex integratorRegistrationMutex;


    IntegratorFactory &IntegratorFactory::getInstance() {
        return RegistrationFactory::getInstance<IntegratorFactory>(integratorFactoryMutex);
    }

    void IntegratorFactory::Register() {
        static bool flag = false;
        if (!flag) {
            std::lock_guard<std::mutex> lockGuard(integratorRegistrationMutex);
            flag = true;
            IntegratorFactory::getInstance().registerSolver(new CVODEIntegrator);
            IntegratorFactory::getInstance().registerSolver(new GillespieIntegrator);
            IntegratorFactory::getInstance().registerSolver(new RK4Integrator);
            IntegratorFactory::getInstance().registerSolver(new RK45Integrator);
            IntegratorFactory::getInstance().registerSolver(new EulerIntegrator);
        }
    }

}