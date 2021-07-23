
#include "SensitivitySolverFactory.h"

namespace rr {

    /**
     * To make the singleton SensitivitySolverFactory thread safe
     */
    static std::mutex sensitivitySolverMutex;
    static std::mutex sensitivityRegistrationMutex;


    SensitivitySolverFactory &SensitivitySolverFactory::getInstance() {
        return RegistrationFactory::getInstance<SensitivitySolverFactory>(sensitivitySolverMutex);
    }

    void SensitivitySolverFactory::Register() {
        static bool flag = false;
        if (!flag) {
            std::lock_guard<std::mutex> mtx(sensitivityRegistrationMutex);
            SensitivitySolverFactory::getInstance().registerSolver(new ForwardSensitivitySolver);
            flag = true;
        }
    }
}

