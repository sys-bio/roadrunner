// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

// == FILEDOC =================================================

/** @file SolverRegistration.cpp
* @author JKM
* @date 07/31/2015
* @copyright Apache License, Version 2.0
* @brief Registers the solvers at startup
**/

// == INCLUDES ================================================

# include "rrConfigure.h"
# include "SolverRegistration.h"
# include "Solver.h"
# include "NLEQ1Solver.h"
# include "NLEQ2Solver.h"

# if RR_USE_CXX14
#   include <mutex>
# endif
// == CODE ====================================================

namespace rr
{
    // call exactly once
    static void register_solvers_at_init() {
        SteadyStateSolverFactory::getInstance().registerSteadyStateSolver(new NLEQ1SolverRegistrar());
        SteadyStateSolverFactory::getInstance().registerSteadyStateSolver(new NLEQ2SolverRegistrar());
    }

    void SolverRegistrationMgr::Register() {
# if RR_USE_CXX14
        static std::once_flag flag;
        std::call_once(flag, register_solvers_at_init());
# else
        // WARNING: not thread safe
        static bool flag = false;
        if (!flag) {
            flag = true;
            register_solvers_at_init();
        }
# endif
    }
}