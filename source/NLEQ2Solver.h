// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

// == FILEDOC =================================================

/** @file NLEQ2Solver.h
* @author ETS, WBC, JKM, KC
* @date 08/03/2015
* @copyright Apache License, Version 2.0
* @brief Provides an interface to NLEQ
**/

#ifndef rrNLEQ2SolverH
#define rrNLEQ2SolverH

#include <vector>
#include "NLEQSolver.h"
#include "rrExecutableModel.h"

using std::vector;

namespace rr {

/**
 * @internal
 */
    class RR_DECLSPEC NLEQ2Solver : public NLEQSolver {
    public:

        using NLEQSolver::NLEQSolver;

        /**
        * @author WBC, JKM
        * @brief Get the name for this Solver
        * @note Delegates to @ref getName
        */
        std::string getName() const override;

        /**
        * @author WBC
        * @brief Get the description for this Solver
        * @note Delegates to @ref getDescription
        */
        std::string getDescription() const override;

        /**
        * @author WBC
        * @brief Get the hint for this Solver
        * @note Delegates to @ref getHint
        */
        std::string getHint() const override;

        // ** Solver routines
        double solve() override;

        /**
         * @brief construct an instance of NLEQ2Solver
         * @details Caller is responsible for deleting memory
         * associated with returned Solver*.
         */
        Solver *construct(ExecutableModel *executableModel) const override;
    };
}

#endif
