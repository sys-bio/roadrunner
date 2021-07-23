// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

// == FILEDOC =================================================

/** @file NLEQ1Solver.h
* @author ETS, WBC, JKM, KC
* @date 08/03/2015
* @copyright Apache License, Version 2.0
* @brief Provides an interface to NLEQ1
**/

#ifndef rrNLEQ1SolverH
#define rrNLEQ1SolverH

#include <vector>
#include "NLEQSolver.h"
#include "rrExecutableModel.h"

using std::vector;

namespace rr {

/**
 * @internal
 */
    class RR_DECLSPEC NLEQ1Solver : public NLEQSolver {
    public:
        using NLEQSolver::NLEQSolver;

        explicit NLEQ1Solver(ExecutableModel *_model);

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

        /**
         * @brief construct an instance of NLEQ1Solver
         * @details Caller is responsible for deleting memory
         * associated with returned Solver*.
         */
        Solver* construct(ExecutableModel* executableModel) const override;

        // ** Solver routines
        double solve() override;
    };

}

#endif
