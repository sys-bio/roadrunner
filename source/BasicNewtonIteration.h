//
// Created by Ciaran on 26/03/2021.
//

#ifndef ROADRUNNER_BASICNEWTONITERATION_H
#define ROADRUNNER_BASICNEWTONITERATION_H

#include "NewtonIteration.h"

namespace rr {

    /**
     * @brief Solve for steady state using Kinsol's implementation 
     * of Newton Iteration. 
     * @details The kinsol docs refer to this algorithm as the "basic" 
     * newton iteration to differentiate it from the "linesearch" strategy 
     * which is a variant of the "basic" newton iteration. 
     * @see LinesearchNewtonIteration
     */
    class BasicNewtonIteration : public NewtonIteration {

    public:
        using NewtonIteration::NewtonIteration;

        ~BasicNewtonIteration() override = default ;

        explicit BasicNewtonIteration(ExecutableModel *executableModel);

        /**
        * @brief Get the name of this solver
        */
        std::string getName() const override;

        /**
        * @brief Get the description of this solver
        */
        std::string getDescription() const override;

        /**
        * @brief Get a (user-readable) hint for this solver
        */
        std::string getHint() const override;

        /**
         * @brief construct a new BasicNewtonIterator
         * @details caller is responsible for deleting memory returned.
         */
        Solver *construct(ExecutableModel *model) const override;

        /**
         * @brief solve the currently defined problem
         */
        double solve() override;

    };
}

#endif //ROADRUNNER_BASICNEWTONITERATION_H
