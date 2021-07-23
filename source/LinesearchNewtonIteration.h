//
// Created by Ciaran on 26/03/2021.
//

#ifndef ROADRUNNER_LINESEARCHNEWTONITERATION_H
#define ROADRUNNER_LINESEARCHNEWTONITERATION_H

#include "NewtonIteration.h"

namespace rr {

    /**
     * @brief Solve for steady state using Kinsol's implementation 
     * of Newton Iteration with linesearch globalization
     * @see BasicNewtonIteration
     * @code
     *  RoadRunner rr(sbml);
     *  rr.setSteadyStateSolver("newton_linesearch");
     *  rr.solve();
     * @endcode
     */
    class LinesearchNewtonIteration : public NewtonIteration {

    public:
        using NewtonIteration::NewtonIteration;

        ~LinesearchNewtonIteration() override = default ;

        explicit LinesearchNewtonIteration(ExecutableModel *executableModel);

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
         * @brief construct a new LinesearchNewtonIteration
         * @details caller is responsible for deleting memory returned.
         */
        Solver *construct(ExecutableModel *model) const override;

        /**
         * @brief solve the currently defined problem
         */
        double solve() override;
        
    };

}

#endif //ROADRUNNER_LINESEARCHNEWTONITERATION_H
