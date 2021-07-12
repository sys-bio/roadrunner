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
         * @brief static version of getName
         */
        static std::string getLinesearchNewtonIterationName();

        /**
         * @brief static version of getDescription
         */
        static std::string getLinesearchNewtonIterationDescription();

        /**
         * @brief static version of getHint
         */
        static std::string getLinesearchNewtonIterationHint();

        /**
         * @brief solve the currently defined problem
         */
        double solve() override;
        
    };
    
    
    class LinesearchNewtonIterationRegistrar : public SteadyStateSolverRegistrar {
    public:
        /**
        * @brief Gets the name associated with this integrator type
        */
        std::string getName() const override {
            return LinesearchNewtonIteration::getLinesearchNewtonIterationName();
        }

        /**
        * @brief Gets the description associated with this integrator type
        */
        std::string getDescription() const override {
            return LinesearchNewtonIteration::getLinesearchNewtonIterationDescription();
        }

        /**
        * @brief Gets the hint associated with this integrator type
        */
        std::string getHint() const override {
            return LinesearchNewtonIteration::getLinesearchNewtonIterationHint();
        }

        /**
        * @brief Constructs a new integrator of a given type
        */
        SteadyStateSolver *construct(ExecutableModel *model) const override {
            return new LinesearchNewtonIteration(model);
        }
    };


}

#endif //ROADRUNNER_LINESEARCHNEWTONITERATION_H
