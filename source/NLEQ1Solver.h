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
#include "SteadyStateSolver.h"
#include "rrExecutableModel.h"
using std::vector;

namespace rr
{

/**
 * @internal
 */
class RR_DECLSPEC NLEQ1Solver : public SteadyStateSolver
{
    public:
        /**
        * Creates a new Instance of NLEQ1 for the given Model
        */
        NLEQ1Solver(ExecutableModel *_model = NULL);
        ~NLEQ1Solver();

        /**
        * @author JKM
        * @brief Called whenever a new model is loaded to allow integrator
        * to reset internal state
        */
        virtual void syncWithModel(ExecutableModel* m);

        void loadConfigSettings();

        /**
        * @author JKM
        * @brief Reset all integrator settings to their respective default values
        */
        void resetSettings();

        /**
        * @author WBC, JKM
        * @brief Get the name for this Solver
        * @note Delegates to @ref getName
        */
        std::string getName() const;

        /**
        * @author JKM
        * @brief Get the name for this Solver
        */
        static std::string getNLEQName();

        /**
        * @author WBC
        * @brief Get the description for this Solver
        * @note Delegates to @ref getDescription
        */
        std::string getDescription() const;

        /**
        * @author JKM
        * @brief Get the description for this Solver
        */
        static std::string getNLEQDescription();

        /**
        * @author WBC
        * @brief Get the hint for this Solver
        * @note Delegates to @ref getHint
        */
        std::string getHint() const;

        /**
        * @author JKM
        * @brief Get the hint for this Solver
        */
        static std::string getNLEQHint();

        // ** Solver routines
        double solve();

    private:
        ExecutableModel *model; // Model generated from the SBML. Static so we can access it from standalone function
};


    // ** Registration *********************************************************


    class NLEQ1SolverRegistrar : public SteadyStateSolverRegistrar {
        public:
            /**
            * @author JKM
            * @brief Gets the name associated with this integrator type
            */
            virtual std::string getName() const {
                return NLEQ1Solver::getNLEQName();
            }

            /**
            * @author JKM
            * @brief Gets the description associated with this integrator type
            */
            virtual std::string getDescription() const {
                return NLEQ1Solver::getNLEQDescription();
            }

            /**
            * @author JKM
            * @brief Gets the hint associated with this integrator type
            */
            virtual std::string getHint() const {
                return NLEQ1Solver::getNLEQHint();
            }

            /**
            * @author JKM
            * @brief Constructs a new integrator of a given type
            */
            virtual SteadyStateSolver* construct(ExecutableModel *model) const {
                return new NLEQ1Solver(model);
            }
    };
}

#endif
