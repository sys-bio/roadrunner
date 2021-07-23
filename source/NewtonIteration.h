//
// Created by Ciaran on 26/02/2021.
//

#ifndef ROADRUNNER_NEWTONITERATION_H
#define ROADRUNNER_NEWTONITERATION_H

#include "KinsolSteadyStateSolver.h"
#include <kinsol/kinsol.h>           /* access to KINSOL func., consts. */
#include <nvector/nvector_serial.h>  /* access to serial N_Vector       */
#include <sundials/sundials_types.h> /* defs. of realtype, sunindextype */

namespace rr {

    class RR_DECLSPEC NewtonIteration : public KinsolSteadyStateSolver {

    public:

        using KinsolSteadyStateSolver::KinsolSteadyStateSolver;

        ~NewtonIteration() override ;

        explicit NewtonIteration(ExecutableModel *executableModel);

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
         * @brief reset settings back to default.
         */
        void resetSettings() override;

        void getSolverStatsFromKinsol() override;

    protected:
        /**
         * @brief solve the currently defined problem
         */
        double solveNewtonIteration(const std::string& strategy);

        void createKinsol() override;

        void freeKinsol() override;

        void updateKinsol() override;

        /**
         * @brief matrix for jacobian
         */
        SUNMatrix jac = nullptr;

        /**
         * @brief linear solver for newton methods
         */
        SUNLinearSolver linearSolver = nullptr;
    };


}


#endif //ROADRUNNER_NEWTONITERATION_H
