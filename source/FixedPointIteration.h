//
// Created by Ciaran on 26/02/2021.
//

#ifndef ROADRUNNER_FIXEDPOINTITERATION_H
#define ROADRUNNER_FIXEDPOINTITERATION_H

#include "KinsolSteadyStateSolver.h"
#include <kinsol/kinsol.h>           /* access to KINSOL func., consts. */
#include <nvector/nvector_serial.h>  /* access to serial N_Vector       */
#include <sundials/sundials_types.h> /* defs. of sunrealtype, sunindextype */

namespace rr {

    /** @class FixedPointIteration
    * @brief Implementation of the fixed point iteration form of the kinsol steady state solver.
    */
    class RR_DECLSPEC FixedPointIteration : public KinsolSteadyStateSolver {

    public:
        using KinsolSteadyStateSolver::KinsolSteadyStateSolver;

        explicit FixedPointIteration(ExecutableModel* executableModel) ;

        ~FixedPointIteration() noexcept override;
        /**
        * 
        * @brief Get the name of this solver
        */
        std::string getName() const override;

        /**
        * 
        * @brief Get the description of this solver
        */
        std::string getDescription() const override;

        /**
        * 
        * @brief Get a (user-readable) hint for this solver
        */
        std::string getHint() const override;

        double solve() override;

    public:
        
        void createKinsol() override;

        void freeKinsol() override;

        void updateKinsol() override;

        void resetSettings() override;

    };

}


#endif //ROADRUNNER_FIXEDPOINTITERATION_H
