//
// Created by Ciaran on 08/03/2021.
//

#ifndef ROADRUNNER_NLEQSOLVER_H
#define ROADRUNNER_NLEQSOLVER_H

#include "rrExporter.h"
#include "SteadyStateSolver.h"
#include "LLVMExecutableModel.h"

namespace rr {

    /**
     * @brief Base class for NLEQ type steady state solvers.
     */
    class RR_DECLSPEC NLEQSolver : public SteadyStateSolver {
    public:
        using SteadyStateSolver::SteadyStateSolver;

        /**
        * Creates a new Instance of NLEQ for the given Model
        */
        explicit NLEQSolver(ExecutableModel *model);

        ~NLEQSolver() override;

        /**
        * @brief Called whenever a new model is loaded to allow integrator
        * to reset internal state
        */
        void syncWithModel(ExecutableModel *m) override;

        void loadConfigSettings() override;

        /**
        * @brief Reset all integrator settings to their respective default values
        */
        void resetSettings() override;

        /**
         * @brief main solver method.
         * @details NLEQ1 and NLEQ2 have the same interface, which
         * means we only need 1 method for both.
         * @param which either "nleq1" or "nleq2"
         */
        template<class NLEQSolverType>
        double solveNLEQ() {

            int size = mModel->getStateVector(nullptr);
            double* states = new double[size];
            for (int i=0; i<size; i++)
                states[i] = i;
            mModel->getStateVector(states);

            delete[] states;

            auto nleq = std::unique_ptr<NLEQSolverType>( new NLEQSolverType(mModel));
            rrLog(Logger::LOG_DEBUG) << "NLEQSolver::solve: " << std::endl;
            nleq->allowPreSim = getValue("allow_presimulation");
            nleq->preSimMaximumSteps = getValue("presimulation_maximum_steps");
            nleq->preSimTime = getValue("presimulation_time");
            nleq->allowApprox = getValue("allow_approx");
            nleq->approxTolerance = getValue("approx_tolerance");
            nleq->approxMaximumSteps = getValue("approx_maximum_steps");
            nleq->approxTime = getValue("approx_time");
            nleq->relativeTolerance = getValue("relative_tolerance");
            nleq->maxIterations = getValue("maximum_iterations");
            nleq->minDamping = getValue("minimum_damping");
            nleq->broyden = getValue("broyden_method");
            nleq->linearity = getValue("linearity");

            double result = nleq->solve();
            return result;
        }

    };

}


#endif //ROADRUNNER_NLEQSOLVER_H
