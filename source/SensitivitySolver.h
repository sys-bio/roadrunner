//
// Created by Ciaran on 08/07/2021.
//

#ifndef ROADRUNNER_SENSITIVITYSOLVER_H
#define ROADRUNNER_SENSITIVITYSOLVER_H

#include "Solver.h"
#include "Matrix3D.h"


namespace rr {

    template <typename T>
    class Matrix;

    class ForwardSensitivitySolver;

    /**
     * @brief generic interface for all SensitivitySolvers
     */
    class SensitivitySolver : public Solver {
    public:
        using Solver::Solver;

        virtual ~SensitivitySolver() = default;

        /**
         * @brief get current values of sensitivities of model variables
         * to parameters.
         * @param kth derivative of the sensitivities.
         */
        virtual rr::Matrix<double> getSensitivityMatrix(int k = 0) = 0;

    };

    /**
     * @brief generic interface for sensitivity solvers that
     * integrate the model and compute sensitivities at
     * each time point
     */
    class TimeSeriesSensitivitySolver : public SensitivitySolver {
    public:
        using SensitivitySolver::SensitivitySolver;

        ~TimeSeriesSensitivitySolver() override = default;

        /**
         * @brief integrate the model from t0 to t0 + hstep.
         * @note this signature is the same as that found in
         * Integrator().
         * @details integrating the model should update the
         * sensitivities, which are available from getSensitivities()
         */
        virtual double integrate(double t0, double hstep) = 0;

        /**
         * @brief simulate a timeseries with sensitivities from start to step with num
         * data points.
         * @details Matrix3D indexed by time. Each element of the 3D matrix is a
         * Matrix<double> with rows and columns parameters and model variables respectively.
         * The parameter k determines the kth order derivative of the sensitivity information
         * that will be returned
         * @param start starting time for time series simulation
         * @param stop last time point for time series simulation
         * @param num number of data points to simulate. Determines Z of Matrix3D.
         * @param params vector of parameters that you want sensitivity for. When empty (default), compute
         * sensitivities for all parameters vs all variables.
         * @param k (default 0) return the kth other derivative of the sensitivity data.
         */
        virtual Matrix3D<double, double> solveSensitivities(
                double start, double stop, int num,
                const std::vector<std::string> &params = std::vector<std::string>(),
                int k = 0) = 0;

    };

    /**
     * @brief genetic interface for sensitivity solvers that
     * solve for steady state before computing model sensitivities
     */
    class SteadyStateSensitivitySolver : public SensitivitySolver {
    public:
        using SensitivitySolver::SensitivitySolver;

        ~SteadyStateSensitivitySolver() override = default;

        /**
         * @brief solves the model for steady state.
         * @details this method is called prior to computing sensitivities
         * to get sensitivities at steady state.
         */
        virtual double solveSteadyState() = 0;

        /**
         * @brief compute sensitivities at steady state
         * @param params which parameters do you want sensitivities for? Each
         * string must be a valid model parameter. If left empty (default)
         * then all model parameters are selected.
         * @param k returned double matrix will contain the kth order derivative
         * of the sensitivities, where 0 (default) is just the sensitivity values
         * @return the k'th order derivitive of the Sensitivities for this model
         * at steady state. Rows (number of model variables), columns (Np, number
         * of parameters user has requested sensitivites for).
         */
        virtual rr::Matrix<double> solveSensitivities(
                const std::vector<std::string> &params = std::vector<std::string>(),
                        int k = 0) = 0;

    };

}

#endif //ROADRUNNER_SENSITIVITYSOLVER_H
