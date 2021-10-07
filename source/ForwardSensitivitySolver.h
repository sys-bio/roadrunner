//
// Created by Ciaran Welsh on 09/06/2021.
//

#ifndef ROADRUNNER_SENSITIVITIES_H
#define ROADRUNNER_SENSITIVITIES_H

#include <iostream>
#include "CVODEIntegrator.h"
#include "SensitivitySolver.h"
#include <unordered_map>
#include <vector>
#include "LLVMExecutableModel.h"
#include "Matrix.h"
#include "Matrix3D.h"
#include "nvector/nvector_serial.h"

namespace rr {

    using ParameterMap = std::unordered_map<std::string, double>;

    class CVODEIntegrator;


    /**
     * @brief Time based sensivitity solver.
     * @details Uses CVODEIntegrator to integrate the ExecutableModel
     * and cvodes to compute sensitivity information at each time point.
     * Implements the TimeSeriesSensitivitySolver interface
     */
    class ForwardSensitivitySolver : public TimeSeriesSensitivitySolver {
    public:

        ForwardSensitivitySolver() = default;

        ~ForwardSensitivitySolver() override;

        explicit ForwardSensitivitySolver(ExecutableModel *executableModel);

        ForwardSensitivitySolver(ExecutableModel *executableModel, std::vector<std::string> whichParameters);

        double integrate(double tStart, double hstep) override;


        /**
         * @brief instantiate the code necessary to use cvodes
         */
        void create();

        /**
         * @brief free sundials memory associated with sensitivities
         */
        void freeSundialsMemory();

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

        Solver *construct(ExecutableModel *executableModel) const override;

        /**
         * @brief resets all settings back to default values
         */
        void resetSettings() override;

        void syncWithModel(ExecutableModel *executableModel) override;

        std::string toRepr() const override;

        void loadConfigSettings();

        std::string toString() const override;

        /**
         * @brief returns a vector of global parameter names
         * extracted from the model
         */
        std::vector<std::string> getGlobalParameterNames();

        /**
         * @brief get a vector of variable names in the order that
         * they appear in the model.
         * @details variables refers to each equation in the model
         * and are obtained with ExecutableModel::getStateVectorId.
         */
        std::vector<std::string> getVariableNames();

        /**
         * @brief get global parameters as an unordered map, strings as
         * keys and parameter values as values
         */
        ParameterMap getModelParametersAsMap();

        /**
         * @brief return a std::vector<double> of model parameters
         * in the order they appear in the model.
         */
        std::vector<double> getModelParametersAsVector();

        /**
         * @brief returns the indexes of parameters that user wants
         * sensitivities for, based off of whichParameters.
         */
        void deducePlist();

        /**
         * @brief retuns pointer to the state vector
         * used by sundials for solving ODE.
         */
        N_Vector getStateVector();

        /**
         * @brief retuns pointer to the state vector
         * used by sundials for storing sensitivity matrix.
         */
        N_Vector *getSensitivityNVectorPtr();

        /**
         * @brief get current values of sensitivities of model variables
         * to parameters.
         * @param kth derivative of the sensitivities.
         */
        rr::Matrix<double> getSensitivityMatrix(int k = 0);

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
         * @note for developers. Pass by value for the params is easier to wrap with swig.
         */
        Matrix3D<double, double> solveSensitivities(double start, double stop, int num,
                                                    std::vector<std::string> params = std::vector<std::string>(),
                                                    std::vector<std::string> species = std::vector<std::string>(),
                                                    int k = 0);

        void setValue(const std::string &key, Setting val);

        /**
         * @brief parameters in the model as a member variable
         * which enables passing the underlying data pointer to
         * sundials for finite difference approx
         * @details This is the full vector of model global
         * parameters in the order indexed in the model.
         * @see plist
         */
        std::vector<double> p;

        /**
         * @brief scaling factors. Set to the values of
         * the parameters for sensitivity. This should
         * be identical to p as acceptible default.
         */
        std::vector<double> pbar;

        /**
         * @brief which parameters to get sensitivity for as int?
         * @details indexes parameters in p and pbar.
         */
        std::vector<int> plist;

        /**
         * @brief which parameters to get sensitivities for, as strings
         */
        std::vector<std::string> whichParameters;

        /**
         * @brief a map containing model parameter names to values
         */
        ParameterMap globalParameterMap;

        /**
         * @brief number of global parameters in the model
         */
        int Np = 0;

        /**
         * @brief number of parameters we want to find sensitivities for
         */
        int Ns = 0;

        /**
         * @brief the number of state variables in the model
         * @details aka the size of the mStateVector
         */
        int numModelVariables = 0;

    private:

        /**
         * @brief An instance of CVODEIntegrator is needed
         * for the forward sensitivity solver
         * @details The ForwardSensitivitySolver is in essence a CVODEIntegrator and a
         * SensitivitySolver. However we avoid the diamond inheritance problem that
         * would arise from inheriting from both and the consequent need to use
         * virtual inheritance, which may come with hidden problems;
         */
        std::unique_ptr<CVODEIntegrator> cvodeIntegrator = nullptr;

        void constructorOperations();

        /**
         * @brief set whichParameters to all model parameters in the order
         * they appear in the model.
         * @details whichParameters is a string vector that determines which
         * parameters we compute sensitivities for. When user does not provide
         * explicit parameters for sensitivities, all are used.
         * @details for developers. The whichParameters variable needs
         * updating when a parameter is added or removed via RoadRunner::addParameter or
         * RoadRunner::RemoveParameter.
         */
        void setDefaultWhichParameters();

        /**
         * @brief indicator variable that specifies whether a user has manually
         * chosen which parameters to compute sensitivities for.
         * @details when true, the ForwardSensitivitySolver needs to
         * update the default list of parameters to compute sensitivities for
         * because its possible the user may modify the model between instantiation
         * of the RoadRunner instance and solving for sensitivities.
         * @details this indicator is needed, since the test for empty
         * whichParameter variable only works on instantiation of the FFS class
         * @details inverse of usingUseWhichParametersFromConstructor.
         * Used for readibility in some places - helps prevent need
         * to do mental aerobics with booleans
         */
        bool usingDefaultWhichParameters = true;

        /**
         * @brief indicator variable which is true when
         * the constructor that has "whichParameters" argument
         * is used. This is needed for the dual control over which
         * parameters to solve sensitivities for - construtor and
         * main solveSensitivities method.
         * @details inverse of usingDefaultWhichParameters.
         * Used for readibility in some places - helps prevent need
         * to do mental aerobics with booleans
         */
        bool usingUseWhichParametersFromConstructor = false;

        /**
         * @brief Non-linear solver for sensitivity analysis
         */
        SUNNonlinearSolver NLSsens = nullptr;

        /**
         * @brief place to store the sensitivities
         */
        N_Vector *mSensitivityMatrix = nullptr;
//        using NVectorArrayPtr = std::unique_ptr<N_Vector*, std::function<void( N_Vector*, int)>>;
//        using NVectorArrayPtr = std::unique_ptr<N_Vector*, decltype(&N_VDestroyVectorArray_Serial)>;
//        NVectorArrayPtr mSensitivityMatrixUnique = std::make_unique<N_Vector*, decltype(&N_VDestroyVectorArray_Serial)>(nullptr);

        int mSensitivityMatrixSize = 0;

        /**
         * @brief indicator for whether model has state vector variables or not
         * @details mirrors CVODEIntegrator
         */
        bool stateVectorVariables = false;

        /**
         * @brief get the parameter names that correspond to the current values of plist
         */
        std::vector<std::string> getParameterNamesFromPlist();

        /**
         * @brief get the parameter values that correspond to the current values of plist
         */
        std::vector<double> getParameterValuesFromPlist();

        friend int FFSDyDtFcn(realtype time, N_Vector cv_y, N_Vector cv_ydot, void *userData);

        friend int FFSRootFcn(realtype time, N_Vector y_vector, realtype *gout, void *user_data);

    };



}

#endif //ROADRUNNER_SENSITIVITIES_H
