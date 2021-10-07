//
// Created by Ciaran Welsh on 09/06/2021.
//

#include "ForwardSensitivitySolver.h"
#include "ForwardSensitivitySolver.h"
#include "LLVMExecutableModel.h"
#include <nvector/nvector_serial.h>               /* access to serial N_Vector                    */
#include <cvodes/cvodes.h>                        /* prototypes for CVODE fcts., consts.          */
#include <nvector/nvector_serial.h>               /* access to serial N_Vector                    */
#include "sundials/sundials_dense.h"
#include "sundials/sundials_linearsolver.h"
#include "sundials/sundials_nonlinearsolver.h"
#include "sunnonlinsol/sunnonlinsol_newton.h"
#include "sunnonlinsol/sunnonlinsol_fixedpoint.h"
#include "sunmatrix/sunmatrix_dense.h"
#include "sunlinsol/sunlinsol_dense.h"
#include <sundials/sundials_types.h>

#include <utility>
#include "Matrix.h"
#include "Matrix3D.h"

namespace rr {

#define FFSHandleError(errorCode) { \
    std::string _what = decodeSundialsError<ForwardSensitivitySolver>(this, err, true);\
    std::ostringstream _os;\
    _os << "Error: " << __FUNC__ <<": " << _what;\
    throw std::logic_error(_os.str());                                \
    }

    int FFSDyDtFcn(realtype time, N_Vector cv_y, N_Vector cv_ydot, void *userData) {
        double *y = NV_DATA_S(cv_y);
        double *ydot = NV_DATA_S(cv_ydot);
        ForwardSensitivitySolver *inst = (ForwardSensitivitySolver *) userData;

        assert(inst && "userData pointer is NULL in cvode dydt callback");

        ExecutableModel *model = inst->getModel();

        // for cases when Ns < Np we should ensure p maps to the indexes in plist
        // just in case cvodes only reads the first Ns values (aka uses positional index)
        std::vector<double> pVector = inst->getParameterValuesFromPlist();
        model->setGlobalParameterValues(inst->Ns, inst->plist.data(), pVector.data());

        model->getStateVectorRate(time, y, ydot);

        if (!inst->stateVectorVariables &&
            inst->getStateVector() &&
            NV_LENGTH_S(inst->getStateVector()) == 1) {
            ydot[0] = 0.0;
        }

        rrLog(Logger::LOG_TRACE) << __FUNC__ << ", model: " << model;

        return CV_SUCCESS;
    }

    int FFSRootFcn(realtype time, N_Vector y_vector, realtype *gout, void *user_data) {
        ForwardSensitivitySolver *cvInstance = (ForwardSensitivitySolver *) user_data;

        assert(cvInstance && "user data pointer is NULL on CVODE root callback");

        ExecutableModel *model = cvInstance->getModel();

        double *y = NV_DATA_S(y_vector);

        model->getEventRoots(time, y, gout);

        return CV_SUCCESS;
    }

    void ffsErrHandler(int error_code, const char *module, const char *function,
                       char *msg, void *eh_data) {
        ForwardSensitivitySolver *i = (ForwardSensitivitySolver *) eh_data;

        if (error_code < 0) {
            rrLog(Logger::LOG_ERROR) << "ForwardSensitivitySolver Error: " << decodeSundialsError(i, error_code, false)
                                     << ", Module: " << module << ", Function: " << function
                                     << ", Message: " << msg;

        } else if (error_code == CV_WARNING) {
            rrLog(Logger::LOG_WARNING) << "CVODE Warning: "
                                       << ", Module: " << module << ", Function: " << function
                                       << ", Message: " << msg;
        }
    }

    void ForwardSensitivitySolver::setDefaultWhichParameters() {
        Ns = Np;
        whichParameters = getGlobalParameterNames();
    }


    void ForwardSensitivitySolver::constructorOperations() {

        if (mModel) {

            if (!cvodeIntegrator) {
                cvodeIntegrator = std::make_unique<CVODEIntegrator>(mModel);
            }

            // free old cvode memory in favor of new
            // we ensure we free sundials objects before
            // modifying values of Np and Ns.
            if (cvodeIntegrator->mCVODE_Memory) {
                cvodeIntegrator->freeSundialsMemory();
            }
            if (mSensitivityMatrix) {
                freeSundialsMemory();
            }

            numModelVariables = mModel->getStateVector(nullptr); // returns num when arg is null
            Np = mModel->getNumGlobalParameters();

            // when no argument to whichParameter specified, so we assume all parameters
            if (usingDefaultWhichParameters) {
                setDefaultWhichParameters();
            } else {
                Ns = (int) whichParameters.size();
            }
            // figure out indexes of parameters that we want sens for
            deducePlist();

            // populate p and pbar
            p = getModelParametersAsVector();
            pbar = getModelParametersAsVector(); // scaling factors, use value scaling. See sundials docs.

            // pbar are scaling factors and therefore cannot be 0. However,
            // we still need the flexibility to be able to turn model
            // parameters to 0.
            for (double &i : pbar) {
                if (i == 0) {
                    i = 1e-30; // arbitrarily small. Doesn't matter since 0/x=0
                }
            }

            if (settings.empty()) {
                resetSettings();
            }

            ForwardSensitivitySolver::create();
        }
    }

    ForwardSensitivitySolver::ForwardSensitivitySolver(ExecutableModel *executableModel)
            : TimeSeriesSensitivitySolver(executableModel) {
        // improves readibility in some places
        // to have two flags here, even though
        // they are always inverses
        usingDefaultWhichParameters = true;
        usingUseWhichParametersFromConstructor = false;
        constructorOperations();
    }

    ForwardSensitivitySolver::ForwardSensitivitySolver(ExecutableModel *executableModel,
                                                       std::vector<std::string> whichParameters)
            : TimeSeriesSensitivitySolver(executableModel), whichParameters(std::move(whichParameters)) {
        usingDefaultWhichParameters = false;
        usingUseWhichParametersFromConstructor = true;
        constructorOperations();
    }

    double ForwardSensitivitySolver::integrate(double tStart, double hstep) {
        return cvodeIntegrator->integrate(tStart, hstep);
    }


    ForwardSensitivitySolver::~ForwardSensitivitySolver() {
        freeSundialsMemory();
    }

    void ForwardSensitivitySolver::freeSundialsMemory() {
        if (!mModel)
            return;
        if (NLSsens) {
            SUNNonlinSolFree(NLSsens);
            NLSsens = nullptr;
        }

        if (mSensitivityMatrix) {
            N_VDestroyVectorArray_Serial(mSensitivityMatrix, mSensitivityMatrixSize);
            mSensitivityMatrix = nullptr;
        }
    }

    void ForwardSensitivitySolver::create() {
        if (!mModel) {
            return;
        }

        assert(cvodeIntegrator->mStateVector == nullptr && cvodeIntegrator->mCVODE_Memory == nullptr &&
               "calling create, but cvode objects already exist");

        // still need cvode state std::vector size if we have no vars, but have
        // events, needed so root finder works.
        int allocStateVectorSize = 0;
        // when argument is null, returns size of state std::vector (see rrExecutableModel::getStateVector)
        int realStateVectorSize = mModel->getStateVector(nullptr);

        // cvode return code
        int err;

        if (realStateVectorSize > 0) {
            cvodeIntegrator->stateVectorVariables = true;
            allocStateVectorSize = realStateVectorSize;
        } else if (mModel->getNumEvents() > 0) {
            allocStateVectorSize = 1;
            cvodeIntegrator->stateVectorVariables = false;
        } else {
            cvodeIntegrator->stateVectorVariables = false;
            return;
        }

        // allocate and init the cvode arrays
        cvodeIntegrator->mStateVector = N_VNew_Serial(allocStateVectorSize);
        cvodeIntegrator->variableStepPostEventState.resize(allocStateVectorSize);

        // set mStateVector to the values that are currently in the model
        std::vector<double> states(allocStateVectorSize);
        mModel->getStateVector(states.data());

        auto getArrayPtr = cvodeIntegrator->mStateVector->ops->nvgetarraypointer;
        for (int i = 0; i < allocStateVectorSize; i++) {
            getArrayPtr(cvodeIntegrator->mStateVector)[i] = states[i];
        }

        if ((bool) getValue("stiff")) {
            rrLog(Logger::LOG_INFORMATION) << "using stiff integrator";
            cvodeIntegrator->mCVODE_Memory = (void *) CVodeCreate(CV_BDF);
        } else {
            rrLog(Logger::LOG_INFORMATION) << "using non-stiff integrator";
            cvodeIntegrator->mCVODE_Memory = (void *) CVodeCreate(CV_ADAMS);
        }

        assert(cvodeIntegrator->mCVODE_Memory && "could not create Cvode, CVodeCreate failed");

        if ((err = CVodeSetErrHandlerFn(cvodeIntegrator->mCVODE_Memory, ffsErrHandler, this)) !=
            CV_SUCCESS) {
            FFSHandleError(err);
        }

        // use non default CVODE value here, default is too short
        // for some sbml tests.
        CVodeSetMaxNumSteps(cvodeIntegrator->mCVODE_Memory, cvodeIntegrator->mDefaultMaxNumSteps);

        double t0 = 0.0;

        if ((err = CVodeSetUserData(cvodeIntegrator->mCVODE_Memory, (void *) this)) != CV_SUCCESS) {
            FFSHandleError(err);
        }

        if ((err = CVodeInit(cvodeIntegrator->mCVODE_Memory, FFSDyDtFcn, t0, cvodeIntegrator->mStateVector)) !=
            CV_SUCCESS) {
            FFSHandleError(err);
        }

        if (mModel->getNumEvents() > 0) {
            if ((err = CVodeRootInit(cvodeIntegrator->mCVODE_Memory, mModel->getNumEvents(), FFSRootFcn)) !=
                CV_SUCCESS) {
                FFSHandleError(err);
            }
            rrLog(Logger::LOG_TRACE) << "CVRootInit executed.....";
        }

        /**
         * note the cvDirectDemo.c example from sundials is useful
         * to see the different cvode options in action
         */

        /* create fixed point nonlinear solver object */
        if ((bool) getValue("stiff")) {
            // as per the cvode docs (look closely at docs for CVodeCreate)
            // we use the default Newton iteration for stiff

            cvodeIntegrator->nonLinSolver = SUNNonlinSol_Newton(cvodeIntegrator->mStateVector);

            if (cvodeIntegrator->nonLinSolver == nullptr) {
                throw std::runtime_error("CVODEIntegrator::createCVODE: nonLinearSolver_ is nullptr\n");
            }

            if ((err = CVodeSetNonlinearSolver(cvodeIntegrator->mCVODE_Memory, cvodeIntegrator->nonLinSolver)) !=
                CV_SUCCESS) {
                FFSHandleError(err);
            }

            // the newton method requires use of a linear solver, which we set up here.
            cvodeIntegrator->jac = SUNDenseMatrix(allocStateVectorSize, allocStateVectorSize);
            cvodeIntegrator->linSolver = SUNLinSol_Dense(cvodeIntegrator->mStateVector, cvodeIntegrator->jac);
            if (cvodeIntegrator->linSolver == nullptr) {
                throw std::runtime_error("CVODEIntegrator::createCVODE: call to SunLinSol_Dense returned nullptr. "
                                         "The size of the sundials matrix (created with SUNDenseMatrix) used for the jacobian "
                                         "(" + std::to_string(allocStateVectorSize) + "x" +
                                         std::to_string(allocStateVectorSize)
                                         + ") is inappropriate for this model\n");
            }
            if ((err = CVodeSetLinearSolver(cvodeIntegrator->mCVODE_Memory, cvodeIntegrator->linSolver,
                                            cvodeIntegrator->jac)) !=
                CV_SUCCESS) {
                FFSHandleError(err);
            };

            // Use a difference quotient Jacobian by not passing Jac to CVodeSetJacFn
            if ((err = CVodeSetJacFn(cvodeIntegrator->mCVODE_Memory, nullptr)) != CV_SUCCESS) {
                FFSHandleError(err);
            }
        } else {
            // and fixed point solver (which used to be called functional iteration)
            // for nonstiff problems
            cvodeIntegrator->nonLinSolver = SUNNonlinSol_FixedPoint(cvodeIntegrator->mStateVector, 0);
            if ((err = CVodeSetNonlinearSolver(cvodeIntegrator->mCVODE_Memory, cvodeIntegrator->nonLinSolver)) !=
                CV_SUCCESS) {
                FFSHandleError(err);
            }
        }
        cvodeIntegrator->setCVODETolerances();
        mModel->resetEvents();

        /**
         * Sensitivity setup
         * Only do this if there are model variables and parameters
         * since its possible to have an empty RoadRunner object
         * from which we create a model using the direct interface.
         */
        if (numModelVariables > 0 && Np > 0) {
            // pointer to N_Vector - i.e. a matrix
            mSensitivityMatrix = N_VCloneVectorArray_Serial(Ns, cvodeIntegrator->mStateVector);
            mSensitivityMatrixSize = Ns; //Need to keep the original--'Ns' may change.
//            mSensitivityMatrixUnique = NVectorArrayPtr(N_VCloneVectorArray_Serial, [](N_Vector* nvec, int num){
//                N_VDestroyVectorArray_Serial(nvec, num);
//            });
            assert(mSensitivityMatrix && "Sensitivity vector is nullptr");

            for (int i = 0; i < Ns; i++) {
                auto data = mSensitivityMatrix[i]->ops->nvgetarraypointer(mSensitivityMatrix[i]);
                for (int j = 0; j < numModelVariables; j++) {
                    data[j] = 0;
                }
            }

            int sensi_meth;
            if (getValue("sensitivity_method") == "simultaneous") {
                sensi_meth = CV_SIMULTANEOUS;
            } else if (getValue("sensitivity_method") == "staggered") {
                sensi_meth = CV_STAGGERED;
            }

            if ((err = CVodeSensInit1(cvodeIntegrator->mCVODE_Memory, Ns, sensi_meth, NULL, mSensitivityMatrix))) {
                FFSHandleError(err);
            }

            if ((err = CVodeSensEEtolerances(cvodeIntegrator->mCVODE_Memory))) {
                FFSHandleError(err);
            }

            if ((err = CVodeSetSensErrCon(cvodeIntegrator->mCVODE_Memory, true))) {
                FFSHandleError(err);
            }

            int dq_method;
            if (getValue("DQ_method") == "forward") {
                dq_method = CV_FORWARD;
            } else if (getValue("DQ_method") == "centered") {
                dq_method = CV_CENTERED;
            }

            if ((err = CVodeSetSensDQMethod(cvodeIntegrator->mCVODE_Memory, dq_method, 1))) {
                FFSHandleError(err);
            }

            if ((err = CVodeSetSensParams(cvodeIntegrator->mCVODE_Memory, p.data(), pbar.data(), plist.data()))) {
                FFSHandleError(err);
            }

            if (sensi_meth == CV_SIMULTANEOUS) {
                if (getValue("nonlinear_solver") == "newton") {
                    NLSsens = SUNNonlinSol_NewtonSens(Ns + 1, cvodeIntegrator->mStateVector);
                }
                if (getValue("nonlinear_solver") == "fixed_point") {
                    NLSsens = SUNNonlinSol_FixedPointSens(Ns + 1, cvodeIntegrator->mStateVector, 1);
                }
            } else if (sensi_meth == CV_STAGGERED) {
                if (getValue("nonlinear_solver") == "newton") {
                    NLSsens = SUNNonlinSol_NewtonSens(Ns, cvodeIntegrator->mStateVector);
                } else {
                    NLSsens = SUNNonlinSol_FixedPointSens(Ns, cvodeIntegrator->mStateVector, 0);
                }
            } else {
                if (getValue("nonlinear_solver") == "newton") {
                    NLSsens = SUNNonlinSol_Newton(cvodeIntegrator->mStateVector);
                } else {
                    NLSsens = SUNNonlinSol_FixedPoint(cvodeIntegrator->mStateVector, 0);
                }
            }

            /* attach nonlinear solver object to CVode */
            if (sensi_meth == CV_SIMULTANEOUS) {
                if ((err = CVodeSetNonlinearSolverSensSim(cvodeIntegrator->mCVODE_Memory, NLSsens))) {
                    FFSHandleError(err);
                }
            } else if (sensi_meth == CV_STAGGERED) {
                if ((err = CVodeSetNonlinearSolverSensStg(cvodeIntegrator->mCVODE_Memory, NLSsens))) {
                    FFSHandleError(err);
                }
            } else {
                if ((err = CVodeSetNonlinearSolverSensStg1(cvodeIntegrator->mCVODE_Memory, NLSsens))) {
                    FFSHandleError(err);
                }
            }
        }
    }

    void ForwardSensitivitySolver::syncWithModel(ExecutableModel *executableModel) {
        if (executableModel) {
            mModel = executableModel;

            // if we have no cvodeIntegrate, we call constructorOperations to
            // create one/
            if (!cvodeIntegrator) {
                constructorOperations();
            }

            cvodeIntegrator->syncWithModel(executableModel);

            // requirement for calling this again is a code smell.
            // The issue is that we need FFS to work both when using
            // the constructor and when you have no model followed
            // by a call to syncWithModel.
            // This added complexity is a little akward but we get around
            // the issue by calling constructorOperations again here.
            constructorOperations();
        }
    }

    std::string ForwardSensitivitySolver::getName() const {
        return "forward";
    }

    std::string ForwardSensitivitySolver::getDescription() const {
        return "Interface to the sundials cvodes forward sensitivities solver";
    }

    std::string ForwardSensitivitySolver::getHint() const {
        return "forward sensitivities solver";
    }

    Solver *ForwardSensitivitySolver::construct(ExecutableModel *executableModel) const {
        return new ForwardSensitivitySolver(executableModel);
    }

    std::string ForwardSensitivitySolver::toString() const {
        std::stringstream ss;
        ss << "< roadrunner.SensitivitySolver() >\n";
        ss << "  name: " << getName() << "\n";
        ss << "  settings:\n";
        ss << getSettingsRepr();
        return ss.str();
    }

    std::string ForwardSensitivitySolver::toRepr() const {
        std::stringstream ss;
        ss << "< roadrunner.SensitivitySolver() \"" << getName() << "\" " << settingsPyDictRepr() << " >\n";
        return ss.str();
    }

    void ForwardSensitivitySolver::resetSettings() {
        // reuse settings from cvodeIntegrator since cvodes is a superset of cvode
        settings.insert(cvodeIntegrator->getSettingsMap().begin(), cvodeIntegrator->getSettingsMap().end());
        sorted_settings = cvodeIntegrator->sorted_settings;
        hints.insert(cvodeIntegrator->hints.begin(), cvodeIntegrator->hints.end());
        descriptions.insert(cvodeIntegrator->descriptions.begin(), cvodeIntegrator->descriptions.end());

        addSetting("nonlinear_solver", Setting("newton"), "Non-Linear Sovler", "newton or fixed_point",
                   "which non-linear solver to use for sensitivities");
        addSetting("sensitivity_method", Setting("simultaneous"), "Sensitivity Method", "simultaneous or staggered",
                   "Solve the sensitivity equations simultaneously with state variable equations "
                   "or stagger them");
        addSetting("DQ_method", Setting("centered"), "DQ method", "centered or forward",
                   "Use the centered or forward difference quotient approximation of the sensitivities rhs");
    }

    void ForwardSensitivitySolver::loadConfigSettings() {
        cvodeIntegrator->loadConfigSettings();
    }

    std::vector<std::string> ForwardSensitivitySolver::getGlobalParameterNames() {
        std::vector<std::string> out;
        for (int i = 0; i < Np; i++) {
            std::string name = mModel->getGlobalParameterId(i);
            out.push_back(name);
        }
        return out;
    }

    std::vector<std::string> ForwardSensitivitySolver::getVariableNames() {
        std::vector<std::string> out;
        for (int i = 0; i < numModelVariables; i++) {
            std::string name = mModel->getStateVectorId(i);
            out.push_back(name);
        }
        return out;
    }

    ParameterMap ForwardSensitivitySolver::getModelParametersAsMap() {
        ParameterMap m;
        int *idx = new int[Np];
        for (int i = 0; i < Np; i++) idx[i] = i;
        double *vals = new double[Np];
        mModel->getGlobalParameterValues(Np, idx, vals);
        // first get the parameter ids
        for (int i = 0; i < Np; i++) {
            std::string name = mModel->getGlobalParameterId(i);
            m.insert(std::pair<std::string, double>(name, vals[i]));
        }
        delete[] idx;
        delete[] vals;
        return m;
    }

    std::vector<double> ForwardSensitivitySolver::getModelParametersAsVector() {
        std::vector<double> vec(Np);
        auto *idx = new int[Np];
        auto *vals = new double[Np];
        for (int i = 0; i < Np; i++) {
            idx[i] = i;
        }
        mModel->getGlobalParameterValues(Np, idx, vals);
        for (int i = 0; i < Np; i++) {
            vec[i] = vals[i];
        }
        delete[] idx;
        delete[] vals;
        return vec;
    }

    void ForwardSensitivitySolver::deducePlist() {
        // collect valid strings for validating user input
        std::vector<std::string> validParameterStrings(Np);
        for (int i = 0; i < Np; i++) {
            validParameterStrings[i] = mModel->getGlobalParameterId(i);
        }
        // make sure we resize both Ns, and plist
        Ns = (int) whichParameters.size();
        plist.clear();
        plist.resize(whichParameters.size());
        for (int i = 0; i < whichParameters.size(); i++) {
            const std::string &paramName = whichParameters.at(i);
            if (std::find(validParameterStrings.begin(), validParameterStrings.end(), paramName) ==
                validParameterStrings.end()) {
                std::ostringstream err;
                err << "Parameter \"" << paramName << "\" is not a valid parameter. ";
                err << "These are valid parameters: ";
                for (auto para: validParameterStrings) {
                    err << para << ", ";
                }
                throw std::invalid_argument(err.str());
            }
            int idx = mModel->getGlobalParameterIndex(paramName);
            plist[i] = idx;
        }
    }

    N_Vector ForwardSensitivitySolver::getStateVector() {
        return cvodeIntegrator->mStateVector;
    }

    N_Vector *ForwardSensitivitySolver::getSensitivityNVectorPtr() {
        return mSensitivityMatrix;
    }

    rr::Matrix<double> ForwardSensitivitySolver::getSensitivityMatrix(int k) {
        auto stateVector = cvodeIntegrator->mStateVector;
        auto print = stateVector->ops->nvprint;
        auto getArrayPtr = stateVector->ops->nvgetarraypointer;
        auto len = stateVector->ops->nvgetlength;
        double t = mModel->getTime();
        CVodeGetSensDky(cvodeIntegrator->mCVODE_Memory, t, k, mSensitivityMatrix);

        Matrix<double> sensResults(Ns, numModelVariables);

        for (int i = 0; i < Ns; i++) {
            N_Vector arrayVector = mSensitivityMatrix[i];
            for (int j = 0; j < numModelVariables; j++) {
                if (t == 0) {
                    // sensitivities at t=0 are always 0
                    // but they appear as nan instead. Its more
                    // convenient to show 0
                    sensResults(i, j) = 0;
                } else {
                    sensResults(i, j) = getArrayPtr(arrayVector)[j];
                }
            }
        }
        return sensResults;
    }


    void ForwardSensitivitySolver::setValue(const std::string &key, Setting val) {
        if (cvodeIntegrator->getSettingsMap().find(key) != cvodeIntegrator->getSettingsMap().end()) {
            cvodeIntegrator->setValue(key, val);
        } else {
            Solver::setValue(key, val);
        }

        if (key == "stiff") {
            // If the integrator is changed from stiff to standard, we must re-create CVode.
            rrLog(Logger::LOG_INFORMATION) << "Integrator stiffness has been changed. Re-creating CVode.";
            /**
             *
             */
            freeSundialsMemory();
            cvodeIntegrator->freeSundialsMemory();
//            cvodeIntegrator->create();
            create();
        }
    }

    std::vector<std::string> ForwardSensitivitySolver::getParameterNamesFromPlist() {
        std::vector<std::string> pVector(Ns);
        const auto &allParamNames = getGlobalParameterNames();
        for (int i = 0; i < Ns; i++) {
            auto plist_idx = plist[i];
            pVector[i] = allParamNames[plist_idx];
        }
        return pVector;
    }

    std::vector<double> ForwardSensitivitySolver::getParameterValuesFromPlist() {
        std::vector<double> pVector(Ns);
        for (int i = 0; i < Ns; i++) {
            auto plist_idx = plist[i];
            pVector[i] = p[plist_idx];
        }
        return pVector;
    }

    Matrix3D<double, double> ForwardSensitivitySolver::solveSensitivities(
            double start, double stop, int num,
            std::vector<std::string> params,
            std::vector<std::string> species,
            int k) {
        if (!params.empty()) {
            // turn off the indicator flag so that
            // subsequent calls to solveSensitivities
            // accurately deduce plist parameters from params
            usingDefaultWhichParameters = false;
            whichParameters = params;
        } else {
            // in cases where ForwardSensitivitySolver has already
            // been used with an non empty params argument
            // we need to reset the parameters to default
            //
            // this doesn't apply when the user provided the whichParameters argument to
            // the constructor. In this case the usingUseWhichParametersFromConstructor
            // will be true (and the usingDefaultWhichParameters will be false).
            if (!usingUseWhichParametersFromConstructor) {
                whichParameters = getGlobalParameterNames();
            }
            /**
             * Note this is slightly contrived logic that is necessary
             * for enabling users to supply whichParameters in the constructor
             * and here via the params argument. This was an "accident of design"
             * and it would be better to just have the whichParameters variable
             * set in this method only.
             */
        }

        deducePlist();
        cvodeIntegrator->restart(start);
        int intervals = num - 1;
        double stepSize = (stop - start) / intervals;

        Matrix3D<double, double> results(Ns, numModelVariables, num);
        //        std::cout << results.numRows() << "; " << results.numCols() << "; " << results.numZ() << std::endl;

        // collect initial data
        double t = start;
        results.setKthMatrix(0, t, getSensitivityMatrix(k));

        int numberOfPoints = 1;
        for (int i = 1; i < num; i++) {
            t = integrate(t, stepSize);
            results.setKthMatrix(numberOfPoints, t, getSensitivityMatrix(k));
            numberOfPoints++;
        }

        results.setRowNames(whichParameters);
        results.setColNames(getVariableNames());

        // in the situation where user would like to
        // only return result for selected species
        // we now delete unselected species.
        if (!species.empty()){
            // species contains items we want to keep.
            // so we iterate over column names.

            std::vector<std::string> itemsToRemove;
            for (auto& s: results.getColNames()){
                // if an item from column names
                // is not in users chosen species, delete
                // that column
                if (std::find(species.begin(), species.end(), s) == species.end()){
                    // add it to itemsToRemove
                    results.deleteCol(s);
                }
            }

        }

        return results;
    }


}
