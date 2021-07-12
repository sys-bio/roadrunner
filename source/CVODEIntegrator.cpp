#pragma hdrstop

#include "CVODEIntegrator.h"
#include "rrExecutableModel.h"
#include "rrException.h"
#include "rrLogger.h"
#include "rrStringUtils.h"
#include "rrException.h"
#include "rrConfig.h"
#include "rrUtils.h"


#include <cvodes/cvodes.h>                          /* prototypes for CVODE fcts., consts.          */
#include <nvector/nvector_serial.h>               /* access to serial N_Vector                    */
#include <sunnonlinsol/sunnonlinsol_fixedpoint.h> /* access to the fixed point SUNNonlinearSolver */
#include <sunnonlinsol/sunnonlinsol_newton.h> /* access to the fixed point SUNNonlinearSolver */
#include <sunlinsol/sunlinsol_dense.h>

#include <cstring>
#include <iomanip>
#include <cmath>
#include <map>
#include <algorithm>
#include <limits>
#include <cassert>
#include <Poco/Logger.h>

#include <iostream>

#define CVODE_INT_TYPECODE 0x7799ff00

namespace rr {
    const int CVODEIntegrator::mDefaultMaxNumSteps = 20000;
    const int CVODEIntegrator::mDefaultMaxAdamsOrder = 12;
    const int CVODEIntegrator::mDefaultMaxBDFOrder = 5;

    int cvodeDyDtFcn(realtype t, N_Vector cv_y, N_Vector cv_ydot, void *userData);

    int cvodeRootFcn(realtype t, N_Vector y, realtype *gout, void *userData);

    // Sets the value of an element in a N_Vector object
    inline void SetVector(N_Vector v, int Index, double Value) {
        double *data = NV_DATA_S(v);
        data[Index] = Value;
    }

    inline double GetVector(N_Vector v, int Index) {
        double *data = NV_DATA_S(v);
        return data[Index];
    }

    /**
    * Purpose
    * This function processes error and warning messages from CVODE and its
    * sub-modules.
    *
    * Arguments:
    * error_code: is the error code.
    * module: is the name of the CVODE module reporting the error.
    * function: is the name of the function in which the error occurred.
    * msg: is the error message.
    * eh_data: is a pointer to user data, the same as the eh_data parameter
    * passed to CVodeSetErrHandlerFn.
    * Return value: A CVErrHandlerFn function has no return value.
    *
    * Notes:
    * error_code is negative for errors and positive (CV_WARNING) for warnings.
    * If a function that returns a pointer to memory encounters an error,
    * it sets error_code to 0.
    */
    static void cvodeErrHandler(int error_code, const char *module, const char *function, char *msg, void *eh_data);

    /**
    * macro to throw a (hopefully) usefull error message
    */
#define handleCVODEError(errCode) \
    { std::string _err_what = std::string("CVODE Error: ") + \
    cvodeDecodeError(errCode); \
    throw IntegratorException(_err_what, std::string(__FUNC__)); }


    CVODEIntegrator::CVODEIntegrator(ExecutableModel *aModel)
            :
            Integrator(aModel),
            mStateVector(nullptr),
            mCVODE_Memory(nullptr),
            lastEventTime(0),
            stateVectorVariables(false),
            variableStepPendingEvent(false),
            variableStepTimeEndEvent(false),
            typecode_(CVODE_INT_TYPECODE) {
        rrLog(Logger::LOG_INFORMATION) << "creating CVODEIntegrator";

        CVODEIntegrator::resetSettings();

        if (aModel) {
            createCVode();

            // allocate space for the event status array
            eventStatus = std::vector<unsigned char>(mModel->getEventTriggers(0, nullptr, nullptr), false);
        }

        // Update parameter settings within CVODE.
        updateCVODE();
    }

    CVODEIntegrator::~CVODEIntegrator() {
        freeCVode();
    }


    void CVODEIntegrator::setListener(IntegratorListenerPtr p) {
        listener = p;
    }

    IntegratorListenerPtr CVODEIntegrator::getListener() {
        return listener;
    }

    void CVODEIntegrator::checkType() const {
        if (typecode_ != CVODE_INT_TYPECODE)
            throw std::runtime_error("CVODEIntegrator::checkType failed, memory bug");
    }

    void CVODEIntegrator::checkVectorSize(int expected, size_t real) const {
        if (expected != real)
            throw std::runtime_error(
                    "CVODEIntegrator::checkVectorSize failed, real size is " + CVODEIntegrator::ToString(real) +
                    ", whereas expected size is " + CVODEIntegrator::ToString(expected));
    }

    void CVODEIntegrator::checkIndex(int index, int size) const {
        if (index < 0 || index >= size)
            throw std::out_of_range("CVODEIntegrator::checkIndex failed, index " + CVODEIntegrator::ToString(index) +
                                    " out of range from " + CVODEIntegrator::ToString(0) + " to " +
                                    CVODEIntegrator::ToString(size - 1));
    }

    std::string CVODEIntegrator::ToString(int val) const {
        std::stringstream stream;
        stream << val;
        return stream.str();
    }

    std::string CVODEIntegrator::ToString(size_t val) const {
        std::stringstream stream;
        stream << val;
        return stream.str();
    }


    void CVODEIntegrator::resetSettings() {
        Solver::resetSettings();

        // Set default integrator settings.
        addSetting("relative_tolerance", Setting(Config::getDouble(Config::CVODE_MIN_RELATIVE)), "Relative Tolerance",
                   "Specifies the scalar relative tolerance (double).",
                   "(double) CVODE calculates a std::vector of error weights which is used in all error and convergence tests. The weighted RMS norm for the relative tolerance should not become smaller than this value.");
        addSetting("absolute_tolerance", Setting(Config::getDouble(Config::CVODE_MIN_ABSOLUTE)), "Absolute Tolerance",
                   "Specifies the scalar or std::vector absolute tolerance based on amounts (double or double std::vector).",
                   "(double or double std::vector) CVODE calculates a std::vector of error weights which is used in all error and convergence tests. The weighted RMS norm for the absolute tolerance should not become smaller than this value.");
        addSetting("stiff", Setting(true), "Stiff", "Specifies whether the integrator attempts to solve stiff equations. (bool)",
                   "(bool) Specifies whether the integrator attempts to solve stiff equations. Ensure the integrator can solver stiff differential equations by setting this value to true.");
        addSetting("maximum_bdf_order", Setting(mDefaultMaxBDFOrder), "Maximum BDF Order",
                   "Specifies the maximum order for Backward Differentiation Formula integration. (int)",
                   "(int) Specifies the maximum order for Backward Differentiation Formula integration. This integration method is used for stiff problems. Default value is 5.");
        addSetting("maximum_adams_order", Setting(mDefaultMaxAdamsOrder), "Maximum Adams Order",
                   "Specifies the maximum order for Adams-Moulton intergration. (int)",
                   "(int) Specifies the maximum order for Adams-Moulton intergration. This integration method is used for non-stiff problems. Default value is 12.");
        addSetting("maximum_num_steps", Setting(mDefaultMaxNumSteps), "Maximum Number of Steps",
                   "Specifies the maximum number of steps to be taken by the CVODE solver in its attempt to reach tout. (int)",
                   "(int) Maximum number of steps to be taken by the CVODE solver in its attempt to reach tout.");
        addSetting("maximum_time_step", Setting(0.0), "Maximum Time Step",
                   "Specifies the maximum absolute value of step size allowed. (double)",
                   "(double) The maximum absolute value of step size allowed.");
        addSetting("minimum_time_step", Setting(0.0), "Minimum Time Step",
                   "Specifies the minimum absolute value of step size allowed. (double)",
                   "(double) The minimum absolute value of step size allowed.");
        addSetting("initial_time_step", Setting(0.0), "Initial Time Step", "Specifies the initial time step size. (double)",
                   "(double) Specifies the initial time step size. If inappropriate, CVODE will attempt to estimate a better initial time step.");
        addSetting("multiple_steps", Setting(false), "Multiple Steps", "Perform a multiple time step simulation. (bool)",
                   "(bool) Perform a multiple time step simulation.");
        addSetting("variable_step_size", Setting(false), "Variable Step Size", "Perform a variable time step simulation. (bool)",
                   "(bool) Enabling this setting will allow the integrator to adapt the size of each time step. This will result in a non-uniform time column.  The number of steps or points will be ignored, and the max number of output rows will be used instead.");
        addSetting("max_output_rows", Setting(Config::getInt(Config::MAX_OUTPUT_ROWS)), "Maximum Output Rows",
                   "For variable step size simulations, the maximum number of output rows produced (int).",
                   "(int) This will set the maximum number of output rows for variable step size integration.  This may truncate some simulations that may not reach the desired end time, but prevents infinite or massive output for simulations where the variable step size ends up decreasing too much.");
        CVODEIntegrator::loadConfigSettings();
    }


    void CVODEIntegrator::syncWithModel(ExecutableModel *m) {
        freeCVode();

        mModel = m;

        lastEventTime = 0;
        stateVectorVariables = false;
        variableStepPendingEvent = false;
        variableStepTimeEndEvent = false;

        if (m) {
            createCVode();

            // allocate space for the event status array
            eventStatus = std::vector<unsigned char>(mModel->getEventTriggers(0, 0, 0), false);
        }

        // Update parameter settings within CVODE.
        updateCVODE();
    }

    void CVODEIntegrator::loadConfigSettings() {
        //  VARIABLE STEP SIZE
        bool bVal = false;
        if (getIntegrationMethod() == Integrator::Deterministic) {
            bVal = Config::getBool(Config::SIMULATEOPTIONS_DETERMINISTIC_VARIABLE_STEP);
            Integrator::setValue("variable_step_size", Setting(bVal));
        }
        // Why is this here? This is CVODE, a *deterministic* integrator.
        else if (getIntegrationMethod() == Integrator::Stochastic) {
            bVal = Config::getBool(Config::SIMULATEOPTIONS_STOCHASTIC_VARIABLE_STEP);
            Integrator::setValue("variable_step_size", Setting(bVal));
        }
        Integrator::setValue("max_output_rows", Setting(Config::getInt(Config::MAX_OUTPUT_ROWS)));

        // STIFFNESS
        bVal = Config::getBool(Config::SIMULATEOPTIONS_STIFF);
        Integrator::setValue("stiff", Setting(bVal));

        // MULTIPLE STEPS
        bVal = Config::getBool(Config::SIMULATEOPTIONS_MULTI_STEP);
        Integrator::setValue("multiple_steps", Setting(bVal));
    }

    void CVODEIntegrator::loadSBMLSettings(std::string const &filename) {
        if (filename.empty()) {
            rrLog(Logger::LOG_ERROR) << "Empty file name for settings file";
        } else {
            std::map<std::string, std::string> options;
            std::map<std::string, std::string>::iterator it;
            //Read each line in the settings file
            std::vector<std::string> lines = getLinesInFile(filename);
            for (int i = 0; i < lines.size(); i++) {
                std::vector<std::string> line = splitString(lines[i], ":");
                if (line.size() == 2) {
                    options.insert(std::pair<std::string, std::string>(line[0], line[1]));
                } else {
                    rrLog(lDebug2) << "Empty line in settings file: " << lines[i];
                }
            }

            rrLog(lDebug3) << "Settings File =============";
            for (it = options.begin(); it != options.end(); it++) {
                rrLog(lDebug) << (*it).first << " => " << (*it).second;
            }
            rrLog(lDebug) << "===========================";

            //Assign values
            it = options.find("absolute");
            if (it != options.end()) {
                if ((*it).second.find("[") == std::string::npos) {
                    // scalar absolute tolerance
                    CVODEIntegrator::setValue("absolute_tolerance", Setting(std::abs(toDouble((*it).second))));
                } else {
                    // std::vector absolute tolerance

                    std::vector<double> v = toDoubleVector((*it).second);
                    // take absolute value of each element
                    for (unsigned int i = 0; i < v.size(); i++)
                        v[i] = std::abs(v[i]);
                    CVODEIntegrator::setValue("absolute_tolerance", Setting(v));
                }
            }

            it = options.find("relative");
            if (it != options.end()) {
                CVODEIntegrator::setValue("relative_tolerance", Setting(std::abs(toDouble((*it).second))));
            }
        }
    }

    std::string CVODEIntegrator::getName() const {
        return CVODEIntegrator::getCVODEIntegratorName();
    }

    std::string CVODEIntegrator::getCVODEIntegratorName() {
        return "cvode";
    }

    std::string CVODEIntegrator::getDescription() const {
        return CVODEIntegrator::getCVODEIntegratorDescription();
    }

    std::string CVODEIntegrator::getCVODEIntegratorDescription() {
        return "CVODE is a deterministic ODE solver from the SUNDIALS suite "
               "of timecourse integrators. It implements an Adams-Moulton solver "
               "for non-stiff problems and a backward differentiation formula "
               "(BDF) solver for stiff problems.";
    }

    std::string CVODEIntegrator::getHint() const {
        return CVODEIntegrator::getCVODEIntegratorHint();
    }

    std::string CVODEIntegrator::getCVODEIntegratorHint() {
        return "Deterministic ODE solver";
    }

    Integrator::IntegrationMethod CVODEIntegrator::getIntegrationMethod() const {
        return Integrator::Deterministic;
    }


    void CVODEIntegrator::setIndividualTolerance(std::string sid, double value) {

        // the tolerance std::vector that will be stored
        // [0, numIndFloatingSpecies) stores tolerances for independent floating species
        // [numIndFloatingSpecies, numIndFloatingSpecies+numRateRule) stores tolerances for variables that have rate rule
        std::vector<double> v;

        int speciesIndex = mModel->getFloatingSpeciesIndex(sid);
        std::ptrdiff_t index;
        if (speciesIndex > -1 && speciesIndex < mModel->getNumIndFloatingSpecies()) {
            // sid is an independent floating species
            index = speciesIndex;
        } else {
            // sid might has a rate rule
            std::vector<std::string> symbols = mModel->getRateRuleSymbols();
            std::vector<std::string>::iterator it = std::find(symbols.begin(), symbols.end(), sid);
            if (it != symbols.end()) {
                // found it
                index = mModel->getNumIndFloatingSpecies() + std::distance(symbols.begin(), it);

            } else {
                throw std::invalid_argument("CVODEIntegrator::setIndividualTolerance failed, given sid " + sid +
                                            " is neither a floating species nor has a rate rule.");
            }
        }


        switch (getType("absolute_tolerance")) {

            // all cases below could be convert to a double type
            case Setting::INT32:
            case Setting::INT64:
            case Setting::UINT32:
            case Setting::UINT64:
            case Setting::FLOAT:
            case Setting::DOUBLE: {
                // scalar tolerance
                // need to be converted to std::vector tolerance since tolerance of individual variables is set

                double abstol =(double) CVODEIntegrator::getValue("absolute_tolerance");
                for (int i = 0; i < mModel->getNumFloatingSpecies(); i++)
                    v.push_back(i == index ? value : abstol);
            }

            case Setting::DOUBLEVECTOR: {
                // std::vector tolerance
                Setting absTolSetting = CVODEIntegrator::getValue("absolute_tolerance");
                if (auto vec = absTolSetting.get_if<std::vector<double>>()){
                    v = *vec;
                }
                // only need to update the corresponding index
                v[index] = value;
                break;
            }

            default:
                throw std::runtime_error(
                        "CVODEIntegrator::setIndividualTolerance failed, double or double std::vector expected");
                break;
        }

        // update the tolerance
        CVODEIntegrator::setValue("absolute_tolerance", Setting(v));
    }


    void CVODEIntegrator::setConcentrationTolerance(Setting value) {
        uint ncomp = mModel->getNumCompartments();

        double *volumes = (double *) calloc(ncomp, sizeof(double));
        mModel->getCompartmentVolumes(ncomp, 0, volumes);

        // the tolerance std::vector that will be stored
        std::vector<double> v;

        switch (value.type()) {

            // all cases below could be convert to a double type
            case Setting::INT32:
            case Setting::INT64:
            case Setting::UINT32:
            case Setting::UINT64:
            case Setting::FLOAT:
            case Setting::DOUBLE: {
                // scalar concentration tolerance
                // need to be converted to std::vector tolerance since speices might have various compartment sizes
                double abstol = value.get<double>();
                int index;
                for (int i = 0; i < mModel->getNumIndFloatingSpecies(); i++) {
                    // get the compartment volume of each species
                    index = mModel->getCompartmentIndexForFloatingSpecies(i);
                    if (volumes[index] == 0) {
                        // when the compartment volume is 0, simply set the amount tolerance as the volume tolerance
                        // since the tolerance cannot be 0 (ILLEGAL_INPUT)
                        v.push_back(abstol);
                    } else {
                        // compare the concentration tolerance with the adjusted amount tolerace
                        // store whichever is smaller
                        v.push_back(std::min(abstol, abstol * volumes[index]));
                    }
                }

                std::vector<std::string> symbols = mModel->getRateRuleSymbols();
                for (int i = 0; i < mModel->getNumRateRules(); i++) {
                    int speciesIndex = mModel->getFloatingSpeciesIndex(symbols[i]);
                    if (speciesIndex > -1) {
                        // the symbol defined by the rate rule is a species
                        index = mModel->getCompartmentIndexForFloatingSpecies(i);
                        if (volumes[index] == 0) {
                            // when the compartment volume is 0, simply set the amount tolerance as the volume tolerance
                            // since the tolerance cannot be 0 (ILLEGAL_INPUT)
                            v.push_back(abstol);
                        } else {
                            // compare the concentration tolerance with the adjusted amount tolerace
                            // store whichever is smaller
                            v.push_back(std::min(abstol, abstol * volumes[index]));
                        }
                    } else {
                        // not a species, no need to multiply compartment volume
                        v.push_back(abstol);
                    }
                }


                break;
            }

            case Setting::DOUBLEVECTOR: {
                // std::vector concentration tolerance

                // [0, numIndFloatingSpecies) stores tolerances for independent floating species
                // [numIndFloatingSpecies, numIndFloatingSpecies+numRateRule) stores tolerances for variables that have rate rule

                v = value.get<std::vector<double> >();

                checkVectorSize(mModel->getNumIndFloatingSpecies() + mModel->getNumRateRules(), v.size());

                int index;
                for (int i = 0; i < mModel->getNumIndFloatingSpecies(); i++) {
                    // get the compartment volume of each species
                    index = mModel->getCompartmentIndexForFloatingSpecies(i);
                    if (volumes[index] > 0) {
                        // compare the concentration tolerance with the adjusted amount tolerace
                        // store whichever is smaller
                        v[i] = std::min(v[i], v[i] * volumes[index]);
                    }
                }


                std::vector<std::string> symbols = mModel->getRateRuleSymbols();
                for (int i = mModel->getNumIndFloatingSpecies();
                     i < mModel->getNumRateRules() + mModel->getNumIndFloatingSpecies(); i++) {
                    std::string symbol = symbols[i];
                    int speciesIndex = mModel->getFloatingSpeciesIndex(symbol);
                    if (speciesIndex > -1) {
                        // the symbol defined by the rate rule is a species
                        index = mModel->getCompartmentIndexForFloatingSpecies(i);
                        if (volumes[index] > 0) {
                            // compare the concentration tolerance with the adjusted amount tolerace
                            // store whichever is smaller
                            v[i] = std::min(v[i], v[i] * volumes[index]);
                        }
                    }
                }

                break;
            }

            default:
                throw std::runtime_error(
                        "CVODEIntegrator::setIndividualTolerance failed, double or double std::vector expected");
                break;
        }

        free(volumes);
        // update the tolerance
        CVODEIntegrator::setValue("absolute_tolerance", Setting(v));
    }

    std::vector<double> CVODEIntegrator::getConcentrationTolerance() {
        uint ncomp = mModel->getNumCompartments();

        double *volumes = (double *) calloc(ncomp, sizeof(double));
        mModel->getCompartmentVolumes(ncomp, 0, volumes);

        // the tolerance std::vector
        std::vector<double> v;

        switch (getType("absolute_tolerance")) {

            // all cases below could be convert to a double type
            case Setting::INT32:
            case Setting::INT64:
            case Setting::UINT32:
            case Setting::UINT64:
            case Setting::FLOAT:
            case Setting::DOUBLE: {
                // scalar tolerance
                double abstol = CVODEIntegrator::getValue("absolute_tolerance");
                int index;
                for (int i = 0; i < mModel->getNumIndFloatingSpecies(); i++) {
                    // get the compartment volume of each species
                    index = mModel->getCompartmentIndexForFloatingSpecies(i);
                    if (volumes[index] == 0) {
                        // when the compartment volume is 0, simply set the amount tolerance as the volume tolerance
                        // since the tolerance cannot be 0 (ILLEGAL_INPUT)
                        v.push_back(abstol);
                    } else {
                        // convert the amount tolerance to concentration
                        v.push_back(abstol / volumes[index]);
                    }
                }


                std::vector<std::string> symbols = mModel->getRateRuleSymbols();
                for (int i = 0; i < mModel->getNumRateRules(); i++) {
                    int speciesIndex = mModel->getFloatingSpeciesIndex(symbols[i]);
                    if (speciesIndex > -1) {
                        // the symbol defined by the rate rule is a species
                        index = mModel->getCompartmentIndexForFloatingSpecies(i);
                        if (volumes[index] == 0) {
                            v.push_back(abstol);
                        } else {
                            v.push_back(abstol / volumes[index]);
                        }
                    } else {
                        // not a species, no need to divide by compartment volume
                        v.push_back(abstol);
                    }
                }
                break;
            }


            case Setting::DOUBLEVECTOR: {
                // std::vector concentration tolerance

                // [0, numIndFloatingSpecies) stores tolerances for independent floating species
                // [numIndFloatingSpecies, numIndFloatingSpecies+numRateRule) stores tolerances for variables that have rate rule

                v = CVODEIntegrator::getValue("absolute_tolerance").get<std::vector<double>>();

                int index;
                for (int i = 0; i < mModel->getNumIndFloatingSpecies(); i++) {
                    // get the compartment volume of each species
                    index = mModel->getCompartmentIndexForFloatingSpecies(i);
                    if (volumes[index] > 0) {
                        // compare the concentration tolerance with the adjusted amount tolerace
                        // store whichever is smaller
                        v[i] = v[i] / volumes[index];
                    }
                }


                std::vector<std::string> symbols = mModel->getRateRuleSymbols();
                for (int i = mModel->getNumIndFloatingSpecies();
                     i < mModel->getNumRateRules() + mModel->getNumIndFloatingSpecies(); i++) {
                    std::string symbol = symbols[i];
                    int speciesIndex = mModel->getFloatingSpeciesIndex(symbol);
                    if (speciesIndex > -1) {
                        // the symbol defined by the rate rule is a species
                        index = mModel->getCompartmentIndexForFloatingSpecies(i);
                        if (volumes[index] > 0) {
                            v[i] = v[i] / volumes[index];
                        }
                    }
                }

                break;

            }

            default:
                break;
        }
        free(volumes);
        return v;
    }

    void CVODEIntegrator::setValue(const std::string& key, Setting val) {
        // if std::vector tolerance is set, the size of std::vector must be equal to
        // the number of floating species
        if (key == "absolute_tolerance" && val.type() == Setting::DOUBLEVECTOR)
            checkVectorSize(mModel->getNumIndFloatingSpecies() + mModel->getNumRateRules(),
                            val.get<std::vector<double> >().size());

        Integrator::setValue(key, val);

        /// Values and keys are stored in the settings std::map, which is updated
        /// in every call to @ref setValue. In addition, changing CVODE-specific
        /// parameters requires a call into the CVODE library to synchronize
        /// CVODE's internal memory with the settings std::map.
        if (mCVODE_Memory) {
            if (key == "maximum_bdf_order") {
                CVodeSetMaxOrd(mCVODE_Memory, (int)getValue("maximum_bdf_order"));
            } else if (key == "maximum_adams_order") {
                CVodeSetMaxOrd(mCVODE_Memory, (int)getValue("maximum_adams_order"));
            } else if (key == "initial_time_step") {
                CVodeSetInitStep(mCVODE_Memory, (double)getValue("initial_time_step"));
            } else if (key == "minimum_time_step") {
                CVodeSetMinStep(mCVODE_Memory, (double)getValue("minimum_time_step"));
            } else if (key == "maximum_time_step") {
                CVodeSetMaxStep(mCVODE_Memory, (double)getValue("maximum_time_step"));
            } else if (key == "maximum_num_steps") {
                CVodeSetMaxNumSteps(mCVODE_Memory, (int)getValue("maximum_num_steps"));
            }
            else if (key == "absolute_tolerance" || key == "relative_tolerance") {
                CVodeSetMaxNumSteps(mCVODE_Memory, (double)getValue("maximum_num_steps"));
                setCVODETolerances();

            }
        }
        if (key == "stiff") {
            // If the integrator is changed from stiff to standard, we must re-create CVode.
            rrLog(Logger::LOG_INFORMATION) << "Integrator stiffness has been changed. Re-creating CVode.";
            freeCVode();
            createCVode();
        }
    }

    void CVODEIntegrator::updateCVODE() {
        if (mCVODE_Memory == nullptr) {
            return;
        }

        CVodeSetInitStep(mCVODE_Memory, (double)getValue("initial_time_step"));
        CVodeSetMinStep(mCVODE_Memory, (double)getValue("minimum_time_step"));
        CVodeSetMaxStep(mCVODE_Memory, (double)getValue("maximum_time_step"));
        CVodeSetMaxNumSteps(mCVODE_Memory, (int)getValue("maximum_num_steps") > 0 ? (int)getValue("maximum_num_steps")
                                                                                  : mDefaultMaxNumSteps);
        if ((bool)getValue("stiff"))
            CVodeSetMaxOrd(mCVODE_Memory, (int)getValue("maximum_bdf_order"));
        else
            CVodeSetMaxOrd(mCVODE_Memory, (int)getValue("maximum_adams_order"));
        setCVODETolerances();
    }

    void CVODEIntegrator::reInit(double t0) {
        // if we have no state std::vector variables and no events, we never
        // created an integrator.
        if (mCVODE_Memory == nullptr) {
            return;
        }

        int result = CVodeReInit(mCVODE_Memory, t0, mStateVector);

        if (result != CV_SUCCESS) {
            handleCVODEError(result);
        }

        setCVODETolerances();
    }

    double CVODEIntegrator::integrate(double timeStart, double hstep) {
        static const double epsilon = std::numeric_limits<double>::epsilon();
        // CVODE root tolerance, used for backing up when an event fires (see CVODE User Doc pp. 13)
        static const double roottol = 100. * (32. * epsilon) * (fabs(timeStart) + fabs(hstep));

        rrLog(Logger::LOG_DEBUG) << "CVODEIntegrator::integrate("
                               << timeStart << ", " << hstep << ")";

        if (variableStepPendingEvent || variableStepTimeEndEvent) {
            return applyVariableStepPendingEvents() + roottol;
        }

        double timeEnd = 0.0;
        double tout = timeStart + hstep;
        int strikes = 3;

        // Set itask based on step size settings.
        int itask = CV_NORMAL;
        bool varstep = (bool)getValue("variable_step_size");
        double relTol = (double)getValue("relative_tolerance");

        if ((bool)getValue("multiple_steps") || (bool)getValue("variable_step_size")) {
            itask = CV_ONE_STEP;
        }

        // loop until machine epislon
        while (tout - timeEnd >= epsilon) {
            // here we bail in case we have no ODEs set up with CVODE ... though we should
            // still at least evaluate the model function
            if (!haveVariables() && mModel->getNumEvents() == 0) {
                mModel->getStateVectorRate(tout, nullptr, nullptr);
                return tout;
            }

            double nextTargetEndTime = tout;

            if (mModel->getPendingEventSize() > 0 &&
                mModel->getNextPendingEventTime(false) < nextTargetEndTime) {
                nextTargetEndTime = mModel->getNextPendingEventTime(true);
            }

            // event status before time step
            mModel->getEventTriggers(eventStatus.size(), nullptr, eventStatus.empty() ? nullptr : &eventStatus[0]);

            // time step
            int nResult = CVode(mCVODE_Memory, nextTargetEndTime, mStateVector, &timeEnd, itask);

            if (nResult == CV_ROOT_RETURN) {
                rrLog(Logger::LOG_DEBUG) << "Event detected at time " << timeEnd;

                bool tooCloseToStart = fabs(timeEnd - lastEventTime) > relTol;

                if (tooCloseToStart) {
                    strikes = 3;
                } else {
                    strikes--;
                }

                // the condition that we are to evaluate and apply events
                if (tooCloseToStart || strikes > 0) {
                    lastEventTime = timeEnd;

                    if (varstep
                        && (timeEnd - timeStart > 2. * epsilon)) {
                        variableStepPendingEvent = true;
                        assignResultsToModel();
                        mModel->setTime(timeEnd - epsilon);
                        if (listener) {
                            listener->onTimeStep(this, mModel, timeEnd);
                        }
                        return timeEnd - roottol;
                    }

                    // apply events, copy post event status into integrator state std::vector.
                    applyEvents(timeEnd, eventStatus);

                    if (listener) {
                        listener->onEvent(this, mModel, timeEnd);
                    }
                }
            } else if (nResult == CV_SUCCESS) {
                // copy integrator state std::vector into model
                assignResultsToModel();


                // need to check if an event occured at the exact time step,
                // if so, add an extra point if we're doing variable step
                if (varstep
                    && (timeEnd - timeStart > 2. * epsilon)) {
                    // event status before time step
                    mModel->getEventTriggers(eventStatus.size(), 0, &eventStatus[0]);
                    // apply events and write state to variableStepPostEventState
                    // model state is updated by events.
                    int handled = mModel->applyEvents(timeEnd, &eventStatus[0],
                                                      NULL, &variableStepPostEventState[0]);
                    if (handled > 0) {
                        // write original state back to model
                        mModel->setTime(timeEnd - epsilon);
                        assignResultsToModel();
                        variableStepTimeEndEvent = true;
                        lastEventTime = timeEnd;
                        return timeEnd - epsilon;
                    }
                } else {


                    mModel->setTime(timeEnd);

                    // only needs to be called after a reg time step completes, the applyEvents
                    // called when a event root is found clears out all pending events and applies
                    // them.
                    applyPendingEvents(timeEnd);
                }

                if (listener) {
                    listener->onTimeStep(this, mModel, timeEnd);
                }
            } else {
                handleCVODEError(nResult);
            }

            try {
                mModel->testConstraints();
            }
            catch (const std::exception &e) {
                rrLog(Logger::LOG_WARNING) << "Constraint Violated at time = " << timeEnd << ": " << e.what();
            }

            if (varstep && (timeEnd - timeStart > 2. * epsilon)) {
                return timeEnd;
            }


            if (tout - timeEnd > epsilon) {
                timeStart = timeEnd;
            }
            rrLog(Logger::LOG_TRACE) << "time step, tout: " << tout << ", timeEnd: " << timeEnd;
        }
        return timeEnd;
    }


    void CVODEIntegrator::tweakTolerances() {
        double minAbs = Config::getDouble(Config::CVODE_MIN_ABSOLUTE);
        double minRel = Config::getDouble(Config::CVODE_MIN_RELATIVE);

        switch (getType("absolute_tolerance")) {

            // all cases below could be convert to a double type
            case Setting::INT32:
            case Setting::INT64:
            case Setting::UINT32:
            case Setting::UINT64:
            case Setting::FLOAT:
            case Setting::DOUBLE:
                // scalar tolerance
                CVODEIntegrator::setValue("absolute_tolerance",
                                          Setting(std::min((double)CVODEIntegrator::getValue("absolute_tolerance"), minAbs)));
                break;


            case Setting::DOUBLEVECTOR: {
                // std::vector tolerance
                std::vector<double> v = CVODEIntegrator::getValue("absolute_tolerance");
                for (int i = 0; i < v.size(); i++)
                    v[i] = std::min(v[i], minAbs);
                CVODEIntegrator::setValue("absolute_tolerance", Setting(v));

                break;
            }

            default:
                break;
        }

        CVODEIntegrator::setValue("relative_tolerance",
                                  Setting(std::min((double)CVODEIntegrator::getValue("relative_tolerance"), minRel)));

        rrLog(Logger::LOG_INFORMATION) << "tweaking CVODE tolerances to abs="
                                     << (double)CVODEIntegrator::getValue("absolute_tolerance") << ", rel="
                                     << (double)CVODEIntegrator::getValue("relative_tolerance");
    }


    bool CVODEIntegrator::haveVariables() const {
        return stateVectorVariables;
    }

    void CVODEIntegrator::createCVode() {
        if (!mModel) {
            return;
        }

        assert(mStateVector == nullptr && mCVODE_Memory == nullptr &&
               "calling cvodeCreate, but cvode objects already exist");

        // still need cvode state std::vector size if we have no vars, but have
        // events, needed so root finder works.
        int allocStateVectorSize = 0;
        // when argument is null, returns size of state std::vector (see rrExecutableModel::getStateVector)
        int realStateVectorSize = mModel->getStateVector(nullptr);


        // cvode return code
        int err;

        if (realStateVectorSize > 0) {
            stateVectorVariables = true;
            allocStateVectorSize = realStateVectorSize;
        } else if (mModel->getNumEvents() > 0) {
            allocStateVectorSize = 1;
            stateVectorVariables = false;
        } else {
            stateVectorVariables = false;
            return;
        }

        // allocate and init the cvode arrays
        mStateVector = N_VNew_Serial(allocStateVectorSize);
        variableStepPostEventState.resize(allocStateVectorSize);

        for (int i = 0; i < allocStateVectorSize; i++) {
            SetVector(mStateVector, i, 0.);
        }

        if ((bool)getValue("stiff")) {
            rrLog(Logger::LOG_INFORMATION) << "using stiff integrator";
            mCVODE_Memory = (void *) CVodeCreate(CV_BDF);
        } else {
            rrLog(Logger::LOG_INFORMATION) << "using non-stiff integrator";
            mCVODE_Memory = (void *) CVodeCreate(CV_ADAMS);
        }

        assert(mCVODE_Memory && "could not create Cvode, CVodeCreate failed");

        if ((err = CVodeSetErrHandlerFn(mCVODE_Memory, cvodeErrHandler, this)) != CV_SUCCESS) {
            handleCVODEError(err);
        }

        // use non default CVODE value here, default is too short
        // for some sbml tests.
        CVodeSetMaxNumSteps(mCVODE_Memory, mDefaultMaxNumSteps);

        double t0 = 0.0;

        if ((err = CVodeSetUserData(mCVODE_Memory, (void *) this)) != CV_SUCCESS) {
            handleCVODEError(err);
        }

        if ((err = CVodeInit(mCVODE_Memory, cvodeDyDtFcn, t0, mStateVector)) != CV_SUCCESS) {
            handleCVODEError(err);
        }

        if (mModel->getNumEvents() > 0) {
            if ((err = CVodeRootInit(mCVODE_Memory, mModel->getNumEvents(),
                                     cvodeRootFcn)) != CV_SUCCESS) {
                handleCVODEError(err);
            }
            rrLog(Logger::LOG_TRACE) << "CVRootInit executed.....";
        }

        /**
         * note the cvDirectDemo.c example from sundials is useful
         * to see the different cvode options in action
         */

        /* create fixed point nonlinear solver object */
        if ((bool)getValue("stiff")) {
            // as per the cvode docs (look closely at docs for CVodeCreate)
            // we use the default Newton iteration for stiff

            nonLinSolver = SUNNonlinSol_Newton(mStateVector);

            if (nonLinSolver == nullptr) {
                throw std::runtime_error("CVODEIntegrator::createCVODE: nonLinearSolver_ is nullptr\n");
            }

            if ((err = CVodeSetNonlinearSolver(mCVODE_Memory, nonLinSolver)) != CV_SUCCESS) {
                handleCVODEError(err);
            }

            // the newton method requires use of a linear solver, which we set up here.
            jac = SUNDenseMatrix(allocStateVectorSize, allocStateVectorSize);
            linSolver = SUNLinSol_Dense(mStateVector, jac);
            if (linSolver == nullptr) {
                throw std::runtime_error("CVODEIntegrator::createCVODE: call to SunLinSol_Dense returned nullptr. "
                                         "The size of the sundials matrix (created with SUNDenseMatrix) used for the jacobian "
                                         "(" + std::to_string(allocStateVectorSize) + "x" +
                                         std::to_string(allocStateVectorSize)
                                         + ") is inappropriate for this model\n");
            }
            if ((err = CVodeSetLinearSolver(mCVODE_Memory, linSolver, jac)) != CV_SUCCESS) {
                handleCVODEError(err);
            };

            // Use a difference quotient Jacobian by not passing Jac to CVodeSetJacFn
            if ((err = CVodeSetJacFn(mCVODE_Memory, nullptr)) != CV_SUCCESS) {
                handleCVODEError(err);
            }
        } else {
            // and fixed point solver (which used to be called functional iteration)
            // for nonstiff problems
            nonLinSolver = SUNNonlinSol_FixedPoint(mStateVector, 0);
            if ((err = CVodeSetNonlinearSolver(mCVODE_Memory, nonLinSolver)) != CV_SUCCESS) {
                handleCVODEError(err);
            }
        }

        setCVODETolerances();
        mModel->resetEvents();
    }

    void CVODEIntegrator::testRootsAtInitialTime() {
        std::vector<unsigned char> initialEventStatus(mModel->getEventTriggers(0, 0, 0), false);
        mModel->getEventTriggers(initialEventStatus.size(), 0,
                                 initialEventStatus.empty() ? nullptr : &initialEventStatus[0]);
        applyEvents(0, initialEventStatus);
    }

    void CVODEIntegrator::applyEvents(double timeEnd, std::vector<unsigned char> &previousEventStatus) {
        double *stateVector = mStateVector ? NV_DATA_S(mStateVector) : nullptr;
        mModel->applyEvents(timeEnd, previousEventStatus.empty() ? nullptr : &previousEventStatus[0], stateVector,
                            stateVector);

        if (timeEnd > 0.0) {
            mModel->setTime(timeEnd);

            // copy state std::vector into cvode memory
            if (mStateVector) {
                mModel->getStateVector(NV_DATA_S(mStateVector));
            }

            // set tolerances and so forth.
            reInit(timeEnd);
        }
    }


    void CVODEIntegrator::applyPendingEvents(double timeEnd) {
        if (mModel) {
            // get current event triggered state
            mModel->getEventTriggers(eventStatus.size(), 0, eventStatus.size() == 0 ? NULL : &eventStatus[0]);
            int handled = mModel->applyEvents(timeEnd, eventStatus.size() == 0 ? NULL : &eventStatus[0], NULL, NULL);
            if (handled > 0) {
                rrLog(Logger::LOG_DEBUG) << __FUNC__;
                restart(timeEnd);
            }
        }
    }

    void CVODEIntegrator::assignResultsToModel() const {
        if (mStateVector) {
            mModel->setStateVector(mStateVector->ops->nvgetarraypointer(mStateVector));
        }
    }

    void CVODEIntegrator::setCVODETolerances() {
        if (mStateVector == nullptr || mModel == nullptr) {
            return;
        }

        // If we have a model with only events, cvode still needs a state std::vector of length 1 to integrate.
        if (!haveVariables() && mModel->getNumEvents() > 0) {
            SetVector(mStateVector, 0, 1.0);
        }

        int err;
        // switch on different cases that the absolute tolerance is scalar or std::vector
        switch (getType("absolute_tolerance")) {

            // all cases below could be convert to a double type
            case Setting::INT32:
            case Setting::INT64:
            case Setting::UINT32:
            case Setting::UINT64:
            case Setting::FLOAT:
            case Setting::DOUBLE:
                // scalar tolerance
                err = CVodeSStolerances(mCVODE_Memory, (double)getValue("relative_tolerance"),
                                        (double)getValue("absolute_tolerance"));
                break;


            case Setting::DOUBLEVECTOR: {
                // std::vector tolerance
                // convert a double std::vector to a n_vector?
                std::vector<double> v = getValue("absolute_tolerance").get<std::vector<double>>();
                double *arr = new double[v.size()];
                for (int i = 0; i < v.size(); i++)
                    arr[i] = v[i];
                N_Vector nv = N_VMake_Serial(static_cast<long>(v.size()), arr);


                err = CVodeSVtolerances(mCVODE_Memory, (double)getValue("relative_tolerance"), nv);
                // need to destroy

                N_VDestroy_Serial(nv);
                delete[] arr;
                break;
            }

            default:
                err = CV_ILL_INPUT;
                break;
        }


        if (err != CV_SUCCESS) {
            handleCVODEError(err);
        }

        switch (getType("absolute_tolerance")) {
            // scalar tolerance
            // all cases below could be convert to a double type
            case Setting::INT32:
            case Setting::INT64:
            case Setting::UINT32:
            case Setting::UINT64:
            case Setting::FLOAT:
            case Setting::DOUBLE:
                rrLog(Logger::LOG_INFORMATION) << "Set tolerance to abs: " << std::setprecision(16)
                                             << (double)getValue("absolute_tolerance") << ", rel: "
                                             << (double)getValue("relative_tolerance") << std::endl;
                break;

                // std::vector tolerance
            case Setting::DOUBLEVECTOR: {
                rrLog(Logger::LOG_INFORMATION) << "Set tolerance to abs: " << std::setprecision(16) << "[";
                std::vector<double> v = getValue("absolute_tolerance").get<std::vector<double>>();
                for (int i = 0; i < v.size(); i++) {
                    if (i != 0) {
                        rrLog(Logger::LOG_INFORMATION) << ", ";
                    }
                    rrLog(Logger::LOG_INFORMATION) << v[i];
                }
                rrLog(Logger::LOG_INFORMATION) << "], rel: " << (double)getValue("relative_tolerance") << std::endl;

                break;
            }


            default:
                break;
        }

    }

    void CVODEIntegrator::restart(double time) {
        if (!mModel) {
            return;
        }

        lastEventTime = 0.0;

        // apply any events that trigger before or at time 0.
        // important NOT to set model time before we check get
        // the initial event state, initially time is < 0.
        if (time <= 0.0) {

            // copy state std::vector into cvode memory, need to do this before evaluating
            // roots because the applyEvents method copies the cvode state std::vector
            // into the model
            if (mStateVector) {
                mModel->getStateVector(NV_DATA_S(mStateVector));
            }

            testRootsAtInitialTime();
        }

        mModel->setTime(time);

        // copy state std::vector into cvode memory
        if (mStateVector && mCVODE_Memory) {
            mModel->getStateVector(NV_DATA_S(mStateVector));
        }

        // set tolerances and so forth.
        if (mCVODE_Memory) {
            reInit(time);
        }
    }

    // Cvode calls this to compute the dy/dts. This routine in turn calls the
    // model function which is located in the host application.
    int cvodeDyDtFcn(realtype time, N_Vector cv_y, N_Vector cv_ydot, void *userData) {
        double *y = NV_DATA_S(cv_y);
        double *ydot = NV_DATA_S(cv_ydot);
        CVODEIntegrator *cvInstance = (CVODEIntegrator *) userData;

        assert(cvInstance && "userData pointer is NULL in cvode dydt callback");

        ExecutableModel *model = cvInstance->mModel;

        model->getStateVectorRate(time, y, ydot);

        if (!cvInstance->stateVectorVariables && cvInstance->mStateVector &&
            NV_LENGTH_S(cvInstance->mStateVector) == 1) {
            ydot[0] = 0.0;
        }

        rrLog(Logger::LOG_TRACE) << __FUNC__ << ", model: " << model;

        return CV_SUCCESS;
    }

    void CVODEIntegrator::freeCVode() {
        // cvode does not check for null values.
        if (mStateVector) {
            N_VDestroy_Serial(mStateVector);
        }

        if (mCVODE_Memory) {
            CVodeFree(&mCVODE_Memory);
        }

        if (nonLinSolver) {
            SUNNonlinSolFree(nonLinSolver);
        }

        if (linSolver) {
            SUNLinSolFree(linSolver);
        }

        if (jac){
            SUNMatDestroy(jac);
        }

        mCVODE_Memory = nullptr;
        mStateVector = nullptr;
        nonLinSolver = nullptr;
        linSolver = nullptr;
        jac = nullptr;
    }

    double CVODEIntegrator::applyVariableStepPendingEvents() {
        if (variableStepTimeEndEvent) {
            // post event state allready calcuated.
            mModel->setStateVector(&variableStepPostEventState[0]);
            // copy state std::vector into cvode memory
            if (mStateVector) {
                mModel->getStateVector(NV_DATA_S(mStateVector));
            }
            mModel->setTime(lastEventTime);
            reInit(lastEventTime);
        } else {
            // apply events, copy post event status into integrator state std::vector.
            applyEvents(lastEventTime, eventStatus);
        }

        if (listener) {
            listener->onEvent(this, mModel, lastEventTime);
        }

        variableStepPendingEvent = false;
        variableStepTimeEndEvent = false;
        return lastEventTime;
    }

    // int (*CVRootFn)(realtype t, N_Vector y, realtype *gout, void *user_data)
    // Cvode calls this to check for event changes
    int cvodeRootFcn(realtype time, N_Vector y_vector, realtype *gout, void *user_data) {
        CVODEIntegrator *cvInstance = (CVODEIntegrator *) user_data;

        assert(cvInstance && "user data pointer is NULL on CVODE root callback");

        ExecutableModel *model = cvInstance->mModel;

        double *y = NV_DATA_S(y_vector);

        model->getEventRoots(time, y, gout);

        return CV_SUCCESS;
    }

    std::string CVODEIntegrator::cvodeDecodeError(int cvodeError, bool exInfo) {
        std::string result;
        std::stringstream ss;
        ss << (int)getValue("maximum_num_steps");
        std::string max_steps = ss.str();

        switch (cvodeError) {
            case CV_TOO_MUCH_WORK:
                result = "CV_TOO_MUCH_WORK";
                if (exInfo) {
                    result += ": The solver took mxstep (" + max_steps + ") internal steps but " +
                              "could not reach tout.";
                }
                break;
            case CV_TOO_MUCH_ACC:
                result = "CV_TOO_MUCH_ACC";
                if (exInfo) {
                    result += ": The solver could not satisfy the accuracy "
                              "demanded by the user for some internal step.";
                }
                break;
            case CV_ERR_FAILURE:
                result = "CV_ERR_FAILURE";
                if (exInfo) {
                    result += ": Error test failures occurred too many times "
                              "(= MXNEF = 7) during one internal time step or"
                              "occurred with |h| = hmin.";
                }
                break;
            case CV_CONV_FAILURE:
                result = "CV_CONV_FAILURE";
                if (exInfo) {
                    result += ": Convergence test failures occurred too many "
                              "times (= MXNCF = 10) during one internal time"
                              "step or occurred with |h| = hmin.";
                }
                break;
            case CV_LINIT_FAIL:
                result = "CV_LINIT_FAIL";
                if (exInfo) {
                    result += ": The linear solver's initialization function "
                              "failed.";
                }
                break;
            case CV_LSETUP_FAIL:
                result = "CV_LSETUP_FAIL";
                if (exInfo) {
                    result += ": The linear solver's setup routine failed in an "
                              "unrecoverable manner.";
                }
                break;
            case CV_LSOLVE_FAIL:
                result = "CV_LSOLVE_FAIL";
                if (exInfo) {
                    result += ": The linear solver's solve routine failed in an "
                              "unrecoverable manner.";
                }
                break;
            case CV_RHSFUNC_FAIL:
                result = "CV_RHSFUNC_FAIL";
                break;
            case CV_FIRST_RHSFUNC_ERR:
                result = "CV_FIRST_RHSFUNC_ERR";
                break;
            case CV_REPTD_RHSFUNC_ERR:
                result = "CV_REPTD_RHSFUNC_ERR";
                break;
            case CV_UNREC_RHSFUNC_ERR:
                result = "CV_UNREC_RHSFUNC_ERR";
                break;
            case CV_RTFUNC_FAIL:
                result = "CV_RTFUNC_FAIL";
                break;
            case CV_MEM_FAIL:
                result = "CV_MEM_FAIL";
                break;
            case CV_MEM_NULL:
                result = "CV_MEM_NULL";
                if (exInfo) {
                    result += ": The cvode_mem argument was NULL.";
                }
                break;
            case CV_ILL_INPUT:
                result = "CV_ILL_INPUT";
                if (exInfo) {
                    result += ": One of the inputs to CVode is illegal. This "
                              "includes the situation when a component of the "
                              "error weight vectors becomes < 0 during "
                              "internal time-stepping.  It also includes the "
                              "situation where a root of one of the root "
                              "functions was found both at t0 and very near t0. "
                              "The ILL_INPUT flag will also be returned if the "
                              "linear solver routine CV--- (called by the user "
                              "after calling CVodeCreate) failed to set one of "
                              "the linear solver-related fields in cvode_mem or "
                              "if the linear solver's init routine failed. In "
                              "any case, the user should see the printed "
                              "error message for more details.";
                }
                break;
            case CV_NO_MALLOC:
                result = "CV_NO_MALLOC";
                if (exInfo) {
                    result += ": indicating that cvode_mem has not been "
                              "allocated (i.e., CVodeInit has not been "
                              "called).";
                }
                break;
            case CV_BAD_K:
                result = "CV_BAD_K";
                if (exInfo) {
                    result += ": k is not in the range 0, 1, ..., qu.";
                }
                break;
            case CV_BAD_T:
                result = "CV_BAD_T";
                if (exInfo) {
                    result += ": t is not in the interval [tn-hu,tn].";
                }
                break;
            case CV_BAD_DKY:
                result = "CV_BAD_DKY";
                if (exInfo) {
                    result += ": The dky argument was NULL.";
                }
                break;
            case CV_TOO_CLOSE:
                result = "CV_TOO_CLOSE:";
                break;
            default:
                result = "UNKNOWN_CODE";
                break;
        }
        return result;
    }

    void *CVODEIntegrator::getCvodeMemory() const {
        return mCVODE_Memory;
    }

    N_Vector CVODEIntegrator::getStateVector() const {
        return mStateVector;
    }

    SUNNonlinearSolver CVODEIntegrator::getSolver() const {
        return nonLinSolver;
    }

    /**
    * Purpose
    * This function processes error and warning messages from CVODE and its
    * sub-modules.
    *
    * Arguments:
    * error_code: is the error code.
    * module: is the name of the CVODE module reporting the error.
    * function: is the name of the function in which the error occurred.
    * msg: is the error message.
    * eh_data: is a pointer to user data, the same as the eh_data parameter
    * passed to CVodeSetErrHandlerFn.
    * Return value: A CVErrHandlerFn function has no return value.
    *
    * Notes:
    * error_code is negative for errors and positive (CV_WARNING) for warnings.
    * If a function that returns a pointer to memory encounters an error,
    * it sets error_code to 0.
    */
    void cvodeErrHandler(int error_code, const char *module, const char *function,
                         char *msg, void *eh_data) {
        CVODEIntegrator *i = (CVODEIntegrator *) eh_data;
        i->checkType();

        if (error_code < 0) {
            rrLog(Logger::LOG_ERROR) << "CVODE Error: " << i->cvodeDecodeError(error_code, false)
                                   << ", Module: " << module << ", Function: " << function
                                   << ", Message: " << msg;

        } else if (error_code == CV_WARNING) {
            rrLog(Logger::LOG_WARNING) << "CVODE Warning: "
                                     << ", Module: " << module << ", Function: " << function
                                     << ", Message: " << msg;
        }
    }

}

