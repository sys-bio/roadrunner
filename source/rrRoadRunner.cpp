#pragma hdrstop
// in Windows, this needs to go first to get M_PI
#define _USE_MATH_DEFINES

#include "rrOSSpecifics.h"

#include "ApproxSteadyStateDecorator.h"
#include "ExecutableModelFactory.h"
#include "ForwardSensitivitySolver.h"
#include "Integrator.h"
#include "IntegratorFactory.h"
#include "PresimulationDecorator.h"
#include "PresimulationProgramDecorator.h"
#include "SBMLValidator.h"
#include "SVD.h"
#include "SensitivitySolver.h"
#include "SensitivitySolverFactory.h"
#include "SteadyStateSolver.h"
#include "SteadyStateSolverFactory.h"
#include "rr-libstruct/lsLA.h"
#include "rr-libstruct/lsLibla.h"
#include "rrCompiler.h"
#include "rrConfig.h"
#include "rrConstants.h"
#include "rrException.h"
#include "rrExecutableModel.h"
#include "rrLogger.h"
#include "rrRoadRunner.h"
#include "rrSBMLReader.h"
#include "rrUtils.h"
#include "rrVersionInfo.h"
#include "sbml/ListOf.h"
#include "sbml/Model.h"
#include "sbml/common/operationReturnValues.h"
#include "sbml/math/FormulaParser.h"
#include "sbml/packages/fbc/extension/FbcSBMLDocumentPlugin.h"

//Have to include these last because of something to do with min and max in Random.h
#include <rr-libstruct/lsLibStructural.h>
#include <sbml/UnitKind.h>
#include <sbml/conversion/SBMLLevelVersionConverter.h>
#include <sbml/conversion/SBMLLocalParameterConverter.h>
#include <Poco/File.h>
#include <assert.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iostream>
#include <list>
#include <list>
#include <math.h>
#include <memory>
#include <thread>
#include <thread_pool.hpp>
#include <utility>
//Shouldn't need the warning-disabling #pragmas here, since these are all links to *our* llvm, not the offical LLVM source.
#include "llvm/LLVMExecutableModel.h"
#include "llvm/ModelResources.h"
#include "llvm/IR/IRBuilder.h"


#ifdef _MSC_VER
#define isfinite(x) _finite(x)
#endif


namespace rr {

    using ls::ComplexMatrix;
    using ls::DoubleMatrix;
    using ls::LibStructural;

    typedef std::vector<std::string> string_vector;


// we can write a single function to pick the std::string lists out
// of the model instead of duplicating it 6 times with
// fun ptrs.
    typedef std::string (ExecutableModel::*GetNameFuncPtr)(size_t);

    typedef int (ExecutableModel::*GetNumFuncPtr)();

// make this static here, hide our implementation...
    static std::vector<std::string> createModelStringList(ExecutableModel *model,
                                                          GetNumFuncPtr numFunc, GetNameFuncPtr nameFunc) {
        if (!model) {
            return std::vector<std::string>(0);
        }

        const int num = (model->*numFunc)();
        std::vector<std::string> strings(num);

        for (int i = 0; i < num; i++) {
            strings[i] = (model->*nameFunc)(i);
        }

        return strings;
    }

/**
 * convert the names from the sbml settings file into the
 * RoadRunner selection syntax.
 */
    static std::vector<std::string> createSelectionList(const SimulateOptions &o);

/**
 * Some functions needed for the freq. analysis code
 * From Frank Bergmann code, ported to C++ by Totte Karlson
 * Generate log spaced array, algorithm from matlab, note operator precedence
 * y = (10).^ [d1+(0:n-2)*(d2-d1)/(floor(n)-1), d2];
 *
 * d1 = starting value (10^d1)
 * d2 = ending value (10^d2)
 * n = number of values in the generated series
 */
    static std::vector<double> logspace(const double &startW, const double &d2, const int &n);

    static double phase(std::complex<double> &val);

    static double getAdjustment(std::complex<double> &z);

/**
 * variable time step integration data struct
 */
    typedef std::list<std::vector<double> > DoubleVectorList;


/**
 * check if metabolic control analysis is valid for the model.
 *
 * In effect, this checks that the model is a pure
 * reaction-kinetics model with no rate rules, no events.
 *
 * Throws an invaid arg exception if not valid.
 */
    static void metabolicControlCheck(ExecutableModel *model);

/*
* used to write simulation result to out stream. This is likely one batch out of many
* for variable time step stochastic simulation
*/
    static void writeDoubleVectorListToStream(std::ostream &out, const DoubleVectorList &results);

    static void writeDoubleMatrixToStream(std::ostream &out, const ls::DoubleMatrix &results, int nrows);

//The instance count increases/decreases as instances are created/destroyed.
    static int mInstanceCount = 0;

/**
 * The type of sbml element that the RoadRunner::setParameterValue
 * and RoadRunner::getParameterValue method operate on.
 */
    enum ParameterType {
        ptGlobalParameter = 0,
        ptLocalParameter,
        ptBoundaryParameter,
        ptConservationParameter,
        ptFloatingSpecies
    };


/**
 * nicer way to look at the impl object.
 */
#define get_self() RoadRunnerImpl& self = *(this->impl);

/**
 * check if a model is loaded.
 */
#define check_model() { if (!impl->model) { throw std::logic_error(gEmptyModelMessage); } }

/**
 * assert that numbers are similar.
 */
#ifdef abs
#define assert_similar(a, b) assert(a==b || (b != 0 && abs(a/b - 1) < 1e-6))
#else
#define assert_similar(a, b) assert(a==b || (b != 0 && std::abs(a/b - 1) < 1e-6))
#endif

/**
 * implemention class, hide all details here.
 */
    class RoadRunnerImpl {
    public:

        int mInstanceID;

        ls::DoubleMatrix simulationResult;

        /**
         * Points to the current integrator. This is a pointer into the
         * integrators array.
         */
        Integrator *integrator = nullptr;
        std::vector<Integrator *> integrators;

        /**
         * Points to the current steady state solver
         */
        SteadyStateSolver *steady_state_solver = nullptr;
        std::vector<SteadyStateSolver *> steady_state_solvers;

        /**
         * Points to the current sensitivities solver
         */
        SensitivitySolver *sensitivity_solver = nullptr;
        std::vector<SensitivitySolver *> sensitivity_solvers;


        std::vector<SelectionRecord> mSelectionList;

        /**
         * ModelGenerator obtained from the factory
         */
        LoadSBMLOptions loadOpt;

        std::vector<SelectionRecord> mSteadyStateSelection;

        /**
         * here for compatiblity, will go.
         */
        Compiler *compiler;


        /**
         * structural analysis library.
         */
        ls::LibStructural *mLS;

        /**
         * options that are specific to the simulation
         */
        SimulateOptions simulateOpt;

        /**
         * various general options that can be modified by external callers.
         */
        RoadRunnerOptions roadRunnerOptions;

        /**
         * the xml std::string that is given in setConfigurationXML.
         *
         * Needed because the NLEQ is only created in the steadyState method.
         */
        std::string configurationXML;

        /*
        * Has this roadrunner instance been simulated since the last time reset was called?
        */
        bool simulatedSinceReset = false;

        RoadRunnerImpl(const std::string &uriOrSBML, const Dictionary *dict) :
                simulationResult(),
                integrator(0),
                mSelectionList(),
                mSteadyStateSelection(),
                mLS(0),
                simulateOpt(),
                mInstanceID(0),
                loadOpt(dict),
                compiler(Compiler::New()) {
            // have to init integrators the hard way in c++98
            //memset((void*)integrators, 0, sizeof(integrators)/sizeof(char));
        }

        RoadRunnerImpl(const std::string &_compiler, const std::string &_tempDir,
                       const std::string &_supportCodeDir) :
                simulationResult(),
                integrator(NULL),
                integrators(),
                steady_state_solver(NULL),
                steady_state_solvers(),
                mSelectionList(),
                mSteadyStateSelection(),
                mLS(0),
                simulateOpt(),
                mInstanceID(0),
                compiler(Compiler::New()) {
            loadOpt.setItem("compiler", Setting(_compiler));
            loadOpt.setItem("tempDir", Setting(_tempDir));
            loadOpt.setItem("supportCodeDir", Setting(_supportCodeDir));

            // have to init integrators the hard way in c++98
            //memset((void*)integrators, 0, sizeof(integrators)/sizeof(char));
        }

        RoadRunnerImpl(const RoadRunnerImpl &rri) :
                mInstanceID(0),
                simulationResult(rri.simulationResult),
                integrator(NULL),
                integrators(),
                steady_state_solver(NULL),
                steady_state_solvers(),
                mSelectionList(rri.mSelectionList),
                loadOpt(rri.loadOpt),
                mSteadyStateSelection(rri.mSteadyStateSelection),
                //model(NULL), //Create below instead.  Constructing with 'NULL' doesn't work.
                compiler(Compiler::New()),
                mLS(NULL), //Create only if asked.
                simulateOpt(rri.simulateOpt),
                roadRunnerOptions(rri.roadRunnerOptions),
                configurationXML(rri.configurationXML),
                simulatedSinceReset(false),
                document(rri.document->clone()) {
            //There may be an easier way to save and load the model state, but this
            // is at least straightforward.  We call 'saveState', convert it to an
            // input stream, and use that to create a new model.  -LS
            if (rri.model) {
                std::stringstream ss;
                rri.model->saveState(ss);

                std::istringstream istr(ss.str());

                model.reset(ExecutableModelFactory::createModel(istr, loadOpt.modelGeneratorOpt));
                syncAllSolversWithModel(model.get());
            }
            //Set up integrators.
            // We loop through all the integrators in the source, setting the current one to
            //  each in turn, and setting the values of that current one based on the one in
            //  the std::vector of the original rr.
            for (size_t in = 0; in < rri.integrators.size(); in++) {
                string name = rri.integrators[in]->getName();
                Integrator* newint = dynamic_cast<Integrator*>(
                    IntegratorFactory::getInstance().New(name, model.get())
                    );
                integrators.push_back(newint);
                for (const std::string& k : rri.integrators[in]->getSettings()) {
                    auto x = rri.integrators[in]->getValue(k);
                    newint->setValue(k, x);
                }
                if (newint->getName() == name) {
                    integrator = newint;
                    if (model) {
                        integrator->restart(model->getTime());
                    }
                }
            }

            //Set up steady state solvers as we did the integrators.
            for (size_t in = 0; in < rri.steady_state_solvers.size(); in++) {
                string name = rri.steady_state_solvers[in]->getName();
                SteadyStateSolver* newsolver = dynamic_cast<SteadyStateSolver*>(
                    SteadyStateSolverFactory::getInstance().New(name, model.get()));
                steady_state_solvers.push_back(newsolver);
                for (const std::string& k : rri.steady_state_solvers[in]->getSettings()) {
                    auto x = rri.steady_state_solvers[in]->getValue(k);
                    newsolver->setValue(k, x);
                }
                if (newsolver->getName() == name) {
                    steady_state_solver = newsolver;
                }
            }

            //Set up sensitivity solvers as we did the integrators.
            for (size_t in = 0; in < rri.sensitivity_solvers.size(); in++) {
                string name = rri.sensitivity_solvers[in]->getName();
                SensitivitySolver* newsolver = dynamic_cast<SensitivitySolver*>(
                    SensitivitySolverFactory::getInstance().New(name, model.get()));
                sensitivity_solvers.push_back(newsolver);
                for (const std::string& k : rri.sensitivity_solvers[in]->getSettings()) {
                    auto x = rri.sensitivity_solvers[in]->getValue(k);
                    newsolver->setValue(k, x);
                }
                if (newsolver->getName() == name) {
                    sensitivity_solver = newsolver;
                }
            }

        }

        ~RoadRunnerImpl() {
            rrMtx.lock();
            rrLog(Logger::LOG_DEBUG) << __FUNC__ << ", global instance count: " << mInstanceCount;
            rrMtx.unlock();

            delete compiler;
            delete mLS;

            deleteAllSolvers();
            rrMtx.lock();
            mInstanceCount--; // static, so one thread at a time
            rrMtx.unlock();
        }

        void deleteAllSolvers() {
            // remove integrators
            for (auto &integrator: integrators) {
                delete integrator;
                integrator = nullptr;
            }
            integrators.clear();

            // remove steady state solvers
            for (auto &steady_state_solver: steady_state_solvers) {
                delete steady_state_solver;
                steady_state_solver = nullptr;
            }
            steady_state_solvers.clear();

            // remove sensitivity solvers
            for (auto &sensitivity_solver: sensitivity_solvers) {
                delete sensitivity_solver;
                sensitivity_solver = nullptr;
            }
            sensitivity_solvers.clear();
        }

        void syncAllSolversWithModel(ExecutableModel *m) {
            for (auto &integrator: integrators) {
                integrator->syncWithModel(m);
            }
            for (auto &steady_state_solver: steady_state_solvers) {
                steady_state_solver->syncWithModel(m);
            }
            for (auto &sensitivity_solver: sensitivity_solvers) {
                sensitivity_solver->syncWithModel(m);
            }
        }

        void setParameterValue(const ParameterType parameterType,
                               const int parameterIndex, const double value) {
            switch (parameterType) {
                case ptBoundaryParameter:
                    model->setBoundarySpeciesConcentrations(1, &parameterIndex, &value);
                    break;

                case ptGlobalParameter:
                    model->setGlobalParameterValues(1, &parameterIndex, &value);
                    break;

                case ptFloatingSpecies:
                    model->setFloatingSpeciesConcentrations(1, &parameterIndex, &value);
                    break;

                case ptConservationParameter:
                    model->setConservedMoietyValues(1, &parameterIndex, &value);
                    break;

                case ptLocalParameter:
                    throw Exception("Local parameters not permitted in setParameterValue (getCC, getEE)");
            }
        }

        double getParameterValue(const ParameterType parameterType,
                                 const int parameterIndex) {
            switch (parameterType) {
                case ptBoundaryParameter: {
                    double result = 0;
                    model->getBoundarySpeciesConcentrations(1, &parameterIndex, &result);
                    return result;
                }
                    break;
                case ptGlobalParameter: {
                    double result = 0;
                    model->getGlobalParameterValues(1, &parameterIndex, &result);
                    return result;
                }
                    break;

                    // Used when calculating elasticities
                case ptFloatingSpecies: {
                    double result = 0;
                    model->getFloatingSpeciesConcentrations(1, &parameterIndex, &result);
                    return result;
                }
                    break;

                case ptConservationParameter: {
                    double result = 0;
                    model->getConservedMoietyValues(1, &parameterIndex, &result);
                    return result;
                }
                    break;

                case ptLocalParameter:
                    throw Exception("Local parameters not permitted in getParameterValue (getCC?)");
                    break;

                default:
                    return 0.0;
                    break;
            }
            return 0;
        }

        // Changes a given parameter type by the given increment
        void changeParameter(ParameterType parameterType, int reactionIndex, int parameterIndex,
                             double originalValue, double increment) {
            setParameterValue(parameterType, parameterIndex, originalValue + increment);
        }

        //If a kinetic law or two is actually set, even in an FBC model, that means roadrunner can simulate it.  It's possible that someone has loaded an FBC model into rr and has set the kinetic laws for simulation.
        bool isCompleteFBC() {
            libsbml::FbcSBMLDocumentPlugin* fdp = static_cast<libsbml::FbcSBMLDocumentPlugin*>(document->getPlugin("fbc"));
            if (fdp == NULL)
            {
                return false;
            }
            if (fdp->isSetRequired()) {
                libsbml::Model* model = document->getModel();
                for (unsigned int r = 0; r < model->getNumReactions(); r++) {
                    libsbml::Reaction* rxn = model->getReaction(r);
                    if (!rxn->isSetKineticLaw()) {
                        continue;
                    }
                    if (rxn->getKineticLaw()->isSetMath()) {
                        return false;
                    }
                }
                return true;
            }
            return false;
        }

        friend RoadRunner;

    protected:
        std::unique_ptr<ExecutableModel> model;
        std::unique_ptr<libsbml::SBMLDocument> document;

    };


    int RoadRunner::getInstanceCount() {
        std::lock_guard<std::mutex> lock(rrMtx);
        return mInstanceCount;
    }

    int RoadRunner::getInstanceID() {
        return impl->mInstanceID;
    }

    RoadRunner::RoadRunner(unsigned int level, unsigned int version)
        : impl(new RoadRunnerImpl("", NULL))
        , dataVersionNumber(RR_VERSION_MAJOR * 10 + RR_VERSION_MINOR)
    {

        initLLVM();

        // Register integrators, steady state and sensitivity solvers on instantiation
        registerSolvers();

        //Increase instance count..
        rrMtx.lock();
        mInstanceCount++; // static, so one thread at a time
        impl->mInstanceID = mInstanceCount;
        rrMtx.unlock();


        // make CVODE the default integrator
        setIntegrator("cvode");
        // make NLEQ2 the default steady state solver
        setSteadyStateSolver("nleq2");
        // make forward the default sensitivity state solver
        setSensitivitySolver("forward");

        // enable building the model using editing methods
        // and allow simultion without loading SBML files
        impl->document = std::make_unique<libsbml::SBMLDocument>(level, version);
        impl->document->createModel();

    }


    RoadRunner::RoadRunner(const std::string &uriOrSBML, const Dictionary *options)
        : impl(new RoadRunnerImpl(uriOrSBML, options))
        , dataVersionNumber(RR_VERSION_MAJOR * 10 + RR_VERSION_MINOR)
    {
        initLLVM();

        /**
         * Register solvers
         */
        registerSolvers();

        // make CVODE the default integrator
        setIntegrator("cvode");
        // make NLEQ2 the default steady state solver
        setSteadyStateSolver("nleq2");
        // make forward the default steady state solver
        setSensitivitySolver("forward");

        load(uriOrSBML, options);

        //Increase instance count..
        rrMtx.lock();
        mInstanceCount++; // static, so one thread at a time
        impl->mInstanceID = mInstanceCount;
        rrMtx.unlock();


    }


    RoadRunner::RoadRunner(const std::string &_compiler, const std::string &_tempDir,
                           const std::string &supportCodeDir)
        : impl(new RoadRunnerImpl(_compiler, _tempDir, supportCodeDir))
        , dataVersionNumber(RR_VERSION_MAJOR * 10 + RR_VERSION_MINOR)
    {
        initLLVM();
        // must be run to register solvers
        registerSolvers();

        std::string tempDir = _tempDir.empty() ? getTempDir() : _tempDir;

        setTempDir(tempDir);

        //Increase instance count..
        rrMtx.lock();
        mInstanceCount++; // static, so one thread at a time
        impl->mInstanceID = mInstanceCount;
        rrMtx.unlock();


        // make CVODE the default integrator
        setIntegrator("cvode");
        // make NLEQ2 the default steady state solver
        setSteadyStateSolver("nleq2");
        // make forward the default steady state solver
        setSensitivitySolver("forward");
        impl->document = std::make_unique<libsbml::SBMLDocument>(3, 2);
        impl->document->createModel();

    }

    RoadRunner::RoadRunner(const RoadRunner &rr)
        : impl(new RoadRunnerImpl(*rr.impl))
        , dataVersionNumber(RR_VERSION_MAJOR * 10 + RR_VERSION_MINOR)
    {
        reset(SelectionRecord::TIME);

        //Increase instance count..
        rrMtx.lock();
        mInstanceCount++; // static, so one thread at a time
        impl->mInstanceID = mInstanceCount;
        rrMtx.unlock();
    }

    void RoadRunner::operator=(const RoadRunner& rr)
    {
        delete impl;
        impl = new RoadRunnerImpl(*rr.impl);
        reset(SelectionRecord::TIME);

        //Increase instance count..
        rrMtx.lock();
        mInstanceCount++; // static, so one thread at a time
        impl->mInstanceID = mInstanceCount;
        rrMtx.unlock();
    }

    RoadRunner::~RoadRunner() {
        delete impl;
    }

    ExecutableModel *RoadRunner::getModel() {
        return impl->model.get();
    }

    void RoadRunner::setOptions(const RoadRunnerOptions &opt) {
        impl->roadRunnerOptions = opt;
    }

    std::vector<SelectionRecord> RoadRunner::getSelectionList() {
        return impl->mSelectionList;
    }

    std::string RoadRunner::getInfo() {
        applySimulateOptions();

        std::stringstream ss;
        ss << "<roadrunner.RoadRunner() { " << std::endl;
        ss << "'this' : " << (void *) (this) << std::endl;
        ss << "'modelLoaded' : " << (impl->model == NULL ? "false" : "true") << std::endl;

        if (impl->model) {
            ss << "'modelName' : " << impl->model->getModelName() << std::endl;
        }

        ss << "'libSBMLVersion' : " << getVersionStr(VERSIONSTR_LIBSBML) << std::endl;
        ss << "'jacobianStepSize' : " << impl->roadRunnerOptions.jacobianStepSize << std::endl;
        ss << "'steadyStateThreshold' : " << impl->roadRunnerOptions.steadyStateThreshold << std::endl;
        ss << "'fluxThreshold' : " << impl->roadRunnerOptions.fluxThreshold << std::endl;
        ss << "'conservedMoietyAnalysis' : " << rr::toString(impl->loadOpt.getConservedMoietyConversion()) << std::endl;

#if defined(BUILD_LEGACY_C)
                                                                                                                                ss<<"Temporary folder: "        <<    getTempDir()<<std::endl;
    ss<<"Compiler location: "       <<    getCompiler()->getCompilerLocation() << std::endl;
    ss<<"Support Code Folder: "     <<    getCompiler()->getSupportCodeFolder() << std::endl;
    ss<<"Working Directory: "       <<    getCWD() << std::endl;
#endif

        ss << "'simulateOptions' : " << std::endl;
        ss << impl->simulateOpt.toString();
        ss << ", " << std::endl;

        ss << "'integrator' : " << std::endl;
        if (impl->integrator) {
            ss << impl->integrator->toString();
            ss << std::endl;
        } else {
            ss << "Null" << std::endl;
        }

        ss << "}>";

        return ss.str();
    }

    double RoadRunner::getCurrentTime() {
        return getModel()->getTime();
    }

    std::string RoadRunner::getExtendedVersionInfo() {
        std::stringstream info;
        info << getVersionStr(VERSIONSTR_BASIC | VERSIONSTR_COMPILER | VERSIONSTR_DATE | VERSIONSTR_LIBSBML)
             << std::endl;
        info << "Working Directory: " << getCWD() << std::endl;
        return info.str();
    }


    ls::LibStructural *RoadRunner::getLibStruct() {
        std::lock_guard<std::mutex> lock(rrMtx);

        if (impl->mLS) {
            return impl->mLS;
        } else if (impl->document) {
            impl->mLS = new ls::LibStructural(getSBML());
            rrLog(Logger::LOG_INFORMATION) << "created structural analysis, messages: "
                                           << impl->mLS->getAnalysisMsg();
            return impl->mLS;
        } else {
            throw std::invalid_argument("could not create structural analysis with no loaded sbml");
        }
    }

    Compiler *RoadRunner::getCompiler() {
        return impl->compiler;
    }


    void RoadRunner::setCompiler(const std::string &compiler) {
        impl->loadOpt.setItem("compiler", Setting(compiler));
    }

    Integrator *RoadRunner::getIntegrator() {
        //applySimulateOptions();
        return impl->integrator;
    }

    SteadyStateSolver *RoadRunner::getSteadyStateSolver() {
        return impl->steady_state_solver;
    }

    SensitivitySolver *RoadRunner::getSensitivitySolver() {
        //applySimulateOptions();
        return impl->sensitivity_solver;
    }


    Integrator *RoadRunner::getIntegratorByName(const std::string &name) {
        // ensure it exists
        makeIntegrator(name);
        // find it and return
        for (auto integrator: impl->integrators) {
            if (integrator->getName() == name) {
                return integrator;
            }
        }
        throw std::runtime_error("No integrator implemented for \"" + name + "\"");
    }

    SteadyStateSolver *RoadRunner::getSteadyStateSolverByName(const std::string &name) {
        // ensure it exists
        makeSteadyStateSolver(name);
        // find it and return
        for (auto ssSolver: impl->steady_state_solvers) {
            if (ssSolver->getName() == name) {
                return ssSolver;
            }
        }
        throw std::runtime_error("No integrator implemented for \"" + name + "\"");
    }

    SensitivitySolver *RoadRunner::getSensitivitySolverByName(const std::string &name) {
        // ensure it exists
        makeSensitivitySolver(name);
        // find it and return
        for (auto sensitivitySolver: impl->sensitivity_solvers) {
            if (sensitivitySolver->getName() == name) {
                return sensitivitySolver;
            }
        }
        throw std::runtime_error("No integrator implemented for \"" + name + "\"");
    }


    Integrator *RoadRunner::makeIntegrator(const std::string &name) {
        if (integratorExists(name)) {
            rrLog(Logger::LOG_DEBUG) << "Integrator \"" << name << "\" already exists";
            return NULL;
        }
        rrLog(Logger::LOG_DEBUG) << "Creating new integrator for " << name;
        Integrator *result = dynamic_cast<Integrator *>(
                IntegratorFactory::getInstance().New(name, impl->model.get())
        );
        impl->integrators.push_back(result);
        return result;
    }


    SteadyStateSolver *RoadRunner::makeSteadyStateSolver(const std::string &name) {
        if (steadyStateSolverExists(name)) {
            rrLog(Logger::LOG_DEBUG) << "SteadyStateSolver \"" << name << "\" already exists";
            return NULL;
        }
        rrLog(Logger::LOG_DEBUG) << "Creating new SteadyStateSolver for " << name;
        SteadyStateSolver *result = dynamic_cast<SteadyStateSolver *>(
                SteadyStateSolverFactory::getInstance().New(name, impl->model.get())
        );
        impl->steady_state_solvers.push_back(result);
        return result;
    }

    SensitivitySolver *RoadRunner::makeSensitivitySolver(const std::string &name) {
        if (sensitivitySolverExists(name)) {
            rrLog(Logger::LOG_DEBUG) << "SensitivitySolver \"" << name << "\" already exists";
            return NULL;
        }
        rrLog(Logger::LOG_DEBUG) << "Creating new SensitivitySolver for " << name;
        SensitivitySolver *result = dynamic_cast<SensitivitySolver *>(
                SensitivitySolverFactory::getInstance().New(name, impl->model.get())
        );
        impl->sensitivity_solvers.push_back(result);
        return result;
    }

    std::vector<std::string> RoadRunner::getExistingIntegratorNames() {
        std::vector<std::string> result;
        for (auto integrator: impl->integrators) {
            result.push_back(integrator->getName());
        }
        return result;
    }

    std::vector<std::string> RoadRunner::getExistingSteadyStateSolverNames() {
        std::vector<std::string> result;
        for (auto solver: impl->steady_state_solvers) {
            result.push_back(solver->getName());
        }
        return result;
    }

    std::vector<std::string> RoadRunner::getExistingSensitivitySolverNames() {
        std::vector<std::string> result;
        for (auto sensSolver: impl->sensitivity_solvers) {
            result.push_back(sensSolver->getName());
        }
        return result;
    }

    std::vector<std::string> RoadRunner::getRegisteredIntegratorNames() {
        std::vector<std::string> result;
        for (int n = 0; n < IntegratorFactory::getInstance().size(); ++n) {
            result.push_back(IntegratorFactory::getInstance().name(n));
        }
        return result;
    }

    std::vector<std::string> RoadRunner::getRegisteredSteadyStateSolverNames() {
        std::vector<std::string> result;
        for (int n = 0; n < SteadyStateSolverFactory::getInstance().size(); ++n) {
            result.push_back(SteadyStateSolverFactory::getInstance().name(n));
        }
        return result;
    }

    std::vector<std::string> RoadRunner::getRegisteredSensitivitySolverNames() {
        std::vector<std::string> result;
        for (int n = 0; n < SensitivitySolverFactory::getInstance().size(); ++n) {
            result.push_back(SensitivitySolverFactory::getInstance().name(n));
        }
        return result;
    }


    void RoadRunner::setIntegrator(const std::string &name) {
        rrLog(Logger::LOG_DEBUG) << "Setting integrator to " << name;
        // Try to set integrator from an existing reference.
        if (integratorExists(name)) {
            for (auto integrator: impl->integrators) {
                if (integrator->getName() == name) {
                    rrLog(Logger::LOG_DEBUG) << "Using pre-existing integrator for " << name;
                    impl->integrator = integrator;
                }
            }
        }
            // Otherwise, create a new integrator.
        else {
            impl->integrator = makeIntegrator(name);
        }
    }

    void RoadRunner::setSteadyStateSolver(const std::string &name) {
        rrLog(Logger::LOG_DEBUG) << "Setting steady state solver to " << name;
        // Try to set steady_state_solver from an existing reference.
        if (steadyStateSolverExists(name)) {
            for (auto solver: impl->steady_state_solvers) {
                if (solver->getName() == name) {
                    rrLog(Logger::LOG_DEBUG) << "Using pre-existing steady state solver for " << name;
                    impl->steady_state_solver = solver;
                }
            }
        }
            // Otherwise, create a new steady state solver.
        else {
            rrLog(Logger::LOG_DEBUG) << "Creating new steady state solver for " << name;
            impl->steady_state_solver = dynamic_cast<SteadyStateSolver *>(
                    SteadyStateSolverFactory::getInstance().New(name, impl->model.get())
            );
            impl->steady_state_solvers.push_back(impl->steady_state_solver);
        }
    }

    void RoadRunner::setSensitivitySolver(const std::string &name) {
        rrLog(Logger::LOG_DEBUG) << "Setting Sensitivity solver to " << name;
        // Try to set steady_state_solver from an existing reference.
        if (sensitivitySolverExists(name)) {
            for (auto solver: impl->sensitivity_solvers) {
                if (solver->getName() == name) {
                    rrLog(Logger::LOG_DEBUG) << "Using pre-existing sensitivity solver for " << name;
                    impl->sensitivity_solver = solver;
                }
            }
        }
            // Otherwise, create a new sensitivity solver.
        else {
            rrLog(Logger::LOG_DEBUG) << "Creating new sensitivity solver for " << name;
            impl->sensitivity_solver = dynamic_cast<SensitivitySolver *>(
                    SensitivitySolverFactory::getInstance().New(name, impl->model.get())
            );
            impl->sensitivity_solvers.push_back(impl->sensitivity_solver);
        }
    };

    bool RoadRunner::integratorExists(const std::string &name) {
        for (auto it: impl->integrators) {
            if (it->getName() == name) {
                return true;
            }
        }
        return false;
    }

    bool RoadRunner::steadyStateSolverExists(const std::string &name) {
        for (auto it: impl->steady_state_solvers) {
            if (it->getName() == name) {
                return true;
            }
        }
        return false;
    }

    bool RoadRunner::sensitivitySolverExists(const std::string &name) {
        for (auto it: impl->sensitivity_solvers) {
            if (it->getName() == name) {
                return true;
            }
        }
        return false;
    }

    bool RoadRunner::solversRegistered = false;

    void RoadRunner::registerSolvers() {
        // must be run to register solvers at startup
        std::lock_guard<std::mutex> lock(rrMtx);
        if (!solversRegistered) {
            IntegratorFactory::Register();
            SteadyStateSolverFactory::Register();
            SensitivitySolverFactory::Register();
            solversRegistered = true;
        }
    }

    bool RoadRunner::isModelLoaded() {
        return impl->model ? true : false;
    }

    void RoadRunner::setSimulateOptions(const SimulateOptions &settings) {
        impl->simulateOpt = settings;
    }

    SimulateOptions &RoadRunner::getSimulateOptions() {
        return impl->simulateOpt;
    }

    bool RoadRunner::getConservedMoietyAnalysis() {
        return impl->loadOpt.getConservedMoietyConversion();
    }

    /** @cond PRIVATE */
    void RoadRunner::setTempDir(const std::string &folder) {
        impl->loadOpt.setItem("tempDir", Setting(folder));
    }

    std::string RoadRunner::getTempDir() {
        return impl->loadOpt.getItem("tempDir");
    }

    const libsbml::SBase* RoadRunner::getElementWithMathematicalMeaning(const libsbml::Model* model, const std::string& id)
    {
        const libsbml::SBase* element = model->getSpecies(id);
        if (!element) {
            element = model->getParameter(id);
            if (!element) {
                element = model->getCompartment(id);
                if (!element) {
                    libsbml::ListOfReactions* lor = const_cast<libsbml::ListOfReactions*>(model->getListOfReactions());
                    element = lor->getElementBySId(id);
                }
            }
        }
        return element;
    }
    /** @endcond PRIVATE */

    size_t RoadRunner::createDefaultTimeCourseSelectionList() {
        std::vector<std::string> selections;
        std::vector<std::string> oFloating = getFloatingSpeciesIds();
        size_t numFloatingSpecies = oFloating.size();
        //int numIndSpecies = getNumberOfIndependentSpecies();

        // add floating species to the default selection

        selections.push_back("time");
        //for(int i = 0; i < numIndSpecies; i++)
        for (int i = 0; i < numFloatingSpecies; i++) {
            selections.push_back("[" + oFloating[i] + "]");
        }

        // add parameters defined by rate rules to the default selection

        std::vector<std::string> selections_with_ratelaws(selections);
        try {
            std::vector<std::string> raterule_symbols = impl->model->getRateRuleSymbols();
            for (std::vector<std::string>::iterator i = raterule_symbols.begin(); i != raterule_symbols.end(); ++i)
                if (impl->model->getFloatingSpeciesIndex(*i) == -1)
                    // not push the species again
                    selections_with_ratelaws.push_back(*i);
        } catch (NotImplementedException) {
            rrLog(Logger::LOG_WARNING) << "Querying rate rule symbols not supported with this executable model";
        }

        // try to apply selections including rate laws,
        // fall back if unable to use rate laws
        // see https://github.com/sys-bio/roadrunner/issues/88
        try {
            setSelections(selections_with_ratelaws);
        } catch (...) {
            rrLog(Logger::LOG_WARNING) << "Rate laws exist but cannot be added to default selections";
            setSelections(selections);
        }

        rrLog(lDebug) << "The following is selected:";
        for (int i = 0; i < impl->mSelectionList.size(); i++) {
            rrLog(lDebug) << impl->mSelectionList[i];
        }
        return impl->mSelectionList.size();
    }

    size_t RoadRunner::createTimeCourseSelectionList() {
        // make a list out of the values in the settings,
        // will always have at least a "time" at the first item.
        std::vector<std::string> settingsList = createSelectionList(impl->simulateOpt);

        assert(settingsList.size() >= 1 && "selection list from SimulateOptions does does not have time");

        // if settings has any selections, this overrides the selection list.
        if (settingsList.size() > 1) {
            rrLog(Logger::LOG_INFORMATION) << "overriding selection list with values from SimulateOptions.";
            setSelections(settingsList);
        }

        for (int i = 0; i < impl->mSelectionList.size(); i++) {
            rrLog(Logger::LOG_DEBUG) << "Selection Value [" << i << "]: " << impl->mSelectionList[i].to_repr();
        }

        return impl->mSelectionList.size();
    }

    std::string RoadRunner::getParamPromotedSBML(const std::string &sbml) {
        if (SBMLReader::is_sbml(sbml)) {
            libsbml::SBMLDocument *doc = libsbml::readSBMLFromString(sbml.c_str());
            // converts in-place
            libsbml::SBMLLocalParameterConverter converter;

            converter.setDocument(doc);

            converter.convert();

            libsbml::SBMLWriter writer;

            std::stringstream stream;

            writer.writeSBML(doc, stream);

            delete doc;

            return stream.str();
        } else {
            libsbml::SBMLDocument *doc = libsbml::readSBMLFromFile(sbml.c_str());
            // converts in-place
            libsbml::SBMLLocalParameterConverter converter;

            converter.setDocument(doc);

            converter.convert();

            libsbml::SBMLWriter writer;

            std::stringstream stream;

            writer.writeSBML(doc, stream);

            delete doc;

            return stream.str();
        }
    }

    double RoadRunner::getValue(const SelectionRecord &record) {
        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }

        double dResult;

        switch (record.selectionType) {
            case SelectionRecord::FLOATING_CONCENTRATION:

                dResult = 0;
                impl->model->getFloatingSpeciesConcentrations(1, &record.index, &dResult);
                break;

            case SelectionRecord::BOUNDARY_CONCENTRATION:
                impl->model->getBoundarySpeciesConcentrations(1, &record.index, &dResult);
                break;

            case SelectionRecord::REACTION_RATE:
                dResult = 0;
                impl->model->getReactionRates(1, &record.index, &dResult);
                break;

            case SelectionRecord::FLOATING_AMOUNT_RATE:
                dResult = 0;
                try {
                    impl->model->getFloatingSpeciesAmountRates(1, &record.index, &dResult);
                }
                catch (std::out_of_range) {
                    std::stringstream err;
                    err << "No rate available for floating species " << record.p1
                        << ": if conserved moieties are enabled, this species may be defined by an implied assignment rule instead, and its rate cannot be determined.";
                    throw std::invalid_argument(err.str());
                }

                break;

            case SelectionRecord::FLOATING_CONCENTRATION_RATE: {
                dResult = 0;
                std::stringstream err;
                try {
                    impl->model->getFloatingSpeciesAmountRates(1, &record.index, &dResult);
                }
                catch (std::out_of_range) {
                    err << "No rate available for floating species " << record.p1
                        << ": if conserved moieties are enabled, this species may be defined by an implied assignment rule instead, and its rate cannot be determined.";
                    throw std::invalid_argument(err.str());
                }
                try {
                    int compidx = impl->model->getCompartmentIndexForFloatingSpecies(record.index);
                    string compid = impl->model->getCompartmentId(compidx);
                    double comprate = getValue(compid + "'");
                    double compvol = getValue(compid);
                    double species_conc = getValue("[" + record.p1 + "]");
                    dResult = (dResult - (comprate * species_conc)) / compvol;
                }
                catch (std::invalid_argument) {
                    err << "No rate available for the concentration of floating species " << record.p1
                        << ": its compartment is defined by an assignment rule, and its rate of change cannot be calculated.";
                    throw std::invalid_argument(err.str());
                }
                break;
            }
            case SelectionRecord::BOUNDARY_CONCENTRATION_RATE: {
                dResult = 0;
                std::stringstream err;
                try {
                    dResult = getValue(record.p1 + "'");
                }
                catch (std::out_of_range) {
                    err << "No rate available for boundary species " << record.p1
                        << ": if it is defined by an assignment rule, its rate cannot be determined.";
                    throw std::invalid_argument(err.str());
                }
                try {
                    int compidx = impl->model->getCompartmentIndexForBoundarySpecies(record.index);
                    string compid = impl->model->getCompartmentId(compidx);
                    double comprate = getValue(compid + "'");
                    double compvol = getValue(compid);
                    double species_conc = getValue("[" + record.p1 + "]");
                    dResult = (dResult - (comprate * species_conc)) / compvol;
                }
                catch (std::invalid_argument) {
                    err << "No rate available for the concentration of boundary species " << record.p1
                        << ": its compartment is defined by an assignment rule, and its rate of change cannot be calculated.";
                    throw std::invalid_argument(err.str());
                }
                break;
            }
            case SelectionRecord::COMPARTMENT_RATE:
            case SelectionRecord::GLOBAL_PARAMETER_RATE:
            case SelectionRecord::BOUNDARY_AMOUNT_RATE: {
                dResult = 0;
                list<string> rrs, ars;
                impl->model->getRateRuleIds(rrs);
                impl->model->getAssignmentRuleIds(ars);
                for (list<string>::iterator ar = ars.begin(); ar != ars.end(); ar++) {
                    if (*ar == record.p1) {
                        std::stringstream err;
                        err << "No rate available for compartment " << record.p1
                            << " because the volume is defined by an assignment rule.";
                        throw std::invalid_argument(err.str());
                    }
                }
                int index = 0;
                for (list<string>::iterator rr = rrs.begin(); rr != rrs.end(); rr++) {
                    if (*rr == record.p1) {
                        impl->model->getRateRuleRates(1, &index, &dResult);
                        break;
                    }
                    index++;
                }
                break;
            }
            case SelectionRecord::COMPARTMENT:
                impl->model->getCompartmentVolumes(1, &record.index, &dResult);
                break;

            case SelectionRecord::GLOBAL_PARAMETER: {
                if (record.index > ((impl->model->getNumGlobalParameters()) - 1)) {
                    int index = record.index - impl->model->getNumGlobalParameters();
                    impl->model->getConservedMoietyValues(1, &index, &dResult);
                } else {
                    impl->model->getGlobalParameterValues(1, &record.index, &dResult);
                }
            }
                break;

            case SelectionRecord::FLOATING_AMOUNT:
                impl->model->getFloatingSpeciesAmounts(1, &record.index, &dResult);
                break;

            case SelectionRecord::BOUNDARY_AMOUNT:
                impl->model->getBoundarySpeciesAmounts(1, &record.index, &dResult);
                break;

            case SelectionRecord::ELASTICITY:
                dResult = getEE(record.p1, record.p2, false);
                break;

            case SelectionRecord::UNSCALED_ELASTICITY:
                dResult = getuEE(record.p1, record.p2, false);
                break;

            case SelectionRecord::CONTROL:
                dResult = getCC(record.p1, record.p2);
                break;

            case SelectionRecord::UNSCALED_CONTROL:
                dResult = getuCC(record.p1, record.p2);
                break;
            case SelectionRecord::EIGENVALUE_REAL: {
                std::string species = record.p1;
                int index = impl->model->getFloatingSpeciesIndex(species);

                if (index < 0) {
                    throw std::logic_error("Invalid species id" + record.p1 + " for eigenvalue");
                }

                std::vector<std::complex<double>> eig = getEigenValues(JACOBIAN_FULL);

                if (eig.size() <= index) {
                    // this should NEVER happen
                    throw std::runtime_error("Eigenvalues std::vector length less than species id");
                }

                return std::real(eig[index]);
            }
                break;
            case SelectionRecord::EIGENVALUE_IMAG: {
                std::string species = record.p1;
                int index = impl->model->getFloatingSpeciesIndex(species);

                if (index < 0) {
                    throw std::logic_error("Invalid species id" + record.p1 + " for eigenvalue");
                }

                std::vector<std::complex<double>> eig = getEigenValues(JACOBIAN_FULL);

                if (eig.size() <= index) {
                    // this should NEVER happen
                    throw std::runtime_error("Eigenvalues std::vector length less than species id");
                }

                return std::imag(eig[index]);
            }
                break;
            case SelectionRecord::EIGENVALUE_COMPLEX: {
                double real = getValue("eigenReal(" + record.p1 + ")");
                double imag = getValue("eigenImag(" + record.p1 + ")");
                if (imag == 0) {
                    return real;
                }
                std::stringstream err;
                err << "Eigenvalue is complex (" << real << "+" << imag
                    << "j):  unable to return value as a double.  Request the real and imaginary parts separately with 'eigenReal("
                    << record.p1 << ") and 'eigenImag(" << record.p1 << ").";
                throw std::invalid_argument(err.str());
            }
                break;
            case SelectionRecord::INITIAL_FLOATING_CONCENTRATION:
                impl->model->getFloatingSpeciesInitConcentrations(1, &record.index, &dResult);
                break;
            case SelectionRecord::INITIAL_FLOATING_AMOUNT:
                impl->model->getFloatingSpeciesInitAmounts(1, &record.index, &dResult);
                break;
            case SelectionRecord::INITIAL_BOUNDARY_CONCENTRATION:
                impl->model->getBoundarySpeciesInitConcentrations(1, &record.index, &dResult);
                break;
            case SelectionRecord::INITIAL_BOUNDARY_AMOUNT:
                impl->model->getBoundarySpeciesInitAmounts(1, &record.index, &dResult);
                break;
            case SelectionRecord::INITIAL_GLOBAL_PARAMETER:
                impl->model->getGlobalParameterInitValues(1, &record.index, &dResult);
                break;
            case SelectionRecord::INITIAL_COMPARTMENT:
                impl->model->getCompartmentInitVolumes(1, &record.index, &dResult);
                break;
            case SelectionRecord::STOICHIOMETRY: {
                // in case it is entered in the form of stoich(SpeciesId, ReactionId)
                if (impl->model->getFloatingSpeciesIndex(record.p1) != -1 && impl->model->getReactionIndex(record.p2) != -1)
                    dResult = impl->model->getStoichiometry(impl->model->getStoichiometryIndex(record.p1, record.p2));
                // in case it is entered in the form of a stoichiometry parameter
                else
                    dResult = impl->model->getStoichiometry(impl->model->getStoichiometryIndex(record.p1));
                break;
            }
            case SelectionRecord::TIME:
                dResult = getCurrentTime();
                break;
            default:
                dResult = 0.0;
                break;
        }
        return dResult;
    }

    double RoadRunner::getNthSelectedOutput(size_t index, double currentTime) {
        const SelectionRecord &record = impl->mSelectionList[index];

        if (record.selectionType == SelectionRecord::TIME) {
            return currentTime;
        } else {
            return getValue(record);
        }
    }

    int RoadRunner::getTimeRowIndex() {
        for (u_int j = 0; j < impl->mSelectionList.size(); j++) {
            const SelectionRecord &record = impl->mSelectionList[j];
            if (record.selectionType == SelectionRecord::TIME) {
                return (int) j;
            }
        }
        // -1 for failure
        return -1;
    }

    void RoadRunner::getSelectedValues(ls::DoubleMatrix &results, int nRow, double currentTime) {
        for (u_int j = 0; j < impl->mSelectionList.size(); j++) {
            double out = getNthSelectedOutput(j, currentTime);
            results(nRow, j) = out;
        }
    }

    void RoadRunner::getSelectedValues(std::vector<double> &results,
                                       double currentTime) {
        assert(results.size() == impl->mSelectionList.size()
               && "given std::vector and selection list different size");

        size_t size = results.size();
        for (size_t i = 0; i < size; ++i) {
            results[i] = getNthSelectedOutput(i, currentTime);
        }
    }


    std::vector<double> RoadRunner::getConservedMoietyValues() {
        return getLibStruct()->getConservedSums();
    }

    void RoadRunner::load(const std::string &uriOrSbml, const Dictionary *dict) {
        std::string currentSBML = SBMLReader::read(uriOrSbml);

        // chomp any leading or trailing whitespace
        currentSBML = trim(currentSBML);

        impl->model.reset(nullptr);

        delete impl->mLS;
        impl->mLS = NULL;

        if (dict) {
            impl->loadOpt = LoadSBMLOptions(dict);
        }

        // TODO: add documentation for validations
        if ((impl->loadOpt.loadFlags & LoadSBMLOptions::TURN_ON_VALIDATION) != 0) {
            std::string errors = validateSBML(currentSBML, VALIDATE_GENERAL | VALIDATE_IDENTIFIER | VALIDATE_MATHML |
                                                           VALIDATE_OVERDETERMINED);
            if (!errors.empty()) {
                throw std::runtime_error(errors.c_str());
            }
        }
        try {

            // the following lines load and compile the model. If anything fails here,
            // we validate the model to provide explicit details about where it
            // failed. Its *VERY* expensive to pre-validate the model.
            libsbml::SBMLReader reader;
            rrMtx.lock();
            libsbml::SBMLDocument* doc = reader.readSBMLFromString(currentSBML);
            rrMtx.unlock();
            impl->document.reset(doc);
            std::string md5 = rr::getSBMLMD5(currentSBML, (int)impl->loadOpt.modelGeneratorOpt);

            // Fix various problems that might occur in incomplete SBML models.  This makes it out of sync with the md5 hash, but the hash still maps to a unique document.
            fixMissingStoichAndMath(impl->document.get());

            impl->model.reset(ExecutableModelFactory::createModel(impl->document.get(), md5, &impl->loadOpt));
        } catch (const rr::UninitializedValueException &e) {
            // catch specifically for UninitializedValueException, otherwise for some
            // reason the message is erased, and an 'unknown error' is displayed to the user.
            throw e;
        } catch (const rrllvm::LLVMException &e) {
            // catch specifically for LLVMException, otherwise the exception type is removed,
            // and an 'unknown error' is displayed to the user.
            throw e;
        }
        catch (const std::runtime_error &e) {
            throw e;
        }
        catch (const std::domain_error& e) {
            throw e;
        }
        catch (const std::exception &e) {
            std::string errors = validateSBML(impl->document.get());

            if (!errors.empty()) {
                rrLog(Logger::LOG_ERROR) << "Invalid SBML: " << std::endl << errors;
            }

            // re-throw the exception
            throw e;
        }

        impl->syncAllSolversWithModel(impl->model.get());

        reset();
        if ((impl->loadOpt.loadFlags & LoadSBMLOptions::NO_DEFAULT_SELECTIONS) == 0) {
            createDefaultSelectionLists();
        }
    }

    bool RoadRunner::createDefaultSelectionLists() {
        bool result = true;

        //Create a default timecourse selectionlist
        if (!createDefaultTimeCourseSelectionList()) {
            rrLog(lDebug) << "Failed creating default timecourse selectionList.";
            result = false;
        } else {
            rrLog(lDebug) << "Created default TimeCourse selection list.";
        }

        //Create a defualt steady state selectionlist
        if ((impl->loadOpt.loadFlags & LoadSBMLOptions::NO_DEFAULT_STEADY_STATE_SELECTIONS) == 0 && !createDefaultSteadyStateSelectionList()) {
            rrLog(lDebug) << "Failed creating default steady state selectionList.";
            result = false;
        } else {
            rrLog(lDebug) << "Created default SteadyState selection list.";
        }
        return result;
    }

    bool RoadRunner::clearModel() {
        // The model owns the shared library (if it exists), when the model is deleted,
        // its dtor unloads the shared lib.
        impl->document.reset(new libsbml::SBMLDocument(3, 2));
        impl->document->createModel();
        if (impl->model) {
            impl->model = nullptr;

            delete impl->mLS;
            impl->mLS = NULL;
            return true;
        }
        return false;
    }

    void RoadRunner::reset() {
        uint opt1 = rr::Config::getInt(rr::Config::MODEL_RESET);

        reset(opt1);
        //With other fixes, this doesn't work:  it resets all global parameters when it shouldn't.
        //uint opt2 = rr::SelectionRecord::DEPENDENT_INITIAL_GLOBAL_PARAMETER;
        //reset((int) opt2 | opt1);
    }

    void RoadRunner::reset(int options) {
        impl->simulatedSinceReset = false;
        if (impl->model) {
            // model gets set to before time = 0
            impl->model->reset(options);

            if (!impl->integrator)
                throw std::runtime_error("No integrator set");
            impl->integrator->restart(0.0);

            try {
                impl->model->testConstraints();
            }
            catch (const Exception &e) {
                rrLog(lWarning) << "Constraint Violated at time = 0\n" << e.Message();
            }
        }
    }

    void RoadRunner::resetSelectionLists() {
        createDefaultSelectionLists();
    }

    bool RoadRunner::populateResult() {
        std::vector<std::string> list(impl->mSelectionList.size());

        for (int i = 0; i < impl->mSelectionList.size(); ++i) {
            list[i] = impl->mSelectionList[i].to_string();
        }

        return true;
    }


    std::vector<rr::SelectionRecord> &RoadRunner::getSelections() {
        return impl->mSelectionList;
    }


    double RoadRunner::mcaSteadyState() {
        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }

        double start = impl->simulateOpt.start;
        double duration = impl->simulateOpt.duration;
        int steps = impl->simulateOpt.steps;
        impl->simulateOpt.start = 0;
        impl->simulateOpt.duration = 10.0;
        impl->simulateOpt.steps = 100;
        simulate();
        impl->simulateOpt.start = start;
        impl->simulateOpt.duration = duration;
        impl->simulateOpt.steps = steps;

        return steadyState();
    }

    double RoadRunner::steadyState(Dictionary *dict) {
        rrLog (Logger::LOG_DEBUG) << "RoadRunner::steadyState...";
        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }

        // store the name of the solver for later use.
        const std::string &solverName = impl->steady_state_solver->getName();

        // automatic detection of requirement for conserved moiety analysis
        bool currentConservedMoietyAnalysisStatus = getConservedMoietyAnalysis();
        if (getSteadyStateSolver()->getValue("auto_moiety_analysis")) {
            rrLog(Logger::LOG_DEBUG) << "Checking whether moiety conservation analysis is needed" << std::endl;
            if (!impl->loadOpt.getConservedMoietyConversion()) {
                /*
                 * Note this is an expensive operation. The other way
                 * of determining need for moiety conservation is to
                 * compute whether the jacobian matrix is singular
                 * (with SVD::isSingular). This as it turns out, is even more
                 * expensive.
                 */
                setConservedMoietyAnalysis(true);
                int numConservedMoieties = getModel()->getNumConservedMoieties();
                if (numConservedMoieties == 0) {
                    setConservedMoietyAnalysis(false);
                } else {
                    rrLog(Logger::LOG_WARNING) << "Turning on moiety conservation analysis "
                                                  "because this model has " << numConservedMoieties
                                               << "conserved moieties";
                }
            }
        }

        metabolicControlCheck(impl->model.get());

        if (!impl->steady_state_solver) {
            rrLog(Logger::LOG_ERROR) << "No steady state solver";
            throw std::runtime_error("No steady state solver");
        }

        // apply user settings
        impl->steady_state_solver->updateSettings(dict);

        // initialize pointer to decorator
        // Decorators wrap the solve method and modifies its behaviour.
        // The first solve() method attempted is always the undecorated version
        // The first decorator applied is the first attempted to be executed.
        //  i.e. we try Presimulation then approximation.
        PresimulationProgramDecorator *presimDec = nullptr;
        ApproxSteadyStateDecorator *approxDec = nullptr;

        // copy of steady state solver pointer to put back after
        // were done with modifications
//    SteadyStateSolver* copyOfSSSolverPtr;

        // apply presimulation decorator if requested by user
        if (impl->steady_state_solver->getValue("allow_presimulation")) {
            presimDec = new PresimulationProgramDecorator(impl->steady_state_solver);
            impl->steady_state_solver = presimDec;
        }

        // apply approximation decorator if requested by user
        if (impl->steady_state_solver->getValue("allow_approx")) {
            approxDec = new ApproxSteadyStateDecorator(impl->steady_state_solver);
            impl->steady_state_solver = approxDec;
        }

        double ss = impl->steady_state_solver->solve();

        if (presimDec) {
            delete presimDec;
            presimDec = nullptr;
        }

        if (approxDec) {
            delete approxDec;
            approxDec = nullptr;
        }

        // put back to original type before return
        // so the next call to steadyState starts
        // without any decorators.
        setSteadyStateSolver(solverName);
        if (!currentConservedMoietyAnalysisStatus)
            setConservedMoietyAnalysis(currentConservedMoietyAnalysisStatus);

        return ss;
    }


    ls::DoubleMatrix RoadRunner::steadyStateNamedArray(const Dictionary *dict) {
        steadyState();
        return getSteadyStateValuesNamedArray();
    }


    void RoadRunner::setConservedMoietyAnalysis(bool value) {
        get_self();

        if (value == self.loadOpt.getConservedMoietyConversion()) {
            rrLog(lDebug) << "The compute and assign conservation laws flag already set to : " << toString(value);
        }

        self.loadOpt.setConservedMoietyConversion(value);

        if (self.model != NULL) {
            uint32_t savedOpt = self.loadOpt.modelGeneratorOpt;

            // have to reload
            self.loadOpt.modelGeneratorOpt |= LoadSBMLOptions::RECOMPILE;

            //load(getSBML());
            regenerateModel(true);

            // restore original reload value
            self.loadOpt.modelGeneratorOpt = savedOpt;
        }
    }


// [Help("Get scaled elasticity coefficient with respect to a global parameter or species")]
    double RoadRunner::getEE(const std::string &reactionName, const std::string &parameterName) {
        return getEE(reactionName, parameterName, true);
    }

    double
    RoadRunner::getEE(const std::string &reactionName, const std::string &parameterName, bool computeSteadyState) {
        ParameterType parameterType;
        int reactionIndex;
        int parameterIndex;

        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }

        // Check the reaction name
        if ((reactionIndex = impl->model->getReactionIndex(reactionName)) < 0) {
            throw CoreException(format("Unable to locate reaction name: [{0}]", reactionName));
        }

        // Find out what kind of parameter we are dealing with
        if ((parameterIndex = impl->model->getFloatingSpeciesIndex(parameterName)) >= 0) {
            parameterType = ptFloatingSpecies;
        } else if ((parameterIndex = impl->model->getBoundarySpeciesIndex(parameterName)) >= 0) {
            parameterType = ptBoundaryParameter;
        } else if ((parameterIndex = impl->model->getGlobalParameterIndex(parameterName)) >= 0) {
            parameterType = ptGlobalParameter;
        } else if ((parameterIndex = impl->model->getConservedMoietyIndex(parameterName)) >= 0) {
            throw std::invalid_argument("Cannot calculate elasticities for conserved moieties.");
        } else {
            throw CoreException(format("Unable to locate variable: [{0}]", parameterName));
        }


        double variableValue = 0;
        impl->model->getReactionRates(1, &reactionIndex, &variableValue);
        double parameterValue = impl->getParameterValue(parameterType, parameterIndex);
        if (variableValue == 0) {
            variableValue = 1e-12;
        }
        return getuEE(reactionName, parameterName, computeSteadyState) * parameterValue / variableValue;
    }

    double RoadRunner::getDiffStepSize() const {
        if (!impl)
            throw std::runtime_error("Missing impl");
        return impl->roadRunnerOptions.diffStepSize;
    }

    void RoadRunner::setDiffStepSize(double val) {
        if (!impl)
            throw std::runtime_error("Missing impl");
        impl->roadRunnerOptions.diffStepSize = val;
    }

    double RoadRunner::getSteadyStateThreshold() const {
        if (!impl)
            throw std::runtime_error("Missing impl");
        return impl->roadRunnerOptions.steadyStateThreshold;
    }

    void RoadRunner::setSteadyStateThreshold(double val) {
        if (!impl)
            throw std::runtime_error("Missing impl");
        impl->roadRunnerOptions.steadyStateThreshold = val;
    }

    double RoadRunner::getFluxThreshold() const {
        if (!impl)
            throw std::runtime_error("Missing impl");
        return impl->roadRunnerOptions.fluxThreshold;
    }

    void RoadRunner::setFluxThreshold(double val) {
        if (!impl)
            throw std::runtime_error("Missing impl");
        impl->roadRunnerOptions.fluxThreshold = val;
    }

    double RoadRunner::getuEE(const std::string &reactionName, const std::string &parameterName) {
        return getuEE(reactionName, parameterName, true);
    }

    void RoadRunner::fixDependentSpeciesValues(int except, double *ref) {
        int l = impl->model->getNumFloatingSpecies();
        double *vals = new double[l];
        impl->model->getFloatingSpeciesConcentrations(l, NULL, vals);
        for (int i = 0; i < l; ++i) {
            if (i != except && vals[i] != ref[i]) {
                impl->model->setFloatingSpeciesConcentrations(1, &i, &ref[i]);
            }
        }
        delete[] vals;
    }

    bool RoadRunner::llvmInitialized = false;

    void RoadRunner::initLLVM() {
        // only do this once
        std::lock_guard<std::mutex> lock(rrMtx);
        if (!llvmInitialized) {
            /**
             * The main program should call this function to
             * initialize the native target corresponding to the host.  This is useful
             * for JIT applications to ensure that the target gets linked in correctly.
             * It is legal for a client to make multiple calls to this function.
             */
            llvm::InitializeNativeTarget(); // looks like this has been renamed to LLVMInitializeNativeTarget: https://stackoverflow.com/a/21149010/3059024

            /**
             * The main program should call
             * this function to initialize the native target assembly (asm) printer.
             */
            llvm::InitializeNativeTargetAsmPrinter();

            /**
             * The main program should call
             *  this function to initialize the native target assembly (asm) parser.
             */
            llvm::InitializeNativeTargetAsmParser();

            llvmInitialized = true;
        }
    }

    double
    RoadRunner::getuEE(const std::string &reactionName, const std::string &parameterName, bool computeSteadystate) {
        try {
            if (!impl->model) {
                throw CoreException(gEmptyModelMessage);
            }

            ParameterType parameterType;
            double originalParameterValue;
            int reactionIndex;
            int parameterIndex;

            int l = impl->model->getNumFloatingSpecies();
            double* ref = new double[l];
            impl->model->getFloatingSpeciesConcentrations(l, NULL, ref);

            // Check the reaction name
            if ((reactionIndex = impl->model->getReactionIndex(reactionName)) < 0) {
                throw CoreException("Unable to locate reaction name: [" + reactionName + "]");
            }

            // Find out what kind of parameter we are dealing with
            if ((parameterIndex = impl->model->getFloatingSpeciesIndex(parameterName)) >= 0) {
                parameterType = ptFloatingSpecies;
                originalParameterValue = 0;
                impl->model->getFloatingSpeciesConcentrations(1, &parameterIndex, &originalParameterValue);
            }
            else if ((parameterIndex = impl->model->getBoundarySpeciesIndex(parameterName)) >= 0) {
                parameterType = ptBoundaryParameter;
                originalParameterValue = 0;
                impl->model->getBoundarySpeciesConcentrations(1, &parameterIndex, &originalParameterValue);
            }
            else if ((parameterIndex = impl->model->getGlobalParameterIndex(parameterName)) >= 0) {
                if (impl->model->getConservedMoietyIndex(parameterName) >= 0) {
                    throw std::invalid_argument("Cannot calculate elasticities for conserved moieties.");
                }
                parameterType = ptGlobalParameter;
                originalParameterValue = 0;
                impl->model->getGlobalParameterValues(1, &parameterIndex, &originalParameterValue);
            }
            else if ((parameterIndex = impl->model->getConservedMoietyIndex(parameterName)) >= 0) {
                parameterType = ptConservationParameter;
                originalParameterValue = 0;
                impl->model->getConservedMoietyValues(1, &parameterIndex, &originalParameterValue);
            }
            else {
                throw CoreException("Unable to locate variable: [" + parameterName + "]");
            }

            double hstep = impl->roadRunnerOptions.diffStepSize * originalParameterValue;
            if (fabs(hstep) < 1E-12) {
                hstep = impl->roadRunnerOptions.diffStepSize;
            }

            impl->setParameterValue(parameterType, parameterIndex, originalParameterValue + hstep);
            fixDependentSpeciesValues(parameterIndex, ref);
            double fi = 0;
            impl->model->getReactionRates(1, &reactionIndex, &fi);

            impl->setParameterValue(parameterType, parameterIndex, originalParameterValue + 2 * hstep);
            fixDependentSpeciesValues(parameterIndex, ref);
            double fi2 = 0;
            impl->model->getReactionRates(1, &reactionIndex, &fi2);

            impl->setParameterValue(parameterType, parameterIndex, originalParameterValue - hstep);
            fixDependentSpeciesValues(parameterIndex, ref);
            double fd = 0;
            impl->model->getReactionRates(1, &reactionIndex, &fd);

            impl->setParameterValue(parameterType, parameterIndex, originalParameterValue - 2 * hstep);
            fixDependentSpeciesValues(parameterIndex, ref);
            double fd2 = 0;
            impl->model->getReactionRates(1, &reactionIndex, &fd2);

            // restore original value
            impl->setParameterValue(parameterType, parameterIndex, originalParameterValue);
            fixDependentSpeciesValues(-1, ref);

            // Use instead the 5th order approximation double unscaledValue = (0.5/hstep)*(fi-fd);
            // The following separated lines avoid small amounts of roundoff error
            double f1 = fd2 + 8 * fi;
            double f2 = -(8 * fd + fi2);

            delete[] ref;

            return 1 / (12 * hstep) * (f1 + f2);
        }
        catch (const Exception& e) {
            throw CoreException("Unexpected error from getuEE(): " + e.Message());
        }
    }

    // Help("Updates the model based on all recent changes")
    void RoadRunner::evalModel() {
        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }
        impl->model->getStateVectorRate(impl->model->getTime(), 0);
    }

    bool hasTime(const libsbml::ASTNode* astn)
    {
        if (astn->getType() == libsbml::AST_NAME_TIME) {
            return true;
        }
        for (int c = 0; c < astn->getNumChildren(); c++) {
            if (hasTime(astn->getChild(c))) {
                return true;
            }
        }
        return false;
    }

    const ls::DoubleMatrix* RoadRunner::simulate(const SimulateOptions* opt) {
        get_self();
        check_model();

        if (opt) {
            self.simulateOpt = *opt;
        }

        if (impl->isCompleteFBC()) {
            rrLog(Logger::LOG_ERROR) << "FBC model discovered, but not simulatable.";
            throw std::domain_error("This SBML model contains information from the 'fbc' package for Flux Balance Control analysis, or constraint-based modeling.  These models can be analyzed but not simulated by roadrunner or tellurium.  The most popular software package that supports fbc (as of 2023) is COBRA (https://opencobra.github.io/), and other software packages exist as well.");

        }

        if (self.integrator->getName() == "gillespie") {
            // Throw error if model has rate rules.
            if (self.model->getNumRateRules() > 0) {
                std::stringstream err;
                err << "The gillespie integrator is unable to simulate a model with rate rules.  Either change the rate rules into reactions, or remove them from the model.  The IDs of elements with rate rules are: ";
                list<string> rrids;
                self.model->getRateRuleIds(rrids);
                for (auto rr = rrids.begin(); rr != rrids.end(); rr++) {
                    err << *rr;
                    if (rr != rrids.begin()) {
                        err << ", ";
                    }
                }
                err << ".";
                throw std::domain_error(err.str());
            }
            // Warn if event triggers have time in them.
            const libsbml::ListOfEvents* loe = self.document->getModel()->getListOfEvents();
            for (int e = 0; e < loe->size(); e++) {
                const libsbml::Event* event = loe->get(e);
                const libsbml::ASTNode* trigger = event->getTrigger()->getMath();
                if (hasTime(trigger)) {
                    rrLog(Logger::LOG_ERROR) << "An event involving 'time' is present in this model, but time is not treated continuously in a gillespie simulation.  Continuing, but the simulation will not be precise.  Consider changing the trigger to involve species levels instead.";
                }
                if (event->isSetDelay()) {
                    rrLog(Logger::LOG_ERROR) << "An event with a delay is present in this model, but time is not treated continuously in a gillespie simulation.  Continuing, but the delay will not be precise.  Consider changing the trigger to not have a delay instead.";
                }
            }
        }

        applySimulateOptions();

        const double timeStart = self.simulateOpt.start;
        self.integrator->setIntegrationStartTime(self.simulateOpt.start);
        self.model->setIntegrationStartTime(self.simulateOpt.start);

        const double timeEnd = self.simulateOpt.duration + self.simulateOpt.start;

        impl->simulatedSinceReset = true;

        const std::vector<SelectionRecord> &selList = getSelectionList();
        bool writeToFile = !self.simulateOpt.output_file.empty() && !selList.empty();
        std::string outfname = changeFileExtensionTo(self.simulateOpt.output_file, ".csv");
        std::ofstream outfile;
        int kRowsPerWrite = rr::Config::K_ROWS_PER_WRITE;  // only write to outfile every k rows processed
        if (kRowsPerWrite < 1) {
            // there are no checks for validity in rrConfig, so the check is done here.
            throw std::invalid_argument("Config.K_ROWS_PER_WRITE should be at least 1.");
        }
        // used only for initializing double matrices for fixed time step simulations
        // if writing to output file, this is kRowsPerWrite, i.e. the buffer matrix size
        // otherwise this is just the full matrix size
        unsigned int bufSize = writeToFile ? kRowsPerWrite : (self.simulateOpt.steps + 1);
        if (writeToFile) {
            std::string outfname = changeFileExtensionTo(self.simulateOpt.output_file, ".csv");
            rrLog(Logger::LOG_DEBUG) << "Writing simulation result to output file '" << outfname
                                     << "'...";
            outfile = std::ofstream();
            outfile.open(outfname);
            if (!outfile) {
                std::ostringstream ss;
                ss << "cannot open file '" << outfname << "' for writing.";
                throw std::invalid_argument(ss.str());
            }

            // write header
            outfile << selList[0].to_string();
            for (int i = 1; i < selList.size(); i++) {
                outfile << "," << selList[i].to_string();
            }
            outfile << "\n";
        }

        // evalute the model with its current state
        self.model->getStateVectorRate(timeStart, 0, 0);
        // Variable Time Step Integration
        if (self.integrator->hasValue("variable_step_size") && self.integrator->getValue("variable_step_size")) {
            rrLog(Logger::LOG_INFORMATION) << "Performing variable step integration";

            int max_output_rows = Config::getInt(Config::MAX_OUTPUT_ROWS);
            if (self.integrator->hasValue("max_output_rows")) {
                max_output_rows = (int) self.integrator->getValue("max_output_rows");
            }

            if (self.simulateOpt.duration <= 0 && self.simulateOpt.steps <= 0) {
                throw CoreException(
                        "For simulations with variable step sizes, you must either set a time duration, or set the total number of steps to take.  Neither were set.");
            }

            DoubleVectorList results;
            std::vector<double> row(self.mSelectionList.size());

            try {
                // add current state as first row
                getSelectedValues(row, timeStart);
                results.push_back(row);

                self.integrator->restart(timeStart);

                double tout = timeStart;
                double last_tout = tout;

                // optimiziation for certain getValue operations.
                self.model->setIntegration(true);

                int i = 1;

                // If there is an output file, do not place a constraint on max output rows
                while (((self.simulateOpt.duration && tout <= timeEnd) ||
                        (!self.simulateOpt.duration && self.simulateOpt.steps && i < self.simulateOpt.steps)) &&
                       (writeToFile || !max_output_rows || i < max_output_rows)) {
                    // "flush" results to file
                    if (writeToFile && results.size() >= kRowsPerWrite) {
                        writeDoubleVectorListToStream(outfile, results);
                        results.clear();
                    }

                    rrLog(Logger::LOG_DEBUG) << "variable step, start: " << tout
                                             << ", end: " << timeEnd;
                    tout = self.integrator->integrate(tout, timeEnd - tout);

                    //If tout is larger than timeEnd but not infinite, this is actually OK, because
                    // it means that we're just taking a certain number of steps and don't care
                    // about the time.
                    if (!isfinite(tout) || (tout == timeEnd)) {
                        // time step is at infinity or maximum so bail, but get the last value
                        getSelectedValues(row, timeEnd);
                        results.push_back(row);
                        break;
                    }

                    getSelectedValues(row, tout);

                    // use interpolation to ensure last time point is requested end time
                    if (tout > timeEnd) {
                        if (!getIntegrator()) {
                            // should never happen
                            throw CoreException("No integrator selected in call to simulate");
                        }

                        if (getIntegrator()->getName() == "gillespie") {
                            // stochastic simulations use flat interpolation
                            rrLog(Logger::LOG_DEBUG)
                                << "simulate: use flat interpolation for last value with timeEnd = " << timeEnd
                                << ", tout = " << tout << ", last_tout = " << last_tout;

                            for (int n = 0; n < row.size(); ++n) {
                                row.at(n) = results.back().at(n);
                            }

                            int itime = getTimeRowIndex();

                            if (itime >= 0) {
                                row.at(itime) = timeEnd;
                            }
                        } else {
                            // ODE simulations use linear interpolation
                            double alpha = (timeEnd - last_tout) / (tout - last_tout);
                            rrLog(Logger::LOG_DEBUG)
                                << "simulate: use linear interpolation for last value with timeEnd = " << timeEnd
                                << ", tout = " << tout << ", last_tout = " << last_tout;

                            for (int n = 0; n < row.size(); ++n) {
                                row.at(n) = results.back().at(n) + alpha * (row.at(n) - results.back().at(n));
                            }
                        }
                    } else {
                        last_tout = tout;
                    }

                    results.push_back(row);

                    ++i;
                }
                // flush leftover result
                if (writeToFile && results.size() > 0) {
                    writeDoubleVectorListToStream(outfile, results);
                    results.clear();
                }
            }
            catch (EventListenerException &e) {
                rrLog(Logger::LOG_NOTICE) << e.what();
            }

            // stuff list values into result matrix
            self.simulationResult.resize(static_cast<unsigned int>(results.size()),
                                         static_cast<unsigned int>(row.size()));
            uint rowi = 0;
            for (DoubleVectorList::const_iterator i = results.begin();
                 i != results.end(); ++i, ++rowi) {
                // evidently [] operator gets row, go figure...
                double *prow = self.simulationResult[rowi];
                std::copy(i->begin(), i->end(), prow);
            }
        }

            // Stochastic Fixed Step Integration
            // do fixed time step simulation, these are different for deterministic
            // and stochastic.
        else if (self.integrator->getIntegrationMethod() == Integrator::Stochastic) {
            rrLog(Logger::LOG_INFORMATION)
                << "Performing stochastic fixed step integration for "
                << self.simulateOpt.steps + 1 << " steps";

            int numPoints = self.simulateOpt.steps + 1;

            if (numPoints <= 1) {
                numPoints = 2;
            }

            unsigned int nrCols = static_cast<unsigned int>(self.mSelectionList.size());

            rrLog(Logger::LOG_DEBUG) << "starting simulation with " << nrCols << " selected columns";

            // ignored if same
            self.simulationResult.resize(bufSize, nrCols);

            try {
                // add current state as first row
                getSelectedValues(self.simulationResult, 0, timeStart);

                self.integrator->restart(timeStart);

                double tout = timeStart;           // the exact times the integrator returns

                // for writing to file; only used if output_file is specified
                // this gets reset to 0 when the buffer matrix is written to output and cleared
                int bufIndex = 1;
                // index gets bumped in do-while loop.
                for (int i = 1; i < self.simulateOpt.steps + 1;) {
                    double next = self.simulateOpt.getNext(i);
                    rrLog(Logger::LOG_DEBUG) << "step: " << i << "t0: " << tout << "hstep: " << next - tout;

                    // stochastic frequently overshoots time end
                    // may also be infinite
                    tout = self.integrator->integrate(tout, next - tout);

                    assert((tout >= next)
                           && "stochastic integrator did not integrate to end time");
                    // get the output, always get at least one output
                    do {
                        if (writeToFile && bufIndex == kRowsPerWrite) {
                            writeDoubleMatrixToStream(outfile, self.simulationResult, bufIndex);
                            // clear results
                            self.simulationResult = ls::DoubleMatrix(bufSize, nrCols);
                            bufIndex = 0;
                        }
                        getSelectedValues(self.simulationResult, i, next);
                        i++;
                        bufIndex++;
                        next = self.simulateOpt.getNext(i);
                    } while ((i < self.simulateOpt.steps + 1) && tout > next);
                }
                // write leftover stuff
                if (writeToFile) {
                    if (bufIndex >= 1) {
                        writeDoubleMatrixToStream(outfile, self.simulationResult, bufIndex);
                    }
                    self.simulationResult = ls::DoubleMatrix();
                }
            }
            catch (EventListenerException &e) {
                rrLog(Logger::LOG_NOTICE) << e.what();
            }
        }

            // Deterministic Fixed Step Integration
        else {
            rrLog(Logger::LOG_INFORMATION)
                << "Performing deterministic fixed step integration for  "
                << self.simulateOpt.steps + 1 << " steps";
            // always have 1 more number of time points than we do steps (or intervals)
            int numPoints = self.simulateOpt.steps + 1;

            if (numPoints <= 1) {
                numPoints = 2;
            }

            unsigned int nrCols = static_cast<unsigned int>(self.mSelectionList.size());

            rrLog(Logger::LOG_DEBUG) << "starting simulation with " << nrCols << " selected columns";

            // ignored if same
            self.simulationResult.resize(bufSize, nrCols);

            try {
                // add current state as first row
                getSelectedValues(self.simulationResult, 0, timeStart);

                self.integrator->restart(timeStart);

                // optimiziation for certain getValue operations.
                self.model->setIntegration(true);

                double tout = timeStart;

                // for write to file
                int bufIndex = 1;

                // integrate
                for (int i = 1; i < self.simulateOpt.steps + 1; i++, bufIndex++) {
                    if (writeToFile && bufIndex == kRowsPerWrite) {
                        writeDoubleMatrixToStream(outfile, self.simulationResult, bufIndex);
                        // clear buf
                        self.simulationResult = ls::DoubleMatrix(bufSize, nrCols);
                        bufIndex = 0;
                    }
                    rrLog(Logger::LOG_DEBUG) << "Step " << i;
                    double next = self.simulateOpt.getNext(i);
                    double itime = self.integrator->integrate(tout, next - tout);

                    // the test suite is extremly sensitive to time differences,
                    // so need to use the *exact* time here. occasionally the integrator
                    // will return a value just slightly off from the exact time
                    // value.
                    tout = next;
                    getSelectedValues(self.simulationResult, bufIndex, tout);
                }
                // write leftover stuff
                if (writeToFile) {
                    if (bufIndex >= 1) {
                        writeDoubleMatrixToStream(outfile, self.simulationResult, bufIndex);
                    }
                    self.simulationResult = ls::DoubleMatrix();
                }
            }
            catch (EventListenerException &e) {
                rrLog(Logger::LOG_NOTICE) << e.what();
            }
        }

        // done with integration

        self.model->setIntegration(false);

        rrLog(Logger::LOG_DEBUG) << "Simulation done.";

        return &self.simulationResult;
    }

    const ls::DoubleMatrix *RoadRunner::simulate(double start, double stop, int points) {
        SimulateOptions opt;
        opt.start = start;
        opt.duration = stop;
        // substract 1 so that start, stop, num has the same meaning as it does in numpy functions
        // i.e. see https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
        opt.steps = points - 1;
        return simulate(&opt);
    }

    const ls::DoubleMatrix *RoadRunner::simulate(const std::vector<double> &times) {
        SimulateOptions opt;
        opt.times = times;
        return simulate(&opt);
    }

    Matrix3D<double, double> RoadRunner::timeSeriesSensitivities(
            double start, double stop, int num,
            std::vector<std::string> params,
            std::vector<std::string> species,
            int k) {
        auto tsSensSolver = dynamic_cast<TimeSeriesSensitivitySolver *>(getSensitivitySolver());

        return tsSensSolver->solveSensitivities(
                start, stop, num, std::move(params), std::move(species), k);
    }

    double RoadRunner::oneStep(const double currentTime, const double stepSize, const bool reset) {
        get_self();
        check_model();
        applySimulateOptions();

        try {
            if (reset) {
                self.integrator->restart(currentTime);
            }
            return self.integrator->integrate(currentTime, stepSize);
        }
        catch (EventListenerException &e) {
            rrLog(Logger::LOG_NOTICE) << e.what();
            return self.model->getTime();
        }
    }


    double RoadRunner::internalOneStep(const double currentTime, const double stepSize, const bool reset) {
        get_self();
        check_model();
        applySimulateOptions();
        double endTime;

        bool temp_var = self.integrator->getValue("variable_step_size");
        self.integrator->setValue("variable_step_size", Setting(true));

        try {
            if (reset) {
                self.integrator->restart(currentTime);
            }
            endTime = self.integrator->integrate(currentTime, stepSize);
            self.integrator->setValue("variable_step_size", Setting(temp_var));
            rrLog(Logger::LOG_DEBUG) << "internalOneStep: " << endTime;
            return endTime;
        }
        catch (EventListenerException &e) {
            rrLog(Logger::LOG_NOTICE) << e.what();
            endTime = self.model->getTime();
            self.integrator->setValue("variable_step_size", Setting(temp_var));
            return endTime;
        }
    }


    std::vector<std::complex<double>> RoadRunner::getFullEigenValues() {
        return getEigenValues(JACOBIAN_FULL);
    }


    std::vector<std::complex<double>> RoadRunner::getReducedEigenValues() {
        return getEigenValues(JACOBIAN_REDUCED);
    }

    std::vector<std::complex<double> > RoadRunner::getEigenValues(RoadRunner::JacobianMode mode) {
        check_model();
        ls::DoubleMatrix mat;

        if (mode == JACOBIAN_FULL) {
            mat = getFullJacobian();
        } else {
            mat = getReducedJacobian();
        }
        return ls::getEigenValues(mat);
    }


    DoubleMatrix RoadRunner::getFullEigenValuesNamedArray() {
        return getEigenValuesNamedArray(JACOBIAN_FULL);
    }

    DoubleMatrix RoadRunner::getReducedEigenValuesNamedArray() {
        return getEigenValuesNamedArray(JACOBIAN_REDUCED);
    }


    DoubleMatrix RoadRunner::getEigenValuesNamedArray(RoadRunner::JacobianMode mode) {
        check_model();
        DoubleMatrix mat;

        if (mode == JACOBIAN_FULL) {
            mat = getFullJacobian();
        } else {
            mat = getReducedJacobian();
        }
        std::vector<std::complex<double> > eigens = ls::getEigenValues(mat);
        //std::vector< std::complex<double> > eigens = getEigenValues(mode);
        DoubleMatrix v(eigens.size(), 2);
        for (unsigned int i = 0; i < eigens.size(); i++) {
            v(i, 0) = std::real(eigens[i]);
            v(i, 1) = std::imag(eigens[i]);
        }
        v.setRowNames(mat.getRowNames());
        std::vector<std::string> colnames;
        colnames.push_back("real");
        colnames.push_back("imaginary");
        v.setColNames(colnames);
        return v;
    }


    ls::DoubleMatrix RoadRunner::getFloatingSpeciesAmountsNamedArray() {
        check_model();

        int l = impl->model->getNumFloatingSpecies();

        double *vals = new double[l];
        impl->model->getFloatingSpeciesAmounts(l, NULL, vals);

        LibStructural *ls = getLibStruct();
        ls::DoubleMatrix v(1, l);

        for (int i = 0; i < l; ++i)
            v(0, i) = vals[i];

        delete[] vals;

        v.setColNames(getFloatingSpeciesIds());

        return v;
    }

    ls::DoubleMatrix RoadRunner::getFloatingSpeciesConcentrationsNamedArray() {
        check_model();

        int l = impl->model->getNumFloatingSpecies();

        double *vals = new double[l];
        impl->model->getFloatingSpeciesConcentrations(l, NULL, vals);

        LibStructural *ls = getLibStruct();
        ls::DoubleMatrix v(1, l);

        for (int i = 0; i < l; ++i)
            v(0, i) = vals[i];

        delete[] vals;

        v.setColNames(getFloatingSpeciesIds());

        return v;
    }

    std::vector<double> RoadRunner::getIndependentFloatingSpeciesAmountsV() {
        check_model();

        std::vector<double> result(getNumberOfIndependentSpecies(), 0);
        //The independent species are at the beginning of the vector
        if (result.size()) {
            impl->model->getFloatingSpeciesAmounts(result.size(), 0, &result[0]);
        }
        return result;
    }

    std::vector<double> RoadRunner::getDependentFloatingSpeciesAmountsV() {
        check_model();

        std::vector<double> result(getNumberOfDependentSpecies(), 0);
        //The dependent species are after the independent ones
        if (result.size()) {
            int indfloats = getNumberOfIndependentSpecies();
            impl->model->getFloatingSpeciesAmounts(result.size(), &indfloats, &result[0]);
        }
        return result;
    }

    std::vector<double> RoadRunner::getIndependentFloatingSpeciesConcentrationsV() {
        check_model();

        std::vector<double> result(getNumberOfIndependentSpecies(), 0);
        //The independent species are at the beginning of the vector
        if (result.size()) {
            impl->model->getFloatingSpeciesConcentrations(result.size(), 0, &result[0]);
        }
        return result;
    }

    std::vector<double> RoadRunner::getDependentFloatingSpeciesConcentrationsV() {
        check_model();

        std::vector<double> result(getNumberOfDependentSpecies(), 0);
        //The dependent species are after the independent ones
        if (result.size()) {
            int indfloats = getNumberOfIndependentSpecies();
            impl->model->getFloatingSpeciesConcentrations(result.size(), &indfloats, &result[0]);
        }
        return result;
    }

    ls::DoubleMatrix RoadRunner::getIndependentFloatingSpeciesAmountsNamedArray() {
        check_model();

        size_t num_ind = getNumberOfIndependentSpecies();
        ls::DoubleMatrix result(1, num_ind);

        std::vector<double> vals = getIndependentFloatingSpeciesAmountsV();

        for (int i = 0; i < num_ind; i++) {
            result(0, i) = vals[i];
        }

        result.setColNames(getIndependentFloatingSpeciesIds());
        return result;
    }

    ls::DoubleMatrix RoadRunner::getDependentFloatingSpeciesAmountsNamedArray() {
        check_model();

        size_t num_dep = getNumberOfDependentSpecies();
        ls::DoubleMatrix result(1, num_dep);

        std::vector<double> vals = getDependentFloatingSpeciesAmountsV();

        for (int i = 0; i < num_dep; i++) {
            result(0, i) = vals[i];
        }

        result.setColNames(getDependentFloatingSpeciesIds());
        return result;
    }

    ls::DoubleMatrix RoadRunner::getIndependentFloatingSpeciesConcentrationsNamedArray() {
        check_model();

        size_t num_ind = getNumberOfIndependentSpecies();
        ls::DoubleMatrix result(1, num_ind);

        std::vector<double> vals = getIndependentFloatingSpeciesConcentrationsV();

        for (int i = 0; i < num_ind; i++) {
            result(0, i) = vals[i];
        }

        result.setColNames(getIndependentFloatingSpeciesIds());
        return result;
    }

    ls::DoubleMatrix RoadRunner::getDependentFloatingSpeciesConcentrationsNamedArray() {
        check_model();

        size_t num_dep = getNumberOfDependentSpecies();
        ls::DoubleMatrix result(1, num_dep);

        std::vector<double> vals = getDependentFloatingSpeciesConcentrationsV();

        for (int i = 0; i < num_dep; i++) {
            result(0, i) = vals[i];
        }

        result.setColNames(getDependentFloatingSpeciesIds());
        return result;
    }


    std::vector<double> RoadRunner::getRatesOfChange() {
        check_model();

        int ssvl = impl->model->getStateVector(NULL);
        double *vals = new double[ssvl];
        double *ssv = new double[ssvl];
        std::vector<double> v(ssvl);

        impl->model->getStateVector(ssv);
        impl->model->getStateVectorRate(impl->model->getTime(), ssv, vals);

        if (getConservedMoietyAnalysis()) {
            LibStructural *ls = getLibStruct();
            ls::DoubleMatrix lm = *ls->getLinkMatrix();

            for (int i = 0; i < ssvl; ++i) {
                double sum = 0;
                for (int j = 0; j < ssvl; ++j) {
                    sum += lm[i][j] * vals[j];
                }
                v[i] = sum;
            }
        } else {
            for (int i = 0; i < ssvl; ++i)
                v[i] = vals[i];
        }

        delete[] vals;
        delete[] ssv;

        return v;
    }

    ls::DoubleMatrix RoadRunner::getRatesOfChangeNamedArray() {
        check_model();

        int ssvl = impl->model->getStateVector(NULL);
        double *vals = new double[ssvl];
        double *ssv = new double[ssvl];
        ls::DoubleMatrix v(1, ssvl);

        impl->model->getStateVector(ssv);
        impl->model->getStateVectorRate(impl->model->getTime(), ssv, vals);
        if (getConservedMoietyAnalysis()) {
            LibStructural *ls = getLibStruct();
            ls::DoubleMatrix lm = *ls->getLinkMatrix();

            for (int i = 0; i < ssvl; ++i) {
                double sum = 0;
                for (int j = 0; j < ssvl; ++j) {
                    sum += lm[i][j] * vals[j];
                }
                v(0, i) = sum;
            }
        } else {
            for (int i = 0; i < ssvl; ++i)
                v(0, i) = vals[i];
        }

        delete[] vals;
        delete[] ssv;

        v.setColNames(getRateOfChangeIds());

        return v;
    }

    std::vector<double> RoadRunner::getIndependentRatesOfChange() {
        check_model();

        std::vector<std::string> idfsId = getIndependentFloatingSpeciesIds();
        std::vector<std::string> fsId = getFloatingSpeciesIds();
        size_t nindep = idfsId.size();
        std::vector<double> v(nindep);

        std::vector<double> rate = getRatesOfChange();

        for (int i = 0; i < nindep; ++i) {
            std::vector<std::string>::iterator it = find(fsId.begin(), fsId.end(), idfsId[i]);
            size_t index = distance(fsId.begin(), it);

            v[i] = rate[index];
        }

        return v;
    }

    ls::DoubleMatrix RoadRunner::getIndependentRatesOfChangeNamedArray() {
        check_model();

        vector<double> vals = getIndependentRatesOfChange();
        int nindep = vals.size();
        ls::DoubleMatrix v(1, nindep);

        for (int i = 0; i < nindep; ++i) {
            v(0, i) = vals[i];
        }

        v.setColNames(getIndependentFloatingSpeciesIds());

        return v;
    }

    std::vector<double> RoadRunner::getDependentRatesOfChange() {
        check_model();

        std::vector<std::string> dfsId = getDependentFloatingSpeciesIds();
        std::vector<std::string> fsId = getFloatingSpeciesIds();
        size_t ndep = dfsId.size();
        std::vector<double> v(ndep);

        std::vector<double> rate = getRatesOfChange();

        for (int i = 0; i < ndep; ++i) {
            std::vector<std::string>::iterator it = find(fsId.begin(), fsId.end(), dfsId[i]);
            size_t index = distance(fsId.begin(), it);

            v[i] = rate[index];
        }

        return v;
    }

    ls::DoubleMatrix RoadRunner::getDependentRatesOfChangeNamedArray() {
        check_model();

        vector<double> vals = getDependentRatesOfChange();
        int ndep = vals.size();
        ls::DoubleMatrix v(1, ndep);

        for (int i = 0; i < ndep; ++i) {
            v(0, i) = vals[i];
        }

        v.setColNames(getDependentFloatingSpeciesIds());

        return v;
    }

    ls::DoubleMatrix RoadRunner::getFullJacobian() {
        check_model();

        get_self();

        // function pointers to the model get values and get init values based on
        // if we are doing amounts or concentrations.
        typedef int (ExecutableModel::*GetValueFuncPtr)(size_t len, int const *indx,
                                                        double *values);
        typedef int (ExecutableModel::*SetValueFuncPtr)(size_t len, int const *indx,
                                                        double const *values);
        typedef int (ExecutableModel::* SetValueFuncPtrSize)(size_t len, int const *indx,
                                                             double const *values);

        GetValueFuncPtr getValuePtr = 0;
        GetValueFuncPtr getInitValuePtr = 0;
        SetValueFuncPtr setValuePtr = 0;
        SetValueFuncPtrSize setInitValuePtr = 0;

        if (Config::getValue(Config::ROADRUNNER_JACOBIAN_MODE).getAs<std::int32_t>() ==
            Config::ROADRUNNER_JACOBIAN_MODE_AMOUNTS) {
            getValuePtr = &ExecutableModel::getFloatingSpeciesAmounts;
            getInitValuePtr = &ExecutableModel::getFloatingSpeciesInitAmounts;
            setValuePtr = &ExecutableModel::setFloatingSpeciesAmounts;
            setInitValuePtr = &ExecutableModel::setFloatingSpeciesInitAmounts;
        } else {
            getValuePtr = &ExecutableModel::getFloatingSpeciesConcentrations;
            getInitValuePtr = &ExecutableModel::getFloatingSpeciesInitConcentrations;
            setValuePtr = &ExecutableModel::setFloatingSpeciesConcentrations;
            setInitValuePtr = &ExecutableModel::setFloatingSpeciesInitConcentrations;
        }


        if (self.model->getNumReactions() == 0 && self.model->getNumRateRules() > 0) {
            if (self.model->getNumFloatingSpecies() < self.model->getNumRateRules()) {
                std::stringstream errSS;
                errSS << "cannot compute full Jacobian because there are fewer floating species (" <<
                      self.model->getNumFloatingSpecies() << ")" << " than rate rules ("
                      << self.model->getNumRateRules() << ").";
                errSS << " You may need to declare one or more of your variables as a species. See " <<
                      "https://tellurium.readthedocs.io/en/latest/antimony.html#rate-rules if you are using Antimony.";
                throw std::out_of_range(errSS.str());
            }
            ls::DoubleMatrix jac(self.model->getNumRateRules(), self.model->getNumRateRules());

            for (int i = 0; i < self.model->getNumRateRules(); i++) {
                for (int j = 0; j < self.model->getNumRateRules(); j++) {
                    double value;
                    double originalConc = 0;
                    double result = std::numeric_limits<double>::quiet_NaN();

                    // note setting init values auotmatically sets the current values to the
                    // init values

                    // this causes a reset, so need to save the current amounts to set them back
                    // as init conditions.
                    std::vector<double> conc(self.model->getNumFloatingSpecies());
                    if (conc.size()) {
                        (self.model.get()->*getValuePtr)(conc.size(), 0, &conc[0]);
                    }

                    // save the original init values
                    std::vector<double> initConc(self.model->getNumFloatingSpecies());
                    if (initConc.size()) {
                        (self.model.get()->*getInitValuePtr)(initConc.size(), 0, &initConc[0]);
                    }

                    // get the original value
                    (self.model.get()->*getValuePtr)(1, &j, &originalConc);

                    // now we start changing things
                    try {
                        // set init amounts to current amounts, restore them later.
                        // have to do this as this is only way to set conserved moiety values
                        if (conc.size()) {
                            (self.model.get()->*setInitValuePtr)(conc.size(), 0, &conc[0]);
                        }

                        // sanity check
                        assert_similar(originalConc, conc[j]);
                        double tmp = 0;
                        (self.model.get()->*getInitValuePtr)(1, &j, &tmp);
                        assert_similar(originalConc, tmp);
                        (self.model.get()->*getValuePtr)(1, &j, &tmp);
                        assert_similar(originalConc, tmp);

                        // things check out, start fiddling...
                        double hstep = self.roadRunnerOptions.diffStepSize * originalConc;
                        if (fabs(hstep) < 1E-12) {
                            hstep = self.roadRunnerOptions.diffStepSize;
                        }

                        value = originalConc + hstep;
                        setValue(self.model->getFloatingSpeciesId(j), value);
                        double fi = 0;
                        fi = getRatesOfChange()[i];

                        value = originalConc + 2 * hstep;
                        setValue(self.model->getFloatingSpeciesId(j), value);
                        double fi2 = 0;
                        fi2 = getRatesOfChange()[i];

                        value = originalConc - hstep;
                        setValue(self.model->getFloatingSpeciesId(j), value);
                        double fd = 0;
                        fd = getRatesOfChange()[i];

                        value = originalConc - 2 * hstep;
                        setValue(self.model->getFloatingSpeciesId(j), value);
                        double fd2 = 0;
                        fd2 = getRatesOfChange()[i];

                        // Use instead the 5th order approximation
                        // double unscaledElasticity = (0.5/hstep)*(fi-fd);
                        // The following separated lines avoid small amounts of roundoff error
                        double f1 = fd2 + 8 * fi;
                        double f2 = -(8 * fd + fi2);

                        result = 1 / (12 * hstep) * (f1 + f2);
                    }
                    catch (const std::exception &) {
                        // What ever happens, make sure we restore the species level
                        if (initConc.size()) {
                            (self.model.get()->*setInitValuePtr)(initConc.size(), 0, &initConc[0]);
                        }
                        // only set the indep species, setting dep species is not permitted.
                        if (conc.size()) {
                            (self.model.get()->*setValuePtr)(self.model->getNumFloatingSpecies(), 0, &conc[0]);
                        }

                        // re-throw the exception.
                        throw;
                    }

                    // What ever happens, make sure we restore the species level
                    if (initConc.size()) {
                        (self.model.get()->*setInitValuePtr)(initConc.size(), 0, &initConc[0]);
                    }

                    // only set the indep species, setting dep species is not permitted.
                    if (conc.size()) {
                        (self.model.get()->*setValuePtr)(self.model->getNumFloatingSpecies(), 0, &conc[0]);
                    }
                    jac[i][j] = result;
                }
            }

            // get the row/column ids, independent floating species
            std::list<std::string> list;
            self.model->getIds(SelectionRecord::FLOATING_AMOUNT, list);
            std::vector<std::string> ids(list.begin(), list.end());
            assert(ids.size() == jac.RSize() &&
                   ids.size() == jac.CSize() && "independent species ids length != RSize && CSize");
            jac.setColNames(ids);
            jac.setRowNames(ids);

            return jac;
        } else {
            // There might be no model.
            int nr = self.model->getNumReactions();
            if (nr == 0) {
                ls::DoubleMatrix jac(self.model->getNumRateRules(), self.model->getNumRateRules());
                return jac;
            }

            ls::DoubleMatrix uelast = getUnscaledElasticityMatrix();

            // ptr to libstruct owned obj.
            ls::DoubleMatrix *rsm;
            LibStructural *ls = getLibStruct();
            if (self.loadOpt.getConservedMoietyConversion()) {
                rsm = ls->getReorderedStoichiometryMatrix();
            } else {
                rsm = ls->getStoichiometryMatrix();
            }

            ls::DoubleMatrix jac = ls::mult(*rsm, uelast);

            // get the row/column ids, independent floating species
            std::list<std::string> list;
            self.model->getIds(SelectionRecord::FLOATING_AMOUNT, list);
            std::vector<std::string> ids(list.begin(), list.end());
            assert(ids.size() == jac.RSize() &&
                   ids.size() == jac.CSize() && "independent species ids length != RSize && CSize");
            jac.setColNames(ids);
            jac.setRowNames(ids);

            return jac;
        }
    }

    ls::DoubleMatrix RoadRunner::getFullReorderedJacobian() {
        check_model();

        LibStructural *ls = getLibStruct();
        ls::DoubleMatrix uelast = getUnscaledElasticityMatrix();
        ls::DoubleMatrix *rsm = ls->getStoichiometryMatrix();
        return mult(*rsm, uelast);
    }

    ls::DoubleMatrix RoadRunner::getReducedJacobian(double h) {
        get_self();

        check_model();

        if (h <= 0) {
            h = self.roadRunnerOptions.jacobianStepSize;
        }

        int nIndSpecies = self.model->getNumIndFloatingSpecies();

        // result matrix
        ls::DoubleMatrix jac(nIndSpecies, nIndSpecies);

        // get the row/column ids, independent floating species
        std::list<std::string> list;
        self.model->getIds(SelectionRecord::INDEPENDENT_FLOATING_AMOUNT, list);
        std::vector<std::string> ids(list.begin(), list.end());
        assert(ids.size() == nIndSpecies && "independent species ids length != getNumIndFloatingSpecies");
        jac.setColNames(ids);
        jac.setRowNames(ids);

        // need 2 buffers for rate central difference.
        std::vector<double> dy0v(nIndSpecies);
        std::vector<double> dy1v(nIndSpecies);

        double *dy0 = &dy0v[0];
        double *dy1 = &dy1v[0];

        // function pointers to the model get values and get init values based on
        // if we are doing amounts or concentrations.
        typedef int (ExecutableModel::*GetValueFuncPtr)(size_t len, int const *indx,
                                                        double *values);
        typedef int (ExecutableModel::*SetValueFuncPtr)(size_t len, int const *indx,
                                                        double const *values);

        GetValueFuncPtr getValuePtr = 0;
        GetValueFuncPtr getRateValuePtr = 0;
        SetValueFuncPtr setValuePtr = 0;

        if (Config::getValue(Config::ROADRUNNER_JACOBIAN_MODE).getAs<std::int32_t>()
            == Config::ROADRUNNER_JACOBIAN_MODE_AMOUNTS) {
            rrLog(Logger::LOG_DEBUG) << "getReducedJacobian in AMOUNT mode";
            getValuePtr = &ExecutableModel::getFloatingSpeciesAmounts;
            getRateValuePtr = &ExecutableModel::getFloatingSpeciesAmountRates;
            setValuePtr = &ExecutableModel::setFloatingSpeciesAmounts;
        } else {
            rrLog(Logger::LOG_DEBUG) << "getReducedJacobian in CONCENTRATION mode";
            getValuePtr = &ExecutableModel::getFloatingSpeciesConcentrations;
            getRateValuePtr = &ExecutableModel::getFloatingSpeciesAmountRates;
            setValuePtr = &ExecutableModel::setFloatingSpeciesConcentrations;
        }

        for (int i = 0; i < nIndSpecies; ++i) {
            double savedVal = 0;
            double y = 0;

            // current value of species i
            (self.model.get()->*getValuePtr)(1, &i, &savedVal);

            // get the entire rate of change for all the species with
            // species i being value(i) + h;
            y = savedVal + h;
            (self.model.get()->*setValuePtr)(1, &i, &y);
            (self.model.get()->*getRateValuePtr)(nIndSpecies, 0, dy0);


            // get the entire rate of change for all the species with
            // species i being value(i) - h;
            y = savedVal - h;
            (self.model.get()->*setValuePtr)(1, &i, &y);
            (self.model.get()->*getRateValuePtr)(nIndSpecies, 0, dy1);

            // restore original value
            (self.model.get()->*setValuePtr)(1, &i, &savedVal);

            // matrix is row-major, so have to copy by elements
            for (int j = 0; j < nIndSpecies; ++j) {
                jac(j, i) = (dy0[j] - dy1[j]) / (2.0 * h);
            }
        }
        return jac;
    }

    ls::DoubleMatrix RoadRunner::getLinkMatrix() {
        check_model();

        ls::LibStructural *ls = getLibStruct();

        // pointer to owned matrix
        ls::DoubleMatrix res = *ls->getLinkMatrix();

        ls->getLinkMatrixLabels(res.getRowNames(), res.getColNames());

        return res;
    }

    ls::DoubleMatrix RoadRunner::getReducedStoichiometryMatrix() {
        return getNrMatrix();
    }

    ls::DoubleMatrix RoadRunner::getNrMatrix() {
        check_model();

        ls::LibStructural *ls = getLibStruct();

        // pointer to owned matrix
        ls::DoubleMatrix res = *ls->getNrMatrix();

        ls->getNrMatrixLabels(res.getRowNames(), res.getColNames());

        return res;
    }

    ls::DoubleMatrix RoadRunner::getKMatrix() {
        check_model();

        ls::LibStructural *ls = getLibStruct();

        // pointer to owned matrix
        ls::DoubleMatrix res = *ls->getKMatrix();

        ls->getKMatrixLabels(res.getRowNames(), res.getColNames());

        return res;
    }


    ls::DoubleMatrix RoadRunner::getFullStoichiometryMatrix() {
        check_model();
        get_self();
        ls::LibStructural *ls = getLibStruct();

        if (self.loadOpt.getConservedMoietyConversion()) {
            // pointer to mat owned by ls
            ls::DoubleMatrix m = *ls->getReorderedStoichiometryMatrix();
            ls->getReorderedStoichiometryMatrixLabels(
                    m.getRowNames(), m.getColNames());
            return m;
        }

        // pointer to owned matrix
        ls::DoubleMatrix *mptr = ls->getStoichiometryMatrix();
        if (!mptr) {
            throw CoreException("Error: Stoichiometry matrix does not exist for this model");
        }
        ls::DoubleMatrix m = *mptr;
        ls->getStoichiometryMatrixLabels(m.getRowNames(), m.getColNames());
        return m;
    }


    ls::DoubleMatrix RoadRunner::getExtendedStoichiometryMatrix() {
        check_model();
        get_self();
        ls::LibStructural *ls = getLibStruct();

        if (self.loadOpt.getConservedMoietyConversion()) {
            // pointer to mat owned by ls
            ls::DoubleMatrix m = *ls->getReorderedStoichiometryMatrix();
            ls->getReorderedStoichiometryMatrixLabels(
                    m.getRowNames(), m.getColNames());
            return m;
        }

        // pointer to owned matrix
        ls::DoubleMatrix *mptr = ls->getStoichiometryMatrix();
        if (!mptr) {
            throw CoreException("Error: Stoichiometry matrix does not exist for this model");
        }
        ls::DoubleMatrix m = *mptr;
        ls->getStoichiometryMatrixLabels(m.getRowNames(), m.getColNames());

        libsbml::SBMLReader reader;
        libsbml::SBMLDocument *doc = reader.readSBMLFromString(getSBML());
        libsbml::Model *model = doc->getModel();
        typedef cxx11_ns::unordered_map<int, int> RxnIdxToRowMap;
        typedef cxx11_ns::unordered_map<int, libsbml::Reaction *> RxnIdxToRxnMap;
        typedef cxx11_ns::unordered_map<libsbml::Species *, int> SpcToRowMap;
        RxnIdxToRowMap missing_rct_row_map;
        RxnIdxToRowMap missing_prd_row_map;
        RxnIdxToRxnMap rxn_map;
        SpcToRowMap boundary_spc_row_map;

        int n_rows = m.numRows();

        for (int i = 0; i < m.getColNames().size(); ++i) {
            libsbml::Reaction *r = model->getReaction(m.getColNames().at(i));
            assert(r);
            rxn_map[i] = r;
            if (!r->getNumReactants()) {
                missing_rct_row_map[i] = n_rows++;
            } else {
                for (int j = 0; j < r->getNumReactants(); ++j) {
                    libsbml::SpeciesReference *ref = r->getReactant(j);
                    assert(ref);
                    libsbml::Species *s = model->getSpecies(ref->getSpecies());
                    assert(s);
                    if (s->getBoundaryCondition() && boundary_spc_row_map.find(s) == boundary_spc_row_map.end())
                        boundary_spc_row_map[s] = n_rows++;
                }
            }
            if (!r->getNumProducts()) {
                missing_prd_row_map[i] = n_rows++;
            } else {
                for (int j = 0; j < r->getNumProducts(); ++j) {
                    libsbml::SpeciesReference *ref = r->getProduct(j);
                    assert(ref);
                    libsbml::Species *s = model->getSpecies(ref->getSpecies());
                    assert(s);
                    if (s->getBoundaryCondition() && boundary_spc_row_map.find(s) == boundary_spc_row_map.end())
                        boundary_spc_row_map[s] = n_rows++;
                }
            }
        }

        ls::DoubleMatrix extended_matrix(n_rows, m.numCols());
        extended_matrix.setRowNames(m.getRowNames());
        extended_matrix.setColNames(m.getColNames());
        extended_matrix.getRowNames().resize(n_rows);
        for (int i = 0; i < m.numRows(); ++i) {
            for (int j = 0; j < m.numCols(); ++j) {
                extended_matrix(i, j) = m(i, j);
            }
        }
        for (int i = m.numRows(); i < n_rows; ++i) {
            for (int j = 0; j < m.numCols(); ++j) {
                extended_matrix(i, j) = 0;
            }
        }
        for (RxnIdxToRowMap::const_iterator i = missing_rct_row_map.begin(); i != missing_rct_row_map.end(); ++i) {
            extended_matrix(i->second, i->first) = -1;
            RxnIdxToRxnMap::const_iterator r = rxn_map.find(i->first);
            if (r != rxn_map.end())
                extended_matrix.getRowNames().at(i->second) = r->second->getId() + "_source";
        }
        for (RxnIdxToRowMap::const_iterator i = missing_prd_row_map.begin(); i != missing_prd_row_map.end(); ++i) {
            extended_matrix(i->second, i->first) = 1;
            RxnIdxToRxnMap::const_iterator r = rxn_map.find(i->first);
            if (r != rxn_map.end())
                extended_matrix.getRowNames().at(i->second) = r->second->getId() + "_sink";
        }
        for (SpcToRowMap::const_iterator i = boundary_spc_row_map.begin(); i != boundary_spc_row_map.end(); ++i) {
            for (int j = 0; j < m.getColNames().size(); ++j) {
                libsbml::Reaction *r = model->getReaction(m.getColNames().at(j));
                assert(r);
                for (int k = 0; k < r->getNumReactants(); ++k) {
                    libsbml::SpeciesReference *ref = r->getReactant(k);
                    assert(ref);
                    libsbml::Species *s = model->getSpecies(ref->getSpecies());
                    assert(s);
                    if (s == i->first) {
                        extended_matrix(i->second, j) = -1;
                        extended_matrix.getRowNames().at(i->second) = s->getId();
                    }
                }
                for (int k = 0; k < r->getNumProducts(); ++k) {
                    libsbml::SpeciesReference *ref = r->getProduct(k);
                    assert(ref);
                    libsbml::Species *s = model->getSpecies(ref->getSpecies());
                    assert(s);
                    if (s == i->first) {
                        extended_matrix(i->second, j) = 1;
                        extended_matrix.getRowNames().at(i->second) = s->getId();
                    }
                }
            }
        }

        delete doc;

        return extended_matrix;
    }


    ls::DoubleMatrix RoadRunner::getL0Matrix() {
        check_model();
        ls::LibStructural *ls = getLibStruct();

        // the libstruct getL0Matrix returns a NEW matrix,
        // nice consistent wrappers yes?!?!?
        ls::DoubleMatrix *tmp = ls->getL0Matrix();
        ls::DoubleMatrix m = *tmp;
        delete tmp;

        ls->getL0MatrixLabels(m.getRowNames(), m.getColNames());
        return m;
    }


    ls::DoubleMatrix RoadRunner::getConservationMatrix() {
        if (!getConservedMoietyAnalysis())
            throw CoreException("Cannot compute conservation matrix because conserved moieties are not enabled");

        ls::DoubleMatrix mat;

        try {
            if (impl->model) {
                ls::DoubleMatrix *aMat = getLibStruct()->getGammaMatrix();
                if (aMat) {
                    mat.resize(aMat->numRows(), aMat->numCols());
                    for (int row = 0; row < mat.RSize(); row++) {
                        for (int col = 0; col < mat.CSize(); col++) {
                            mat(row, col) = (*aMat)(row, col);
                        }
                    }

                    // Note that LibStruct::getGammaMatrixLabels() is not used because it merely populates
                    // the rows with indices, whereas we want the Moiety IDs
                    mat.setRowNames(getConservedMoietyIds());
                    mat.setColNames(getLibStruct()->getReorderedSpecies());
                }
                return mat;

            }
            throw CoreException(gEmptyModelMessage);
        }
        catch (const Exception &e) {
            throw CoreException("Unexpected error from getConservationMatrix(): ", e.Message());
        }
    }

// Help("Returns the number of dependent species in the model")
    int RoadRunner::getNumberOfDependentSpecies() {
        try {
            if (impl->model) {
                //return mStructAnalysis.GetInstance()->getNumDepSpecies();
                return getLibStruct()->getNumDepSpecies();
            }

            throw CoreException(gEmptyModelMessage);
        }

        catch (Exception &e) {
            throw CoreException("Unexpected error from getNumberOfDependentSpecies(): ", e.Message());
        }
    }

// Help("Returns the number of independent species in the model")
    int RoadRunner::getNumberOfIndependentSpecies() {
        try {
            if (impl->model) {
                return getLibStruct()->getNumIndSpecies();
            }
            //return StructAnalysis.getNumIndSpecies();
            throw CoreException(gEmptyModelMessage);
        }
        catch (Exception &e) {
            throw CoreException("Unexpected error from getNumberOfIndependentSpecies(): ", e.Message());
        }
    }

    double RoadRunner::getVariableValue(const VariableType variableType,
                                        const int variableIndex) {
        switch (variableType) {
            case vtFlux: {
                double result = 0;
                impl->model->getReactionRates(1, &variableIndex, &result);
                return result;
            }
                break;

            case vtSpecies: {
                double result = 0;
                impl->model->getFloatingSpeciesConcentrations(1, &variableIndex, &result);
                return result;
            }
                break;

            default:
                throw CoreException("Unrecognised variable in getVariableValue");
                break;
        }
        return 0;
    }

    size_t RoadRunner::createDefaultSteadyStateSelectionList() {
        impl->mSteadyStateSelection.clear();
        // default should be independent floating species only ...
        std::vector<std::string> floatingSpecies = getFloatingSpeciesIds();
        size_t numFloatingSpecies = floatingSpecies.size();
        //int numIndSpecies = getNumberOfIndependentSpecies();
        //impl->mSteadyStateSelection.resize(numIndSpecies);
        impl->mSteadyStateSelection.resize(numFloatingSpecies);
        //for (int i = 0; i < numIndSpecies; i++)
        for (int i = 0; i < numFloatingSpecies; i++) {
            SelectionRecord aRec;
            aRec.selectionType = SelectionRecord::FLOATING_CONCENTRATION;
            aRec.p1 = floatingSpecies[i];
            aRec.index = i;
            impl->mSteadyStateSelection[i] = aRec;
        }
        return impl->mSteadyStateSelection.size();
    }

    std::vector<SelectionRecord> &RoadRunner::getSteadyStateSelections() {
        return impl->mSteadyStateSelection;
    }

    std::vector<double> RoadRunner::getSteadyStateValues() {
        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }
        if ((impl->loadOpt.loadFlags & LoadSBMLOptions::NO_DEFAULT_STEADY_STATE_SELECTIONS) == 0) {
            createDefaultSteadyStateSelectionList();
        }

        steadyState();

        std::vector<double> result; //= new double[oSelection.Length];
        for (int i = 0; i < impl->mSteadyStateSelection.size(); i++) {
            result.push_back(getValue(impl->mSteadyStateSelection[i]));
        }
        return result;
    }

    std::vector<std::string> RoadRunner::getSteadyStateSelectionStrings() const {
        std::vector<std::string> r;

        for (int i = 0; i < impl->mSteadyStateSelection.size(); i++) {
            r.push_back(impl->mSteadyStateSelection[i].to_string());
        }

        return r;
    }

    ls::DoubleMatrix RoadRunner::getSteadyStateValuesNamedArray() {
        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }
        if ((impl->loadOpt.loadFlags & LoadSBMLOptions::NO_DEFAULT_STEADY_STATE_SELECTIONS) == 0) {
            createDefaultSteadyStateSelectionList();
        }

        steadyState();

        ls::DoubleMatrix v(1, static_cast<unsigned int>(impl->mSteadyStateSelection.size()));
        for (int i = 0; i < impl->mSteadyStateSelection.size(); i++) {
            v(0, i) = getValue(impl->mSteadyStateSelection[i]);
        }

        v.setColNames(getSteadyStateSelectionStrings());

        return v;
    }

    std::string RoadRunner::getModelName() {
        if (impl->document && impl->document->isSetModel()) {
            libsbml::Model *model = impl->document->getModel();
            if (model->isSetName()) {
                return model->getName();
            }
        }
        if (impl->model) {
            return impl->model->getModelName();
        }
        return "";
    }

    void RoadRunner::setModelName(const string &name) {
        if (impl->document && impl->document->isSetModel()) {
            impl->document->getModel()->setName(name);
        }
    }

    std::string RoadRunner::getModelId() {
        if (impl->document && impl->document->isSetModel()) {
            libsbml::Model *model = impl->document->getModel();
            if (model->isSetId()) {
                return model->getId();
            }
        }
        return "";
    }

    void RoadRunner::setModelId(const string &id) {
        if (impl->document && impl->document->isSetModel()) {
            impl->document->getModel()->setId(id);
        }
    }

    /**
    * find an symbol id in the model and set its value.
    */
    static void setSBMLValue(libsbml::Model *model, const std::string &id, double value) {
        if (model == NULL) {
            throw Exception("You need to load the model first");
        }

        libsbml::Species *oSpecies = model->getSpecies(id);
        if (oSpecies != NULL) {
            if (oSpecies->isSetInitialAmount())
                oSpecies->setInitialAmount(value);
            else
                oSpecies->setInitialConcentration(value);
            return;
        }

        libsbml::Compartment *oCompartment = model->getCompartment(id);
        if (oCompartment != NULL) {
            oCompartment->setVolume(value);
            return;
        }

        for (int i = 0; i < model->getNumReactions(); i++) {
            libsbml::Reaction *reaction = model->getReaction(i);
            for (int j = 0; j < reaction->getNumReactants(); j++) {
                libsbml::SpeciesReference *reference = reaction->getReactant(j);
                if (reference->isSetId() && reference->getId() == id) {
                    reference->setStoichiometry(value);
                    return;
                }
            }
            for (int j = 0; j < reaction->getNumProducts(); j++) {
                libsbml::SpeciesReference *reference = reaction->getProduct(j);
                if (reference->isSetId() && reference->getId() == id) {
                    reference->setStoichiometry(value);
                    return;
                }
            }
        }

        throw Exception("Invalid std::string name. The id '" + id + "' does not exist in the model");
    }


    static void setSpeciesAmount(libsbml::Model *model, const std::string &id, double value) {
        if (model == NULL) {
            throw Exception("You need to load the model first");
        }

        libsbml::Species *oSpecies = model->getSpecies(id);
        if (oSpecies == NULL) {
            throw Exception("No such species found in model.");
        }
        if (oSpecies->isSetInitialConcentration()) {
            oSpecies->unsetInitialConcentration();
        }
        oSpecies->setInitialAmount(value);
        return;
    }


// Help("Get the number of reactions")
    int RoadRunner::getNumberOfReactions() {
        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }
        return impl->model->getNumReactions();
    }

// Help("Returns the rate of a reaction by its index")
    double RoadRunner::getReactionRate(const int &index) {
        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }

        if ((index >= 0) && (index < impl->model->getNumReactions())) {
            double result = 0;
            impl->model->getReactionRates(1, &index, &result);
            return result;
        } else {
            throw CoreException(format("Index in getReactionRate out of range: [{0}]", index));
        }
    }


    double RoadRunner::getRateOfChange(const int &index) {
        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }

        if ((index >= 0) && (index < impl->model->getNumFloatingSpecies())) {
            double value = 0;
            impl->model->getFloatingSpeciesAmountRates(1, &index, &value);
            return value;
        }

        throw CoreException(format("Index in getRateOfChange out of range: [{0}]", index));
    }


// Help("Get the number of compartments")
    int RoadRunner::getNumberOfCompartments() {
        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }
        return impl->model->getNumCompartments();
    }


    void RoadRunner::setCompartmentByIndex(const int &index, const double &value) {
        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }

        if ((index >= 0) && (index < impl->model->getNumCompartments())) {
            impl->model->setCompartmentVolumes(1, &index, &value);
        } else {
            throw CoreException(format("Index in getCompartmentByIndex out of range: [{0}]", index));
        }
    }

    double RoadRunner::getCompartmentByIndex(const int &index) {
        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }

        if ((index >= 0) && (index < impl->model->getNumCompartments())) {
            double result = 0;
            impl->model->getCompartmentVolumes(1, &index, &result);
            return result;
        }

        throw CoreException(format("Index in getCompartmentByIndex out of range: [{0}]", index));
    }

    int RoadRunner::getNumberOfBoundarySpecies() {
        if (!impl->model) {
            throw Exception(gEmptyModelMessage);
        }
        return impl->model->getNumBoundarySpecies();
    }

// Help("Sets the value of a boundary species by its index")
    void RoadRunner::setBoundarySpeciesByIndex(const int &index, const double &value) {
        if (!impl->model) {
            throw Exception(gEmptyModelMessage);
        }

        if ((index >= 0) && (index < impl->model->getNumBoundarySpecies())) {
            impl->model->setBoundarySpeciesConcentrations(1, &index, &value);
        } else {
            throw Exception(format("Index in getBoundarySpeciesByIndex out of range: [{0}]", index));
        }
    }

// Help("Returns the value of a boundary species by its index")
    double RoadRunner::getBoundarySpeciesByIndex(const int &index) {
        if (!impl->model) {
            throw Exception(gEmptyModelMessage);
        }
        if ((index >= 0) && (index < impl->model->getNumBoundarySpecies())) {
            double result = 0;
            impl->model->getBoundarySpeciesConcentrations(1, &index, &result);
            return result;
        }
        throw Exception(format("Index in getBoundarySpeciesByIndex out of range: [{0}]", index));
    }


    ls::DoubleMatrix RoadRunner::getBoundarySpeciesConcentrationsNamedArray() {
        check_model();

        int l = impl->model->getNumBoundarySpecies();

        double *vals = new double[l];
        impl->model->getBoundarySpeciesConcentrations(l, NULL, vals);

        LibStructural *ls = getLibStruct();
        ls::DoubleMatrix v(1, l);

        for (int i = 0; i < l; ++i)
            v(0, i) = vals[i];

        delete[] vals;

        v.setColNames(getBoundarySpeciesIds());

        return v;
    }


    ls::DoubleMatrix RoadRunner::getBoundarySpeciesAmountsNamedArray() {
        check_model();

        int l = impl->model->getNumBoundarySpecies();

        double *vals = new double[l];
        impl->model->getBoundarySpeciesAmounts(l, NULL, vals);

        LibStructural *ls = getLibStruct();
        ls::DoubleMatrix v(1, l);

        for (int i = 0; i < l; ++i)
            v(0, i) = vals[i];

        delete[] vals;

        v.setColNames(getBoundarySpeciesIds());

        return v;
    }


// Help("Get the number of floating species")
    int RoadRunner::getNumberOfFloatingSpecies() {
        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }
        return impl->model->getNumFloatingSpecies();
    }


// Help("Sets the value of a floating species by its index")
    void RoadRunner::setFloatingSpeciesInitialConcentrationByIndex(const int &index, const double &value) {
        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }

        if ((index >= 0) && (index < impl->model->getNumFloatingSpecies())) {
            impl->model->setFloatingSpeciesInitConcentrations(1, &index, &value);
            reset();
        } else {
            throw CoreException(
                    format("Index in setFloatingSpeciesInitialConcentrationByIndex out of range: [{0}]", index));
        }
    }

// Help("Sets the value of a floating species by its index")
    void RoadRunner::setFloatingSpeciesByIndex(int index, double value) {
        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }

        if ((index >= 0) && (index < impl->model->getNumFloatingSpecies())) {
            impl->model->setFloatingSpeciesConcentrations(1, &index,
                                                          &value); // This updates the amount std::vector aswell
        } else {
            throw CoreException(format("Index in setFloatingSpeciesByIndex out of range: [{0}]", index));
        }
    }

// Help("Returns the value of a floating species by its index")
    double RoadRunner::getFloatingSpeciesByIndex(const int index) {
        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }

        if ((index >= 0) && (index < impl->model->getNumFloatingSpecies())) {
            double result = 0;
            impl->model->getFloatingSpeciesConcentrations(1, &index, &result);
            return result;
        }
        throw CoreException(format("Index in getFloatingSpeciesByIndex out of range: [{0}]", index));
    }

// Help("Returns an array of floating species concentrations")
    std::vector<double> RoadRunner::getFloatingSpeciesConcentrationsV() {
        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }


        std::vector<double> result(impl->model->getNumFloatingSpecies(), 0);
        if (result.size()) {
            impl->model->getFloatingSpeciesConcentrations(result.size(), 0, &result[0]);
        }
        return result;
    }

// Help("Returns an array of floating species amounts")
    std::vector<double> RoadRunner::getFloatingSpeciesAmountsV() {
        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }


        std::vector<double> result(impl->model->getNumFloatingSpecies(), 0);
        if (result.size()) {
            impl->model->getFloatingSpeciesAmounts(result.size(), 0, &result[0]);
        }
        return result;
    }

// Help("returns an array of floating species initial conditions")
    std::vector<double> RoadRunner::getFloatingSpeciesInitialConcentrations() {
        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }
        std::vector<double> initYs(impl->model->getNumFloatingSpecies());
        if (initYs.size()) {
            impl->model->getFloatingSpeciesInitConcentrations(initYs.size(), 0, &initYs[0]);
        }
        return initYs;
    }

// Help("Sets the initial conditions for all floating species in the model")
    void RoadRunner::setFloatingSpeciesInitialConcentrations(const std::vector<double> &values) {
        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }

        if (values.size()) {
            impl->model->setFloatingSpeciesConcentrations(values.size(), 0, &values[0]);
            impl->model->setFloatingSpeciesInitConcentrations(values.size(), 0, &values[0]);
        }
        reset();
    }

// Help("Set the concentrations for all floating species in the model")
    void RoadRunner::setFloatingSpeciesConcentrations(const std::vector<double>& values) {
        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }

        if (values.size()) {
            impl->model->setFloatingSpeciesConcentrations(values.size(), 0, &values[0]);
        }
    }

    std::vector<double> RoadRunner::getBoundarySpeciesConcentrationsV() {
        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }


        std::vector<double> result(impl->model->getNumBoundarySpecies(), 0);
        if (result.size()) {
            impl->model->getBoundarySpeciesConcentrations(result.size(), 0, &result[0]);
        }
        return result;
    }

    std::vector<double> RoadRunner::getBoundarySpeciesAmountsV() {
        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }


        std::vector<double> result(impl->model->getNumBoundarySpecies(), 0);
        if (result.size()) {
            impl->model->getBoundarySpeciesAmounts(result.size(), 0, &result[0]);
        }
        return result;
    }


// Help("Set the concentrations for all boundary species in the model")
    void RoadRunner::setBoundarySpeciesConcentrations(const std::vector<double>& values) {
        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }

        if (values.size()) {
            impl->model->setBoundarySpeciesConcentrations(values.size(), 0, &values[0]);
        }
    }


// Help("Set the amounts for all boundary species in the model")
    void RoadRunner::setBoundarySpeciesAmounts(const std::vector<double>& values) {
        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }

        if (values.size()) {
            impl->model->setBoundarySpeciesAmounts(values.size(), 0, &values[0]);
        }
    }


// Help("Get the number of global parameters")
    size_t RoadRunner::getNumberOfGlobalParameters() {
        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }
        return getGlobalParameterIds().size();
    }

    void RoadRunner::setGlobalParameterByIndex(const int index, const double value) {
        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }

        impl->model->setGlobalParameterValues(1, &index, &value);
    }

    void RoadRunner::setGlobalParameterByName(const std::string &param, const double value) {
        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }
        int index_of_param;
        std::vector<std::string> global_parameter_ids = getGlobalParameterIds();
        auto iterator = std::find(global_parameter_ids.begin(), global_parameter_ids.end(), param);
        if (iterator == global_parameter_ids.end()) {
            throw std::invalid_argument("std::invalid_argument: RoadRunner::setGlobalParameterByName "
                                        "Parameter \"" + param + "\" not found in model");
        } else {
            index_of_param = static_cast<int>(std::distance(global_parameter_ids.begin(), iterator));
        }
        impl->model->setGlobalParameterValues(1, &index_of_param, &value);
    }

    double RoadRunner::getGlobalParameterByName(const std::string &param) {
        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }
        int index_of_param;
        std::vector<std::string> global_parameter_ids = getGlobalParameterIds();
        auto iterator = std::find(global_parameter_ids.begin(), global_parameter_ids.end(), param);
        if (iterator == global_parameter_ids.end()) {
            throw std::invalid_argument("std::invalid_argument: RoadRunner::setGlobalParameterByName "
                                        "Parameter \"" + param + "\" not found in model");
        } else {
            index_of_param = static_cast<int>(std::distance(global_parameter_ids.begin(), iterator));
        }
        double value;
        impl->model->getGlobalParameterValues(1, &index_of_param, &value);
        return value;
    }

// Help("Returns the value of a global parameter by its index")
    double RoadRunner::getGlobalParameterByIndex(const int &index) {
        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }

        if ((index >= 0) &&
            (index < (impl->model->getNumGlobalParameters() + impl->model->getNumDepFloatingSpecies()))) {
            int arraySize = impl->model->getNumGlobalParameters() + impl->model->getNumDepFloatingSpecies();
            if (impl->model->getNumReactions() == 0 && impl->model->getNumRateRules() > 0) {
                int arraySize = impl->model->getNumGlobalParameters();
            }
            double *data = new double[arraySize];

            if (impl->model->getNumReactions() == 0 && impl->model->getNumRateRules() > 0) {
                impl->model->getGlobalParameterValues(impl->model->getNumGlobalParameters(), 0, data);
            } else {
                impl->model->getGlobalParameterValues(impl->model->getNumGlobalParameters(), 0, data);

                impl->model->getConservedMoietyValues(impl->model->getNumDepFloatingSpecies(), 0,
                                                      data + impl->model->getNumGlobalParameters());
            }

            double result = data[index];
            delete[] data;
            return result;
        }

        throw CoreException(format("Index in getNumGlobalParameters out of range: [{0}]", index));
    }

// Help("Get the values for all global parameters in the model")
    std::vector<double> RoadRunner::getGlobalParameterValues() {
        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }

        if (impl->model->getNumDepFloatingSpecies() > 0) {
            std::vector<double> result(impl->model->getNumGlobalParameters() +
                                       impl->model->getNumDepFloatingSpecies());

            if (result.size()) {
                impl->model->getGlobalParameterValues(
                    impl->model->getNumGlobalParameters(), 0, &result[0]);

                impl->model->getConservedMoietyValues(
                    impl->model->getNumDepFloatingSpecies(), 0,
                    &result[impl->model->getNumGlobalParameters()]);
            }
            return result;
        }

        std::vector<double> result(impl->model->getNumGlobalParameters());
        if (result.size()) {
            impl->model->getGlobalParameterValues(result.size(), 0, &result[0]);
        }
        return result;
    }


//---------------- MCA functions......
//        [Help("Get unscaled control coefficient with respect to a global parameter")]
    double RoadRunner::getuCC(const std::string &variableName, const std::string &parameterName) {
        try {
            if (!impl->model) {
                throw CoreException(gEmptyModelMessage);
            }

            ParameterType parameterType;
            VariableType variableType;
            double originalParameterValue;
            int variableIndex;
            int parameterIndex;

            std::string variableNameMod = variableName;

            variableNameMod.erase(std::remove(variableNameMod.begin(), variableNameMod.end(), '['),
                                  variableNameMod.end());
            variableNameMod.erase(std::remove(variableNameMod.begin(), variableNameMod.end(), ']'),
                                  variableNameMod.end());

            // Check the variable name
            if ((variableIndex = impl->model->getReactionIndex(variableNameMod)) >= 0) {
                variableType = vtFlux;
            } else if ((variableIndex = impl->model->getFloatingSpeciesIndex(variableNameMod)) >= 0) {
                variableType = vtSpecies;
            } else {
                throw CoreException("Unable to locate variable: [" + variableNameMod + "]");
            }

            //Save the current state before calculating the steady state so we can restore it later.
            int nfloat = impl->model->getNumFloatingSpecies();
            double *floats = new double[nfloat];
            impl->model->getFloatingSpeciesAmounts(nfloat, 0, floats);
            int nboundary = impl->model->getNumBoundarySpecies();
            double *boundaries = new double[nboundary];
            impl->model->getBoundarySpeciesAmounts(nboundary, 0, boundaries);
            int ncomp = impl->model->getNumCompartments();
            double *comps = new double[ncomp];
            impl->model->getCompartmentVolumes(ncomp, 0, comps);
            int nparam = impl->model->getNumGlobalParameters();
            double *params = new double[nparam];
            impl->model->getGlobalParameterValues(nparam, 0, params);
            //std::stringstream* orig_state = saveStateS();

            mcaSteadyState();

            // Check for the parameter name
            if ((parameterIndex = impl->model->getGlobalParameterIndex(parameterName)) >= 0) {
                parameterType = ptGlobalParameter;
                originalParameterValue = 0;
                impl->model->getGlobalParameterValues(1, &parameterIndex, &originalParameterValue);
            } else if ((parameterIndex = impl->model->getBoundarySpeciesIndex(parameterName)) >= 0) {
                parameterType = ptBoundaryParameter;
                originalParameterValue = 0;
                impl->model->getBoundarySpeciesConcentrations(1, &parameterIndex, &originalParameterValue);
            } else if ((parameterIndex = impl->model->getConservedMoietyIndex(parameterName)) >= 0) {
                parameterType = ptConservationParameter;
                originalParameterValue = 0;
                impl->model->getConservedMoietyValues(1, &parameterIndex, &originalParameterValue);
            } else {
                throw CoreException("Unable to locate parameter: [" + parameterName + "]");
            }

            // Get the original parameter value
            originalParameterValue = impl->getParameterValue(parameterType, parameterIndex);

            double hstep = impl->roadRunnerOptions.diffStepSize * originalParameterValue;
            if (fabs(hstep) < 1E-12) {
                hstep = impl->roadRunnerOptions.diffStepSize;
            }

            try {
                impl->setParameterValue(parameterType, parameterIndex, originalParameterValue + hstep);
                mcaSteadyState();
                double fi = getVariableValue(variableType, variableIndex);

                impl->setParameterValue(parameterType, parameterIndex, originalParameterValue + 2 * hstep);
                mcaSteadyState();
                double fi2 = getVariableValue(variableType, variableIndex);

                impl->setParameterValue(parameterType, parameterIndex, originalParameterValue - hstep);
                mcaSteadyState();
                double fd = getVariableValue(variableType, variableIndex);

                impl->setParameterValue(parameterType, parameterIndex, originalParameterValue - 2 * hstep);
                mcaSteadyState();
                double fd2 = getVariableValue(variableType, variableIndex);

                // Use instead the 5th order approximation double unscaledValue = (0.5/hstep)*(fi-fd);
                // The following separated lines avoid small amounts of roundoff error
                double f1 = fd2 + 8 * fi;
                double f2 = -(8 * fd + fi2);

                // What ever happens, make sure we restore the original state
                impl->model->setFloatingSpeciesAmounts(nfloat, 0, floats, false);
                impl->model->setBoundarySpeciesAmounts(nboundary, 0, boundaries, false);
                impl->model->setCompartmentVolumes(ncomp, 0, comps, false);
                impl->model->setGlobalParameterValues(nparam, 0, params, false);
                //loadStateS(orig_state);

                return 1 / (12 * hstep) * (f1 + f2);
            }
            catch (...) //Catch anything... and do 'finalizeObject'
            {
                // What ever happens, make sure we restore the original state
                impl->model->setFloatingSpeciesAmounts(nfloat, 0, floats);
                impl->model->setBoundarySpeciesAmounts(nboundary, 0, boundaries);
                impl->model->setCompartmentVolumes(ncomp, 0, comps);
                impl->model->setGlobalParameterValues(nparam, 0, params);
                //loadStateS(orig_state);
                throw;
            }
        }
        catch (const Exception &e) {
            throw CoreException(
                    "Unable to calculate control coefficients in getuCC(), perhaps due to an inability to reach the steady state.  Error from the calculation: ",
                    e.Message());
        }
    }

//        [Help("Get scaled control coefficient with respect to a global parameter")]
    double RoadRunner::getCC(const std::string &variableName, const std::string &parameterName) {
        VariableType variableType;
        ParameterType parameterType;
        int variableIndex;
        int parameterIndex;

        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }

        std::string variableNameMod = variableName;

        variableNameMod.erase(std::remove(variableNameMod.begin(), variableNameMod.end(), '['), variableNameMod.end());
        variableNameMod.erase(std::remove(variableNameMod.begin(), variableNameMod.end(), ']'), variableNameMod.end());

        // Check the variable name
        if ((variableIndex = impl->model->getReactionIndex(variableNameMod)) >= 0) {
            variableType = vtFlux;
        } else if ((variableIndex = impl->model->getFloatingSpeciesIndex(variableNameMod)) >= 0) {
            variableType = vtSpecies;
        } else {
            throw CoreException("Unable to locate variable: [" + variableNameMod + "]");
        }

        // Check for the parameter name
        if ((parameterIndex = impl->model->getGlobalParameterIndex(parameterName)) >= 0) {
            parameterType = ptGlobalParameter;
        } else if ((parameterIndex = impl->model->getBoundarySpeciesIndex(parameterName)) >= 0) {
            parameterType = ptBoundaryParameter;
        } else if ((parameterIndex = impl->model->getConservedMoietyIndex(parameterName)) >= 0) {
            parameterType = ptConservationParameter;
        } else {
            throw CoreException("Unable to locate parameter: [" + parameterName + "]");
        }

        double ucc = getuCC(variableNameMod, parameterName);
        double variableValue = getVariableValue(variableType, variableIndex);
        if (variableType == vtFlux && abs(variableValue) < impl->roadRunnerOptions.fluxThreshold) {
            return 0.0;
        }
        double parameterValue = impl->getParameterValue(parameterType, parameterIndex);
        return ucc * parameterValue / variableValue;
    }


    double RoadRunner::getUnscaledSpeciesElasticity(int reactionId, int speciesIndex) {
        get_self();

        check_model();

        // function pointers to the model get values and get init values based on
        // if we are doing amounts or concentrations.
        typedef int (ExecutableModel::* GetValueFuncPtr)(size_t len, int const *indx,
                                                         double *values);
        typedef int (ExecutableModel::* SetValueFuncPtr)(size_t len, int const *indx,
                                                         double const *values);

        GetValueFuncPtr getValuePtr = 0;
        GetValueFuncPtr getInitValuePtr = 0;
        SetValueFuncPtr setValuePtr = 0;
        SetValueFuncPtr setInitValuePtr = 0;
        GetValueFuncPtr getInitVolumesPtr = &ExecutableModel::getCompartmentInitVolumes;
        SetValueFuncPtr setInitVolumesPtr = &ExecutableModel::setCompartmentInitVolumes;
        GetValueFuncPtr getCurrentVolumesPtr = &ExecutableModel::getCompartmentVolumes;

        if (Config::getValue(Config::ROADRUNNER_JACOBIAN_MODE).getAs<std::int32_t>() ==
            Config::ROADRUNNER_JACOBIAN_MODE_AMOUNTS) {
            getValuePtr = &ExecutableModel::getFloatingSpeciesAmounts;
            getInitValuePtr = &ExecutableModel::getFloatingSpeciesInitAmounts;
            setValuePtr = &ExecutableModel::setFloatingSpeciesAmounts;
            setInitValuePtr = &ExecutableModel::setFloatingSpeciesInitAmounts;
        } else {
            getValuePtr = &ExecutableModel::getFloatingSpeciesConcentrations;
            getInitValuePtr = &ExecutableModel::getFloatingSpeciesInitConcentrations;
            setValuePtr = &ExecutableModel::setFloatingSpeciesConcentrations;
            setInitValuePtr = &ExecutableModel::setFloatingSpeciesInitConcentrations;
        }

        double value;
        double origCurrentVal = 0;
        double result = std::numeric_limits<double>::quiet_NaN();
        //Need to save the time, too, since calling setInitValuePtr will reset time to -inf.
        //  We can't *not* call reset, either, as it's necessary for other things to work.
        double origTime = self.model->getTime();

        // note setting init values auotmatically sets the current values to the
        // init values

        // this causes a reset, so need to save the current amounts to set them back
        // as init conditions.
        std::vector<double> currentVals(self.model->getNumFloatingSpecies());
        if (currentVals.size()) {
            (self.model.get()->*getValuePtr)(currentVals.size(), 0, &currentVals[0]);
        }

        //If the initial volumes have changed, we have to set them, but we'll need to reset them later.
        std::vector<double> origInitVolumes(self.model->getNumCompartments());
        if (origInitVolumes.size()) {
            (self.model.get()->*getInitVolumesPtr)(origInitVolumes.size(), 0, &origInitVolumes[0]);
        }

        std::vector<double> currentVolumes(self.model->getNumCompartments());
        if (currentVolumes.size()) {
            (self.model.get()->*getCurrentVolumesPtr)(currentVolumes.size(), 0, &currentVolumes[0]);
        }

        // Don't try to compute any elasticities if there a numerically suspicious values
        for (int i = 0; i < currentVals.size() - 1; i++)
            if (fabs(currentVals[i]) > 1E100) {
                throw std::runtime_error(
                        "Floating species concentations are of the order of 1E100, unable to compute elasticities");
            }

        // save the original init values
        std::vector<double> origInitVal(self.model->getNumFloatingSpecies());
        if (origInitVal.size()) {
            (self.model.get()->*getInitValuePtr)(origInitVal.size(), 0, &origInitVal[0]);
        }

        // get the original value
        (self.model.get()->*getValuePtr)(1, &speciesIndex, &origCurrentVal);

        // now we start changing things
        try {
            //First set the initial volumes so everything else matches.
            if (currentVolumes.size()) {
                (self.model.get()->*setInitVolumesPtr)(currentVolumes.size(), 0, &currentVolumes[0]);
            }

            // Now set init amounts to current amounts, restore them later.
            // have to do this as this is only way to set conserved moiety values
            if (currentVals.size()) {
                (self.model.get()->*setInitValuePtr)(currentVals.size(), 0, &currentVals[0]);
            }
            self.model->setTime(origTime);

            // sanity check
            assert_similar(origCurrentVal, currentVals[speciesIndex]);
            double tmp = 0;
            (self.model.get()->*getInitValuePtr)(1, &speciesIndex, &tmp);
            assert_similar(origCurrentVal, tmp);
            (self.model.get()->*getValuePtr)(1, &speciesIndex, &tmp);
            assert_similar(origCurrentVal, tmp);

            // things check out, start fiddling...

            double hstep = self.roadRunnerOptions.diffStepSize * origCurrentVal;
            if (fabs(hstep) < 1E-12) {
                hstep = self.roadRunnerOptions.diffStepSize;
            }

            value = origCurrentVal + hstep;
            (self.model.get()->*setInitValuePtr)(1, &speciesIndex, &value);
            self.model->setTime(origTime);

            double fi = 0;
            self.model->getReactionRates(1, &reactionId, &fi);

            value = origCurrentVal + 2 * hstep;
            (self.model.get()->*setInitValuePtr)(1, &speciesIndex, &value);
            self.model->setTime(origTime);
            double fi2 = 0;
            self.model->getReactionRates(1, &reactionId, &fi2);

            value = origCurrentVal - hstep;
            (self.model.get()->*setInitValuePtr)(1, &speciesIndex, &value);
            self.model->setTime(origTime);
            double fd = 0;
            self.model->getReactionRates(1, &reactionId, &fd);

            value = origCurrentVal - 2 * hstep;
            (self.model.get()->*setInitValuePtr)(1, &speciesIndex, &value);
            self.model->setTime(origTime);
            double fd2 = 0;
            self.model->getReactionRates(1, &reactionId, &fd2);

            // Use instead the 5th order approximation
            // double unscaledElasticity = (0.5/hstep)*(fi-fd);
            // The following separated lines avoid small amounts of roundoff error
            double f1 = fd2 + 8 * fi;
            double f2 = -(8 * fd + fi2);

            result = 1 / (12 * hstep) * (f1 + f2);
        }
        catch (const std::exception &e) {
            // Whatever happens, make sure we restore the species and volumes levels, and time.
            if (origInitVolumes.size()) {
                (self.model.get()->*setInitVolumesPtr)(origInitVolumes.size(), 0, &origInitVolumes[0]);
            }
            if (origInitVal.size()) {
                (self.model.get()->*setInitValuePtr)(origInitVal.size(), 0, &origInitVal[0]);
            }
            self.model->setTime(origTime);

            // only set the indep species, setting dep species is not permitted.
            if (currentVals.size()) {
                (self.model.get()->*setValuePtr)(
                    self.model->getNumIndFloatingSpecies(), 0, &currentVals[0]);
            }
            // re-throw the exception.
            throw e;
        }

        // Whatever happens, make sure we restore the species and volumes levels, and time.
        if (origInitVal.size()) {
            (self.model.get()->*setInitValuePtr)(
                origInitVal.size(), 0, &origInitVal[0]);
        }

        // only set the indep species, setting dep species is not permitted.
        if (currentVals.size()) {
            (self.model.get()->*setValuePtr)(
                self.model->getNumIndFloatingSpecies(), 0, &currentVals[0]);
        }
        self.model->setTime(origTime);

        return result;
    }


    ls::DoubleMatrix RoadRunner::getUnscaledElasticityMatrix() {
        get_self();

        check_model();

        ls::DoubleMatrix uElastMatrix(self.model->getNumReactions(), self.model->getNumFloatingSpecies());

        uElastMatrix.setRowNames(getReactionIds());
        uElastMatrix.setColNames(getFloatingSpeciesIds());

        for (int i = 0; i < self.model->getNumReactions(); i++) {
            for (int j = 0; j < self.model->getNumFloatingSpecies(); j++) {
                uElastMatrix[i][j] = getUnscaledSpeciesElasticity(i, j);
            }
        }

        return uElastMatrix;
    }

    ls::DoubleMatrix RoadRunner::getScaledElasticityMatrix() {
        get_self();

        check_model();

        ls::DoubleMatrix uelast = getUnscaledElasticityMatrix();
        ls::DoubleMatrix result(uelast.RSize(), uelast.CSize());

        result.setColNames(uelast.getColNames());
        result.setRowNames(uelast.getRowNames());

        std::vector<double> rates(self.model->getNumReactions());
        if (rates.size()) {
            self.model->getReactionRates(rates.size(), 0, &rates[0]);
        }

        if (uelast.RSize() != rates.size()) {
            // this should NEVER happen
            throw std::runtime_error("row count of unscaled elasticity different "
                                     "than # of reactions");
        }

        for (int i = 0; i < uelast.RSize(); i++) {
            for (int j = 0; j < uelast.CSize(); j++) // Columns are species
            {
                double concentration = 0;
                self.model->getFloatingSpeciesConcentrations(1, &j, &concentration);

                result[i][j] = uelast[i][j] * concentration / rates[i];
            }
        }
        return result;
    }


    double RoadRunner::getScaledFloatingSpeciesElasticity(const std::string &reactionName,
                                                          const std::string &speciesName) {
        get_self();

        check_model();

        int speciesIndex = 0;
        int reactionIndex = 0;

        if ((speciesIndex = self.model->getFloatingSpeciesIndex(speciesName)) < 0) {
            throw std::invalid_argument("invalid species name: " + speciesName);
        }
        if ((reactionIndex = self.model->getReactionIndex(reactionName)) < 0) {
            throw std::invalid_argument("invalid reaction name: " + reactionName);
        }

        double concentration = 0;
        self.model->getFloatingSpeciesConcentrations(1, &speciesIndex, &concentration);

        double reactionRate = 0;
        self.model->getReactionRates(1, &reactionIndex, &reactionRate);

        return getUnscaledSpeciesElasticity(reactionIndex, speciesIndex) *
               concentration / reactionRate;
    }


// Use the formula: ucc = -L Jac^-1 Nr
// [Help("Compute the matrix of unscaled concentration control coefficients")]
    ls::DoubleMatrix RoadRunner::getUnscaledConcentrationControlCoefficientMatrix() {
        get_self();

        check_model();

        if (steadyState() > impl->roadRunnerOptions.steadyStateThreshold) {
            if (steadyState() > 1E-2) {
                //impl->simulateOpt.steps = orig_steps;
                throw CoreException("Unable to locate steady state during control coefficient computation");
            }
        }

        // Compute the Jacobian first
        ls::DoubleMatrix uelast = getUnscaledElasticityMatrix();
        ls::DoubleMatrix Nr = getNrMatrix();
        ls::DoubleMatrix T1 = mult(Nr, uelast);
        ls::DoubleMatrix LinkMatrix = getLinkMatrix();
        ls::DoubleMatrix Jac = mult(T1, LinkMatrix);

        // Compute -Jac
        ls::DoubleMatrix T2 = Jac * (-1.0);

        ComplexMatrix temp(T2); //Get a complex matrix from a double one. Imag part is zero
        ComplexMatrix Inv = GetInverse(temp);

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // Sauro: mult which takes complex matrix need to be implemented
        ls::DoubleMatrix T3 = mult(Inv, Nr); // Compute ( - Jac)^-1 . Nr

        // Finally include the dependent set as well.
        ls::DoubleMatrix T4 = mult(LinkMatrix, T3); // Compute L (iwI - Jac)^-1 . Nr

        T4.setRowNames(getFloatingSpeciesIds());
        T4.setColNames(getReactionIds());

        //impl->simulateOpt.steps = orig_steps;

        return T4;
    }


    ls::DoubleMatrix RoadRunner::getScaledConcentrationControlCoefficientMatrix() {
        get_self();

        check_model();

        ls::DoubleMatrix ucc = getUnscaledConcentrationControlCoefficientMatrix();

        if (ucc.size() > 0) {
            for (int i = 0; i < ucc.RSize(); i++) {
                for (int j = 0; j < ucc.CSize(); j++) {
                    double conc = 0;
                    self.model->getFloatingSpeciesConcentrations(1, &i, &conc);

                    double rate = 0;
                    self.model->getReactionRates(1, &j, &rate);
                    ucc[i][j] = ucc[i][j] * rate / conc;
                }
            }
        }
        return ucc;
    }

// Use the formula: ucc = elast CS + I
// [Help("Compute the matrix of unscaled flux control coefficients")]
    ls::DoubleMatrix RoadRunner::getUnscaledFluxControlCoefficientMatrix() {
        get_self();

        check_model();

        ls::DoubleMatrix ucc = getUnscaledConcentrationControlCoefficientMatrix();
        ls::DoubleMatrix uec = getUnscaledElasticityMatrix();

        ls::DoubleMatrix T1 = mult(uec, ucc);

        // Add an identity matrix I to T1, that is add a 1 to every diagonal of T1
        for (int i = 0; i < T1.RSize(); i++) {
            T1[i][i] = T1[i][i] + 1;
        }

        T1.setRowNames(getReactionIds());
        T1.setColNames(getReactionIds());

        return T1;
    }

// [Help("Compute the matrix of scaled flux control coefficients")]
    ls::DoubleMatrix RoadRunner::getScaledFluxControlCoefficientMatrix() {
        try {
            if (!impl->model) {
                throw CoreException(gEmptyModelMessage);
            }

            ls::DoubleMatrix ufcc = getUnscaledFluxControlCoefficientMatrix();

            if (ufcc.RSize() > 0) {
                for (int i = 0; i < ufcc.RSize(); i++) {
                    double irate = 0;
                    impl->model->getReactionRates(1, &i, &irate);
                    if (abs(irate) < impl->roadRunnerOptions.fluxThreshold) {
                        rrLog(rr::Logger::LOG_INFORMATION) << "The reaction '";
                        rrLog(rr::Logger::LOG_INFORMATION) << impl->model->getReactionId(i);
                        rrLog(rr::Logger::LOG_INFORMATION) << "' value is close to zero, so setting the scaled coefficients to zero as well.";
                        irate = 0;
                    }
                    for (int j = 0; j < ufcc.CSize(); j++) {
                        if (irate != 0) {
                            double jrate = 0;
                            impl->model->getReactionRates(1, &j, &jrate);
                            ufcc[i][j] = ufcc[i][j] * jrate / irate;
                        } else {
                            ufcc[i][j] = 0;
                        }
                    }
                }
            }
            return ufcc;
        }
        catch (const Exception &e) {
            throw CoreException("Unexpected error from getScaledFluxControlCoefficientMatrix(): ", e.Message());
        }
    }

    static std::string convertSBMLVersion(const std::string &str, int level, int version) {
        libsbml::SBMLReader reader;
        std::stringstream stream;
        libsbml::SBMLDocument *doc = 0;


        try {
            // new doc
            doc = reader.readSBMLFromString(str);

            // this does an in-place conversion, at least for the time being
            libsbml::SBMLLevelVersionConverter versionConverter;

            libsbml::ConversionProperties versionProps = versionConverter.getDefaultProperties();

            // this is how the target version is set
            libsbml::SBMLNamespaces targetNamespace(level, version);

            // clones the ns
            versionProps.setTargetNamespaces(&targetNamespace);

            versionConverter.setProperties(&versionProps);

            // converter does an in-place conversion
            doc->setApplicableValidators((unsigned char) Config::getInt(
                    Config::SBML_APPLICABLEVALIDATORS));

            versionConverter.setDocument(doc);

            if (versionConverter.convert() != libsbml::LIBSBML_OPERATION_SUCCESS) {
                rrLog(rr::Logger::LOG_ERROR) << "could not change source sbml level or version";

                const libsbml::SBMLErrorLog *log = doc->getErrorLog();
                std::string errors = log ? log->toString() : std::string(" NULL SBML Error Log");
                rrLog(rr::Logger::LOG_ERROR) << "Conversion Errors: " + errors;

                throw std::logic_error("Error version converting sbml: " + errors);
            }

            libsbml::SBMLWriter writer;
            writer.writeSBML(doc, stream);
            delete doc;
        }
        catch (std::exception &) {
            delete doc;
            throw;
        }

        return stream.str();
    }

    std::string RoadRunner::getSBML(int level, int version) {
        check_model();

        std::stringstream stream;

        libsbml::SBMLWriter writer;
        writer.writeSBML(impl->document.get(), stream);

        if (level > 0) {
            return convertSBMLVersion(stream.str(), level, version);
        }
        return stream.str();
    }

    std::string RoadRunner::getCurrentSBML(int level, int version) {
        check_model();
        get_self();

        std::stringstream stream;
        libsbml::SBMLDocument doc(*impl->document);
        libsbml::Model *model = doc.getModel();

        while (model->getNumInitialAssignments() > 0) {
            model->removeInitialAssignment(0);
        }

        std::vector<std::string> array = getFloatingSpeciesIds();
        for (int i = 0; i < array.size(); i++) {
            double value = 0;
            impl->model->getFloatingSpeciesAmounts(1, &i, &value);
            setSpeciesAmount(model, array[i], value);
        }

        array = getBoundarySpeciesIds();
        for (int i = 0; i < array.size(); i++) {
            double value = 0;
            impl->model->getBoundarySpeciesConcentrations(1, &i, &value);
            setSBMLValue(model, array[i], value);
        }

        array = getCompartmentIds();
        for (int i = 0; i < array.size(); i++) {
            double value = 0;
            impl->model->getCompartmentVolumes(1, &i, &value);
            setSBMLValue(model, array[i], value);
        }

        array = getGlobalParameterIds();
        for (int i = 0; i < impl->model->getNumGlobalParameters(); i++) {
            double value = 0;
            impl->model->getGlobalParameterValues(1, &i, &value);

            libsbml::Parameter *param = model->getParameter(array[i]);
            if (param != NULL) {
                param->setValue(value);
            } else {
                // sanity check, just make sure that this is a conserved moeity
                if (self.model->getConservedMoietyIndex(array[i]) < 0) {
                    throw std::logic_error("The global parameter name "
                                           + array[i] + " could not be found in the SBML model, "
                                                        " and it is not a conserved moiety");
                }
            }
        }

        libsbml::SBMLWriter writer;
        writer.writeSBML(&doc, stream);

        if (level > 0) {
            return convertSBMLVersion(stream.str(), level, version);
        }
        return stream.str();
    }

    void RoadRunner::changeInitialConditions(const std::vector<double>& ic) {
        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }

        if (ic.size()) {
            impl->model->setFloatingSpeciesConcentrations(ic.size(), 0, &ic[0]);
            impl->model->setFloatingSpeciesInitConcentrations(ic.size(), 0, &ic[0]);
        }
    }

// Help("Returns the current std::vector of reactions rates")
    std::vector<double> RoadRunner::getReactionRates() {
        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }

        std::vector<double> rates(impl->model->getNumReactions());
        if (rates.size()) {
            impl->model->getReactionRates(rates.size(), 0, &rates[0]);
        }
        return rates;
    }


    void RoadRunner::setValue(const std::string &sId, double dValue) {
        check_model();

        SelectionRecord sel(sId);

        if (sel.selectionType & SelectionRecord::INITIAL) {
            //Don't worry if it doesn't exist, and don't regenerate yet.
            removeInitialAssignment(sel.p1, true, false);
        }
        
        impl->model->setValue(sId, dValue);

        if (sel.selectionType & SelectionRecord::INITIAL) {
            reset(
                SelectionRecord::TIME |
                SelectionRecord::RATE |
                SelectionRecord::FLOATING |
                SelectionRecord::BOUNDARY |
                SelectionRecord::COMPARTMENT |
                SelectionRecord::GLOBAL_PARAMETER |
                SelectionRecord::STOICHIOMETRY);
        }
    }


    double RoadRunner::getValue(const std::string &sel) {
        return getValue(createSelection(sel));
    }


    std::vector<double> RoadRunner::getSelectedValues() {
        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }

        std::vector<double> result;
        result.resize(impl->mSelectionList.size());

        for (int i = 0; i < impl->mSelectionList.size(); i++) {
            result[i] = getNthSelectedOutput(i, impl->model->getTime());
        }
        return result;
    }


    static std::vector<std::string> createSelectionList(const SimulateOptions &o) {
        //read from settings the variables found in the amounts and concentrations lists
        std::vector<std::string> result;

        // have an implied time
        result.emplace_back("time");

        // need to add them in order
        for (std::vector<std::string>::const_iterator var = o.variables.begin(); var != o.variables.end(); ++var) {
            if (find(o.concentrations.begin(), o.concentrations.end(), *var)
                != o.concentrations.end()) {
                result.push_back("[" + *var + "]");
            } else {
                result.push_back(*var);
            }

        }

        return result;
    }

    SelectionRecord RoadRunner::createSelection(const std::string &str) {
        if (!impl->model) {
            throw Exception("Can not create selection without a model");
        }

        SelectionRecord sel(str);

        if (sel.selectionType == SelectionRecord::UNKNOWN) {
            throw Exception("invalid selection std::string " + str);
        }

        // check to see that we have valid selection ids
        switch (sel.selectionType) {
            case SelectionRecord::TIME:
            case SelectionRecord::UNKNOWN_ELEMENT:
                // check for sbml element types

                if ((sel.index = impl->model->getFloatingSpeciesIndex(sel.p1)) >= 0) {
                    sel.selectionType = SelectionRecord::FLOATING_AMOUNT;
                    break;
                } else if ((sel.index = impl->model->getBoundarySpeciesIndex(sel.p1)) >= 0) {
                    sel.selectionType = SelectionRecord::BOUNDARY_AMOUNT;
                    break;
                } else if ((sel.index = impl->model->getCompartmentIndex(sel.p1)) >= 0) {
                    sel.selectionType = SelectionRecord::COMPARTMENT;
                    break;
                } else if ((sel.index = impl->model->getGlobalParameterIndex(sel.p1)) >= 0) {
                    sel.selectionType = SelectionRecord::GLOBAL_PARAMETER;
                    break;
                } else if ((sel.index = impl->model->getReactionIndex(sel.p1)) >= 0) {
                    sel.selectionType = SelectionRecord::REACTION_RATE;
                    break;
                } else if ((sel.index = impl->model->getStoichiometryIndex(sel.p1)) >= 0) {
                    sel.selectionType = SelectionRecord::STOICHIOMETRY;
                    break;
                } else if (sel.selectionType == SelectionRecord::TIME) {
                    //Need to put this here in case there's an actual SBML variable called 'time'
                    break;
                } else {
                    throw Exception("No sbml element exists for symbol '" + str + "'");
                    break;
                }
            case SelectionRecord::UNKNOWN_CONCENTRATION:
                if ((sel.index = impl->model->getFloatingSpeciesIndex(sel.p1)) >= 0) {
                    sel.selectionType = SelectionRecord::FLOATING_CONCENTRATION;
                    break;
                } else if ((sel.index = impl->model->getBoundarySpeciesIndex(sel.p1)) >= 0) {
                    sel.selectionType = SelectionRecord::BOUNDARY_CONCENTRATION;
                    break;
                } else {
                    std::string msg = "No sbml element exists for concentration selection '" + str + "'";
                    rrLog(Logger::LOG_ERROR) << msg;
                    throw Exception(msg);
                    break;
                }
            case SelectionRecord::FLOATING_AMOUNT_RATE:
                if ((sel.index = impl->model->getFloatingSpeciesIndex(sel.p1)) >= 0) {
                    break;
                } else if ((sel.index = impl->model->getBoundarySpeciesIndex(sel.p1)) >= 0) {
                    sel.selectionType = SelectionRecord::BOUNDARY_AMOUNT_RATE;
                    break;
                } else if ((sel.index = impl->model->getGlobalParameterIndex(sel.p1)) >= 0) {
                    sel.selectionType = SelectionRecord::GLOBAL_PARAMETER_RATE;
                    break;
                } else if ((sel.index = impl->model->getCompartmentIndex(sel.p1)) >= 0) {
                    sel.selectionType = SelectionRecord::COMPARTMENT_RATE;
                    break;
                } else {
                    throw Exception("Invalid id '" + str +
                                    "' for rate of change: must be a floating species, global parameter, or compartment.");
                    break;
                }
            case SelectionRecord::FLOATING_CONCENTRATION_RATE:
                if ((sel.index = impl->model->getFloatingSpeciesIndex(sel.p1)) >= 0) {
                    break;
                } else if ((sel.index = impl->model->getBoundarySpeciesIndex(sel.p1)) >= 0) {
                    sel.selectionType = SelectionRecord::BOUNDARY_CONCENTRATION_RATE;
                    break;
                } else {
                    throw Exception(
                            "Invalid id '" + str + "' for concentration rate of change: must be a floating species.");
                    break;
                }
            case SelectionRecord::CONTROL:
            case SelectionRecord::UNSCALED_CONTROL:
                // check that the control coef args are valid
                if (impl->model->getReactionIndex(sel.p1) >= 0 ||
                    impl->model->getFloatingSpeciesIndex(sel.p1) >= 0) {
                    if (impl->model->getGlobalParameterIndex(sel.p2) >= 0 ||
                        impl->model->getBoundarySpeciesIndex(sel.p2) >= 0 ||
                        impl->model->getConservedMoietyIndex(sel.p2) >= 0) {
                        rrLog(Logger::LOG_INFORMATION) << "Valid metabolic control selection: " << sel.to_repr();
                    } else {
                        throw Exception("The second argument to a metabolic control coefficient selection, " + sel.p2 +
                                        ", must be either a global parameter, boundary species or conserved sum");
                    }
                } else {
                    throw Exception("The first argument to a metabolic control coefficient selection, " + sel.p1 +
                                    ", must be either a reaction or a floating species.");
                }

                break;
            case SelectionRecord::ELASTICITY:
            case SelectionRecord::UNSCALED_ELASTICITY:
                // check that the elasticity coef args are valid
                if (impl->model->getReactionIndex(sel.p1) >= 0) {
                    if (impl->model->getGlobalParameterIndex(sel.p2) >= 0 ||
                        impl->model->getBoundarySpeciesIndex(sel.p2) >= 0 ||
                        impl->model->getFloatingSpeciesIndex(sel.p2) >= 0) {
                        rrLog(Logger::LOG_INFORMATION) << "Valid elasticity coefficient selection: " << sel.to_repr();
                    } else {
                        throw Exception("The second argument to an elasticity coefficient selection, " + sel.p2 +
                                        ", must be either a global parameter or a species.");
                    }
                } else {
                    throw Exception("The first argument to a metabolic control coefficient selection, " + sel.p1 +
                                    ", must be a reaction.");
                }

                break;
            case SelectionRecord::EIGENVALUE_REAL:
            case SelectionRecord::EIGENVALUE_IMAG:
            case SelectionRecord::EIGENVALUE_COMPLEX:
                if ((sel.index = impl->model->getFloatingSpeciesIndex(sel.p1)) >= 0) {
                    break;
                } else {
                    throw Exception("eigen argument '" + sel.p1 + "' is not a floating species.");
                }
                break;
            case SelectionRecord::STOICHIOMETRY:
                if (impl->model->getFloatingSpeciesIndex(sel.p1) >= 0) {
                    if (impl->model->getReactionIndex(sel.p2) >= 0) {
                        break;
                    } else {
                        throw Exception("second argument to stoich '" + sel.p2 + "' is not a reaction id.");
                    }
                } else {
                    throw Exception("first argument to stoich '" + sel.p1 + "' is not a floating species id.");
                }
            case SelectionRecord::INITIAL_CONCENTRATION:
                if ((sel.index = impl->model->getFloatingSpeciesIndex(sel.p1)) >= 0) {
                    sel.selectionType = SelectionRecord::INITIAL_FLOATING_CONCENTRATION;
                    break;
                } else if ((sel.index = impl->model->getBoundarySpeciesIndex(sel.p1)) >= 0) {
                    sel.selectionType = SelectionRecord::INITIAL_BOUNDARY_CONCENTRATION;
                    break;
                } else {
                    throw Exception("Invalid id '" + sel.p1 + "' for initial concentration");
                    break;
                }
            case SelectionRecord::INITIAL_AMOUNT:
                if ((sel.index = impl->model->getFloatingSpeciesIndex(sel.p1)) >= 0) {
                    sel.selectionType = SelectionRecord::INITIAL_FLOATING_AMOUNT;
                    break;
                } else if ((sel.index = impl->model->getBoundarySpeciesIndex(sel.p1)) >= 0) {
                    sel.selectionType = SelectionRecord::INITIAL_BOUNDARY_AMOUNT;
                    break;
                } else if ((sel.index = impl->model->getGlobalParameterIndex(sel.p1)) >= 0) {
                    sel.selectionType = SelectionRecord::INITIAL_GLOBAL_PARAMETER;
                    break;
                } else if ((sel.index = impl->model->getCompartmentIndex(sel.p1)) >= 0) {
                    sel.selectionType = SelectionRecord::INITIAL_COMPARTMENT;
                    break;
                } else {
                    throw Exception("Invalid id '" + sel.p1 + "' for initial amount");
                    break;
                }
            default:
                rrLog(Logger::LOG_ERROR) << "A new SelectionRecord should not have this value: "
                                         << sel.to_repr();
                break;
        }

        return sel;
    }

    bool compareNoCaseRR(const string& str1, const string& str2)
    {
#if defined(_WIN32)
        int res = stricmp(str1.c_str(), str2.c_str());
#else
        int res = strcasecmp(str1.c_str(), str2.c_str());
#endif
        return res == 0 ? true : false;
    }

    void RoadRunner::setSelections(const std::vector<std::string> &_selList) {
        impl->mSelectionList.clear();

        for (int i = 0; i < _selList.size(); ++i) {
            impl->mSelectionList.push_back(createSelection(_selList[i]));
        }

        //Even if there's a 'time' variable in the model, if the first variable is 'time', make it the TIME selection record anyway.
        if (impl->mSelectionList.size() && impl->mSelectionList[0].selectionType != SelectionRecord::TIME && compareNoCaseRR(impl->mSelectionList[0].p1, "time"))
        {
            impl->mSelectionList[0].selectionType = SelectionRecord::TIME;
        }

        std::vector<std::string> selstr = std::vector<std::string>(impl->mSelectionList.size());
        for (int i = 0; i < selstr.size(); ++i) {
            selstr[i] = impl->mSelectionList[i].to_string();
        }
        impl->simulationResult.setColNames(selstr.begin(), selstr.end());
    }

    void RoadRunner::setSelections(const std::vector<rr::SelectionRecord> &ss) {
        impl->mSelectionList = ss;
    }

    void RoadRunner::setSteadyStateSelections(const std::vector<std::string> &ss) {
        impl->mSteadyStateSelection.clear();

        for (int i = 0; i < ss.size(); ++i) {
            impl->mSteadyStateSelection.push_back(createSelection(ss[i]));
        }
        impl->loadOpt.loadFlags = impl->loadOpt.loadFlags | LoadSBMLOptions::NO_DEFAULT_STEADY_STATE_SELECTIONS;
    }

    void RoadRunner::setSteadyStateSelections(const std::vector<rr::SelectionRecord> &ss) {
        impl->mSteadyStateSelection = ss;
    }

//Compute the frequency response, startW, Number Of Decades, Number of Points, parameterName, variableName
    ls::Matrix<double> RoadRunner::getFrequencyResponse(double startFrequency,
                                                        int numberOfDecades, int numberOfPoints,
                                                        const std::string &parameterName,
                                                        const std::string &variableName,
                                                        bool useDB, bool useHz) {
        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }

        try {
            std::vector<std::string> reactionNames = getReactionIds();
            std::vector<std::string> speciesNames = getFloatingSpeciesIds();

            // Prepare the dv/dp array
            Matrix<std::complex<double>> dvdp(static_cast<unsigned int>(reactionNames.size()), 1);

            //Guess we don't need to simulate here?? (TK)
            //        SimulateOptions opt;
            //        opt.start = 0;
            //        opt.duration = 50.0;
            //        opt.steps = 1;
            //        simulate(&opt);
            if (steadyState() > 1E-2) {
                throw Exception("Unable to locate steady state during frequency response computation");
            }

            ComplexMatrix uelast(getUnscaledElasticityMatrix());
            ComplexMatrix Nr = getNrMatrix();
            ComplexMatrix LinkMatrix = getLinkMatrix();

            // Compute dv/dp
            for (int j = 0; j < reactionNames.size(); j++) {
                double val = getUnscaledParameterElasticity(reactionNames[j], parameterName);
                dvdp(j, 0) = std::complex<double>(val, 0.0);
                rrLog(lDebug) << "dv/dp: " << dvdp(j, 0);
            }

            // Compute the Jacobian first
            ComplexMatrix T1(Nr.RSize(), uelast.CSize());
            T1 = mult(Nr, uelast);
            rrLog(lInfo) << T1;

            ComplexMatrix Jac(Nr.RSize(), LinkMatrix.CSize());
            Jac = mult(T1, LinkMatrix);

            ComplexMatrix Id(Nr.RSize(), Nr.RSize());

            //Create Identity matrix
            for (int i = 0; i < Id.RSize(); i++) {
                Id(i, i) = std::complex<double>(1, 0);
            }

            ComplexMatrix T2(Nr.RSize(), LinkMatrix.CSize());   // Stores iwI - Jac  and (iwI - Jac)^-1
            ComplexMatrix T3(LinkMatrix.RSize(), 1);            // Stores (iwI - Jac)^-1 . Nr
            ComplexMatrix T4(Nr.RSize(), Nr.CSize());
            ComplexMatrix T5(LinkMatrix.RSize(), 1);
            ComplexMatrix Inv(Nr.RSize(), LinkMatrix.CSize());

            // resultArray stores frequency, gain and phase
            ls::Matrix<double> resultArray(numberOfPoints, 3);

            // Main loop, generate log spaced frequency numbers and compute the gain and phase at each frequency
            std::vector<double> w(logspace(startFrequency, numberOfDecades, numberOfPoints));

            for (int i = 0; i < numberOfPoints; i++) {
                std::complex<double> diagVal(0.0, w[i]);

                T1 = multDiag(Id, diagVal); // Compute iwI
                T2 = subtract(T1, Jac);     // Compute iwI - Jac //// Compute (iwI - Jac)^-1
                Inv = GetInverse(T2);
                T3 = mult(Inv, Nr);        // Compute (iwI - Jac)^-1 . Nr

                for (int j = 0; j < reactionNames.size(); j++) {
                    double realPart = getUnscaledParameterElasticity(reactionNames[j], parameterName);
                    dvdp(j, 0) = std::complex<double>(realPart, 0.0);
                }

                T4 = mult(T3, dvdp);    // Compute(iwI - Jac)^-1 . Nr . dvdp

                // Finally include the dependent set as well.
                T5 = mult(LinkMatrix, T4); // Compute L (iwI - Jac)^-1 . Nr . dvdp

                for (int j = 0; j < speciesNames.size(); j++) {
                    if (speciesNames[j] == variableName) {
                        double dw = abs(T5(j, 0));
                        if (useDB) {
                            dw = 20.0 * log10(dw);
                        }
                        resultArray(i, 1) = dw;
                        std::complex<double> val(T5(j, 0));

                        double phase = (180.0 / M_PI) * rr::phase(val) + getAdjustment(val);
                        resultArray(i, 2) = phase;
                        break;
                    }
                }
                if (useHz) {
                    // Store frequency, convert to Hz by dividing by 2Pi
                    resultArray[i][0] = w[i] / (2. * M_PI);
                } else {
                    // Store frequency, leave as rad/sec
                    resultArray[i][0] = w[i];
                }
            }
            return resultArray;
        }
        catch (const Exception &e) {
            throw Exception("Unexpected error in getFrequencyResponse(): " + e.Message());
        }
    }


    double
    RoadRunner::getUnscaledParameterElasticity(const std::string &reactionName, const std::string &parameterName) {
        int parameterIndex;

        if (!impl->model) {
            throw CoreException(gEmptyModelMessage);
        }
        try {
            int reactionIndex = impl->model->getReactionIndex(reactionName);

            if (reactionIndex == -1) {
                throw CoreException(
                        "Internal Error: unable to locate reaction name while computing unscaled elasticity");
            }

            // Look for the parameter name, check local parameters first, then global
            ParameterType parameterType;

            if (impl->model->getGlobalParameterIndex(parameterName) != -1) {
                parameterIndex = impl->model->getGlobalParameterIndex(parameterName);
                parameterType = ptGlobalParameter;
            } else if (impl->model->getBoundarySpeciesIndex(parameterName) != -1) {
                parameterIndex = impl->model->getBoundarySpeciesIndex(parameterName);
                parameterType = ptBoundaryParameter;
            } else if (impl->model->getConservedMoietyIndex(parameterName) != -1) {
                parameterIndex = impl->model->getConservedMoietyIndex(parameterName);
                parameterType = ptConservationParameter;
            } else {
                return 0.0;
            }

            double originalParameterValue = 0.0;
            double result = 0;
            switch (parameterType) {
                case ptGlobalParameter:
                    originalParameterValue = getGlobalParameterByIndex(parameterIndex);
                    break;
                case ptBoundaryParameter:
                    originalParameterValue = getBoundarySpeciesByIndex(parameterIndex);
                    break;
                case ptConservationParameter:
                    //originalParameterValue = impl->mModel->getC ;//model.ct[parameterIndex];
                    impl->model->getConservedMoietyValues(1, &parameterIndex, &result);
                    originalParameterValue = result;
                    break;
                default:
                    throw (Exception("This parameterType is not supported in getUnscaledParameterElasticity"));
            }

            double hstep = impl->roadRunnerOptions.diffStepSize * originalParameterValue;
            if (fabs(hstep) < 1E-12) {
                hstep = impl->roadRunnerOptions.diffStepSize;
            }

            double f1, f2, fi, fi2, fd, fd2;
            impl->changeParameter(parameterType, reactionIndex, parameterIndex, originalParameterValue, hstep);
            impl->model->getReactionRates(1, &reactionIndex, &fi);

            impl->changeParameter(parameterType, reactionIndex, parameterIndex, originalParameterValue, 2.0 * hstep);
            impl->model->getReactionRates(1, &reactionIndex, &fi2);

            impl->changeParameter(parameterType, reactionIndex, parameterIndex, originalParameterValue, -hstep);
            impl->model->getReactionRates(1, &reactionIndex, &fd);

            impl->changeParameter(parameterType, reactionIndex, parameterIndex, originalParameterValue, -2.0 * hstep);
            impl->model->getReactionRates(1, &reactionIndex, &fd2);

            // Use instead the 5th order approximation double unscaledElasticity = (0.5/hstep)*(fi-fd);
            // The following separated lines avoid small amounts of roundoff error
            f1 = fd2 + 8.0 * fi;
            f2 = -(8.0 * fd + fi2);

            // What ever happens, make sure we restore the species level
            impl->changeParameter(parameterType, reactionIndex, parameterIndex, originalParameterValue, 0);
            return 1.0 / (12.0 * hstep) * (f1 + f2);
        }
        catch (const Exception &ex) {
            throw (Exception("Problem in getUnscaledParameterElasticity():" + ex.getMessage()));
        }
    }


    std::vector<double> logspace(const double &startW, const double &d2, const int &n) {
        double d1 = 0;
        std::vector<double> y(n);
        for (int i = 0; i <= n - 2; i++) {
            y[i] = i * (d2 - d1);
            y[i] = y[i] / (n - 1);
            y[i] = d1 + y[i];
            y[i] = pow(10, y[i]) * startW;
        }
        y[n - 1] = pow(10, d2) * startW;
        return y;
    }

    double phase(std::complex<double> &val) {
        if ((std::real(val) == 0.0) && (imag(val) == 0.0)) {
            return 0.0;
        } else {
            return atan2(imag(val), std::real(val));
        }
    }

//This function is used in roadrunners freq analysis code...
    double getAdjustment(std::complex<double> &z) {
        double adjustment;
        if (std::real(z) >= 0 && imag(z) >= 0) {
            adjustment = 0;
        } else if (std::real(z) >= 0 && imag(z) < 0) {
            adjustment = 360;
        } else if (std::real(z) < 0 && imag(z) >= 0) {
            adjustment = 0;
        } else {
            adjustment = 360;
        }
        return adjustment;
    }


/************************ Selection Ids Species Section ***********************/
/******************************************************************************/
/******************************************************************************/

    void RoadRunner::getIds(int types, std::list<std::string> &ids) {
        if (impl->model) {
            impl->model->getIds(types, ids);

            /**
         * types is -1 when SelectionRecord is set to ALL.
         */
            if (types & SelectionRecord::EIGENVALUE_REAL) {
                std::list<std::string> eigen;
                impl->model->getIds(SelectionRecord::FLOATING_AMOUNT, eigen);

                for (std::list<std::string>::const_iterator i =
                        eigen.begin(); i != eigen.end(); ++i) {
                    ids.push_back("eigen(" + *i + ")");
                    ids.push_back("eigenReal(" + *i + ")");
                    ids.push_back("eigenImag(" + *i + ")");
                }
            }
        }
    }

    std::vector<std::string> RoadRunner::getIndependentFloatingSpeciesIds() {
        return getLibStruct()->getIndependentSpecies();
    }

    std::vector<std::string> RoadRunner::getDependentFloatingSpeciesIds() {
        return getLibStruct()->getDependentSpecies();
    }

    int RoadRunner::getSupportedIdTypes() {
        int types = impl->model ? impl->model->getSupportedIdTypes() : 0;

        return types | SelectionRecord::ELASTICITY |
               SelectionRecord::UNSCALED_ELASTICITY |
               SelectionRecord::CONTROL |
               SelectionRecord::UNSCALED_CONTROL |
               SelectionRecord::EIGENVALUE_REAL;
    }

    std::vector<std::string> RoadRunner::getRateOfChangeIds() {
        std::list<std::string> rate_list, ind_spec_list;

        if (impl->model) {
            impl->model->getIds(SelectionRecord::RATE, rate_list);
            impl->model->getIds(SelectionRecord::FLOATING_AMOUNT_RATE, ind_spec_list);
        }

        std::vector<std::string> ret(rate_list.begin(), rate_list.end());
        for (auto isl = ind_spec_list.begin(); isl != ind_spec_list.end(); isl++)
        {
            bool found = false;
            for (auto rr = rate_list.begin(); rr != rate_list.end(); rr++)
            {
                if (*rr == *isl) {
                    found = true;
                }
            }
            if (!found)
            {
                ret.push_back(*isl);
            }
        }
        return ret;
    }

    std::vector<std::string> RoadRunner::getCompartmentIds() {
        std::list<std::string> list;

        if (impl->model) {
            impl->model->getIds(SelectionRecord::COMPARTMENT, list);
        }

        return std::vector<std::string>(list.begin(), list.end());
    }

    std::vector<std::string> RoadRunner::getGlobalParameterIds() {
        std::list<std::string> list;

        if (impl->model) {
            impl->model->getIds(SelectionRecord::GLOBAL_PARAMETER, list);
        }

        return std::vector<std::string>(list.begin(), list.end());
    }

    std::vector<std::string> RoadRunner::getBoundarySpeciesIds() {
        std::list<std::string> list;

        if (impl->model) {
            impl->model->getIds(SelectionRecord::BOUNDARY_AMOUNT, list);
        }

        return std::vector<std::string>(list.begin(), list.end());
    }

    std::vector<std::string> RoadRunner::getAssignmentRuleIds() {
        std::list<std::string> list;
        if (impl->model) {
            impl->model->getAssignmentRuleIds(list);
        }

        return std::vector<std::string>(list.begin(), list.end());
    }

    std::vector<std::string> RoadRunner::getRateRuleIds() {
        std::list<std::string> list;
        if (impl->model) {
            impl->model->getRateRuleIds(list);
        }

        return std::vector<std::string>(list.begin(), list.end());
    }

    std::vector<std::string> RoadRunner::getInitialAssignmentIds() {
        std::list<std::string> list;
        if (impl->model) {
            impl->model->getInitialAssignmentIds(list);
        }

        return std::vector<std::string>(list.begin(), list.end());
    }

    std::vector<std::string> RoadRunner::getBoundarySpeciesConcentrationIds() {
        std::list<std::string> list;

        if (impl->model) {
            impl->model->getIds(SelectionRecord::BOUNDARY_CONCENTRATION, list);
        }

        return std::vector<std::string>(list.begin(), list.end());
    }

    std::vector<std::string> RoadRunner::getConservedMoietyIds() {
        return createModelStringList(impl->model.get(), &ExecutableModel::getNumConservedMoieties,
                                     &ExecutableModel::getConservedMoietyId);
    }

    std::vector<std::string> RoadRunner::getFloatingSpeciesIds() {
        std::list<std::string> list;

        if (impl->model) {
            impl->model->getIds(SelectionRecord::FLOATING_AMOUNT, list);
        }

        return std::vector<std::string>(list.begin(), list.end());
    }

    std::vector<std::string> RoadRunner::getFloatingSpeciesConcentrationIds() {
        std::list<std::string> list;

        if (impl->model) {
            impl->model->getIds(SelectionRecord::FLOATING_CONCENTRATION, list);
        }

        return std::vector<std::string>(list.begin(), list.end());
    }

    std::vector<std::string> RoadRunner::getFloatingSpeciesInitialConditionIds() {
        std::list<std::string> list;

        if (impl->model) {
            impl->model->getIds(SelectionRecord::INITIAL_FLOATING_CONCENTRATION, list);
        }

        return std::vector<std::string>(list.begin(), list.end());
    }

    std::vector<std::string> RoadRunner::getFloatingSpeciesInitialConcentrationIds() {
        std::list<std::string> list;

        if (impl->model) {
            impl->model->getIds(SelectionRecord::INITIAL_FLOATING_CONCENTRATION, list);
        }

        return std::vector<std::string>(list.begin(), list.end());
    }

    std::vector<std::string> RoadRunner::getReactionIds() {
        std::list<std::string> list;

        if (impl->model) {
            impl->model->getIds(SelectionRecord::REACTION_RATE, list);
        }

        return std::vector<std::string>(list.begin(), list.end());
    }

    std::vector<std::string> RoadRunner::getEigenValueIds() {
        std::list<std::string> list;

        getIds(SelectionRecord::EIGENVALUE_REAL, list);

        return std::vector<std::string>(list.begin(), list.end());
    }


/************************ End Selection Ids Species Section *******************/
/******************************************************************************/
/******************************************************************************/

    RoadRunnerOptions &RoadRunner::getOptions() {
        return impl->roadRunnerOptions;
    }

    const ls::DoubleMatrix *RoadRunner::getSimulationData() const {
        return &impl->simulationResult;
    }

    void RoadRunner::applySimulateOptions() {
        get_self();

        if (self.simulateOpt.duration < 0 || self.simulateOpt.steps < 0) {
            throw std::invalid_argument("duration and steps must be non-negative");
        }

        // This one creates the list of what we will look at in the result
        // uses values (potentially) from simulate options.
        createTimeCourseSelectionList();
        self.simulateOpt.initialize();

        if (self.simulateOpt.reset_model) {
            reset(); // reset back to initial conditions
            self.simulateOpt.reset_model = false;
        }
    }


    static void metabolicControlCheck(ExecutableModel *model) {
        if (model->getNumEvents() > 0 && !Config::getBool(Config::ALLOW_EVENTS_IN_STEADY_STATE_CALCULATIONS)) {
            throw std::invalid_argument(
                    "The steady state cannot be calculated in a model with events, which this model has. To override set ALLOW_EVENTS_IN_STEADY_STATE_CALCULATIONS to True."
                    "To override, run 'roadrunner.Config.setValue(roadrunner.Config.ALLOW_EVENTS_IN_STEADY_STATE_CALCULATIONS, True)'.");
        }
    }

    void RoadRunner::saveState(std::string filename, char opt) {
        check_model();
        std::stringstream *state = saveStateS(opt);
        std::ofstream of(filename, std::iostream::binary);
        of << state->rdbuf();
        of.close();
        delete state;
    }

    std::stringstream *RoadRunner::saveStateS(char opt) {
        check_model();
        switch (opt) {
            case 'b': {
                // binary mode
                // can be loaded later
                auto outPtr = new std::stringstream(
                        std::iostream::binary | std::stringstream::out | std::stringstream::in);
                rr::saveBinary(*outPtr, fileMagicNumber);
                rr::saveBinary(*outPtr, dataVersionNumber);
                //Save all of roadrunner's data to the file
                rr::saveBinary(*outPtr, impl->mInstanceID);

                saveSelectionVector(*outPtr, impl->mSelectionList);

                rr::saveBinary(*outPtr, impl->loadOpt.version);
                rr::saveBinary(*outPtr, impl->loadOpt.size);
                rr::saveBinary(*outPtr, impl->loadOpt.modelGeneratorOpt);
                rr::saveBinary(*outPtr, impl->loadOpt.loadFlags);

                rr::saveBinary(*outPtr, impl->loadOpt.getKeys().size());

                for (std::string k: impl->loadOpt.getKeys()) {
                    rr::saveBinary(*outPtr, k);
                    rr::saveBinary(*outPtr, impl->loadOpt.getItem(k));
                }

                saveSelectionVector(*outPtr, impl->mSteadyStateSelection);

                rr::saveBinary(*outPtr, impl->simulationResult.getColNames());
                rr::saveBinary(*outPtr, impl->simulationResult.getRowNames());

                rr::saveBinary(*outPtr, impl->simulateOpt.reset_model);
                rr::saveBinary(*outPtr, impl->simulateOpt.structured_result);
                rr::saveBinary(*outPtr, impl->simulateOpt.copy_result);
                rr::saveBinary(*outPtr, impl->simulateOpt.steps);
                rr::saveBinary(*outPtr, impl->simulateOpt.start);
                rr::saveBinary(*outPtr, impl->simulateOpt.duration);
                rr::saveBinary(*outPtr, impl->simulateOpt.variables);
                rr::saveBinary(*outPtr, impl->simulateOpt.amounts);
                rr::saveBinary(*outPtr, impl->simulateOpt.concentrations);
                rr::saveBinary(*outPtr, impl->simulateOpt.times);

                //rr::saveBinary(out, impl->simulateOpt.getKeys().size());

                //for (std::string k : impl->simulateOpt.getKeys()) {
                //    rr::saveBinary(out, k);
                //    rr::saveBinary(out, impl->simulateOpt.getItem(k));
                //}

                rr::saveBinary(*outPtr, impl->roadRunnerOptions.flags);
                rr::saveBinary(*outPtr, impl->roadRunnerOptions.jacobianStepSize);
                rr::saveBinary(*outPtr, impl->roadRunnerOptions.diffStepSize);
                rr::saveBinary(*outPtr, impl->roadRunnerOptions.steadyStateThreshold);
                rr::saveBinary(*outPtr, impl->roadRunnerOptions.fluxThreshold);

                rr::saveBinary(*outPtr, impl->configurationXML);
                //Save the model (which saves the model data symbols and model resources)
                impl->model->saveState(*outPtr);

                rr::saveBinary(*outPtr, impl->integrator->getName());
                rr::saveBinary(*outPtr, static_cast<unsigned long>(impl->integrator->getNumParams()));
                for (std::string k: impl->integrator->getSettings()) {
                    rr::saveBinary(*outPtr, k);
                    rr::saveBinary(*outPtr, impl->integrator->getValue(k));
                }

                rr::saveBinary(*outPtr, impl->steady_state_solver->getName());
                rr::saveBinary(*outPtr, static_cast<unsigned long>(impl->steady_state_solver->getNumParams()));
                for (std::string k: impl->steady_state_solver->getSettings()) {
                    rr::saveBinary(*outPtr, k);
                    rr::saveBinary(*outPtr, impl->steady_state_solver->getValue(k));
                }


                rr::saveBinary(*outPtr, impl->sensitivity_solver->getName());
                rr::saveBinary(*outPtr, static_cast<unsigned long>(impl->sensitivity_solver->getNumParams()));
                for (std::string k: impl->sensitivity_solver->getSettings()) {
                    rr::saveBinary(*outPtr, k);
                    rr::saveBinary(*outPtr, impl->sensitivity_solver->getValue(k));
                }

                //Currently I save and reload the SBML that was used to create the model
                //It is not parsed however, unless a instance of LibStructural needs to be
                //created
                //It might also be possible to construct LibStructural without SBML, but I'm not familiar with it
                //If this implementation is too slow we can change that
                char *sbmlToSave = impl->document->toSBML();
                rr::saveBinary(*outPtr, std::string(sbmlToSave));
                free(sbmlToSave);

                return outPtr;
            }

            case 'r': {
                // human-readble mode
                // for user debugging
                auto outPtr = new std::stringstream(
                        std::iostream::binary | std::stringstream::out | std::stringstream::in);
                //                if (!*outPtr) {
                //                    throw std::invalid_argument("Error opening file " + filename + ": " + std::string(strerror(errno)));
                //                }

                *outPtr << "mInstanceID: " << impl->mInstanceID << std::endl;
                *outPtr << "roadRunnerOptions: " << std::endl;
                *outPtr << "	flags: " << impl->roadRunnerOptions.flags << std::endl;
                *outPtr << "	jacobianStepSize: " << impl->roadRunnerOptions.jacobianStepSize << std::endl;
                *outPtr << "    diffStepSize: " << impl->roadRunnerOptions.diffStepSize << std::endl;
                *outPtr << "    steadyStateThreshold: " << impl->roadRunnerOptions.steadyStateThreshold << std::endl << std::endl;

                *outPtr << "loadOpt: " << std::endl;
                *outPtr << "	version: " << impl->loadOpt.version << std::endl;
                *outPtr << "	modelGeneratorOpt: " << impl->loadOpt.modelGeneratorOpt << std::endl;
                *outPtr << "	loadFlags: " << impl->loadOpt.loadFlags << std::endl;
                for (std::string k: impl->loadOpt.getKeys()) {
                    *outPtr << "	" << k << ": ";

                    switch (impl->loadOpt.getItem(k).type()) {
                        case Setting::BOOL:
                            *outPtr << impl->loadOpt.getItem(k).get<bool>();
                            break;
                        case Setting::CHAR:
                            *outPtr << impl->loadOpt.getItem(k).get<char>();
                            break;
                        case Setting::DOUBLE:
                            *outPtr << impl->loadOpt.getItem(k).get<double>();
                            break;
                        case Setting::FLOAT:
                            *outPtr << impl->loadOpt.getItem(k).get<float>();
                            break;
                        case Setting::INT32:
                            *outPtr << impl->loadOpt.getItem(k).get<std::int32_t>();
                            break;
                        case Setting::INT64:
                            *outPtr << impl->loadOpt.getItem(k).get<std::int64_t>();
                            break;
                        case Setting::STRING:
                            *outPtr << impl->loadOpt.getItem(k).get<std::string>();
                            break;
                        case Setting::UCHAR:
                            *outPtr << impl->loadOpt.getItem(k).get<unsigned char>();
                            break;
                        case Setting::UINT32:
                            *outPtr << impl->loadOpt.getItem(k).get<unsigned int>();
                            break;
                        case Setting::UINT64:
                            *outPtr << impl->loadOpt.getItem(k).get<std::uint64_t>();
                            break;
                        default:
                            break;
                    }
                    *outPtr << std::endl;
                }
                *outPtr << std::endl;

                *outPtr << "simulateOpt: " << std::endl;
                *outPtr << impl->simulateOpt.toString() << std::endl << std::endl;

                *outPtr << "mSelectionList: " << std::endl;
                for (SelectionRecord sr: impl->mSelectionList) {
                    *outPtr << sr.to_string() << std::endl;
                }
                *outPtr << std::endl;

                *outPtr << "mSteadyStateSelection: " << std::endl;
                for (SelectionRecord sr: impl->mSteadyStateSelection) {
                    *outPtr << sr.to_string() << std::endl;
                }
                *outPtr << std::endl;

                *outPtr << impl->integrator->toString();
                *outPtr << std::endl;
                *outPtr << impl->steady_state_solver->toString();
                *outPtr << std::endl;

                *outPtr << "simulationResult: " << std::endl;
                *outPtr << impl->simulationResult;
                *outPtr << std::endl;

                *outPtr << std::dec << impl->model.get();

                //*outPtr << "configurationXML" << impl->configurationXML << std::endl;
                //*outPtr << impl->currentSBML;
                return outPtr;

            }
            default:
                throw std::invalid_argument("Invalid option for saveState(), 'b' or 'r' expected");
                break;
        }
        return {};
    }

    void RoadRunner::saveSelectionVector(std::ostream &out, std::vector<SelectionRecord> &v) {
        rr::saveBinary(out, v.size());
        for (SelectionRecord sr: v) {
            rr::saveBinary(out, sr.index);
            rr::saveBinary(out, sr.p1);
            rr::saveBinary(out, sr.p2);
            rr::saveBinary(out, sr.selectionType);
        }
    }

    void RoadRunner::loadState(const std::string &filename) {
        if (!std::filesystem::exists(filename)) {
            std::string err = "Input argument filename doesn't exist: " + filename;
            rrLogDebug << err;
            throw std::invalid_argument(err);
        }
        std::ifstream ifs(filename, std::ios::binary);
        std::stringstream *ss = new std::stringstream(
                std::iostream::binary |
                std::stringstream::out |
                std::stringstream::in);
        *ss << ifs.rdbuf();
        loadStateS(ss);
    }

    void RoadRunner::loadStateS(std::stringstream *in) {
        int inMagicNumber;
        rr::loadBinary(*in, inMagicNumber);
        std::string x = in->str();
        if (inMagicNumber != fileMagicNumber) {
            throw std::invalid_argument(
                    "The state has the wrong magic number. Are you sure it is a roadrunner save state?");
        }

        int inVersionNumber;
        rr::loadBinary(*in, inVersionNumber);
        rrLogDebug << "inVersionNumber:" << inVersionNumber;
        if (inVersionNumber < dataVersionNumber) {
            throw std::invalid_argument(
                    "The file state was saved with a previous version of roadrunner");
        }
        if (inVersionNumber > dataVersionNumber) {
            throw std::invalid_argument(
                    "The file state was saved with a version of roadrunner more recent than this executable.");
        }
        //load roadrunner's data in the same order saveState saves it in
        int oldInstanceID;
        rr::loadBinary(*in, oldInstanceID); //Keep our current one; it's supposed to be unique.

        loadSelectionVector(*in, impl->mSelectionList);

        rr::loadBinary(*in, impl->loadOpt.version);
        rr::loadBinary(*in, impl->loadOpt.size);
        rr::loadBinary(*in, impl->loadOpt.modelGeneratorOpt);
        rr::loadBinary(*in, impl->loadOpt.loadFlags);

        size_t loadOptSize;
        rr::loadBinary(*in, loadOptSize);

        for (int i = 0; i < loadOptSize; i++) {
            std::string k;
            rr::loadBinary(*in, k);
            rr::Setting v;
            rr::loadBinary(*in, v);
            impl->loadOpt.setItem(k, v);
        }
        loadSelectionVector(*in, impl->mSteadyStateSelection);
        std::vector<std::string> colNames;
        rr::loadBinary(*in, colNames);
        impl->simulationResult.setColNames(colNames.begin(), colNames.end());
        std::vector<std::string> rowNames;
        rr::loadBinary(*in, rowNames);
        impl->simulationResult.setRowNames(rowNames.begin(), rowNames.end());
        rr::loadBinary(*in, impl->simulateOpt.reset_model);
        rr::loadBinary(*in, impl->simulateOpt.structured_result);
        rr::loadBinary(*in, impl->simulateOpt.copy_result);
        rr::loadBinary(*in, impl->simulateOpt.steps);
        rr::loadBinary(*in, impl->simulateOpt.start);
        rr::loadBinary(*in, impl->simulateOpt.duration);
        rr::loadBinary(*in, impl->simulateOpt.variables);
        rr::loadBinary(*in, impl->simulateOpt.amounts);
        rr::loadBinary(*in, impl->simulateOpt.concentrations);
        rr::loadBinary(*in, impl->simulateOpt.times);

        //size_t simulateOptSize;
        //rr::loadBinary(*in, simulateOptSize);
        //for (int i = 0; i < simulateOptSize; i++) {
        //    std::string k;
        //    rr::loadBinary(*in, k);
        //    rr::Setting v;
        //    rr::loadBinary(*in, v);
        //    impl->simulateOpt.setItem(k, v);
        //}
        rr::loadBinary(*in, impl->roadRunnerOptions.flags);
        rr::loadBinary(*in, impl->roadRunnerOptions.jacobianStepSize);
        rr::loadBinary(*in, impl->roadRunnerOptions.diffStepSize);
        rr::loadBinary(*in, impl->roadRunnerOptions.steadyStateThreshold);
        rr::loadBinary(*in, impl->roadRunnerOptions.fluxThreshold);

        rr::loadBinary(*in, impl->configurationXML);
        //Create a new model from the stream
        //impl->model = new rrllvm::LLVMExecutableModel(*in, impl->loadOpt.modelGeneratorOpt);
        impl->model.reset(ExecutableModelFactory::createModel(*in, impl->loadOpt.modelGeneratorOpt));

        impl->syncAllSolversWithModel(impl->model.get());
        if (impl->mLS)
            delete impl->mLS;

        std::string integratorName;
        rr::loadBinary(*in, integratorName);
        setIntegrator(integratorName);
        unsigned long integratorNumParams;
        rr::loadBinary(*in, integratorNumParams);
        for (int i = 0; i < integratorNumParams; i++) {
            std::string k;
            rr::loadBinary(*in, k);
            rr::Setting v;
            rr::loadBinary(*in, v);
            if (k != "maximum_adams_order")
                impl->integrator->setValue(k, v);
        }

        std::string steadyStateSolverName;
        rr::loadBinary(*in, steadyStateSolverName);
        setSteadyStateSolver(steadyStateSolverName);
        unsigned long solverNumParams;
        rr::loadBinary(*in, solverNumParams);
        for (int i = 0; i < solverNumParams; i++) {
            std::string k;
            rr::loadBinary(*in, k);
            rr::Setting v;
            rr::loadBinary(*in, v);
            impl->steady_state_solver->setValue(k, v);
        }


        std::string sensitivitySolverName;
        rr::loadBinary(*in, sensitivitySolverName);
        setSensitivitySolver(sensitivitySolverName);
        unsigned long sensSolverNumParams;
        rr::loadBinary(*in, sensSolverNumParams);
        for (int i = 0; i < sensSolverNumParams; i++) {
            std::string k;
            rr::loadBinary(*in, k);
            rr::Setting v;
            rr::loadBinary(*in, v);
            impl->sensitivity_solver->setValue(k, v);
        }

        //Currently the SBML is saved with the binary data, see saveState above
        std::string savedSBML;
        rr::loadBinary(*in, savedSBML);
        libsbml::SBMLReader reader;
        impl->document.reset(reader.readSBMLFromString(savedSBML));

        //Restart the integrator and reset the model time
        impl->integrator->restart(impl->model->getTime());
        reset(SelectionRecord::TIME);
        delete in;
    }

    void RoadRunner::loadSelectionVector(std::istream &in, std::vector<SelectionRecord> &v) {
        size_t vsize;
        rr::loadBinary(in, vsize);
        v.clear();
        for (int i = 0; i < vsize; i++) {
            SelectionRecord sr;
            rr::loadBinary(in, sr.index);
            rr::loadBinary(in, sr.p1);
            rr::loadBinary(in, sr.p2);
            rr::loadBinary(in, sr.selectionType);
            v.push_back(sr);
        }
    }


    void RoadRunner::addSpeciesConcentration(const std::string &sid, const std::string &compartment,
                                             double initConcentration, bool hasOnlySubstanceUnits,
                                             bool boundaryCondition, const std::string &substanceUnits,
                                             bool forceRegenerate) {
        checkID("addSpeciesConcentration", sid);

        libsbml::Model *model = impl->document->getModel();

        if (forceRegenerate && model->getCompartment(compartment) == NULL) {
            throw std::invalid_argument(
                    "Roadrunner::addaddSpeciesConcentrationSpecies failed, no compartment " + compartment +
                    " existed in the model");
        }

        rrLog (Logger::LOG_DEBUG) << "Adding species " << sid << " in compartment " << compartment << "..."
                                  << std::endl;
        libsbml::Species *newSpecies = impl->document->getModel()->createSpecies();

        int ret = newSpecies->setId(sid);
        if (ret != libsbml::LIBSBML_OPERATION_SUCCESS) {
            newSpecies->removeFromParentAndDelete();
            throw std::invalid_argument(
                    "Roadrunner::addSpeciesConcentration failed: invalid species id '" + sid + "'.");
        }
        newSpecies->setCompartment(compartment);

        // if InitialAssignment is set for the species, then initialAmount will be ignored, but set it anyway.

        newSpecies->setInitialConcentration(initConcentration);
        newSpecies->setHasOnlySubstanceUnits(hasOnlySubstanceUnits);
        newSpecies->setBoundaryCondition(boundaryCondition);

        bool validUnit = false;
        if (!substanceUnits.empty()) {
            if (model->getUnitDefinition(substanceUnits) != NULL)
                validUnit = true;
            else {
                validUnit = libsbml::UnitKind_forName(substanceUnits.c_str()) != libsbml::UNIT_KIND_INVALID;
            }
        }

        if (validUnit) {
            newSpecies->setSubstanceUnits(substanceUnits);
        }

        newSpecies->setConstant(false);

        regenerateModel(forceRegenerate);
    }

    void RoadRunner::addSpeciesAmount(const std::string &sid, const std::string &compartment, double initAmount,
                                      bool hasOnlySubstanceUnits, bool boundaryCondition,
                                      const std::string &substanceUnits, bool forceRegenerate) {
        checkID("addSpeciesAmount", sid);
        checkID("addSpeciesAmount", sid);

        libsbml::Model *model = impl->document->getModel();

        if (forceRegenerate && model->getCompartment(compartment) == NULL) {
            throw std::invalid_argument("Roadrunner::addSpaddSpeciesAmountecies failed, no compartment " + compartment +
                                        " existed in the model");
        }

        rrLog(Logger::LOG_DEBUG) << "Adding species " << sid << " in compartment " << compartment << "..." << std::endl;
        libsbml::Species *newSpecies = impl->document->getModel()->createSpecies();

        int ret = newSpecies->setId(sid);
        if (ret != libsbml::LIBSBML_OPERATION_SUCCESS) {
            newSpecies->removeFromParentAndDelete();
            throw std::invalid_argument("Roadrunner::addSpeciesAmount failed: invalid species id '" + sid + "'.");
        }
        newSpecies->setCompartment(compartment);

        // if InitialAssignment is set for the species, then initialAmount will be ignored, but set it anyway.

        newSpecies->setInitialAmount(initAmount);
        newSpecies->setHasOnlySubstanceUnits(hasOnlySubstanceUnits);
        newSpecies->setBoundaryCondition(boundaryCondition);

        bool validUnit = false;
        if (!substanceUnits.empty()) {
            if (model->getUnitDefinition(substanceUnits) != NULL)
                validUnit = true;
            else {
                validUnit = libsbml::UnitKind_forName(substanceUnits.c_str()) != libsbml::UNIT_KIND_INVALID;
            }
        }

        if (validUnit) {
            newSpecies->setSubstanceUnits(substanceUnits);
        }

        newSpecies->setConstant(false);

        regenerateModel(forceRegenerate);
    }

    void RoadRunner::removeSpecies(const std::string &sid, bool forceRegenerate) {

        using namespace libsbml;
        Model *sbmlModel = impl->document->getModel();
        Species *s = sbmlModel->removeSpecies(sid);
        if (s == NULL) {
            throw std::invalid_argument(
                    "Roadrunner::removeSpecies failed, no species with ID " + sid + " existed in the model");
        }

        rrLog(Logger::LOG_DEBUG) << "Removing species " << sid << "..." << std::endl;


        // delete all related reactions as well
        int index = 0;
        int numReaction = sbmlModel->getNumReactions();

        // TODO: will deleting one reaction interrupt with the order?
        // currently we are assuming that the order will keep the same
        for (uint i = 0; i < numReaction; i++) {
            Reaction *reaction = sbmlModel->getReaction(index);
            Reaction *toDelete = nullptr;

            const ListOfSpeciesReferences *reactants = reaction->getListOfReactants();
            for (uint j = 0; j < reactants->size(); j++) {

                if (reactants->get(j)->getSpecies() == sid) {
                    toDelete = sbmlModel->removeReaction(index);
                    break;
                }
            }

            if (toDelete != nullptr) {
                delete toDelete;
                // no need to check anymore
                continue;
            }

            const libsbml::ListOfSpeciesReferences *products = reaction->getListOfProducts();
            for (uint j = 0; j < products->size(); j++) {
                if (products->get(j)->getSpecies() == sid) {
                    toDelete = sbmlModel->removeReaction(index);
                    break;
                }
            }

            if (toDelete != nullptr) {
                delete toDelete;
                // no need to check anymore
                continue;
            }

            const libsbml::ListOfSpeciesReferences *modifiers = reaction->getListOfModifiers();
            for (uint j = 0; j < modifiers->size(); j++) {
                if (modifiers->get(j)->getSpecies() == sid) {
                    toDelete = sbmlModel->removeReaction(index);
                    break;
                }
            }

            if (toDelete != nullptr) {
                delete toDelete;
                // no need to check anymore
                continue;
            }

            // this reaction is not related to the deleted species
            index++;
        }

        removeVariable(sid);
        delete s;
        regenerateModel(forceRegenerate);
    }


    void RoadRunner::setBoundary(const std::string &sid, bool boundaryCondition, bool forceRegenerate) {
        using namespace libsbml;
        Model *sbmlModel = impl->document->getModel();
        Species *species = sbmlModel->getSpecies(sid);

        if (species == NULL) {
            throw std::invalid_argument(
                    "Roadrunner::setBoundarySpecies failed, no species with ID " + sid + " existed in the model");
        }

        rrLog(Logger::LOG_DEBUG) << "Setting boundary condition for species " << sid << "..." << std::endl;
        species->setBoundaryCondition(boundaryCondition);

        regenerateModel(forceRegenerate);
    }

    void
    RoadRunner::setHasOnlySubstanceUnits(const std::string &sid, bool hasOnlySubstanceUnits, bool forceRegenerate) {
        using namespace libsbml;
        Model *sbmlModel = impl->document->getModel();
        Species *species = sbmlModel->getSpecies(sid);

        if (species == NULL) {
            throw std::invalid_argument(
                    "Roadrunner::setHasOnlySubstanceUnits failed, no species with ID " + sid + " existed in the model");
        }

        rrLog(Logger::LOG_DEBUG) << "Setting hasOnlySubstanceUnits attribute for species " << sid << "..." << std::endl;
        species->setHasOnlySubstanceUnits(hasOnlySubstanceUnits);

        regenerateModel(forceRegenerate);
    }

    bool RoadRunner::getHasOnlySubstanceUnits(const std::string &sid) {
        using namespace libsbml;
        Model *sbmlModel = impl->document->getModel();
        Species *species = sbmlModel->getSpecies(sid);

        if (species == NULL) {
            throw std::invalid_argument(
                    "Roadrunner::getHasOnlySubstanceUnits failed, no species with ID " + sid + " existed in the model");
        }

        return species->getHasOnlySubstanceUnits();
    }


    void RoadRunner::setInitAmount(const std::string &sid, double initAmount, bool forceRegenerate) {
        using namespace libsbml;
        Model *sbmlModel = impl->document->getModel();
        Species *species = sbmlModel->getSpecies(sid);

        if (species == NULL) {
            throw std::invalid_argument(
                    "Roadrunner::setInitAmount failed, no species with ID " + sid + " existed in the model");
        }

        rrLog(Logger::LOG_DEBUG) << "Setting initial amount for species " << sid << "..." << std::endl;
        if (species->isSetInitialAmount()) {
            species->unsetInitialAmount();
        }

        species->setInitialAmount(initAmount);

        regenerateModel(forceRegenerate);

        // recover the updated init amount
        int index = impl->model->getFloatingSpeciesIndex(sid);
        if (index >= 0 && index < impl->model->getNumIndFloatingSpecies()) {
            impl->model->setFloatingSpeciesInitAmounts(1, &index, &initAmount);
        }
    }

    void RoadRunner::setInitConcentration(const std::string &sid, double initConcentration, bool forceRegenerate) {
        using namespace libsbml;
        Model *sbmlModel = impl->document->getModel();
        Species *species = sbmlModel->getSpecies(sid);

        if (species == NULL) {
            throw std::invalid_argument(
                    "Roadrunner::setInitConcentration failed, no species with ID " + sid + " existed in the model");
        }

        rrLog(Logger::LOG_DEBUG) << "Setting initial concentration for species " << sid << "..." << std::endl;
        if (species->isSetInitialConcentration()) {
            species->unsetInitialConcentration();
        }
        species->setInitialConcentration(initConcentration);

        regenerateModel(forceRegenerate);

        // recover the updated init concentration
        int index = impl->model->getFloatingSpeciesIndex(sid);

        if (index >= 0 && index < impl->model->getNumIndFloatingSpecies()) {

            int compartment = impl->model->getCompartmentIndex(species->getCompartment());
            double compartmentSize = 1;
            impl->model->getCompartmentVolumes(1, &compartment, &compartmentSize);

            double initValue = initConcentration * compartmentSize;
            impl->model->setFloatingSpeciesInitAmounts(1, &index, &initValue);
        }
    }

    void RoadRunner::setConstant(const std::string &sid, bool constant, bool forceRegenerate) {
        using namespace libsbml;
        Model *sbmlModel = impl->document->getModel();
        Species *species = sbmlModel->getSpecies(sid);
        Parameter *parameter = sbmlModel->getParameter(sid);
        Compartment *compartment = sbmlModel->getCompartment(sid);

        if (species != NULL) {
            rrLog(Logger::LOG_DEBUG) << "Setting constant attribute for species " << sid << "..." << std::endl;
            species->setConstant(constant);
        } else if (parameter != NULL) {
            rrLog(Logger::LOG_DEBUG) << "Setting constant attribute for parameter " << sid << "..." << std::endl;
            parameter->setConstant(constant);
        } else if (compartment != NULL) {
            rrLog(Logger::LOG_DEBUG) << "Setting constant attribute for compartment " << sid << "..." << std::endl;
            compartment->setConstant(constant);
        } else {
            throw std::invalid_argument(
                    "Roadrunner::setConstant failed, no species/ parameter/ compartment with ID " + sid +
                    " existed in the model");
        }

        regenerateModel(forceRegenerate);
    }


    void RoadRunner::addReaction(const std::string &sbmlRep, bool forceRegenerate) {

        rrLog(Logger::LOG_DEBUG) << "Adding new reaction ..." << std::endl;
        libsbml::Reaction *newReaction = impl->document->getModel()->createReaction();
        //libsbml's use of the expat parser requires you to declare the XML at the beginning of any string.
        string input = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" + sbmlRep;
        libsbml::XMLInputStream stream(input.c_str(), false);
        newReaction->read(stream);
        // TODO: ERROR HANDLING

        regenerateModel(forceRegenerate);
    }

    void RoadRunner::addReaction(const std::string &rid, std::vector<std::string> reactants,
                                 std::vector<std::string> products, const std::string &kineticLaw,
                                 bool forceRegenerate) {

        using namespace libsbml;
        Model *sbmlModel = impl->document->getModel();

        checkID("addReaction", rid);

        rrLog(Logger::LOG_DEBUG) << "Adding reaction " << rid << "..." << std::endl;
        Reaction *newReaction = sbmlModel->createReaction();

        int ret = newReaction->setId(rid);
        if (ret != libsbml::LIBSBML_OPERATION_SUCCESS) {
            newReaction->removeFromParentAndDelete();
            throw std::invalid_argument("Roadrunner::addReaction failed: invalid reaction id '" + rid + "'.");
        }

        // no need to check reactants.size() + products.size() > 0
        for (int i = 0; i < reactants.size(); i++) {
            double stoichiometry = 1;
            char *sid = NULL;
            parseSpecies(reactants[i], &stoichiometry, &sid);
            if (sid == NULL) {
                throw std::invalid_argument(
                        "Roadrunner::addReaction failed, invalid stoichiomety value and species ID " + reactants[0]);
            }
            if (forceRegenerate && sbmlModel->getSpecies(sid) == NULL) {
                throw std::invalid_argument(
                        "Roadrunner::addReaction failed, no species with ID " + reactants[i] + " existed in the model");
            }
            SpeciesReference sr(newReaction->getSBMLNamespaces());
            sr.setStoichiometry(stoichiometry);
            sr.setConstant(true);
            sr.setSpecies(sid);
            newReaction->addReactant(&sr);
        }

        for (int i = 0; i < products.size(); i++) {
            char *sid = 0;
            double stoichiometry = 1;
            parseSpecies(products[i], &stoichiometry, &sid);
            if (sid == NULL) {
                throw std::invalid_argument(
                        "Roadrunner::addReaction failed, invalid stoichiomety value and species ID " + products[i]);
            }
            if (forceRegenerate && sbmlModel->getSpecies(sid) == NULL) {
                throw std::invalid_argument(
                        "Roadrunner::addReaction failed, no species with ID " + products[i] + " existed in the model");
            }
            SpeciesReference sr(newReaction->getSBMLNamespaces());
            sr.setStoichiometry(stoichiometry);
            sr.setConstant(true);
            sr.setSpecies(sid);
            newReaction->addProduct(&sr);
        }

        // illegal formulas will be caught during the regeneration
        ASTNode *mathRoot = SBML_parseL3Formula(kineticLaw.c_str());
        KineticLaw *kLaw = newReaction->createKineticLaw();
        kLaw->setMath(mathRoot);
        delete mathRoot;

        std::vector<std::string> kLawSpeciesIds;
        getSpeciesIdsFromAST(kLaw->getMath(), kLawSpeciesIds);
        for (std::string s: kLawSpeciesIds) {
            if (std::find(products.begin(), products.end(), s) == products.end()
                && std::find(reactants.begin(), reactants.end(), s) == reactants.end()) {
                newReaction->addModifier(sbmlModel->getSpecies(s));
            }
        }


        // set required attributes to default
        newReaction->setReversible(false);
        newReaction->setFast(false);

        regenerateModel(forceRegenerate);
    }


    void RoadRunner::removeReaction(const std::string &rid, bool deleteUnusedParameters, bool forceRegenerate) {
        using namespace libsbml;
        libsbml::Reaction *toDelete = impl->document->getModel()->removeReaction(rid);
        if (toDelete == NULL) {
            throw std::invalid_argument(
                    "Roadrunner::removeReaction failed, no reaction with ID " + rid + " existed in the model");
        }
        rrLog(Logger::LOG_DEBUG) << "Removing reaction " << rid << "..." << std::endl;
        if (deleteUnusedParameters) {
            std::set<std::string> toCheck;
            getAllVariables(toDelete->getKineticLaw()->getMath(), toCheck);
            for (uint j = 0; j < toDelete->getNumReactants(); j++) {
                SpeciesReference *reactant = toDelete->getReactant(j);
                if (reactant->isSetId()) {
                    toCheck.insert(reactant->getId());
                }
                // only Level 2 support StoichiometryMath
                if (reactant->getStoichiometryMath() != NULL) {
                    getAllVariables(reactant->getStoichiometryMath()->getMath(), toCheck);
                }
            }

            for (uint j = 0; j < toDelete->getNumProducts(); j++) {
                SpeciesReference *product = toDelete->getProduct(j);
                if (product->isSetId()) {
                    toCheck.insert(product->getId());
                }
                if (product->getStoichiometryMath() != NULL) {
                    getAllVariables(product->getStoichiometryMath()->getMath(), toCheck);
                }
            }
            for (std::string sid: toCheck) {
                if (impl->document->getModel()->getParameter(sid) != NULL) {
                    if (!isParameterUsed(sid)) {
                        removeParameter(sid, false);
                    }
                }
            }
        }
        delete toDelete;
        regenerateModel(forceRegenerate);
    }

    bool RoadRunner::isParameterUsed(const std::string &sid) {
        using namespace libsbml;
        Model *sbmlModel = impl->document->getModel();
        int num = sbmlModel->getNumReactions();
        //Check if this parameter occurs in a reaction
        for (uint i = 0; i < num; i++) {
            Reaction *reaction = sbmlModel->getReaction(i);
            if (impl->document->getLevel() == 2) {
                // only Level 2 support StoichiometryMath
                const ListOfSpeciesReferences *reactants = reaction->getListOfReactants();
                for (uint j = 0; j < reactants->size(); j++) {
                    // TODO: better way to cast?
                    SpeciesReference *reactant = (SpeciesReference *) reactants->get(j);
                    if (reactant->getStoichiometryMath() != NULL) {
                        if (hasVariable(reactant->getStoichiometryMath()->getMath(), sid)) {
                            return true;
                        }
                    }

                }

                const libsbml::ListOfSpeciesReferences *products = reaction->getListOfProducts();
                for (uint j = 0; j < products->size(); j++) {
                    SpeciesReference *product = (SpeciesReference *) products->get(j);
                    if (product->getStoichiometryMath() != NULL) {
                        if (hasVariable(product->getStoichiometryMath()->getMath(), sid)) {
                            return true;
                        }
                    }

                }
            }

            // TODO: check if getMath work with level 1
            if (reaction->getKineticLaw() && hasVariable(reaction->getKineticLaw()->getMath(), sid)) {
                return true;
            }
        }
        // Check if this parameter occurs in an assignment rule or rate rule
        for (uint i = 0; i < sbmlModel->getNumRules(); i++) {
            Rule *rule = sbmlModel->getRule(i);
            if (rule->getId() == sid) {
                return true;
            }
            if (hasVariable(rule->getMath(), sid)) {
                return true;
            }
        }
        // Check if this parameter occurs in an initial assigment
        for (uint i = 0; i < sbmlModel->getNumInitialAssignments(); i++) {
            InitialAssignment *initialAssignment = sbmlModel->getInitialAssignment(i);
            if (initialAssignment->getId() == sid) {
                return true;
            }
            if (hasVariable(initialAssignment->getMath(), sid)) {
                return true;
            }
        }
        return false;
    }


    void RoadRunner::setReversible(const std::string &rid, bool reversible, bool forceRegenerate) {

        using namespace libsbml;
        Model *sbmlModel = impl->document->getModel();
        Reaction *reaction = sbmlModel->getReaction(rid);

        if (reaction == NULL) {
            throw std::invalid_argument(
                    "Roadrunner::setReversible failed, no reaction with ID " + rid + " existed in the model");
        }

        rrLog(Logger::LOG_DEBUG) << "Setting reversible attribute for reaction " << rid << "..." << std::endl;
        reaction->setReversible(reversible);

        regenerateModel(forceRegenerate);
    }


    void RoadRunner::setKineticLaw(const std::string &rid, const std::string &kineticLaw, bool forceRegenerate) {

        using namespace libsbml;
        Model *sbmlModel = impl->document->getModel();
        Reaction *reaction = sbmlModel->getReaction(rid);

        if (reaction == NULL) {
            throw std::invalid_argument(
                    "Roadrunner::setKineticLaw failed, no reaction with ID " + rid + " existed in the model");
        }

        rrLog(Logger::LOG_DEBUG) << "Setting kinetic law for reaction " << rid << "..." << std::endl;


        KineticLaw *law = reaction->getKineticLaw();
        if (law == NULL) {
            law = reaction->createKineticLaw();
        }

        // need to do this instead of setFormula() for L3 parsing
        ASTNode *mathRoot = SBML_parseL3Formula(kineticLaw.c_str());
        law->setMath(mathRoot);
        delete mathRoot;

        std::vector<std::string> kLawSpeciesIds;
        getSpeciesIdsFromAST(law->getMath(), kLawSpeciesIds);
        for (std::string s: kLawSpeciesIds) {
            if (reaction->getProduct(s) != NULL) {
                continue;
            }

            if (reaction->getReactant(s) != NULL) {
                continue;
            }
            reaction->addModifier(sbmlModel->getSpecies(s));
        }

        regenerateModel(forceRegenerate);
    }


    std::string RoadRunner::getKineticLaw(const std::string &rid) {

        using namespace libsbml;
        Model *sbmlModel = impl->document->getModel();
        Reaction *reaction = sbmlModel->getReaction(rid);

        if (reaction == NULL) {
            throw std::invalid_argument(
                    "Roadrunner::getKineticLaw failed, no reaction with ID " + rid + " exists in the model");
        }

        rrLog(Logger::LOG_DEBUG) << "Getting kinetic law for reaction " << rid << std::endl;


        KineticLaw *law = reaction->getKineticLaw();
        if (law == NULL) {
            return "";
        }

        const ASTNode *math = law->getMath();
        if (math == NULL) {
            return "";
        }

        char *mathchr = SBML_formulaToL3String(math);
        std::string ret(mathchr);
        free(mathchr);
        return ret;
    }


    void RoadRunner::addParameter(const std::string &pid, double value, bool forceRegenerate) {
        checkID("addParameter", pid);

        rrLog(Logger::LOG_DEBUG) << "Adding parameter " << pid << " with value " << value << std::endl;
        libsbml::Parameter *newParameter = impl->document->getModel()->createParameter();


        int ret = newParameter->setId(pid);
        if (ret != libsbml::LIBSBML_OPERATION_SUCCESS) {
            newParameter->removeFromParentAndDelete();
            throw std::invalid_argument("Roadrunner::addParameter failed: invalid parameter id '" + pid + "'.");
        }
        newParameter->setValue(value);
        // set required attributes to default
        newParameter->setConstant(false);

        regenerateModel(forceRegenerate);
    }

    void RoadRunner::removeParameter(const std::string &pid, bool forceRegenerate) {
        libsbml::Parameter *toDelete = impl->document->getModel()->removeParameter(pid);
        if (toDelete == NULL) {
            throw std::invalid_argument(
                    "Roadrunner::removeParameter failed, no parameter with ID " + pid + " existed in the model");
        }
        rrLog(Logger::LOG_DEBUG) << "Removing parameter " << pid << "..." << std::endl;
        removeVariable(pid);
        delete toDelete;

        regenerateModel(forceRegenerate);
    }


    void RoadRunner::addCompartment(const std::string &cid, double initVolume, bool forceRegenerate) {
        checkID("addCompartment", cid);
        rrLog(Logger::LOG_DEBUG) << "Adding compartment " << cid << " with initial volume " << initVolume << std::endl;
        libsbml::Compartment *newCompartment = impl->document->getModel()->createCompartment();

        int ret = newCompartment->setId(cid);
        if (ret != libsbml::LIBSBML_OPERATION_SUCCESS) {
            newCompartment->removeFromParentAndDelete();
            throw std::invalid_argument("Roadrunner::addCompartment failed: invalid compartment id '" + cid + "'.");
        }
        newCompartment->setVolume(initVolume);
        // set required attributes to default
        newCompartment->setConstant(false);

        regenerateModel(forceRegenerate);
    }


    void RoadRunner::removeCompartment(const std::string &cid, bool forceRegenerate) {

        libsbml::Model *model = impl->document->getModel();
        libsbml::Compartment *toDelete = model->removeCompartment(cid);

        if (toDelete == NULL) {
            throw std::invalid_argument(
                    "Roadrunner::removeCompartment failed, no compartment with ID " + cid + " existed in the model");
        }
        rrLog(Logger::LOG_DEBUG) << "Removing compartment " << cid << "..." << std::endl;


        // remove all species in the compartment
        int index = 0;
        int numSpecies = model->getNumSpecies();
        for (int i = 0; i < numSpecies; i++) {
            if (model->getSpecies(index)->getCompartment() == cid) {
                std::string temp = model->getSpecies(index)->getId();
                removeSpecies(model->getSpecies(index)->getId(), false);
            } else {
                index++;
            }
        }

        removeVariable(cid);
        delete toDelete;
        regenerateModel(forceRegenerate);
    }

    void checkAddRule(const std::string &vid, libsbml::Model *sbmlModel) {
        libsbml::SBase *sbase = const_cast<libsbml::SBase*>(RoadRunner::getElementWithMathematicalMeaning(sbmlModel, vid));
        if (sbase == NULL) {
            throw std::invalid_argument(
                    "Unable to add rule because no variable with ID " + vid + " was found in the model.");
        }
        switch (sbase->getTypeCode()) {
            case libsbml::SBML_COMPARTMENT: {
                libsbml::Compartment *comp = static_cast<libsbml::Compartment *>(sbase);
                comp->setConstant(false);
            }
                break;
            case libsbml::SBML_SPECIES: {
                libsbml::Species *species = static_cast<libsbml::Species *>(sbase);
                species->setConstant(false);
                if (species->isSetBoundaryCondition() && species->getBoundaryCondition() == false) {
                    throw std::invalid_argument("Unable to add rule because the species with ID " + vid +
                                                " is not a boundary species, and therefore may only change due to reactions.  To change this, the boundary species flag can be set with the function 'setBoundary(id, value)' or when added with 'addSpecies'.");
                }
                species->setBoundaryCondition(true);
            }
                break;
            case libsbml::SBML_PARAMETER: {
                libsbml::Parameter *param = static_cast<libsbml::Parameter *>(sbase);
                param->setConstant(false);
            }
                break;
            case libsbml::SBML_SPECIES_REFERENCE: {
                libsbml::SpeciesReference *sr = static_cast<libsbml::SpeciesReference *>(sbase);
                sr->setConstant(false);
            }
                break;
            default: {
                const char *code = libsbml::SBMLTypeCode_toString(sbase->getTypeCode(), "core");
                throw std::invalid_argument("Unable to add rule because the variable with ID " + vid + " is a " + code +
                                            ", which does not have mathematical meaning.");
            }
        }

        if (sbmlModel->getRule(vid) != NULL) {
            throw std::invalid_argument(
                    "Unable to add rule because the variable " + vid + " already has a rule existing in the model.");
        }
    }


    void RoadRunner::addAssignmentRule(const std::string &vid, const std::string &formula, bool forceRegenerate) {
        using namespace libsbml;
        Model *sbmlModel = impl->document->getModel();
        checkAddRule(vid, sbmlModel);


        rrLog(Logger::LOG_DEBUG) << "Adding assignment rule for" << vid << "..." << std::endl;
        AssignmentRule *newRule = sbmlModel->createAssignmentRule();

        // potential errors with these two inputs will be detected during regeneration and ignored

        newRule->setVariable(vid);
        // need to do this instead of setFormula() for L3 parsing
        ASTNode *mathRoot = SBML_parseL3Formula(formula.c_str());
        newRule->setMath(mathRoot);
        delete mathRoot;

        regenerateModel(forceRegenerate, true);
    }

    void RoadRunner::addRateRule(const std::string &vid, const std::string &formula, bool forceRegenerate) {
        using namespace libsbml;
        Model *sbmlModel = impl->document->getModel();

        checkAddRule(vid, sbmlModel);

        rrLog(Logger::LOG_DEBUG) << "Adding rate rule for" << vid << "..." << std::endl;
        RateRule *newRule = sbmlModel->createRateRule();

        // potential errors with these two inputs will be detected during regeneration

        newRule->setVariable(vid);
        // need to do this instead of setFormula() for L3 parsing
        ASTNode *mathRoot = SBML_parseL3Formula(formula.c_str());
        newRule->setMath(mathRoot);
        delete mathRoot;

        regenerateModel(forceRegenerate);
    }

// TODO: update C wrappers
    void RoadRunner::removeRules(const std::string &vid, bool useInitialValueAsCurrent, bool forceRegenerate) {
        using namespace libsbml;
        Model *sbmlModel = impl->document->getModel();

        bool assignment = false;
        //check if the rule to remove is an assignment rule
        if (sbmlModel->getAssignmentRule(vid) != NULL) {
            assignment = true;
        }
        Rule *toDelete = sbmlModel->removeRule(vid);
        if (toDelete == NULL) {
            throw std::invalid_argument(
                    "Roadrunner::removeRules failed, no rules for variable " + vid + " existed in the model");
        }
        rrLog(Logger::LOG_DEBUG) << "Removing rule for variable" << vid << "..." << std::endl;
        delete toDelete;

        regenerateModel(forceRegenerate);
        // grab the initial initial value from sbml model
        if (assignment) {
            int index = impl->model->getFloatingSpeciesIndex(vid);
            if (index >= 0 && index < impl->model->getNumIndFloatingSpecies()) {

                double initValue = 0;
                if (sbmlModel->getSpecies(vid)->isSetInitialAmount()) {
                    initValue = sbmlModel->getSpecies(vid)->getInitialAmount();
                } else if (sbmlModel->getSpecies(vid)->isSetInitialConcentration()) {
                    double initConcentration = sbmlModel->getSpecies(vid)->getInitialConcentration();
                    int compartment = impl->model->getCompartmentIndex(sbmlModel->getSpecies(vid)->getCompartment());
                    double compartmentSize = 1;
                    impl->model->getCompartmentVolumes(1, &compartment, &compartmentSize);

                    initValue = initConcentration * compartmentSize;
                }

                impl->model->setFloatingSpeciesInitAmounts(1, &index, &initValue);
                if (useInitialValueAsCurrent) {
                    impl->model->setFloatingSpeciesAmounts(1, &index, &initValue);
                }
            }

            index = impl->model->getBoundarySpeciesIndex(vid);
            if (index >= 0 && index < impl->model->getNumBoundarySpecies()) {

                double initValue = 0;
                if (sbmlModel->getSpecies(vid)->isSetInitialAmount()) {
                    initValue = sbmlModel->getSpecies(vid)->getInitialAmount();
                } else if (sbmlModel->getSpecies(vid)->isSetInitialConcentration()) {
                    double initConcentration = sbmlModel->getSpecies(vid)->getInitialConcentration();
                    int compartment = impl->model->getCompartmentIndex(sbmlModel->getSpecies(vid)->getCompartment());
                    double compartmentSize = 1;
                    impl->model->getCompartmentVolumes(1, &compartment, &compartmentSize);

                    initValue = initConcentration * compartmentSize;
                }

                impl->model->setBoundarySpeciesInitAmounts(1, &index, &initValue);
                if (useInitialValueAsCurrent) {
                    impl->model->setBoundarySpeciesAmounts(1, &index, &initValue);
                }
            }

            index = impl->model->getCompartmentIndex(vid);
            if (index >= 0 && index < impl->model->getNumCompartments()) {
                double initValue = 0;
                if (sbmlModel->getCompartment(vid)->isSetSize()) {
                    initValue = sbmlModel->getCompartment(vid)->getSize();
                }
                impl->model->setCompartmentInitVolumes(1, &index, &initValue);
                if (useInitialValueAsCurrent) {
                    impl->model->setCompartmentVolumes(1, &index, &initValue);
                }
            }

            index = impl->model->getGlobalParameterIndex(vid);
            if (index >= 0 && index < impl->model->getNumGlobalParameters()) {
                double initValue = 0;
                if (sbmlModel->getParameter(vid)->isSetValue()) {
                    initValue = sbmlModel->getParameter(vid)->getValue();
                }
                impl->model->setGlobalParameterInitValues(1, &index, &initValue);
                if (useInitialValueAsCurrent) {
                    impl->model->setGlobalParameterValues(1, &index, &initValue);
                }
            }
        } else if (useInitialValueAsCurrent) {
            // recover the initial value for rate rules
            int index = impl->model->getFloatingSpeciesIndex(vid);
            if (index >= 0 && index < impl->model->getNumIndFloatingSpecies()) {
                double initValue = 0;
                impl->model->getFloatingSpeciesInitAmounts(1, &index, &initValue);
                impl->model->setFloatingSpeciesAmounts(1, &index, &initValue);

            }

            if (index >= 0 && index < impl->model->getNumBoundarySpecies()) {
                double initValue = 0;
                impl->model->getBoundarySpeciesInitAmounts(1, &index, &initValue);
                impl->model->setBoundarySpeciesAmounts(1, &index, &initValue);

            }

            index = impl->model->getCompartmentIndex(vid);
            if (index >= 0 && index < impl->model->getNumCompartments()) {
                double initValue = 0;
                impl->model->getCompartmentInitVolumes(1, &index, &initValue);
                impl->model->setCompartmentVolumes(1, &index, &initValue);
            }

            index = impl->model->getGlobalParameterIndex(vid);
            if (index >= 0 && index < impl->model->getNumGlobalParameters()) {
                double initValue = 0;
                impl->model->getGlobalParameterInitValues(1, &index, &initValue);
                impl->model->setGlobalParameterValues(1, &index, &initValue);
            }
        }
    }

    void RoadRunner::addInitialAssignment(const std::string &vid, const std::string &formula, bool forceRegenerate) {
        using namespace libsbml;
        Model *sbmlModel = impl->document->getModel();

        if (sbmlModel->getCompartment(vid) == NULL && sbmlModel->getSpecies(vid) == NULL &&
            sbmlModel->getParameter(vid) == NULL && sbmlModel->getSpeciesReference(vid) == NULL) {
            throw std::invalid_argument(
                    "Roadrunner::addInitialAssignment failed, no symbol with ID " + vid + " existed in the model");
        }

        if (sbmlModel->getAssignmentRule(vid) != NULL) {
            throw std::invalid_argument("Roadrunner::addInitialAssignment failed, symbol " + vid +
                                        " already has an assignment rule existing in the model");
        }

        if (sbmlModel->getInitialAssignment(vid) != NULL) {
            throw std::invalid_argument("Roadrunner::addInitialAssignment failed, symbol " + vid +
                                        " already has an initial assignment existing in the model");
        }

        rrLog(Logger::LOG_DEBUG) << "Adding initial assignment for" << vid << "..." << std::endl;
        InitialAssignment *newAssignment = sbmlModel->createInitialAssignment();

        newAssignment->setSymbol(vid);
        ASTNode_t *math = libsbml::SBML_parseL3Formula(formula.c_str());
        if (math == NULL) {
            throw std::invalid_argument(
                    "Roadrunner::addInitialAssignment failed, an error occurred in parsing the formula");
        }
        newAssignment->setMath(math);

        delete math;

        regenerateModel(forceRegenerate, true);
    }

    void RoadRunner::removeInitialAssignment(const std::string &vid, bool forceRegenerate, bool errIfNotExist) {
        using namespace libsbml;
        Model *sbmlModel = impl->document->getModel();

        InitialAssignment *toDelete = sbmlModel->removeInitialAssignment(vid);
        if (toDelete == NULL) {
            if (errIfNotExist) {
                throw std::invalid_argument(
                    "Roadrunner::removeInitialAssignment failed, no initial assignment for symbol " + vid +
                    " existed in the model");
            }
            // Otherwise, we don't need to do anything.
            return;
        }
        rrLog(Logger::LOG_DEBUG) << "Removing initial assignment for variable" << vid << "..." << std::endl;
        delete toDelete;

        regenerateModel(forceRegenerate);
        reset(
            SelectionRecord::TIME |
            SelectionRecord::RATE |
            SelectionRecord::FLOATING |
            SelectionRecord::BOUNDARY |
            SelectionRecord::COMPARTMENT |
            SelectionRecord::GLOBAL_PARAMETER);
    }


    void RoadRunner::addEvent(const std::string &eid, bool useValuesFromTriggerTime, const std::string &trigger,
                              bool forceRegenerate) {
        using namespace libsbml;
        Model *sbmlModel = impl->document->getModel();
        checkID("addEvent", eid);

        rrLog(Logger::LOG_DEBUG) << "Adding event " << eid << "..." << std::endl;
        Event *newEvent = sbmlModel->createEvent();
        newEvent->setId(eid);
        newEvent->setUseValuesFromTriggerTime(useValuesFromTriggerTime);

        Trigger *newTrigger = newEvent->createTrigger();
        ASTNode_t *formula = libsbml::SBML_parseL3Formula(trigger.c_str());
        if (formula == NULL) {
            throw std::invalid_argument(
                    "Roadrunner::addEvent failed, an error occurred in parsing the trigger formula");
        }
        newTrigger->setMath(formula);
        delete formula;
        // set required attribute to default
        // TODO: should default be true or false?
        if (sbmlModel->getLevel() > 2) {
            newTrigger->setPersistent(false);
            newTrigger->setInitialValue(false);
        }

        // model regeneration will throw RuntimeError if the given formula is not valid
        regenerateModel(forceRegenerate);
    }


    void RoadRunner::addTrigger(const std::string &eid, const std::string &trigger, bool forceRegenerate) {
        using namespace libsbml;
        Event *event = impl->document->getModel()->getEvent(eid);

        if (event == NULL) {
            throw std::invalid_argument("Roadrunner::addTrigger failed, no event " + eid + " existed in the model");
        }

        Trigger *newTrigger = event->createTrigger();

        rrLog(Logger::LOG_DEBUG) << "Adding trigger for event " << eid << "..." << std::endl;
        ASTNode_t *formula = libsbml::SBML_parseL3Formula(trigger.c_str());
        if (formula == NULL) {
            throw std::invalid_argument(
                    "Roadrunner::addTrigger failed, an error occurred in parsing the trigger formula");
        }
        newTrigger->setMath(formula);
        delete formula;
        // set required attributes to default
        if (impl->document->getModel()->getLevel() > 3) {
            newTrigger->setPersistent(false);
            newTrigger->setInitialValue(false);
        }

        // model regeneration will throw RuntimeError if the given formula is not valid
        regenerateModel(forceRegenerate, true);
    }

    void RoadRunner::setPersistent(const std::string &eid, bool persistent, bool forceRegenerate) {
        using namespace libsbml;
        Event *event = impl->document->getModel()->getEvent(eid);

        if (impl->document->getLevel() < 3) {
            throw std::runtime_error(
                    "Roadrunner::setPersistent failed, current SBML level and version does not have peresistent attribute in the trigger");
        }

        if (event == NULL) {
            throw std::invalid_argument("Roadrunner::setPersistent failed, no event " + eid + " existed in the model");
        }

        Trigger *trigger = event->getTrigger();
        if (trigger == NULL) {
            throw std::invalid_argument(
                    "Roadrunner::setPersistent failed, given event " + eid + " does not have a trigger");
        }

        rrLog(Logger::LOG_DEBUG) << "Setting persistent for trigger of " << eid << "..." << std::endl;
        trigger->setPersistent(persistent);

        regenerateModel(forceRegenerate, true);
    }

    void RoadRunner::setTriggerInitialValue(const std::string &eid, bool initValue, bool forceRegenerate) {
        using namespace libsbml;
        Event *event = impl->document->getModel()->getEvent(eid);

        if (impl->document->getLevel() < 3) {
            throw std::runtime_error(
                    "Roadrunner::setTriggerInitialValue failed, current SBML level and version does not have initialValue attribute in the trigger");
        }

        if (event == NULL) {
            throw std::invalid_argument(
                    "Roadrunner::setTriggerInitialValue failed, no event " + eid + " existed in the model");
        }

        Trigger *trigger = event->getTrigger();
        if (trigger == NULL) {
            throw std::invalid_argument(
                    "Roadrunner::setTriggerInitialValue failed, given event " + eid + " does not have a trigger");
        }

        rrLog(Logger::LOG_DEBUG) << "Setting initial value for trigger of " << eid << "..." << std::endl;
        trigger->setInitialValue(initValue);

        regenerateModel(forceRegenerate, true);
    }


    void RoadRunner::addPriority(const std::string &eid, const std::string &priority, bool forceRegenerate) {
        using namespace libsbml;
        Event *event = impl->document->getModel()->getEvent(eid);

        if (event == NULL) {
            throw std::invalid_argument("Roadrunner::addPriority failed, no event " + eid + " existed in the model");
        }

        Priority *newPriority = event->createPriority();
        if (newPriority == NULL) {
            throw std::runtime_error(
                    "Roadrunner::addPriority failed, current SBML level and version does not support Priority in event");
        }
        rrLog(Logger::LOG_DEBUG) << "Adding priority for event " << eid << "..." << std::endl;
        ASTNode_t *formula = libsbml::SBML_parseL3Formula(priority.c_str());
        if (formula == NULL) {
            throw std::invalid_argument(
                    "Roadrunner::addPriority failed, an error occurred in parsing the priority formula");
        }
        newPriority->setMath(formula);
        delete formula;

        // model regeneration will throw RuntimeError if the given formula is not valid
        regenerateModel(forceRegenerate, true);
    }


    void RoadRunner::addDelay(const std::string &eid, const std::string &delay, bool forceRegenerate) {
        using namespace libsbml;
        Event *event = impl->document->getModel()->getEvent(eid);

        if (event == NULL) {
            throw std::invalid_argument("Roadrunner::addDelay failed, no event " + eid + " existed in the model");
        }

        rrLog(Logger::LOG_DEBUG) << "Adding delay for event " << eid << "..." << std::endl;
        Delay *newDelay = event->createDelay();
        ASTNode_t *formula = libsbml::SBML_parseL3Formula(delay.c_str());
        if (formula == NULL) {
            throw std::invalid_argument("Roadrunner::addDelay failed, an error occurred in parsing the delay formula");
        }
        newDelay->setMath(formula);
        delete formula;

        // model regeneration will throw RuntimeError if the given formula is not valid
        regenerateModel(forceRegenerate, true);

    }


    void RoadRunner::addEventAssignment(const std::string &eid, const std::string &vid, const std::string &formula,
                                        bool forceRegenerate) {
        using namespace libsbml;
        Model *sbmlModel = impl->document->getModel();
        Event *event = sbmlModel->getEvent(eid);

        if (event == NULL) {
            throw std::invalid_argument(
                    "Roadrunner::addEventAssignment failed, no event " + eid + " existed in the model");
        }

        if (sbmlModel->getCompartment(vid) == NULL && sbmlModel->getSpecies(vid) == NULL &&
            sbmlModel->getParameter(vid) == NULL && sbmlModel->getSpeciesReference(vid) == NULL) {
            throw std::invalid_argument(
                    "Roadrunner::addEventAssignment failed, no variable with ID " + vid + " existed in the model");
        }

        if (sbmlModel->getAssignmentRule(vid) != NULL) {
            throw std::invalid_argument("Roadrunner::addEventAssignment failed, variable " + vid +
                                        " already has a rule existing in the model");
        }


        if (event->getEventAssignment(vid) != NULL) {
            throw std::invalid_argument("Roadrunner::addEventAssignment failed, variable " + vid +
                                        " already has an assignment in the event " + eid);
        }


        rrLog(Logger::LOG_DEBUG) << "Adding event assignment for variable " << vid << " to event " << eid << "..."
                                 << std::endl;
        EventAssignment *assignment = event->createEventAssignment();
        assignment->setVariable(vid);

        ASTNode_t *math = libsbml::SBML_parseL3Formula(formula.c_str());
        if (math == NULL) {
            throw std::invalid_argument(
                    "Roadrunner::addEventAssignment failed, an error occurred in parsing the math formula");
        }
        assignment->setMath(math);
        delete math;

        // model regeneration will throw RuntimeError if the given formula is not valid
        regenerateModel(forceRegenerate, true);
    }


    void RoadRunner::removeEventAssignments(const std::string &eid, const std::string &vid, bool forceRegenerate) {
        libsbml::Event *event = impl->document->getModel()->getEvent(eid);
        if (event == NULL) {
            throw std::invalid_argument(
                    "Roadrunner::removeEventAssignment failed, no event with ID " + eid + " existed in the model");
        }

        libsbml::EventAssignment *toDelete = event->removeEventAssignment(vid);
        if (toDelete == NULL) {
            throw std::invalid_argument(
                    "Roadrunner::removeEventAssignment failed, no event assignment for variable " + vid +
                    " existed in the event " + eid);
        }

        rrLog(Logger::LOG_DEBUG) << "Removing event assignment for variable" << vid << " in event " << eid << "..."
                                 << std::endl;
        delete toDelete;

        regenerateModel(forceRegenerate, true);

    }


    void RoadRunner::removeEvent(const std::string &eid, bool forceRegenerate) {
        libsbml::Event *toDelete = impl->document->getModel()->removeEvent(eid);
        if (toDelete == NULL) {
            throw std::invalid_argument(
                    "Roadrunner::removeEvent failed, no event with ID " + eid + " existed in the model");
        }
        rrLog(Logger::LOG_DEBUG) << "Removing event " << eid << "..." << std::endl;
        delete toDelete;
        regenerateModel(forceRegenerate, true);
    }

    void RoadRunner::validateCurrentSBML() {
        // turn off unit validation for now
        // TODO: passing options as parameters
        char *documentSBML = impl->document->toSBML();
        std::string errors = validateSBML(std::string(documentSBML),
                                          VALIDATE_GENERAL | VALIDATE_IDENTIFIER | VALIDATE_MATHML |
                                          VALIDATE_OVERDETERMINED);
        free(documentSBML);
        if (!errors.empty()) {
            throw std::runtime_error(errors.c_str());
        }
    }


    void RoadRunner::checkID(const std::string &functionName, const std::string &sid) {
        if (impl->document->getModel()->getElementBySId(sid) != NULL) {
            throw std::invalid_argument(
                    "Roadrunner::" + functionName + " failed, identifier " + sid + " already existed in the model");
        }
    }


    void RoadRunner::regenerateModel(bool forceRegenerate, bool reset) {
        if (forceRegenerate) {
            rrLog(Logger::LOG_DEBUG) << "Regenerating model..." << std::endl;
            std::unordered_map<std::string, double> indTolerances;

//		bool toleranceVector = (impl->integrator->getType("absolute_tolerance") == Setting::DOUBLEVECTOR);
            Setting absTol;
            if (impl->integrator->hasValue("absolute_tolerance")) {
                absTol = impl->integrator->getValue("absolute_tolerance");
                Setting::TypeId tolType = absTol.type();
            }

            // if absolute_tolerance is a double vector
            if (auto v1 = absTol.get_if<std::vector<double>>()) {
                for (int i = 0; i < getNumberOfFloatingSpecies(); i++) {
                    indTolerances.emplace(getFloatingSpeciesIds()[i], (*v1)[i]);
                }
            }


            // regenerate the model
            impl->model.reset(ExecutableModelFactory::regenerateModel(
                    impl->model.get(),
                    impl->document.get(),
                    impl->loadOpt.modelGeneratorOpt));


            impl->syncAllSolversWithModel(impl->model.get());

            if (auto v1 = absTol.get_if<std::vector<double>>()) {
                //Force setIndividualTolerance to construct a std::vector of the correct size
                impl->integrator->setValue("absolute_tolerance", Setting(Config::CVODE_MIN_ABSOLUTE));
                for (const auto &p: indTolerances) {
                    auto ids = getFloatingSpeciesIds();
                    if (std::find(ids.begin(), ids.end(), p.first) != ids.end())
                        impl->integrator->setIndividualTolerance(p.first, p.second);
                }
            }

            // reset();
            if ((impl->loadOpt.loadFlags & LoadSBMLOptions::NO_DEFAULT_SELECTIONS) == 0) {
                createDefaultSelectionLists();
            }

            if (reset) {
                //we need to reset before we simulate so we can reapply events which might be T0-firing
                if (!impl->simulatedSinceReset) {
                    this->reset();
                }
            }
        }
    }


    void RoadRunner::parseSpecies(const std::string &species, double *stoichiometry, char **sid) {
        const char *input = species.c_str();

        double d = strtod(input, sid);
        if (*input != '\0' && *sid != input) {
            *stoichiometry = d;
        }

    }

    void RoadRunner::removeVariable(const std::string &sid) {
        using namespace libsbml;
        Model *sbmlModel = impl->document->getModel();

        rrLog(Logger::LOG_DEBUG) << "Removing reactions related to " << sid << "..." << std::endl;

        SBase *toDelete = nullptr;
        int index = 0;
        int num = sbmlModel->getNumReactions();
        for (uint i = 0; i < num; i++) {
            Reaction *reaction = sbmlModel->getReaction(index);

            if (impl->document->getLevel() == 2) {
                // only Level 2 support StoichiometryMath
                const ListOfSpeciesReferences *reactants = reaction->getListOfReactants();
                for (uint j = 0; j < reactants->size(); j++) {
                    // TODO: better way to cast?
                    SpeciesReference *reactant = (SpeciesReference *) reactants->get(j);
                    if (reactant->getStoichiometryMath() != NULL) {
                        if (hasVariable(reactant->getStoichiometryMath()->getMath(), sid)) {
                            toDelete = sbmlModel->removeReaction(index);
                            break;
                        }
                    }

                }

                if (toDelete != nullptr) {
                    delete toDelete;
                    // no need to check anymore
                    continue;
                }

                const libsbml::ListOfSpeciesReferences *products = reaction->getListOfProducts();
                for (uint j = 0; j < products->size(); j++) {
                    SpeciesReference *product = (SpeciesReference *) products->get(j);
                    if (product->getStoichiometryMath() != NULL) {
                        if (hasVariable(product->getStoichiometryMath()->getMath(), sid)) {
                            toDelete = sbmlModel->removeReaction(index);
                            break;
                        }
                    }

                }

                if (toDelete != nullptr) {
                    delete toDelete;
                    // no need to check anymore
                    continue;
                }

            }

            // TODO: check if getMath work with level 1
            if (reaction->getKineticLaw() && hasVariable(reaction->getKineticLaw()->getMath(), sid)) {
                sbmlModel->getReaction(i)->unsetKineticLaw();
            }

            // not remove this reaction
            index++;
        }

        rrLog(Logger::LOG_DEBUG) << "Removing rules related to " << sid << "..." << std::endl;
        // remove all rules that use this as variable
        toDelete = sbmlModel->removeRule(sid);
        while (toDelete != NULL) {
            delete toDelete;
            toDelete = sbmlModel->removeRule(sid);
        }

        // check for other rules that use this variable in math formula
        index = 0;
        num = sbmlModel->getNumRules();
        for (uint i = 0; i < num; i++) {
            // TODO: check if getMath work with level 1
            if (hasVariable(sbmlModel->getRule(index)->getMath(), sid)) {
                toDelete = sbmlModel->removeRule(index);
                delete toDelete;
            } else {
                index++;
            }
        }

        rrLog(Logger::LOG_DEBUG) << "Removing function definitions related to " << sid << "..." << std::endl;
        index = 0;
        num = sbmlModel->getNumFunctionDefinitions();
        for (uint i = 0; i < num; i++) {
            if (hasVariable(sbmlModel->getFunctionDefinition(index)->getMath(), sid)) {
                toDelete = sbmlModel->removeFunctionDefinition(index);
                delete toDelete;
            } else {
                index++;
            }
        }

        rrLog(Logger::LOG_DEBUG) << "Removing constraints related to " << sid << "..." << std::endl;
        index = 0;
        num = sbmlModel->getNumConstraints();
        for (uint i = 0; i < num; i++) {
            if (hasVariable(sbmlModel->getConstraint(index)->getMath(), sid)) {
                toDelete = sbmlModel->removeConstraint(index);
                delete toDelete;
            } else {
                index++;
            }
        }

        rrLog(Logger::LOG_DEBUG) << "Removing initial assignments related to " << sid << "..." << std::endl;
        // fisrt remove initial assignment that use this variable as symbol
        toDelete = sbmlModel->removeInitialAssignment(sid);
        while (toDelete != NULL) {
            delete toDelete;
            toDelete = sbmlModel->removeInitialAssignment(sid);
        }

        // also other assignments using this variable in the math formula
        index = 0;
        num = sbmlModel->getNumInitialAssignments();
        for (uint i = 0; i < num; i++) {
            if (hasVariable(sbmlModel->getInitialAssignment(index)->getMath(), sid)) {
                toDelete = sbmlModel->removeInitialAssignment(index);
                delete toDelete;
            } else {
                index++;
            }
        }


        rrLog(Logger::LOG_DEBUG) << "Removing event elements related to " << sid << "..." << std::endl;
        index = 0;
        num = sbmlModel->getNumEvents();
        for (uint i = 0; i < num; i++) {
            Event *event = sbmlModel->getListOfEvents()->get(index);
            // check for trigger
            if (hasVariable(event->getTrigger()->getMath(), sid)) {
                // LLVMModelDataSymbol require trigger for event, so we have to delete the event
                toDelete = sbmlModel->removeEvent(index);
                delete toDelete;
                // continue to the next event
                continue;
            }

            // check for priority
            if (event->getPriority()) {
                if (hasVariable(event->getPriority()->getMath(), sid)) {
                    event->unsetPriority();
                }
            }

            // check for delay
            if (event->getDelay()) {
                if (hasVariable(event->getDelay()->getMath(), sid)) {
                    event->unsetDelay();
                }
            }


            // check for event assignment
            toDelete = event->removeEventAssignment(sid);
            while (toDelete != NULL) {
                delete toDelete;
                toDelete = event->removeEventAssignment(sid);
            }
            // look for other assignments that have a math formula using this variable
            int assignmentIndex = 0;
            int numAssign = event->getNumEventAssignments();
            for (uint j = 0; j < numAssign; j++) {
                EventAssignment *assignment = event->getListOfEventAssignments()->get(assignmentIndex);
                if (hasVariable(assignment->getMath(), sid)) {
                    toDelete = event->removeEventAssignment(assignmentIndex);
                    delete toDelete;
                } else {
                    assignmentIndex++;
                }

            }
            // not remove this event
            index++;
        }
    }

    void RoadRunner::getAllVariables(const libsbml::ASTNode *node, std::set<std::string> &ids) {
        if (node == NULL) return;
        if (!node->isOperator() && !node->isNumber()) {
            ids.insert(std::string(node->getName()));
        }
        for (uint i = 0; i < node->getNumChildren(); i++) {
            getAllVariables(node->getChild(i), ids);
        }
    }

    bool RoadRunner::hasVariable(const libsbml::ASTNode *node, const std::string &sid) {
        // DFS
        // TODO: faster wrappers to iterate all childeren node?
        if (node == NULL) return false;
        const char *temp = node->getName();
        if (!node->isOperator() && !node->isNumber() && sid.compare(node->getName()) == 0) {
            return true;
        }
        for (uint i = 0; i < node->getNumChildren(); i++) {
            if (hasVariable(node->getChild(i), sid)) {
                return true;
            }
        }
        return false;
    }


    void RoadRunner::getSpeciesIdsFromAST(const libsbml::ASTNode *node, std::vector<std::string> &species) {
        libsbml::ListOfSpecies *sbmlSpecies = impl->document->getModel()->getListOfSpecies();
        std::vector<std::string> speciesNames;
        for (int i = 0; i < sbmlSpecies->size(); i++) {
            speciesNames.push_back(sbmlSpecies->get(i)->getId());
        }
        getSpeciesIdsFromAST(node, species, speciesNames);
    }

    void RoadRunner::getSpeciesIdsFromAST(const libsbml::ASTNode *node, std::vector<std::string> &species,
                                          std::vector<std::string> &instanceSpeciesNames) {
        if (node == NULL) return;
        if (!node->isOperator() && !node->isNumber()) {
            if (std::find(instanceSpeciesNames.begin(), instanceSpeciesNames.end(), node->getName()) !=
                instanceSpeciesNames.end()) {
                species.push_back(node->getName());
            }
        }
        for (uint i = 0; i < node->getNumChildren(); i++) {
            getSpeciesIdsFromAST(node->getChild(i), species, instanceSpeciesNames);
        }
    }

    void RoadRunner::setSeed(long int seed, bool resetModel) {
        Config::setValue(Config::RANDOM_SEED, seed);
        if (resetModel) {
            regenerateModel(true);
            reset(SelectionRecord::TIME |
            SelectionRecord::RATE |
            SelectionRecord::FLOATING |
            SelectionRecord::BOUNDARY |
            SelectionRecord::COMPARTMENT |
            SelectionRecord::GLOBAL_PARAMETER);
        }
        else {
            impl->model->setRandomSeed(seed);
            for (auto integrator : impl->integrators) {
                if (integrator->getName() == "gillespie")
                    integrator->setValue("seed", Setting(seed));
            }
        }
    }

    int64_t RoadRunner::getSeed(const std::string &integratorName) {
        if (integratorName.empty()) {
            return impl->model->getRandomSeed();
        }
        else if (integratorName == "gillespie") {
            for (auto integrator : impl->integrators) {
                if (integrator->getName() == integratorName) {
                    return integrator->getValue("seed").getAs<int64_t>();
                }
            }
        }
        throw std::invalid_argument(integratorName + " is not set as the current integrator.");
    }

    void RoadRunner::resetSeed() {
        if (Config::getValue(Config::RANDOM_SEED).getAs<long int>() != -1)
            setSeed(-1, false);
        else {
            for (auto integrator : impl->integrators) {
                if (integrator->getName() == "gillespie")
                    integrator->setValue("seed", Setting(-1));
            }
        }
    }

    void writeDoubleVectorListToStream(std::ostream &out, const DoubleVectorList &results) {
        for (const std::vector<double> &row: results) {
            out << row[0];
            for (int i = 1; i < row.size(); i++) {
                out << "," << row[i];
            }
            out << "\n";
        }
        // clear buffer
        out.flush();
    }

    void writeDoubleMatrixToStream(std::ostream &out, const ls::DoubleMatrix &results, int nrows) {
        for (int row = 0; row < nrows; row++) {
            const double *prow = results[row];
            out << prow[0];
            for (int col = 1; col < results.numCols(); col++) {
                out << "," << prow[col];
            }
            out << "\n";
        }
        // clear buffer
        out.flush();
    }

} //namespace

#if defined(_WIN32)
#pragma comment(lib, "IPHLPAPI.lib") //Becuase of poco needing this
#endif
