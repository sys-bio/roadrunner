#pragma hdrstop
// on Windows, this needs to go first to get M_PI
#define _USE_MATH_DEFINES

#include "rrOSSpecifics.h"

// TODO will clean these up in the future
#undef RR_DEPRECATED
#define RR_DEPRECATED(func) func

#include "rrRoadRunner.h"
#include "rrException.h"
#include "ExecutableModelFactory.h"
#include "rrCompiler.h"
#include "rrLogger.h"
#include "rrUtils.h"
#include "rrExecutableModel.h"
#include "rrSBMLModelSimulation.h"
#include "rr-libstruct/lsLA.h"
#include "rr-libstruct/lsLibla.h"
#include "rrConstants.h"
#include "rrVersionInfo.h"
#include "Integrator.h"
#include "IntegratorRegistration.h"
#include "SteadyStateSolver.h"
#include "SolverRegistration.h"
#include "rrSBMLReader.h"
#include "rrConfig.h"
#include "SBMLValidator.h"
#include "llvm/LLVMExecutableModel.h"
#include "sbml/ListOf.h"
#include "sbml/Model.h"
#include "sbml/math/FormulaParser.h"
#include "llvm/ModelResources.h"
#include <llvm/IR/IRBuilder.h>

#include <sbml/conversion/SBMLLocalParameterConverter.h>
#include <sbml/conversion/SBMLLevelVersionConverter.h>

#include <iostream>
#include <math.h>
#include <assert.h>
#include <rr-libstruct/lsLibStructural.h>
#include <Poco/File.h>
#include <Poco/Mutex.h>
#include <list>


#ifdef _MSC_VER
#define isfinite(x) _finite(x)
#endif


namespace rr
{
using namespace std;
using namespace ls;
using Poco::Mutex;

static Mutex roadRunnerMutex;

typedef std::vector<std::string> string_vector;


// we can write a single function to pick the string lists out
// of the model instead of duplicating it 6 times with
// fun ptrs.
typedef string (ExecutableModel::*GetNameFuncPtr)(int);
typedef int (ExecutableModel::*GetNumFuncPtr)();

// make this static here, hide our implementation...
static vector<string> createModelStringList(ExecutableModel *model,
        GetNumFuncPtr numFunc, GetNameFuncPtr nameFunc)
{
    if (!model)
    {
        return vector<string>(0);
    }

    const int num = (model->*numFunc)();
    vector<string> strings(num);

    for(int i = 0; i < num; i++)
    {
        strings[i] = (model->*nameFunc)(i);
    }

    return strings;
}

/**
 * convert the names from the sbml settings file into the
 * RoadRunner selection syntax.
 */
static std::vector<std::string> createSelectionList(const SimulateOptions& o);

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
static vector<double>  logspace(const double& startW, const double& d2, const int& n);
static double          phase(Complex& val);
static double          getAdjustment(Complex& z);

/**
 * variable time step integration data struct
 */
typedef std::list<std::vector<double> > DoubleVectorList;


/**
 * check if metabolic control analysis is valid for the model.
 *
 * In effect, this checks that the the model is a pure
 * reaction-kinetics model with no rate rules, no events.
 *
 * Throws an invaid arg exception if not valid.
 */
static void metabolicControlCheck(ExecutableModel *model);



//The instance count increases/decreases as instances are created/destroyed.
static int mInstanceCount = 0;

/**
 * The type of sbml element that the RoadRunner::setParameterValue
 * and RoadRunner::getParameterValue method operate on.
 *
 * @deprecated use the ExecutableModel methods directly.
 */
enum ParameterType
{
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
#define assert_similar(a, b) assert(std::abs(a - b) < 1e-13)

/**
 * implemention class, hide all details here.
 */
class RoadRunnerImpl {
public:

    int mInstanceID;
    double mDiffStepSize;

    double mSteadyStateThreshold;
    ls::DoubleMatrix simulationResult;

    /**
     * Points to the current integrator. This is a pointer into the
     * integtators array.
     */
    Integrator* integrator;
	std::vector<Integrator*> integrators;

    /**
     * Points to the current steady state solver
     */
    SteadyStateSolver* steady_state_solver;
    std::vector<SteadyStateSolver*> steady_state_solvers;


	std::vector<SelectionRecord> mSelectionList;

    /**
     * ModelGenerator obtained from the factory
     */
    LoadSBMLOptions loadOpt;

    std::vector<SelectionRecord> mSteadyStateSelection;

    ExecutableModel* model;

    /**
     * here for compatiblity, will go.
     */
    Compiler* compiler;

    std::string mCurrentSBML;

    /**
     * structural analysis library.
     */
    LibStructural* mLS;

    /**
     * options that are specific to the simulation
     */
    SimulateOptions simulateOpt;

    /**
     * various general options that can be modified by external callers.
     */
    RoadRunnerOptions roadRunnerOptions;

    /**
     * the xml string that is given in setConfigurationXML.
     *
     * Needed because the NLEQ is only created in the steadyState method.
     */
    std::string configurationXML;

    /**
     * TODO get rid of this garbage
     */
    friend class aFinalizer;

	libsbml::SBMLDocument* document;

    RoadRunnerImpl(const std::string& uriOrSBML,
            const Dictionary* dict) :
                mDiffStepSize(0.05),
                mSteadyStateThreshold(1.E-2),
                simulationResult(),
                integrator(0),
                mSelectionList(),
                mSteadyStateSelection(),
                model(0),
                mCurrentSBML(),
                mLS(0),
                simulateOpt(),
                mInstanceID(0),
                loadOpt(dict),
                compiler(Compiler::New())
    {
        // have to init integrators the hard way in c++98
        //memset((void*)integrators, 0, sizeof(integrators)/sizeof(char));
    }

	RoadRunnerImpl(const std::istream& in) :
		mDiffStepSize(0.05)
	{

	}


    RoadRunnerImpl(const string& _compiler, const string& _tempDir,
            const string& _supportCodeDir) :
                mDiffStepSize(0.05),
                mSteadyStateThreshold(1.E-2),
                simulationResult(),
                integrator(0),
                mSelectionList(),
                mSteadyStateSelection(),
                model(0),
                mCurrentSBML(),
                mLS(0),
                simulateOpt(),
                mInstanceID(0),
                compiler(Compiler::New())
    {
        loadOpt.setItem("compiler", _compiler);
        loadOpt.setItem("tempDir", _tempDir);
        loadOpt.setItem("supportCodeDir", _supportCodeDir);

        // have to init integrators the hard way in c++98
        //memset((void*)integrators, 0, sizeof(integrators)/sizeof(char));
    }

    ~RoadRunnerImpl()
    {
        Log(Logger::LOG_DEBUG) << __FUNC__ << ", global instance count: " << mInstanceCount;

        delete compiler;
        delete model;
        delete mLS;

		deleteAllSolvers();

        mInstanceCount--;
    }

    void deleteAllSolvers()
    {
        for (std::vector<Integrator*>::iterator it = integrators.begin(); it != integrators.end(); ++it)
        {
            delete *it;
            *it = NULL;
        }
        integrators.clear();
        for (std::vector<SteadyStateSolver*>::iterator it = steady_state_solvers.begin(); it != steady_state_solvers.end(); ++it)
        {
            delete *it;
            *it = NULL;
        }
        steady_state_solvers.clear();
	}

    void syncAllSolversWithModel(ExecutableModel* m)
    {
        for (std::vector<Integrator*>::iterator it = integrators.begin(); it != integrators.end(); ++it)
        {
            (*it)->syncWithModel(m);
        }
        for (std::vector<SteadyStateSolver*>::iterator it = steady_state_solvers.begin(); it != steady_state_solvers.end(); ++it)
        {
            (*it)->syncWithModel(m);
        }
    }

    void setParameterValue(const ParameterType parameterType,
            const int parameterIndex, const double value)
    {
        switch (parameterType)
        {
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
            const int parameterIndex)
    {
        switch (parameterType)
        {
        case ptBoundaryParameter:
        {
            double result = 0;
            model->getBoundarySpeciesConcentrations(1, &parameterIndex, &result);
            return result;
        }
        break;
        case ptGlobalParameter:
        {
            double result = 0;
            model->getGlobalParameterValues(1, &parameterIndex, &result);
            return result;
        }
        break;

        // Used when calculating elasticities
        case ptFloatingSpecies:
        {
            double result = 0;
            model->getFloatingSpeciesConcentrations(1, &parameterIndex, &result);
            return result;
        }
        break;

        case ptConservationParameter:
        {
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
                                        double originalValue, double increment)
    {
        setParameterValue(parameterType, parameterIndex, originalValue + increment);
    }

};



int RoadRunner::getInstanceCount()
{
    return mInstanceCount;
}

int RoadRunner::getInstanceID()
{
    return impl->mInstanceID;
}

RoadRunner::RoadRunner() : impl(new RoadRunnerImpl("", NULL))
{
    // must be run to register integrators at startup
    IntegratorRegistrationMgr::Register();
    // must be run to register solvers at startup
    SolverRegistrationMgr::Register();

    //Increase instance count..
    mInstanceCount++;
    impl->mInstanceID = mInstanceCount;

    // make CVODE the default integrator
    setIntegrator("cvode");
    // make NLEQ2 the default steady state solver
    setSteadyStateSolver("nleq2");
}

RoadRunner::RoadRunner(const std::string& uriOrSBML,
        const Dictionary* options) :
            impl(new RoadRunnerImpl(uriOrSBML, options))
{
    // must be run to register integrators at startup
    IntegratorRegistrationMgr::Register();
    // must be run to register solvers at startup
    SolverRegistrationMgr::Register();

    // make CVODE the default integrator
    setIntegrator("cvode");
    // make NLEQ2 the default steady state solver
    setSteadyStateSolver("nleq2");

    load(uriOrSBML, options);

    //Increase instance count..
    mInstanceCount++;
    impl->mInstanceID = mInstanceCount;
}


RoadRunner::RoadRunner(const string& _compiler, const string& _tempDir,
        const string& supportCodeDir) :
        impl(new RoadRunnerImpl(_compiler, _tempDir, supportCodeDir))
{
    // must be run to register integrators at startup
    IntegratorRegistrationMgr::Register();
    // must be run to register solvers at startup
    SolverRegistrationMgr::Register();

    string tempDir = _tempDir.empty() ? getTempDir() : _tempDir;

    setTempDir(tempDir);

    //Increase instance count..
    mInstanceCount++;
    impl->mInstanceID = mInstanceCount;

    // make CVODE the default integrator
    setIntegrator("cvode");
    // make NLEQ2 the default steady state solver
    setSteadyStateSolver("nleq2");
}

RoadRunner::~RoadRunner()
{
    delete impl;
}

ExecutableModel* RoadRunner::getModel()
{
    return impl->model;
}

void RoadRunner::setOptions(const RoadRunnerOptions& opt)
{
    impl->roadRunnerOptions = opt;
}

vector<SelectionRecord> RoadRunner::getSelectionList()
{
    return impl->mSelectionList;
}

string RoadRunner::getInfo()
{
    applySimulateOptions();

    stringstream ss;
    ss << "<roadrunner.RoadRunner() { " << std::endl;
    ss << "'this' : " << (void*)(this) << std::endl;
    ss << "'modelLoaded' : " << (impl->model == NULL ? "false" : "true") << std::endl;

    if(impl->model)
    {
        ss << "'modelName' : " <<  impl->model->getModelName() << std::endl;
    }

    ss << "'libSBMLVersion' : " << getVersionStr(VERSIONSTR_LIBSBML) << std::endl;
    ss << "'jacobianStepSize' : " << impl->roadRunnerOptions.jacobianStepSize << std::endl;
    ss << "'conservedMoietyAnalysis' : " << rr::toString(impl->loadOpt.getConservedMoietyConversion()) << std::endl;

#if defined(BUILD_LEGACY_C)
    ss<<"Temporary folder: "        <<    getTempDir()<<endl;
    ss<<"Compiler location: "       <<    getCompiler()->getCompilerLocation() << endl;
    ss<<"Support Code Folder: "     <<    getCompiler()->getSupportCodeFolder() << endl;
    ss<<"Working Directory: "       <<    getCWD() << endl;
#endif

    ss << "'simulateOptions' : " << endl;
    ss << impl->simulateOpt.toString();
    ss << ", " << endl;

    ss << "'integrator' : " << endl;
    if(impl->integrator) {
        ss << impl->integrator->toString();
        ss << endl;
    }
    else {
        ss << "Null" << endl;
    }

    ss << "}>";

    return ss.str();
}

string RoadRunner::getExtendedVersionInfo()
{
    stringstream info;
    info << getVersionStr(VERSIONSTR_BASIC | VERSIONSTR_COMPILER | VERSIONSTR_DATE | VERSIONSTR_LIBSBML) << endl;
    info << "Working Directory: " << getCWD() << endl;
    return info.str();
}



ls::LibStructural* RoadRunner::getLibStruct()
{
    Mutex::ScopedLock lock(roadRunnerMutex);

    if (impl->mLS)
    {
        return impl->mLS;
    }
    else if (!impl->mCurrentSBML.empty())
    {
        impl->mLS = new ls::LibStructural(impl->mCurrentSBML);
        Log(Logger::LOG_INFORMATION) << "created structural analysis, messages: "
                << impl->mLS->getAnalysisMsg();
        return impl->mLS;
    }
    else
    {
        throw std::invalid_argument("could not create structural analysis with no loaded sbml");
    }
}

Compiler* RoadRunner::getCompiler()
{
    return impl->compiler;
}


void RoadRunner::setCompiler(const string& compiler)
{
    impl->loadOpt.setItem("compiler", compiler);
}

bool RoadRunner::isModelLoaded()
{
    return impl->model ? true : false;
}

void RoadRunner::setSimulateOptions(const SimulateOptions& settings)
{
    impl->simulateOpt = settings;
}

SimulateOptions& RoadRunner::getSimulateOptions()
{
    return impl->simulateOpt;
}

bool RoadRunner::getConservedMoietyAnalysis()
{
    return impl->loadOpt.getConservedMoietyConversion();
}

void RoadRunner::setTempDir(const string& folder)
{
    impl->loadOpt.setItem("tempDir", folder);
}

string RoadRunner::getTempDir()
{
    return impl->loadOpt.getItem("tempDir");
}

int RoadRunner::createDefaultTimeCourseSelectionList()
{
    vector<string> selections;
    vector<string> oFloating  = getFloatingSpeciesIds();
    int numFloatingSpecies = oFloating.size();
    //int numIndSpecies = getNumberOfIndependentSpecies();

    // add floating species to the default selection

    selections.push_back("time");
    //for(int i = 0; i < numIndSpecies; i++)
    for (int i = 0; i < numFloatingSpecies; i++)
    {
        selections.push_back("[" + oFloating[i] + "]");
    }

    // add parameters defined by rate rules to the default selection

    vector<string> selections_with_ratelaws(selections);
    try {
        vector<string> raterule_symbols = impl->model->getRateRuleSymbols();
        for (vector<string>::iterator i = raterule_symbols.begin(); i != raterule_symbols.end(); ++i)
            selections_with_ratelaws.push_back(*i);
    } catch (NotImplementedException) {
        Log(Logger::LOG_WARNING) << "Querying rate rule symbols not supported with this executable model";
    }

    // try to apply selections including rate laws,
    // fall back if unable to use rate laws
    // see https://github.com/sys-bio/roadrunner/issues/88
    try {
        setSelections(selections_with_ratelaws);
    } catch (...) {
        Log(Logger::LOG_WARNING) << "Rate laws exist but cannot be added to default selections";
        setSelections(selections);
    }

    Log(lDebug)<<"The following is selected:";
    for(int i = 0; i < impl->mSelectionList.size(); i++)
    {
        Log(lDebug)<<impl->mSelectionList[i];
    }
    return impl->mSelectionList.size();
}

int RoadRunner::createTimeCourseSelectionList()
{
    // make a list out of the values in the settings,
    // will always have at least a "time" at the first item.
    vector<string> settingsList = createSelectionList(impl->simulateOpt);

    assert(settingsList.size() >= 1 && "selection list from SimulateOptions does does not have time");

    // if settings has any selections, this overrides the selection list.
    if (settingsList.size() > 1)
    {
        Log(Logger::LOG_INFORMATION) << "overriding selection list with values from SimulateOptions.";
        setSelections(settingsList);
    }

    for(int i = 0; i < impl->mSelectionList.size(); i++)
    {
        Log(Logger::LOG_DEBUG) << "Selection Value [" << i << "]: " << impl->mSelectionList[i].to_repr();
    }

    return impl->mSelectionList.size();
}

string RoadRunner::getParamPromotedSBML(const string& sbml)
{
    if (SBMLReader::is_sbml(sbml))
    {
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
    }
    else
    {
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

double RoadRunner::getValue(const SelectionRecord& record)
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }

    double dResult;

    switch (record.selectionType)
    {
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
        impl->model->getFloatingSpeciesAmountRates(1, &record.index, &dResult);
        break;

    case SelectionRecord::COMPARTMENT:
        impl->model->getCompartmentVolumes(1, &record.index, &dResult);
        break;

    case SelectionRecord::GLOBAL_PARAMETER:
    {
        if (record.index > ((impl->model->getNumGlobalParameters()) - 1))
        {
            int index = record.index - impl->model->getNumGlobalParameters();
            impl->model->getConservedMoietyValues(1, &index, &dResult);
        }
        else
        {
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
    case SelectionRecord::EIGENVALUE_REAL:
    {
        string species = record.p1;
        int index = impl->model->getFloatingSpeciesIndex(species);

        if (index < 0)
        {
            throw std::logic_error("Invalid species id" + record.p1 + " for eigenvalue");
        }

        vector<Complex> eig = getEigenValues(JACOBIAN_FULL);

        if (eig.size() <= index)
        {
            // this should NEVER happen
            throw std::runtime_error("Eigenvalues vector length less than species id");
        }

        return std::real(eig[index]);
    }
    break;
    case SelectionRecord::EIGENVALUE_IMAG:
    {
        string species = record.p1;
        int index = impl->model->getFloatingSpeciesIndex(species);

        if (index < 0)
        {
            throw std::logic_error("Invalid species id" + record.p1 + " for eigenvalue");
        }

        vector<Complex> eig = getEigenValues(JACOBIAN_FULL);

        if (eig.size() <= index)
        {
            // this should NEVER happen
            throw std::runtime_error("Eigenvalues vector length less than species id");
        }

        return std::imag(eig[index]);
    }
    break;
    case SelectionRecord::INITIAL_CONCENTRATION:
    {
        impl->model->getFloatingSpeciesInitConcentrations(1, &record.index, &dResult);
    }
    break;
    case SelectionRecord::INITIAL_AMOUNT:
    {
        impl->model->getFloatingSpeciesInitAmounts(1, &record.index, &dResult);
    }
    break;
    case SelectionRecord::INITIAL_GLOBAL_PARAMETER:
    {
        impl->model->getGlobalParameterInitValues(1, &record.index, &dResult);
    }
    break;
    case SelectionRecord::INITIAL_COMPARTMENT:
    {
        impl->model->getCompartmentInitVolumes(1, &record.index, &dResult);
    }
    break;
    case SelectionRecord::STOICHIOMETRY:
    {
        int speciesIndex = impl->model->getFloatingSpeciesIndex(record.p1);
        int reactionIndex = impl->model->getReactionIndex(record.p2);
        return impl->model->getStoichiometry(speciesIndex, reactionIndex);
    }
        break;

    default:
        dResult = 0.0;
        break;
    }
    return dResult;
}

double RoadRunner::getNthSelectedOutput(unsigned index, double currentTime)
{
    const SelectionRecord &record = impl->mSelectionList[index];

    if (record.selectionType == SelectionRecord::TIME)
    {
        return currentTime;
    }
    else
    {
        return getValue(record);
    }
}

int RoadRunner::getTimeRowIndex()
{
  for (u_int j = 0; j < impl->mSelectionList.size(); j++)
  {
      const SelectionRecord &record = impl->mSelectionList[j];
      if (record.selectionType == SelectionRecord::TIME)
      {
          return (int)j;
      }
  }
  // -1 for failure
  return -1;
}

void RoadRunner::getSelectedValues(DoubleMatrix& results, int nRow, double currentTime)
{
    for (u_int j = 0; j < impl->mSelectionList.size(); j++)
    {
        double out =  getNthSelectedOutput(j, currentTime);
        results(nRow,j) = out;
    }
}

void RoadRunner::getSelectedValues(std::vector<double>& results,
        double currentTime)
{
    assert(results.size() == impl->mSelectionList.size()
            && "given vector and selection list different size");

    u_int size = results.size();
    for (u_int i = 0; i < size; ++i)
    {
        results[i] = getNthSelectedOutput(i, currentTime);
    }
}


vector<double> RoadRunner::getConservedMoietyValues()
{
    return getLibStruct()->getConservedSums();
}

void RoadRunner::load(const string& uriOrSbml, const Dictionary *dict)
{
    Mutex::ScopedLock lock(roadRunnerMutex);

    get_self();

    self.mCurrentSBML = SBMLReader::read(uriOrSbml);

    delete impl->model;
    impl->model = 0;

    delete impl->mLS;
    impl->mLS = NULL;

    if(dict) {
        self.loadOpt = LoadSBMLOptions(dict);
    }

    // TODO: streamline so SBML document is not read several times
    // check that stoichiometry is defined
    if (!isStoichDefined(self.mCurrentSBML)) {
        // if any reactions are missing stoich, the simulation results will be wrong
        // fix sbml by assuming unit stoich where missing
        self.mCurrentSBML = fixMissingStoich(self.mCurrentSBML);
        Log(Logger::LOG_WARNING)<<"Stoichiometry is not defined for all reactions; assuming unit stoichiometry where missing";
    }
    try {

    // the following lines load and compile the model. If anything fails here,
    // we validate the model to provide explicit details about where it
    // failed. Its *VERY* expensive to pre-validate the model.
		libsbml::SBMLReader reader;
		impl->document = reader.readSBMLFromString(impl->mCurrentSBML);
        self.model = ExecutableModelFactory::createModel(self.mCurrentSBML, &self.loadOpt);
    } catch (std::exception&) {
        string errors = validateSBML(impl->mCurrentSBML);

        if(!errors.empty()) {
            Log(Logger::LOG_ERROR) << "Invalid SBML: " << endl << errors;
        }

        // re-throw the exception
        throw;
    }

    impl->syncAllSolversWithModel(impl->model);

    reset();
    if ((self.loadOpt.loadFlags & LoadSBMLOptions::NO_DEFAULT_SELECTIONS) == 0)
    {
        createDefaultSelectionLists();
    }
}

bool RoadRunner::createDefaultSelectionLists()
{
    bool result = true;

    //Create a default timecourse selectionlist
    if(!createDefaultTimeCourseSelectionList())
    {
        Log(lDebug)<<"Failed creating default timecourse selectionList.";
        result = false;
    }
    else
    {
        Log(lDebug)<<"Created default TimeCourse selection list.";
    }

    //Create a defualt steady state selectionlist
    if(!createDefaultSteadyStateSelectionList())
    {
        Log(lDebug)<<"Failed creating default steady state selectionList.";
        result = false;
    }
    else
    {
        Log(lDebug)<<"Created default SteadyState selection list.";
    }
    return result;
}

bool RoadRunner::clearModel()
{
    // The model owns the shared library (if it exists), when the model is deleted,
    // its dtor unloads the shared lib.
    if(impl->model)
    {
        delete impl->model;
        impl->model = NULL;

		delete impl->mLS;
		impl->mLS = NULL;
        return true;
    }
    return false;
}

void RoadRunner::reset()
{
    uint opt1 = rr::Config::getInt(rr::Config::MODEL_RESET);
    reset(opt1);
    uint opt2 = rr::SelectionRecord::DEPENDENT_INITIAL_GLOBAL_PARAMETER;
    reset(opt2);
}

void RoadRunner::reset(int options)
{
    if (impl->model)
    {
        // model gets set to before time = 0
        impl->model->reset(options);

        if (!impl->integrator)
            throw std::runtime_error("No integrator set");
        impl->integrator->restart(0.0);

        try
        {
            impl->model->testConstraints();
        }
        catch (const Exception& e)
        {
            Log(lWarning)<<"Constraint Violated at time = 0\n"<<e.Message();
        }
    }
}

void RoadRunner::resetSelectionLists()
{
    createDefaultSelectionLists();
}

bool RoadRunner::populateResult()
{
    vector<string> list(impl->mSelectionList.size());

    for(int i = 0; i < impl->mSelectionList.size(); ++i)
    {
        list[i] = impl->mSelectionList[i].to_string();
    }

    return true;
}


std::vector<rr::SelectionRecord>& RoadRunner::getSelections()
{
    return impl->mSelectionList;
}


double RoadRunner::mcaSteadyState()
{
    if (!impl->model)
    {
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


double RoadRunner::steadyState(const Dictionary* dict)
{
    Log(Logger::LOG_DEBUG)<<"RoadRunner::steadyState...";
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }

    if (!impl->loadOpt.getConservedMoietyConversion() &&
            (Config::getBool(Config::ROADRUNNER_DISABLE_WARNINGS)
                & Config::ROADRUNNER_DISABLE_WARNINGS_STEADYSTATE) == 0)
    {
        Log(Logger::LOG_WARNING) << "Conserved Moiety Analysis is not enabled, steady state may fail with singular Jacobian";
        Log(Logger::LOG_WARNING) << "Conserved Moiety Analysis may be enabled via the conservedMoeityAnalysis property or "
                                    "via the configuration file or the Config class setValue, see roadrunner documentation";
        Log(Logger::LOG_WARNING) << "to remove this warning, set ROADRUNNER_DISABLE_WARNINGS to 1 or 3 in the config file";
    }

    metabolicControlCheck(impl->model);

    if (!impl->steady_state_solver) {
        Log(Logger::LOG_ERROR)<<"No steady state solver";
        throw std::runtime_error("No steady state solver");
    }

    Log(Logger::LOG_DEBUG)<<"Attempting to find steady state using solver '" << impl->steady_state_solver->getName() << "'...";

    double ss;

    // Rough estimation
    if (impl->steady_state_solver->getValueAsBool("allow_presimulation"))
    {
        std::string currint = impl->integrator->getName();

        // use cvode
        setIntegrator("cvode");

        double start_temp = impl->simulateOpt.start;
        double duration_temp = impl->simulateOpt.duration;
        int steps_temp = impl->simulateOpt.steps;

        impl->simulateOpt.start = 0;
        impl->simulateOpt.duration = impl->steady_state_solver->getValueAsDouble("presimulation_time");
        impl->simulateOpt.steps = impl->steady_state_solver->getValueAsInt("presimulation_maximum_steps");

        try
        {
            simulate();
        }
        catch (const CoreException& e)
        {
            impl->simulateOpt.start = start_temp;
            impl->simulateOpt.duration = duration_temp;
            impl->simulateOpt.steps = steps_temp;

            setIntegrator(currint);

            throw CoreException("Steady state presimulation failed. Try turning off allow_presimulation flag to False "
                "via r.steadyStateSolver.allow_presimulation = False where r is an roadrunner instance; ", e.Message());
        }

        impl->simulateOpt.start = start_temp;
        impl->simulateOpt.duration = duration_temp;
        impl->simulateOpt.steps = steps_temp;

        setIntegrator(currint);

        Log(Logger::LOG_DEBUG) << "Steady state presimulation done";
    }

    // NLEQ
    if (impl->steady_state_solver->getValueAsBool("allow_approx"))
    {
        int l = impl->model->getNumFloatingSpecies();
        double* vals = new double[l];
        impl->model->getFloatingSpeciesConcentrations(l, NULL, vals);

        try
        {
            ss = impl->steady_state_solver->solve();

            if (ss < 0)
            {
                Log(Logger::LOG_ERROR) << "Steady State solver failed...";
            }

            return ss;
        }
        catch (NLEQException& e1)
        {
            try
            {
                // Reset to t = 0;
                reset();

                // Restore initial concentrations
                for (int i = 0; i<l; ++i) {
                    impl->model->setFloatingSpeciesConcentrations(1, &i, &vals[i]);
                }

                ss = steadyStateApproximate();

                Log(Logger::LOG_WARNING) << "Steady state solver failed. However, RoadRunner approximated the solution successfully.";

                return ss;
            }
            catch (CoreException& e2)
            {
                throw CoreException("Both steady state solver and approximation routine failed: ", e1.Message() + "; " + e2.Message());
            }
        }
    }
    else
    {
        ss = impl->steady_state_solver->solve();

        if (ss < 0)
        {
            Log(Logger::LOG_ERROR) << "Steady State solver failed...";
        }

        return ss;
    }
    
}


double RoadRunner::steadyStateApproximate(const Dictionary* dict)
{
    Log(Logger::LOG_DEBUG) << "RoadRunner::steadStateApproximate";

    get_self();
    check_model();

    // store current integrator info
    std::string currint = self.integrator->getName();

    // use cvode
    setIntegrator("cvode");

    // create selections
    vector<string> ss_selections_with_time;
    vector<string> ss_selections = getSteadyStateSelectionStrings();
    int num_ss_sel = ss_selections.size();

    ss_selections_with_time.push_back("time");
    for (int i = 0; i < num_ss_sel; i++)
    {
        ss_selections_with_time.push_back(ss_selections[i]);
    }

    // steady state selection
    std::vector<rr::SelectionRecord> currsel = self.mSelectionList;
    setSelections(ss_selections_with_time);

    double start_temp = self.simulateOpt.start;
    double duration_temp = self.simulateOpt.duration;
    int steps_temp = self.simulateOpt.steps;

    // initialize
    double duration = self.steady_state_solver->getValueAsDouble("approx_time");
    int steps = self.steady_state_solver->getValueAsInt("approx_maximum_steps");
    self.simulateOpt.start = 0;
    self.simulateOpt.duration = duration;
    self.simulateOpt.steps = steps;
    double tol = 0;
    int l = self.mSelectionList.size();
    double* vals1 = new double[l];
    double* vals2 = new double[l];

    Log(Logger::LOG_DEBUG) << "tol thres: " << self.steady_state_solver->getValueAsDouble("approx_tolerance");
    Log(Logger::LOG_DEBUG) << "Max steps: " << self.steady_state_solver->getValueAsInt("approx_maximum_steps");
    Log(Logger::LOG_DEBUG) << "Max time: " << self.steady_state_solver->getValueAsDouble("approx_time");
    
    try
    {
        simulate();
        vals1 = self.simulationResult[steps - 1];
        vals2 = self.simulationResult[steps - 2];

        for (int i = 1; i < l; i++)
        {
            tol += sqrt(pow((vals2[i] - vals1[i]) / (duration / steps), 2));
        }
    }
    catch (EventListenerException& e)
    {
        Log(Logger::LOG_ERROR) << e.what();
    }
    
    self.simulateOpt.start = start_temp;
    self.simulateOpt.duration = duration_temp;
    self.simulateOpt.steps = steps_temp;
    setIntegrator(currint);
    setSelections(currsel);

    Log(Logger::LOG_DEBUG) << "Steady state approximation done";

    if (tol > self.steady_state_solver->getValueAsDouble("approx_tolerance"))
    {
        throw CoreException("Failed to converge while running approximation routine. Try increasing "
            "the time or maximum number of iteration via changing the settings under r.steadyStateSolver "
            "where r is an roadrunner instance. Model might not have a steady state.");
    }

    return tol;
}


DoubleMatrix RoadRunner::steadyStateNamedArray(const Dictionary* dict)
{
    steadyState();
    return getSteadyStateValuesNamedArray();
}




void RoadRunner::setConservedMoietyAnalysis(bool value)
{
    get_self();

    if(value == self.loadOpt.getConservedMoietyConversion())
    {
        Log(lDebug)<<"The compute and assign conservation laws flag already set to : " << toString(value);
    }

    self.loadOpt.setConservedMoietyConversion(value);

    if(self.model != NULL)
    {
        uint32_t savedOpt = self.loadOpt.modelGeneratorOpt;

        // have to reload
        self.loadOpt.modelGeneratorOpt |= LoadSBMLOptions::RECOMPILE;

        load(impl->mCurrentSBML);

        // restore original reload value
        self.loadOpt.modelGeneratorOpt = savedOpt;
    } 
}


// [Help("Get scaled elasticity coefficient with respect to a global parameter or species")]
double RoadRunner::getEE(const string& reactionName, const string& parameterName)
{
    return getEE(reactionName, parameterName, true);
}

double RoadRunner::getEE(const string& reactionName, const string& parameterName, bool computeSteadyState)
{
    ParameterType parameterType;
    int reactionIndex;
    int parameterIndex;

    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }

    // Check the reaction name
    if ((reactionIndex = impl->model->getReactionIndex(reactionName)) < 0)
    {
        throw CoreException(format("Unable to locate reaction name: [{0}]", reactionName));
    }

    // Find out what kind of parameter we are dealing with
    if (( parameterIndex = impl->model->getFloatingSpeciesIndex(parameterName)) >= 0)
    {
        parameterType = ptFloatingSpecies;
    }
    else if ((parameterIndex = impl->model->getBoundarySpeciesIndex(parameterName)) >= 0)
    {
        parameterType = ptBoundaryParameter;
    }
    else if ((parameterIndex = impl->model->getGlobalParameterIndex(parameterName)) >= 0)
    {
        parameterType = ptGlobalParameter;
    }
    else if ((parameterIndex = impl->model->getConservedMoietyIndex(parameterName)) >= 0)
    {
        parameterType = ptConservationParameter;
    }
    else
    {
        throw CoreException(format("Unable to locate variable: [{0}]", parameterName));
    }



    double variableValue = 0;
    impl->model->getReactionRates(1, &reactionIndex, &variableValue);
    double parameterValue = impl->getParameterValue(parameterType, parameterIndex);
    if (variableValue == 0)
    {
        variableValue = 1e-12;
    }
    return getuEE(reactionName, parameterName, computeSteadyState) * parameterValue / variableValue;
}

double RoadRunner::getDiffStepSize() const
{
    if (!impl)
        throw std::runtime_error("Missing impl");
    return impl->mDiffStepSize;
}

void RoadRunner::setDiffStepSize(double val)
{
    if (!impl)
        throw std::runtime_error("Missing impl");
    impl->mDiffStepSize = val;
}

double RoadRunner::getSteadyStateThreshold() const
{
    if (!impl)
        throw std::runtime_error("Missing impl");
    return impl->mSteadyStateThreshold;
}

void RoadRunner::setSteadyStateThreshold(double val)
{
    if (!impl)
        throw std::runtime_error("Missing impl");
    impl->mSteadyStateThreshold = val;
}

double RoadRunner::getuEE(const string& reactionName, const string& parameterName)
{
    return getuEE(reactionName, parameterName, true);
}

void RoadRunner::fixDependentSpeciesValues(int except, double* ref) {
    int l = impl->model->getNumFloatingSpecies();
    double* vals = new double[l];
    impl->model->getFloatingSpeciesConcentrations(l, NULL, vals);
    for (int i=0; i<l; ++i) {
        if (i != except && vals[i] != ref[i]) {
            impl->model->setFloatingSpeciesConcentrations(1, &i, &ref[i]);
        }
    }
    delete vals;
}

double RoadRunner::getuEE(const string& reactionName, const string& parameterName, bool computeSteadystate)
{
    try
    {
        if (!impl->model)
        {
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
        if ((reactionIndex = impl->model->getReactionIndex(reactionName)) < 0)
        {
            throw CoreException("Unable to locate reaction name: [" + reactionName + "]");
        }

        // Find out what kind of parameter we are dealing with
        if ((parameterIndex = impl->model->getFloatingSpeciesIndex(parameterName)) >= 0)
        {
            parameterType = ptFloatingSpecies;
            originalParameterValue = 0;
            impl->model->getFloatingSpeciesConcentrations(1, &parameterIndex, &originalParameterValue);
        }
        else if ((parameterIndex = impl->model->getBoundarySpeciesIndex(parameterName)) >= 0)
        {
            parameterType = ptBoundaryParameter;
            originalParameterValue = 0;
            impl->model->getBoundarySpeciesConcentrations(1, &parameterIndex, &originalParameterValue);
        }
        else if ((parameterIndex = impl->model->getGlobalParameterIndex(parameterName)) >= 0)
        {
            parameterType = ptGlobalParameter;
            originalParameterValue = 0;
            impl->model->getGlobalParameterValues(1, &parameterIndex, &originalParameterValue);
        }
        else if ((parameterIndex = impl->model->getConservedMoietyIndex(parameterName)) >= 0)
        {
            parameterType = ptConservationParameter;
            originalParameterValue = 0;
            impl->model->getConservedMoietyValues(1, &parameterIndex, &originalParameterValue);
        }
        else
        {
            throw CoreException("Unable to locate variable: [" + parameterName + "]");
        }

        double hstep = impl->mDiffStepSize*originalParameterValue;
        if (fabs(hstep) < 1E-12)
        {
            hstep = impl->mDiffStepSize;
        }

        impl->setParameterValue(parameterType, parameterIndex, originalParameterValue + hstep);
        fixDependentSpeciesValues(parameterIndex, ref);
        double fi = 0;
        impl->model->getReactionRates(1, &reactionIndex, &fi);
        
        impl->setParameterValue(parameterType, parameterIndex, originalParameterValue + 2*hstep);
        fixDependentSpeciesValues(parameterIndex, ref);
        double fi2 = 0;
        impl->model->getReactionRates(1, &reactionIndex, &fi2);

        impl->setParameterValue(parameterType, parameterIndex, originalParameterValue - hstep);
        fixDependentSpeciesValues(parameterIndex, ref);
        double fd = 0;
        impl->model->getReactionRates(1, &reactionIndex, &fd);

        impl->setParameterValue(parameterType, parameterIndex, originalParameterValue - 2*hstep);
        fixDependentSpeciesValues(parameterIndex, ref);
        double fd2 = 0;
        impl->model->getReactionRates(1, &reactionIndex, &fd2);

        // restore original value
        impl->setParameterValue(parameterType, parameterIndex, originalParameterValue);
        fixDependentSpeciesValues(-1, ref);

        // Use instead the 5th order approximation double unscaledValue = (0.5/hstep)*(fi-fd);
        // The following separated lines avoid small amounts of roundoff error
        double f1 = fd2 + 8*fi;
        double f2 = -(8*fd + fi2);

        delete ref;
        
        return 1/(12*hstep)*(f1 + f2);
    }
    catch(const Exception& e)
    {
        throw CoreException("Unexpected error from getuEE(): " +  e.Message());
    }
}

// Help("Updates the model based on all recent changes")
void RoadRunner::evalModel()
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }
    impl->model->getStateVectorRate(impl->model->getTime(), 0);
}


const DoubleMatrix* RoadRunner::simulate(const Dictionary* dict)
{
    get_self();
    check_model();

    const SimulateOptions *opt = dynamic_cast<const SimulateOptions*>(dict);

    if (opt) {
        self.simulateOpt = *opt;
    }

    applySimulateOptions();

    const double timeEnd = self.simulateOpt.duration + self.simulateOpt.start;
    const double timeStart = self.simulateOpt.start;

    // evalute the model with its current state
    self.model->getStateVectorRate(timeStart, 0, 0);
    // Variable Time Step Integration
    if (self.integrator->hasValue("variable_step_size") && self.integrator->getValueAsBool("variable_step_size"))
    {
        Log(Logger::LOG_INFORMATION) << "Performing variable step integration";

        DoubleVectorList results;
        std::vector<double> row(self.mSelectionList.size());

        try
        {
            // add current state as first row
            getSelectedValues(row, timeStart);
            results.push_back(row);

            self.integrator->restart(timeStart);

            double tout = timeStart;
            double last_tout = tout;

            // optimiziation for certain getValue operations.
            self.model->setIntegration(true);

            int n=0;

            while( tout <= timeEnd &&
              ( !self.simulateOpt.steps || n < self.simulateOpt.steps) &&
              ( !rr::Config::getInt(rr::Config::MAX_OUTPUT_ROWS) || n < rr::Config::getInt(rr::Config::MAX_OUTPUT_ROWS)) )
            {
                Log(Logger::LOG_DEBUG) << "variable step, start: " << tout
                        << ", end: " << timeEnd;
                tout = self.integrator->integrate(tout, timeEnd - tout);


                if (!isfinite(tout) || (tout == timeEnd))
                {
                    // time step is at infinity or zero so bail, but get the last value
                    getSelectedValues(row, timeEnd);
                    results.push_back(row);
                    break;
                }

                getSelectedValues(row, tout);

                // use interpolation to ensure last time point is requested end time
                if( tout > timeEnd ) {
                    if (!getIntegrator())
                    {
                        // should never happen
                        throw CoreException("No integrator selected in call to simulate");
                    }

                    if (getIntegrator()->getName() == "gillespie")
                    {
                        // stochastic simulations use flat interpolation
                        Log(Logger::LOG_DEBUG) << "simulate: use flat interpolation for last value with timeEnd = " <<  timeEnd << ", tout = " << tout << ", last_tout = " << last_tout;

                        for (int n = 0; n<row.size(); ++n) {
                            row.at(n) = results.back().at(n);
                        }

                        int itime = getTimeRowIndex();

                        if (itime >= 0)
                        {
                            row.at(itime) = timeEnd;
                        }
                    }
                    else
                    {
                        // ODE simulations use linear interpolation
                        double alpha = (timeEnd - last_tout)/(tout - last_tout);
                        Log(Logger::LOG_DEBUG) << "simulate: use linear interpolation for last value with timeEnd = " <<  timeEnd << ", tout = " << tout << ", last_tout = " << last_tout;

                        for(int n = 0; n<row.size(); ++n) {
                            row.at(n) = results.back().at(n) + alpha*(row.at(n) - results.back().at(n));
                        }
                    }
                } else {
                    last_tout = tout;
                }

                results.push_back(row);

                ++n;
            }
        }
        catch (EventListenerException& e)
        {
            Log(Logger::LOG_NOTICE) << e.what();
        }

        // stuff list values into result matrix
        self.simulationResult.resize(results.size(), row.size());
        uint rowi = 0;
        for (DoubleVectorList::const_iterator i = results.begin();
                i != results.end(); ++i, ++rowi)
        {
            // evidently [] operator gets row, go figure...
            double* prow = self.simulationResult[rowi];
            std::copy(i->begin(), i->end(), prow);
        }
    }

    // Stochastic Fixed Step Integration
    // do fixed time step simulation, these are different for deterministic
    // and stochastic.
	else if (self.integrator->getIntegrationMethod() == Integrator::Stochastic)
    {
        Log(Logger::LOG_INFORMATION)
                << "Performing stochastic fixed step integration for "
                << self.simulateOpt.steps + 1 << " steps";

        int numPoints = self.simulateOpt.steps + 1;

        if (numPoints <= 1)
        {
            numPoints = 2;
        }

        const double hstep = (timeEnd - timeStart) / (numPoints - 1);
        int nrCols = self.mSelectionList.size();

        Log(Logger::LOG_DEBUG) << "starting simulation with " << nrCols << " selected columns";

        // ignored if same
        self.simulationResult.resize(self.simulateOpt.steps + 1, nrCols);

        try
        {
            // add current state as first row
            getSelectedValues(self.simulationResult, 0, timeStart);

            self.integrator->restart(timeStart);

            double tout = timeStart;           // the exact times the integrator returns
            double next = timeStart + hstep;   // because of fixed steps, the times when the
                                               // value is recorded.

            // index gets bumped in do-while loop.
            for (int i = 1; i < self.simulateOpt.steps + 1;)
            {
                Log(Logger::LOG_DEBUG) << "step: " << i << "t0: " << tout << "hstep: " << next - tout;

                // stochastic frequently overshoots time end
                // may also be infinite
                tout = self.integrator->integrate(tout, next - tout);

                assert((tout >= next)
                        && "stochastic integrator did not integrate to end time");

                // get the output, always get at least one output
                do
                {
                    getSelectedValues(self.simulationResult, i, next);
                    i++;
                    next = timeStart + i * hstep;
                }
                while((i < self.simulateOpt.steps + 1) && tout > next);
            }
        }
        catch (EventListenerException& e)
        {
            Log(Logger::LOG_NOTICE) << e.what();
        }
    }

    // Deterministic Fixed Step Integration
    else
    {
        Log(Logger::LOG_INFORMATION)
                << "Perfroming deterministic fixed step integration for  "
                << self.simulateOpt.steps + 1 << " steps";

        int numPoints = self.simulateOpt.steps + 1;

        if (numPoints <= 1)
        {
            numPoints = 2;
        }

        double hstep = (timeEnd - timeStart) / (numPoints - 1);
        int nrCols = self.mSelectionList.size();

        Log(Logger::LOG_DEBUG) << "starting simulation with " << nrCols << " selected columns";

        // ignored if same
        self.simulationResult.resize(self.simulateOpt.steps + 1, nrCols);

        try
        {
            // add current state as first row
            getSelectedValues(self.simulationResult, 0, timeStart);

            self.integrator->restart(timeStart);

            // optimiziation for certain getValue operations.
            self.model->setIntegration(true);

            double tout = timeStart;

            for (int i = 1; i < self.simulateOpt.steps + 1; i++)
            {
                Log(Logger::LOG_DEBUG)<<"Step "<<i;
                double itime = self.integrator->integrate(tout, hstep);

                // the test suite is extremly sensetive to time differences,
                // so need to use the *exact* time here. occasionally the integrator
                // will return a value just slightly off from the exact time
                // value.
                tout = timeStart + i * hstep;
                getSelectedValues(self.simulationResult, i, tout);
            }
        }
        catch (EventListenerException& e)
        {
            Log(Logger::LOG_NOTICE) << e.what();
        }
    }

    // done with integration

    self.model->setIntegration(false);

    Log(Logger::LOG_DEBUG) << "Simulation done..";

    return &self.simulationResult;
}


double RoadRunner::oneStep(const double currentTime, const double stepSize, const bool reset)
{
    get_self();
    check_model();
    applySimulateOptions();

    try
    {
        if (reset)
        {
            self.integrator->restart(currentTime);
        }    
        return self.integrator->integrate(currentTime, stepSize);
    }
    catch (EventListenerException& e)
    {
        Log(Logger::LOG_NOTICE) << e.what();
        return self.model->getTime();
    }
}


double RoadRunner::internalOneStep(const double currentTime, const double stepSize, const bool reset)
{
    get_self();
    check_model();
    applySimulateOptions();
    double endTime;

    bool temp_var = self.integrator->getValue("variable_step_size");
    self.integrator->setValue("variable_step_size", true);

    try
    {
        if (reset)
        {
            self.integrator->restart(currentTime);
        }
        endTime = self.integrator->integrate(currentTime, stepSize);
        self.integrator->setValue("variable_step_size", temp_var);
        Log(Logger::LOG_DEBUG) << "internalOneStep: " << endTime;
        return endTime;
    }
    catch (EventListenerException& e)
    {
        Log(Logger::LOG_NOTICE) << e.what();
        endTime = self.model->getTime();
        self.integrator->setValue("variable_step_size", temp_var);
        return endTime;
    }
}


std::vector<ls::Complex> RoadRunner::getFullEigenValues()
{
    return getEigenValues(JACOBIAN_FULL);
}


std::vector<ls::Complex> RoadRunner::getReducedEigenValues()
{
    return getEigenValues(JACOBIAN_REDUCED);
}

std::vector< std::complex<double> > RoadRunner::getEigenValues(RoadRunner::JacobianMode mode)
{
    check_model();
    DoubleMatrix mat;

    if(mode == JACOBIAN_FULL)
    {
        mat = getFullJacobian();
    }
    else
    {
        mat = getReducedJacobian();
    }
    return ls::getEigenValues(mat);
}

DoubleMatrix RoadRunner::getFloatingSpeciesAmountsNamedArray()
{
    check_model();

    int l = impl->model->getNumFloatingSpecies();
    
    double* vals = new double[l];
    impl->model->getFloatingSpeciesAmounts(l, NULL, vals);

    LibStructural *ls = getLibStruct();
    DoubleMatrix v(1, l);

    for (int i = 0; i<l; ++i)
        v(0, i) = vals[i];

    delete vals;

    v.setColNames(getFloatingSpeciesIds());

    return v;
}

DoubleMatrix RoadRunner::getFloatingSpeciesConcentrationsNamedArray()
{
    check_model();

    int l = impl->model->getNumFloatingSpecies();

    double* vals = new double[l];
    impl->model->getFloatingSpeciesConcentrations(l, NULL, vals);

    LibStructural *ls = getLibStruct();
    DoubleMatrix v(1, l);

    for (int i = 0; i<l; ++i)
        v(0, i) = vals[i];

    delete vals;

    v.setColNames(getFloatingSpeciesIds());

    return v;
}

std::vector<double> RoadRunner::getRatesOfChange()
{
    check_model();

    int ssvl = impl->model->getStateVector(NULL);
    double* vals = new double[ssvl];
    double* ssv = new double[ssvl];
    std::vector<double> v(ssvl);

    impl->model->getStateVector(ssv);
    impl->model->getStateVectorRate(impl->model->getTime(), ssv, vals);

    if (getConservedMoietyAnalysis())
    {
        LibStructural *ls = getLibStruct();
        DoubleMatrix lm = *ls->getLinkMatrix();

        for (int i = 0; i < ssvl; ++i)
        {
            double sum = 0;
            for (int j = 0; j < ssvl; ++j)
            {
                sum += lm[i][j] * vals[j];
            }
            v[i] = sum;
        }
    }
    else
    {
        for (int i = 0; i<ssvl; ++i)
            v[i] = vals[i];
    }

    delete vals;
    delete ssv;

    return v;
}

DoubleMatrix RoadRunner::getRatesOfChangeNamedArray()
{
    check_model();

    int ssvl = impl->model->getStateVector(NULL);
    double* vals = new double[ssvl];
    double* ssv = new double[ssvl];
    DoubleMatrix v(1, ssvl);

    impl->model->getStateVector(ssv);
    impl->model->getStateVectorRate(impl->model->getTime(), ssv, vals);
    if (getConservedMoietyAnalysis())
    {
        LibStructural *ls = getLibStruct();
        DoubleMatrix lm = *ls->getLinkMatrix();

        for (int i = 0; i < ssvl; ++i)
        {
            double sum = 0;
            for (int j = 0; j < ssvl; ++j)
            {
                sum += lm[i][j] * vals[j];
            }
            v(0, i) = sum;
        }
    }
    else
    {
        for (int i = 0; i<ssvl; ++i)
            v(0, i) = vals[i];
    }

    delete vals;
    delete ssv;

    v.setColNames(getFloatingSpeciesIds());

    return v;
}

std::vector<double> RoadRunner::getIndependentRatesOfChange()
{
    check_model();

    vector<string> idfsId = getIndependentFloatingSpeciesIds();
    vector<string> fsId = getFloatingSpeciesIds();
    int nindep = idfsId.size();
    std::vector<double> v(nindep);

    std::vector<double> rate = getRatesOfChange();

    for (int i = 0; i < nindep; ++i)
    {
        vector<string>::iterator it = find(fsId.begin(), fsId.end(), idfsId[i]);
        int index = distance(fsId.begin(), it);

        v[i] = rate[index];
    }

    return v;
}

DoubleMatrix RoadRunner::getIndependentRatesOfChangeNamedArray()
{
    check_model();

    vector<string> idfsId = getIndependentFloatingSpeciesIds();
    vector<string> fsId = getFloatingSpeciesIds();
    int nindep = idfsId.size();
    DoubleMatrix v(1, nindep);

    DoubleMatrix rate = getRatesOfChangeNamedArray();

    for (int i = 0; i < nindep; ++i)
    {
        vector<string>::iterator it = find(fsId.begin(), fsId.end(), idfsId[i]);
        int index = distance(fsId.begin(), it);

        v(0, i) = rate[0][index];
    }

    v.setColNames(idfsId);

    return v;
}

std::vector<double> RoadRunner::getDependentRatesOfChange()
{
    check_model();

    vector<string> dfsId = getDependentFloatingSpeciesIds();
    vector<string> fsId = getFloatingSpeciesIds();
    int ndep = dfsId.size();
    std::vector<double> v(ndep);

    std::vector<double> rate = getRatesOfChange();

    for (int i = 0; i < ndep; ++i)
    {
        vector<string>::iterator it = find(fsId.begin(), fsId.end(), dfsId[i]);
        int index = distance(fsId.begin(), it);

        v[i] = rate[index];
    }

    return v;
}

DoubleMatrix RoadRunner::getDependentRatesOfChangeNamedArray()
{
    check_model();

    vector<string> dfsId = getDependentFloatingSpeciesIds();
    vector<string> fsId = getFloatingSpeciesIds();
    int ndep = dfsId.size();
    DoubleMatrix v(1, ndep);

    DoubleMatrix rate = getRatesOfChangeNamedArray();

    for (int i = 0; i < ndep; ++i)
    {
        vector<string>::iterator it = find(fsId.begin(), fsId.end(), dfsId[i]);
        int index = distance(fsId.begin(), it);

        v(0, i) = rate[0][index];
    }

    v.setColNames(dfsId);

    return v;
}

DoubleMatrix RoadRunner::getFullJacobian()
{
    check_model();

    get_self();

    if (self.model->getNumReactions() == 0 && self.model->getNumRateRules() > 0)
    {
        DoubleMatrix jac(self.model->getNumRateRules(), self.model->getNumRateRules());

        for (int i = 0; i < self.model->getNumRateRules(); i++)
        {
            for (int j = 0; j < self.model->getNumRateRules(); j++)
            {
                // function pointers to the model get values and get init values based on
                // if we are doing amounts or concentrations.
                typedef int (ExecutableModel::*GetValueFuncPtr)(int len, int const *indx,
                    double *values);
                typedef int (ExecutableModel::*SetValueFuncPtr)(int len, int const *indx,
                    double const *values);

                GetValueFuncPtr getValuePtr = 0;
                GetValueFuncPtr getInitValuePtr = 0;
                SetValueFuncPtr setValuePtr = 0;
                SetValueFuncPtr setInitValuePtr = 0;

                if (Config::getValue(Config::ROADRUNNER_JACOBIAN_MODE).convert<unsigned>()
                    == Config::ROADRUNNER_JACOBIAN_MODE_AMOUNTS)
                {
                    getValuePtr = &ExecutableModel::getFloatingSpeciesAmounts;
                    getInitValuePtr = &ExecutableModel::getFloatingSpeciesInitAmounts;
                    setValuePtr = &ExecutableModel::setFloatingSpeciesAmounts;
                    setInitValuePtr = &ExecutableModel::setFloatingSpeciesInitAmounts;
                }
                else
                {
                    getValuePtr = &ExecutableModel::getFloatingSpeciesConcentrations;
                    getInitValuePtr = &ExecutableModel::getFloatingSpeciesInitConcentrations;
                    setValuePtr = &ExecutableModel::setFloatingSpeciesConcentrations;
                    setInitValuePtr = &ExecutableModel::setFloatingSpeciesInitConcentrations;
                }

                double value;
                double originalConc = 0;
                double result = std::numeric_limits<double>::quiet_NaN();

                // note setting init values auotmatically sets the current values to the
                // init values

                // this causes a reset, so need to save the current amounts to set them back
                // as init conditions.
                std::vector<double> conc(self.model->getNumFloatingSpecies());
                (self.model->*getValuePtr)(conc.size(), 0, &conc[0]);

                // save the original init values
                std::vector<double> initConc(self.model->getNumFloatingSpecies());
                (self.model->*getInitValuePtr)(initConc.size(), 0, &initConc[0]);

                // get the original value
                (self.model->*getValuePtr)(1, &j, &originalConc);

                // now we start changing things
                try
                {
                    // set init amounts to current amounts, restore them later.
                    // have to do this as this is only way to set conserved moiety values
                    (self.model->*setInitValuePtr)(conc.size(), 0, &conc[0]);

                    // sanity check
                    assert_similar(originalConc, conc[j]);
                    double tmp = 0;
                    (self.model->*getInitValuePtr)(1, &j, &tmp);
                    assert_similar(originalConc, tmp);
                    (self.model->*getValuePtr)(1, &j, &tmp);
                    assert_similar(originalConc, tmp);

                    // things check out, start fiddling...
                    double hstep = self.mDiffStepSize*originalConc;
                    if (fabs(hstep) < 1E-12)
                    {
                        hstep = self.mDiffStepSize;
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

                    result = 1 / (12 * hstep)*(f1 + f2);
                }
                catch (const std::exception& e)
                {
                    // What ever happens, make sure we restore the species level
                    (self.model->*setInitValuePtr)(
                        initConc.size(), 0, &initConc[0]);

                    // only set the indep species, setting dep species is not permitted.
                    (self.model->*setValuePtr)(
                        self.model->getNumFloatingSpecies(), 0, &conc[0]);

                    // re-throw the exception.
                    throw;
                }

                // What ever happens, make sure we restore the species level
                (self.model->*setInitValuePtr)(
                    initConc.size(), 0, &initConc[0]);

                // only set the indep species, setting dep species is not permitted.
                (self.model->*setValuePtr)(
                    self.model->getNumFloatingSpecies(), 0, &conc[0]);

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
    }
    else
    {
        DoubleMatrix uelast = getUnscaledElasticityMatrix();

        // ptr to libstruct owned obj.
        DoubleMatrix *rsm;
        LibStructural *ls = getLibStruct();
        if (self.loadOpt.getConservedMoietyConversion())
        {
            rsm = ls->getReorderedStoichiometryMatrix();
        }
        else
        {
            rsm = ls->getStoichiometryMatrix();
        }

        DoubleMatrix jac = ls::mult(*rsm, uelast);

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

DoubleMatrix RoadRunner::getFullReorderedJacobian()
{
    check_model();

    LibStructural *ls = getLibStruct();
    DoubleMatrix uelast = getUnscaledElasticityMatrix();
    DoubleMatrix *rsm = ls->getStoichiometryMatrix();
    return mult(*rsm, uelast);
}

DoubleMatrix RoadRunner::getReducedJacobian(double h)
{   
    get_self();

    check_model();

    if (h <= 0)
    {
        h = self.roadRunnerOptions.jacobianStepSize;
    }

    int nIndSpecies = self.model->getNumIndFloatingSpecies();

    // result matrix
    DoubleMatrix jac(nIndSpecies, nIndSpecies);

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

    double* dy0 = &dy0v[0];
    double* dy1 = &dy1v[0];

    // function pointers to the model get values and get init values based on
    // if we are doing amounts or concentrations.
    typedef int (ExecutableModel::*GetValueFuncPtr)(int len, int const *indx,
            double *values);
    typedef int (ExecutableModel::*SetValueFuncPtr)(int len, int const *indx,
                    double const *values);

    GetValueFuncPtr getValuePtr = 0;
    GetValueFuncPtr getRateValuePtr = 0;
    SetValueFuncPtr setValuePtr = 0;

    if (Config::getValue(Config::ROADRUNNER_JACOBIAN_MODE).convert<unsigned>()
            == Config::ROADRUNNER_JACOBIAN_MODE_AMOUNTS)
    {
        Log(Logger::LOG_DEBUG) << "getReducedJacobian in AMOUNT mode";
        getValuePtr =     &ExecutableModel::getFloatingSpeciesAmounts;
        getRateValuePtr = &ExecutableModel::getFloatingSpeciesAmountRates;
        setValuePtr =     &ExecutableModel::setFloatingSpeciesAmounts;
    }
    else
    {
        Log(Logger::LOG_DEBUG) << "getReducedJacobian in CONCENTRATION mode";
        getValuePtr =     &ExecutableModel::getFloatingSpeciesConcentrations;
        getRateValuePtr = &ExecutableModel::getFloatingSpeciesAmountRates;
        setValuePtr =     &ExecutableModel::setFloatingSpeciesConcentrations;
    }

    for (int i = 0; i < nIndSpecies; ++i)
    {
        double savedVal = 0;
        double y = 0;

        // current value of species i
        (self.model->*getValuePtr)(1, &i, &savedVal);

        // get the entire rate of change for all the species with
        // species i being value(i) + h;
        y = savedVal + h;
        (self.model->*setValuePtr)(1, &i, &y);
        (self.model->*getRateValuePtr)(nIndSpecies, 0, dy0);


        // get the entire rate of change for all the species with
        // species i being value(i) - h;
        y = savedVal - h;
        (self.model->*setValuePtr)(1, &i, &y);
        (self.model->*getRateValuePtr)(nIndSpecies, 0, dy1);

        // restore original value
        (self.model->*setValuePtr)(1, &i, &savedVal);

        // matrix is row-major, so have to copy by elements
       for (int j = 0; j < nIndSpecies; ++j)
       {
           jac(j, i) = (dy0[j] - dy1[j]) / (2.0*h) ;
       }
    }
    return jac;
}

DoubleMatrix RoadRunner::getLinkMatrix()
{
    check_model();

    ls::LibStructural *ls = getLibStruct();

    // pointer to owned matrix
    DoubleMatrix res = *ls->getLinkMatrix();

    ls->getLinkMatrixLabels(res.getRowNames(), res.getColNames());

    return res;
}

DoubleMatrix RoadRunner::getReducedStoichiometryMatrix()
{
    return getNrMatrix();
}

DoubleMatrix RoadRunner::getNrMatrix()
{
    check_model();

    ls::LibStructural *ls = getLibStruct();

    // pointer to owned matrix
    DoubleMatrix res = *ls->getNrMatrix();

    ls->getNrMatrixLabels(res.getRowNames(), res.getColNames());

    return res;
}

DoubleMatrix RoadRunner::getKMatrix()
{
    check_model();

    ls::LibStructural *ls = getLibStruct();

    // pointer to owned matrix
    DoubleMatrix res = *ls->getKMatrix();

    ls->getKMatrixLabels(res.getRowNames(), res.getColNames());

    return res;
}


DoubleMatrix RoadRunner::getFullStoichiometryMatrix()
{
    check_model();
    get_self();
    ls::LibStructural *ls = getLibStruct();

    if (self.loadOpt.getConservedMoietyConversion()) {
        // pointer to mat owned by ls
        DoubleMatrix m = *ls->getReorderedStoichiometryMatrix();
        ls->getReorderedStoichiometryMatrixLabels(
                m.getRowNames(), m.getColNames());
        return m;
    }

    // pointer to owned matrix
    DoubleMatrix *mptr = ls->getStoichiometryMatrix();
    if (!mptr)
    {
        throw CoreException("Error: Stoichiometry matrix does not exist for this model");
    }
    DoubleMatrix m = *mptr;
    ls->getStoichiometryMatrixLabels(m.getRowNames(), m.getColNames());
    return m;
}


DoubleMatrix RoadRunner::getExtendedStoichiometryMatrix()
{
    check_model();
    get_self();
    ls::LibStructural *ls = getLibStruct();

    if (self.loadOpt.getConservedMoietyConversion()) {
        // pointer to mat owned by ls
        DoubleMatrix m = *ls->getReorderedStoichiometryMatrix();
        ls->getReorderedStoichiometryMatrixLabels(
                m.getRowNames(), m.getColNames());
        return m;
    }

    // pointer to owned matrix
    DoubleMatrix *mptr = ls->getStoichiometryMatrix();
    if (!mptr)
    {
        throw CoreException("Error: Stoichiometry matrix does not exist for this model");
    }
    DoubleMatrix m = *mptr;
    ls->getStoichiometryMatrixLabels(m.getRowNames(), m.getColNames());

    libsbml::SBMLReader reader;
    libsbml::SBMLDocument *doc = reader.readSBMLFromString(self.mCurrentSBML);
    libsbml::Model *model = doc->getModel();
    typedef cxx11_ns::unordered_map<int,int> RxnIdxToRowMap;
    typedef cxx11_ns::unordered_map<int,libsbml::Reaction*> RxnIdxToRxnMap;
    typedef cxx11_ns::unordered_map<libsbml::Species*,int> SpcToRowMap;
    RxnIdxToRowMap missing_rct_row_map;
    RxnIdxToRowMap missing_prd_row_map;
    RxnIdxToRxnMap rxn_map;
    SpcToRowMap boundary_spc_row_map;

    int n_rows = m.numRows();

    for (int i=0; i<m.getColNames().size(); ++i) {
      libsbml::Reaction* r = model->getReaction(m.getColNames().at(i));
      assert(r);
      rxn_map[i] = r;
      if (!r->getNumReactants()) {
        missing_rct_row_map[i] = n_rows++;
      } else {
        for (int j=0; j<r->getNumReactants(); ++j) {
          libsbml::SpeciesReference* ref = r->getReactant(j);
          assert(ref);
          libsbml::Species* s = model->getSpecies(ref->getSpecies());
          assert(s);
          if (s->getBoundaryCondition() && boundary_spc_row_map.find(s) == boundary_spc_row_map.end())
            boundary_spc_row_map[s] = n_rows++;
        }
      }
      if (!r->getNumProducts()) {
        missing_prd_row_map[i] = n_rows++;
      } else {
        for (int j=0; j<r->getNumProducts(); ++j) {
          libsbml::SpeciesReference* ref = r->getProduct(j);
          assert(ref);
          libsbml::Species* s = model->getSpecies(ref->getSpecies());
          assert(s);
          if (s->getBoundaryCondition() && boundary_spc_row_map.find(s) == boundary_spc_row_map.end())
            boundary_spc_row_map[s] = n_rows++;
        }
      }
    }

    DoubleMatrix extended_matrix(n_rows, m.numCols());
    extended_matrix.setRowNames(m.getRowNames());
    extended_matrix.setColNames(m.getColNames());
    extended_matrix.getRowNames().resize(n_rows);
    for (int i=0; i<m.numRows(); ++i) {
      for (int j=0; j<m.numCols(); ++j) {
        extended_matrix(i,j) = m(i,j);
      }
    }
    for (int i=m.numRows(); i<n_rows; ++i) {
      for (int j=0; j<m.numCols(); ++j) {
        extended_matrix(i,j) = 0;
      }
    }
    for (RxnIdxToRowMap::const_iterator i=missing_rct_row_map.begin(); i!=missing_rct_row_map.end(); ++i){
      extended_matrix(i->second,i->first) = -1;
      RxnIdxToRxnMap::const_iterator r = rxn_map.find(i->first);
      if (r != rxn_map.end())
        extended_matrix.getRowNames().at(i->second) = r->second->getId() + "_source";
    }
    for (RxnIdxToRowMap::const_iterator i=missing_prd_row_map.begin(); i!=missing_prd_row_map.end(); ++i){
      extended_matrix(i->second,i->first) = 1;
      RxnIdxToRxnMap::const_iterator r = rxn_map.find(i->first);
      if (r != rxn_map.end())
        extended_matrix.getRowNames().at(i->second) = r->second->getId() + "_sink";
    }
    for (SpcToRowMap::const_iterator i=boundary_spc_row_map.begin(); i!=boundary_spc_row_map.end(); ++i) {
      for (int j=0; j<m.getColNames().size(); ++j) {
        libsbml::Reaction* r = model->getReaction(m.getColNames().at(j));
        assert(r);
        for (int k=0; k<r->getNumReactants(); ++k) {
          libsbml::SpeciesReference* ref = r->getReactant(k);
          assert(ref);
          libsbml::Species* s = model->getSpecies(ref->getSpecies());
          assert(s);
          if (s == i->first) {
            extended_matrix(i->second,j) = -1;
            extended_matrix.getRowNames().at(i->second) = s->getId();
          }
        }
        for (int k=0; k<r->getNumProducts(); ++k) {
          libsbml::SpeciesReference* ref = r->getProduct(k);
          assert(ref);
          libsbml::Species* s = model->getSpecies(ref->getSpecies());
          assert(s);
          if (s == i->first) {
            extended_matrix(i->second,j) = 1;
            extended_matrix.getRowNames().at(i->second) = s->getId();
          }
        }
      }
    }

    return extended_matrix;
}


DoubleMatrix RoadRunner::getL0Matrix()
{
    check_model();
    ls::LibStructural *ls = getLibStruct();

    // the libstruct getL0Matrix returns a NEW matrix,
    // nice consistent API yes?!?!?
    DoubleMatrix *tmp = ls->getL0Matrix();
    DoubleMatrix m = *tmp;
    delete tmp;

    ls->getL0MatrixLabels(m.getRowNames(), m.getColNames());
    return m;
}


DoubleMatrix RoadRunner::getConservationMatrix()
{
    if (!getConservedMoietyAnalysis())
        throw CoreException("Cannot compute conservation matrix because conserved moieties are not enabled");

    DoubleMatrix mat;

    try
    {
       if (impl->model)
       {
           DoubleMatrix* aMat = getLibStruct()->getGammaMatrix();
            if (aMat)
            {
                mat.resize(aMat->numRows(), aMat->numCols());
                for(int row = 0; row < mat.RSize(); row++)
                {
                    for(int col = 0; col < mat.CSize(); col++)
                    {
                        mat(row,col) = (*aMat)(row,col);
                    }
                }

                mat.setRowNames(getConservedMoietyIds());
                mat.setColNames(getLibStruct()->getReorderedSpecies());
            }
            return mat;

       }
       throw CoreException(gEmptyModelMessage);
    }
    catch (const Exception& e)
    {
         throw CoreException("Unexpected error from getConservationMatrix(): ", e.Message());
    }
}

// Help("Returns the number of dependent species in the model")
int RoadRunner::getNumberOfDependentSpecies()
{
    try
    {
        if (impl->model)
        {
            //return mStructAnalysis.GetInstance()->getNumDepSpecies();
            return getLibStruct()->getNumDepSpecies();
        }

        throw CoreException(gEmptyModelMessage);
    }

    catch(Exception &e)
    {
        throw CoreException("Unexpected error from getNumberOfDependentSpecies(): ", e.Message());
    }
}

// Help("Returns the number of independent species in the model")
int RoadRunner::getNumberOfIndependentSpecies()
{
    try
    {
        if (impl->model)
        {
            return getLibStruct()->getNumIndSpecies();
        }
        //return StructAnalysis.getNumIndSpecies();
        throw CoreException(gEmptyModelMessage);
    }
    catch (Exception &e)
    {
        throw CoreException("Unexpected error from getNumberOfIndependentSpecies(): ", e.Message());
    }
}

double RoadRunner::getVariableValue(const VariableType variableType,
        const int variableIndex)
{
    switch (variableType)
    {
    case vtFlux:
    {
        double result = 0;
        impl->model->getReactionRates(1, &variableIndex, &result);
        return result;
    }
    break;

    case vtSpecies:
    {
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

int RoadRunner::createDefaultSteadyStateSelectionList()
{
	impl->mSteadyStateSelection.clear();
	// default should be independent floating species only ...
	vector<string> floatingSpecies = getFloatingSpeciesIds();
	int numFloatingSpecies = floatingSpecies.size();
	//int numIndSpecies = getNumberOfIndependentSpecies();
	//impl->mSteadyStateSelection.resize(numIndSpecies);
	impl->mSteadyStateSelection.resize(numFloatingSpecies);
	//for (int i = 0; i < numIndSpecies; i++)
	for (int i = 0; i < numFloatingSpecies; i++)
	{
		SelectionRecord aRec;
		aRec.selectionType = SelectionRecord::FLOATING_CONCENTRATION;
		aRec.p1 = floatingSpecies[i];
		aRec.index = i;
		impl->mSteadyStateSelection[i] = aRec;
	}
	return impl->mSteadyStateSelection.size();
}

vector<SelectionRecord>& RoadRunner::getSteadyStateSelections()
{
    return impl->mSteadyStateSelection;
}

vector<double> RoadRunner::getSteadyStateValues()
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }
    if(impl->mSteadyStateSelection.size() == 0)
    {
        createDefaultSteadyStateSelectionList();
    }

    steadyState();

    vector<double> result; //= new double[oSelection.Length];
    for (int i = 0; i < impl->mSteadyStateSelection.size(); i++)
    {
        result.push_back(getValue(impl->mSteadyStateSelection[i]));
    }
    return result;
}

std::vector<std::string> RoadRunner::getSteadyStateSelectionStrings() const
{
    std::vector<std::string> r;

    for (int i = 0; i < impl->mSteadyStateSelection.size(); i++)
    {
        r.push_back(impl->mSteadyStateSelection[i].to_string());
    }

    return r;
}

DoubleMatrix RoadRunner::getSteadyStateValuesNamedArray()
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }
    if(impl->mSteadyStateSelection.size() == 0)
    {
        createDefaultSteadyStateSelectionList();
    }

    steadyState();

    DoubleMatrix v(1,impl->mSteadyStateSelection.size());
    for (int i = 0; i < impl->mSteadyStateSelection.size(); i++)
    {
        v(0,i) = getValue(impl->mSteadyStateSelection[i]);
    }

    v.setColNames(getSteadyStateSelectionStrings());

    return v;
}

string RoadRunner::getModelName()
{
    return impl->model ? impl->model->getModelName() : string("");
}

/**
 * find an symbol id in the model and set its value.
 */
static void setSBMLValue(libsbml::Model* model, const string& id, double value)
{
    if (model == NULL)
    {
        throw Exception("You need to load the model first");
    }

    libsbml::Species* oSpecies = model->getSpecies(id);
    if (oSpecies != NULL)
    {
        if (oSpecies->isSetInitialAmount())
            oSpecies->setInitialAmount(value);
        else
            oSpecies->setInitialConcentration(value);
        return;
    }

    libsbml::Compartment* oCompartment = model->getCompartment(id);
    if (oCompartment != NULL)
    {
        oCompartment->setVolume(value); return;
    }

    for (int i = 0; i < model->getNumReactions(); i++)
    {
        libsbml::Reaction* reaction = model->getReaction(i);
        for (int j = 0; j < reaction->getNumReactants(); j++)
        {
            libsbml::SpeciesReference* reference = reaction->getReactant(j);
            if (reference->isSetId() && reference->getId() == id)
            {
                reference->setStoichiometry(value);
                return;
            }
        }
        for (int j = 0; j < reaction->getNumProducts(); j++)
        {
            libsbml::SpeciesReference* reference = reaction->getProduct(j);
            if (reference->isSetId() && reference->getId() == id)
            {
                reference->setStoichiometry(value);
                return;
            }
        }
    }

    throw Exception("Invalid string name. The id '" + id + "' does not exist in the model");
}




// Help("Get the number of reactions")
int RoadRunner::getNumberOfReactions()
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }
    return impl->model->getNumReactions();
}

// Help("Returns the rate of a reaction by its index")
double RoadRunner::getReactionRate(const int& index)
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }

    if ((index >= 0) && (index < impl->model->getNumReactions()))
    {
        double result = 0;
        impl->model->getReactionRates(1, &index, &result);
        return result;
    }
    else
    {
        throw CoreException(format("Index in getReactionRate out of range: [{0}]", index));
    }
}


double RoadRunner::getRateOfChange(const int& index)
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }

    if ((index >= 0) && (index < impl->model->getNumFloatingSpecies()))
    {
        double value = 0;
        impl->model->getFloatingSpeciesAmountRates(1, &index, &value);
        return value;
    }

    throw CoreException(format("Index in getRateOfChange out of range: [{0}]", index));
}


// Help("Get the number of compartments")
int RoadRunner::getNumberOfCompartments()
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }
    return impl->model->getNumCompartments();
}


void RoadRunner::setCompartmentByIndex(const int& index, const double& value)
{
    if (!impl->model)
    {
         throw CoreException(gEmptyModelMessage);
    }

    if ((index >= 0) && (index < impl->model->getNumCompartments()))
    {
        impl->model->setCompartmentVolumes(1, &index, &value);
    }
    else
    {
        throw CoreException(format("Index in getCompartmentByIndex out of range: [{0}]", index));
    }
}

double RoadRunner::getCompartmentByIndex(const int& index)
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }

    if ((index >= 0) && (index < impl->model->getNumCompartments()))
    {
        double result = 0;
        impl->model->getCompartmentVolumes(1, &index, &result);
        return result;
    }

    throw CoreException(format("Index in getCompartmentByIndex out of range: [{0}]", index));
}

int RoadRunner::getNumberOfBoundarySpecies()
{
    if (!impl->model)
    {
        throw Exception(gEmptyModelMessage);
    }
    return impl->model->getNumBoundarySpecies();
}

// Help("Sets the value of a boundary species by its index")
void RoadRunner::setBoundarySpeciesByIndex(const int& index, const double& value)
{
    if (!impl->model)
    {
        throw Exception(gEmptyModelMessage);
    }

    if ((index >= 0) && (index < impl->model->getNumBoundarySpecies()))
    {
        impl->model->setBoundarySpeciesConcentrations(1, &index, &value);
    }
    else
    {
        throw Exception(format("Index in getBoundarySpeciesByIndex out of range: [{0}]", index));
    }
}

// Help("Returns the value of a boundary species by its index")
double RoadRunner::getBoundarySpeciesByIndex(const int& index)
{
    if (!impl->model)
    {
        throw Exception(gEmptyModelMessage);
    }
    if ((index >= 0) && (index < impl->model->getNumBoundarySpecies()))
    {
        double result = 0;
        impl->model->getBoundarySpeciesConcentrations(1, &index, &result);
        return result;
    }
    throw Exception(format("Index in getBoundarySpeciesByIndex out of range: [{0}]", index));
}


DoubleMatrix RoadRunner::getBoundarySpeciesConcentrationsNamedArray()
{
    check_model();

    int l = impl->model->getNumBoundarySpecies();

    double* vals = new double[l];
    impl->model->getBoundarySpeciesConcentrations(l, NULL, vals);

    LibStructural *ls = getLibStruct();
    DoubleMatrix v(1, l);

    for (int i = 0; i<l; ++i)
        v(0, i) = vals[i];

    delete vals;

    v.setColNames(getBoundarySpeciesIds());

    return v;
}


DoubleMatrix RoadRunner::getBoundarySpeciesAmountsNamedArray()
{
    check_model();

    int l = impl->model->getNumBoundarySpecies();

    double* vals = new double[l];
    impl->model->getBoundarySpeciesAmounts(l, NULL, vals);

    LibStructural *ls = getLibStruct();
    DoubleMatrix v(1, l);

    for (int i = 0; i<l; ++i)
        v(0, i) = vals[i];

    delete vals;

    v.setColNames(getBoundarySpeciesIds());

    return v;
}


// Help("Get the number of floating species")
int RoadRunner::getNumberOfFloatingSpecies()
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }
    return impl->model->getNumFloatingSpecies();
}


// Help("Sets the value of a floating species by its index")
void RoadRunner::setFloatingSpeciesInitialConcentrationByIndex(const int& index, const double& value)
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }

    if ((index >= 0) && (index < impl->model->getNumFloatingSpecies()))
    {
        impl->model->setFloatingSpeciesInitConcentrations(1, &index, &value);
        reset();
    }
    else
    {
        throw CoreException(format("Index in setFloatingSpeciesInitialConcentrationByIndex out of range: [{0}]", index));
    }
}

// Help("Sets the value of a floating species by its index")
void RoadRunner::setFloatingSpeciesByIndex(int index, double value)
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }

    if ((index >= 0) && (index < impl->model->getNumFloatingSpecies()))
    {
        impl->model->setFloatingSpeciesConcentrations(1, &index, &value); // This updates the amount vector aswell
    }
    else
    {
        throw CoreException(format("Index in setFloatingSpeciesByIndex out of range: [{0}]", index));
    }
}

// Help("Returns the value of a floating species by its index")
double RoadRunner::getFloatingSpeciesByIndex(const int index)
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }

    if ((index >= 0) && (index < impl->model->getNumFloatingSpecies()))
    {
        double result = 0;
        impl->model->getFloatingSpeciesConcentrations(1, &index, &result);
        return result;
    }
    throw CoreException(format("Index in getFloatingSpeciesByIndex out of range: [{0}]", index));
}

// Help("Returns an array of floating species concentrations")
vector<double> RoadRunner::getFloatingSpeciesConcentrationsV()
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }


    vector<double> result(impl->model->getNumFloatingSpecies(), 0);
    impl->model->getFloatingSpeciesConcentrations(result.size(), 0, &result[0]);
    return result;
}

// Help("Returns an array of floating species amounts")
vector<double> RoadRunner::getFloatingSpeciesAmountsV()
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }


    vector<double> result(impl->model->getNumFloatingSpecies(), 0);
    impl->model->getFloatingSpeciesAmounts(result.size(), 0, &result[0]);
    return result;
}

// Help("returns an array of floating species initial conditions")
vector<double> RoadRunner::getFloatingSpeciesInitialConcentrations()
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }
    vector<double> initYs(impl->model->getNumFloatingSpecies());
    impl->model->getFloatingSpeciesInitConcentrations(initYs.size(), 0, &initYs[0]);
    return initYs;
}

// Help("Sets the initial conditions for all floating species in the model")
void RoadRunner::setFloatingSpeciesInitialConcentrations(const vector<double>& values)
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }

    impl->model->setFloatingSpeciesConcentrations(values.size(), 0, &values[0]);
    impl->model->setFloatingSpeciesInitConcentrations(values.size(), 0, &values[0]);

    reset();
}

// Help("Set the concentrations for all floating species in the model")
void RoadRunner::setFloatingSpeciesConcentrations(const vector<double>& values)
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }

    impl->model->setFloatingSpeciesConcentrations(values.size(), 0, &values[0]);
}

vector<double> RoadRunner::getBoundarySpeciesConcentrationsV()
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }


    vector<double> result(impl->model->getNumBoundarySpecies(), 0);
    impl->model->getBoundarySpeciesConcentrations(result.size(), 0, &result[0]);
    return result;
}

vector<double> RoadRunner::getBoundarySpeciesAmountsV()
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }


    vector<double> result(impl->model->getNumBoundarySpecies(), 0);
    impl->model->getBoundarySpeciesAmounts(result.size(), 0, &result[0]);
    return result;
}


// Help("Set the concentrations for all floating species in the model")
void RoadRunner::setBoundarySpeciesConcentrations(const vector<double>& values)
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }

    impl->model->setBoundarySpeciesConcentrations(values.size(), 0, &values[0]);
}




// Help("Get the number of global parameters")
int RoadRunner::getNumberOfGlobalParameters()
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }
    return getGlobalParameterIds().size();
}

void RoadRunner::setGlobalParameterByIndex(const int index, const double value)
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }

    impl->model->setGlobalParameterValues(1, &index, &value);
}

// Help("Returns the value of a global parameter by its index")
double RoadRunner::getGlobalParameterByIndex(const int& index)
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }

    if ((index >= 0) && (index < (impl->model->getNumGlobalParameters() + impl->model->getNumDepFloatingSpecies())))
    {
        int arraySize = impl->model->getNumGlobalParameters() + impl->model->getNumDepFloatingSpecies();
        if (impl->model->getNumReactions() == 0 && impl->model->getNumRateRules() > 0)
        {
            int arraySize = impl->model->getNumGlobalParameters();
        }
        double* data = new double[arraySize];

        if (impl->model->getNumReactions() == 0 && impl->model->getNumRateRules() > 0)
        {
            impl->model->getGlobalParameterValues(impl->model->getNumGlobalParameters(), 0, data);
        }
        else
        {
            impl->model->getGlobalParameterValues(impl->model->getNumGlobalParameters(), 0, data);

            impl->model->getConservedMoietyValues(impl->model->getNumDepFloatingSpecies(), 0, data + impl->model->getNumGlobalParameters());
        }        

        double result = data[index];
        delete[] data;
        return result;
    }

    throw CoreException(format("Index in getNumGlobalParameters out of range: [{0}]", index));
}

// Help("Get the values for all global parameters in the model")
vector<double> RoadRunner::getGlobalParameterValues()
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }

    if (impl->model->getNumDepFloatingSpecies() > 0)
    {
        vector<double> result(impl->model->getNumGlobalParameters() +
                impl->model->getNumDepFloatingSpecies());

        impl->model->getGlobalParameterValues(
                impl->model->getNumGlobalParameters(), 0, &result[0]);

        impl->model->getConservedMoietyValues(
                impl->model->getNumDepFloatingSpecies(), 0,
                &result[impl->model->getNumGlobalParameters()]);

        return result;
    }

    vector<double> result(impl->model->getNumGlobalParameters());
    impl->model->getGlobalParameterValues(result.size(), 0, &result[0]);
    return result;
}




//---------------- MCA functions......
//        [Help("Get unscaled control coefficient with respect to a global parameter")]
double RoadRunner::getuCC(const string& variableName, const string& parameterName)
{
    try
    {
        if (!impl->model)
        {
            throw CoreException(gEmptyModelMessage);
        }

        mcaSteadyState();

        ParameterType parameterType;
        VariableType variableType;
        double originalParameterValue;
        int variableIndex;
        int parameterIndex;

        std::string variableNameMod = variableName;

        variableNameMod.erase(std::remove(variableNameMod.begin(), variableNameMod.end(), '['), variableNameMod.end());
        variableNameMod.erase(std::remove(variableNameMod.begin(), variableNameMod.end(), ']'), variableNameMod.end());

        // Check the variable name
        if ((variableIndex = impl->model->getReactionIndex(variableNameMod)) >= 0)
        {
            variableType = vtFlux;
        }
        else if ((variableIndex = impl->model->getFloatingSpeciesIndex(variableNameMod)) >= 0)
        {
            variableType = vtSpecies;
        }
        else
        {
            throw CoreException("Unable to locate variable: [" + variableNameMod + "]");
        }

        // Check for the parameter name
        if ((parameterIndex = impl->model->getGlobalParameterIndex(parameterName)) >= 0)
        {
            parameterType = ptGlobalParameter;
            originalParameterValue = 0;
            impl->model->getGlobalParameterValues(1, &parameterIndex, &originalParameterValue);
        }
        else if ((parameterIndex = impl->model->getBoundarySpeciesIndex(parameterName)) >= 0)
        {
            parameterType = ptBoundaryParameter;
            originalParameterValue = 0;
            impl->model->getBoundarySpeciesConcentrations(1, &parameterIndex, &originalParameterValue);
        }
        else if ((parameterIndex = impl->model->getConservedMoietyIndex(parameterName)) >= 0)
        {
            parameterType = ptConservationParameter;
            originalParameterValue = 0;
            impl->model->getConservedMoietyValues(1, &parameterIndex, &originalParameterValue);
        }
        else
        {
            throw CoreException("Unable to locate parameter: [" + parameterName + "]");
        }

        // Get the original parameter value
        originalParameterValue = impl->getParameterValue(parameterType, parameterIndex);

        double hstep = impl->mDiffStepSize*originalParameterValue;
        if (fabs(hstep) < 1E-12)
        {
            hstep = impl->mDiffStepSize;
        }

        try
        {
            impl->setParameterValue(parameterType, parameterIndex, originalParameterValue + hstep);
            mcaSteadyState();
            double fi = getVariableValue(variableType, variableIndex);

            impl->setParameterValue(parameterType, parameterIndex, originalParameterValue + 2*hstep);
            mcaSteadyState();
            double fi2 = getVariableValue(variableType, variableIndex);

            impl->setParameterValue(parameterType, parameterIndex, originalParameterValue - hstep);
            mcaSteadyState();
            double fd = getVariableValue(variableType, variableIndex);

            impl->setParameterValue(parameterType, parameterIndex, originalParameterValue - 2*hstep);
            mcaSteadyState();
            double fd2 = getVariableValue(variableType, variableIndex);

            // Use instead the 5th order approximation double unscaledValue = (0.5/hstep)*(fi-fd);
            // The following separated lines avoid small amounts of roundoff error
            double f1 = fd2 + 8*fi;
            double f2 = -(8*fd + fi2);

            // What ever happens, make sure we restore the parameter level
            impl->setParameterValue(parameterType, parameterIndex, originalParameterValue);
            steadyState();

            return 1/(12*hstep)*(f1 + f2);
        }
        catch(...) //Catch anything... and do 'finalize'
        {
            // What ever happens, make sure we restore the parameter level
            impl->setParameterValue(parameterType, parameterIndex, originalParameterValue);
            mcaSteadyState();
            throw;
        }
    }
    catch (const Exception& e)
    {
        throw CoreException("Unexpected error from getuCC(): ", e.Message());
    }
}

//        [Help("Get scaled control coefficient with respect to a global parameter")]
double RoadRunner::getCC(const string& variableName, const string& parameterName)
{
    VariableType variableType;
    ParameterType parameterType;
    int variableIndex;
    int parameterIndex;

    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }

    std::string variableNameMod = variableName;

    variableNameMod.erase(std::remove(variableNameMod.begin(), variableNameMod.end(), '['), variableNameMod.end());
    variableNameMod.erase(std::remove(variableNameMod.begin(), variableNameMod.end(), ']'), variableNameMod.end());

    // Check the variable name
    if ((variableIndex = impl->model->getReactionIndex(variableNameMod)) >= 0)
    {
        variableType = vtFlux;
    }
    else if ((variableIndex = impl->model->getFloatingSpeciesIndex(variableNameMod)) >= 0)
    {
        variableType = vtSpecies;
    }
    else
    {
        throw CoreException("Unable to locate variable: [" + variableNameMod + "]");
    }

    // Check for the parameter name
    if ((parameterIndex = impl->model->getGlobalParameterIndex(parameterName)) >= 0)
    {
        parameterType = ptGlobalParameter;
    }
    else if ((parameterIndex = impl->model->getBoundarySpeciesIndex(parameterName)) >= 0)
    {
        parameterType = ptBoundaryParameter;
    }
    else if ((parameterIndex = impl->model->getConservedMoietyIndex(parameterName)) >= 0)
    {
        parameterType = ptConservationParameter;
    }
    else
    {
        throw CoreException("Unable to locate parameter: [" + parameterName + "]");
    }

    double variableValue = getVariableValue(variableType, variableIndex);
    double parameterValue = impl->getParameterValue(parameterType, parameterIndex);
    return getuCC(variableNameMod, parameterName)*parameterValue/variableValue;
}


double RoadRunner::getUnscaledSpeciesElasticity(int reactionId, int speciesIndex)
{
    get_self();

    check_model();

    // make sure no rate rules or events
    metabolicControlCheck(self.model);

    // function pointers to the model get values and get init values based on
    // if we are doing amounts or concentrations.
    typedef int (ExecutableModel::*GetValueFuncPtr)(int len, int const *indx,
            double *values);
    typedef int (ExecutableModel::*SetValueFuncPtr)(int len, int const *indx,
                    double const *values);

    GetValueFuncPtr getValuePtr = 0;
    GetValueFuncPtr getInitValuePtr = 0;
    SetValueFuncPtr setValuePtr = 0;
    SetValueFuncPtr setInitValuePtr = 0;

    if (Config::getValue(Config::ROADRUNNER_JACOBIAN_MODE).convert<unsigned>()
            == Config::ROADRUNNER_JACOBIAN_MODE_AMOUNTS)
    {
        getValuePtr = &ExecutableModel::getFloatingSpeciesAmounts;
        getInitValuePtr = &ExecutableModel::getFloatingSpeciesInitAmounts;
        setValuePtr = &ExecutableModel::setFloatingSpeciesAmounts;
        setInitValuePtr = &ExecutableModel::setFloatingSpeciesInitAmounts;
    }
    else
    {
        getValuePtr = &ExecutableModel::getFloatingSpeciesConcentrations;
        getInitValuePtr = &ExecutableModel::getFloatingSpeciesInitConcentrations;
        setValuePtr = &ExecutableModel::setFloatingSpeciesConcentrations;
        setInitValuePtr = &ExecutableModel::setFloatingSpeciesInitConcentrations;
    }

    double value;
    double originalConc = 0;
    double result = std::numeric_limits<double>::quiet_NaN();

    // note setting init values auotmatically sets the current values to the
    // init values

    // this causes a reset, so need to save the current amounts to set them back
    // as init conditions.
    std::vector<double> conc(self.model->getNumFloatingSpecies());
    (self.model->*getValuePtr)(conc.size(), 0, &conc[0]);

    // save the original init values
    std::vector<double> initConc(self.model->getNumFloatingSpecies());
    (self.model->*getInitValuePtr)(initConc.size(), 0, &initConc[0]);

    // get the original value
    (self.model->*getValuePtr)(1, &speciesIndex, &originalConc);

    // now we start changing things
    try
    {
        // set init amounts to current amounts, restore them later.
        // have to do this as this is only way to set conserved moiety values
        (self.model->*setInitValuePtr)(conc.size(), 0, &conc[0]);

        // sanity check
        assert_similar(originalConc, conc[speciesIndex]);
        double tmp = 0;
        (self.model->*getInitValuePtr)(1, &speciesIndex, &tmp);
        assert_similar(originalConc, tmp);
        (self.model->*getValuePtr)(1, &speciesIndex, &tmp);
        assert_similar(originalConc, tmp);

        // things check out, start fiddling...

        double hstep = self.mDiffStepSize*originalConc;
        if (fabs(hstep) < 1E-12)
        {
            hstep = self.mDiffStepSize;
        }

        value = originalConc + hstep;
        (self.model->*setInitValuePtr)(1, &speciesIndex, &value);

        double fi = 0;
        self.model->getReactionRates(1, &reactionId, &fi);

        value = originalConc + 2*hstep;
        (self.model->*setInitValuePtr)(1, &speciesIndex, &value);
        double fi2 = 0;
        self.model->getReactionRates(1, &reactionId, &fi2);

        value = originalConc - hstep;
        (self.model->*setInitValuePtr)(1, &speciesIndex, &value);
        double fd = 0;
        self.model->getReactionRates(1, &reactionId, &fd);

        value = originalConc - 2*hstep;
        (self.model->*setInitValuePtr)(1, &speciesIndex, &value);
        double fd2 = 0;
        self.model->getReactionRates(1, &reactionId, &fd2);

        // Use instead the 5th order approximation
        // double unscaledElasticity = (0.5/hstep)*(fi-fd);
        // The following separated lines avoid small amounts of roundoff error
        double f1 = fd2 + 8*fi;
        double f2 = -(8*fd + fi2);

        result = 1/(12*hstep)*(f1 + f2);
    }
    catch(const std::exception& e)
    {
        // What ever happens, make sure we restore the species level
        (self.model->*setInitValuePtr)(
                initConc.size(), 0, &initConc[0]);

        // only set the indep species, setting dep species is not permitted.
        (self.model->*setValuePtr)(
                self.model->getNumIndFloatingSpecies(), 0, &conc[0]);

        // re-throw the exception.
        throw;
    }

    // What ever happens, make sure we restore the species level
    (self.model->*setInitValuePtr)(
            initConc.size(), 0, &initConc[0]);

    // only set the indep species, setting dep species is not permitted.
    (self.model->*setValuePtr)(
            self.model->getNumIndFloatingSpecies(), 0, &conc[0]);

    return result;
}



DoubleMatrix RoadRunner::getUnscaledElasticityMatrix()
{
    get_self();

    check_model();

    DoubleMatrix uElastMatrix(self.model->getNumReactions(), self.model->getNumFloatingSpecies());

    uElastMatrix.setRowNames(getReactionIds());
    uElastMatrix.setColNames(getFloatingSpeciesIds());

    for (int i = 0; i < self.model->getNumReactions(); i++)
    {
        for (int j = 0; j < self.model->getNumFloatingSpecies(); j++)
        {
            uElastMatrix[i][j] = getUnscaledSpeciesElasticity(i, j);
        }
    }

    return uElastMatrix;
}

DoubleMatrix RoadRunner::getScaledElasticityMatrix()
{
    get_self();

    check_model();

    DoubleMatrix uelast = getUnscaledElasticityMatrix();
    DoubleMatrix result(uelast.RSize(), uelast.CSize());

    result.setColNames(uelast.getColNames());
    result.setRowNames(uelast.getRowNames());

    vector<double> rates(self.model->getNumReactions());
    self.model->getReactionRates(rates.size(), 0, &rates[0]);

    if (uelast.RSize() != rates.size())
    {
        // this should NEVER happen
        throw std::runtime_error("row count of unscaled elasticity different "
                "than # of reactions");
    }

    for (int i = 0; i < uelast.RSize(); i++)
    {
        for (int j = 0; j < uelast.CSize(); j++) // Columns are species
        {
            double concentration = 0;
            self.model->getFloatingSpeciesConcentrations(1, &j, &concentration);

            result[i][j] = uelast[i][j]*concentration/rates[i];
        }
    }
    return result;
}


double RoadRunner::getScaledFloatingSpeciesElasticity(const string& reactionName,
        const string& speciesName)
{
    get_self();

    check_model();

    int speciesIndex = 0;
    int reactionIndex = 0;

    if ((speciesIndex = self.model->getFloatingSpeciesIndex(speciesName)) < 0)
    {
        throw std::invalid_argument("invalid species name: " + speciesName);
    }
    if ((reactionIndex = self.model->getReactionIndex(reactionName)) < 0)
    {
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
DoubleMatrix RoadRunner::getUnscaledConcentrationControlCoefficientMatrix()
{
    get_self();

    check_model();

    //int orig_steps = impl->simulateOpt.steps;

    //impl->simulateOpt.start = 0;
    //impl->simulateOpt.duration = 50.0;
    //impl->simulateOpt.steps = 100;

    //// TODO why is simulate called here???
    //simulate();
    if (steadyState() > impl->mSteadyStateThreshold)
    {
        if (steadyState() > 1E-2)
        {
            //impl->simulateOpt.steps = orig_steps;
            throw CoreException("Unable to locate steady state during control coefficient computation");
        }
    }

    // Compute the Jacobian first
    DoubleMatrix uelast     = getUnscaledElasticityMatrix();
    DoubleMatrix Nr         = getNrMatrix();
    DoubleMatrix T1 = mult(Nr, uelast);
    DoubleMatrix LinkMatrix = getLinkMatrix();
    DoubleMatrix Jac = mult(T1, LinkMatrix);

    // Compute -Jac
    DoubleMatrix T2 = Jac * (-1.0);

    ComplexMatrix temp(T2); //Get a complex matrix from a double one. Imag part is zero
    ComplexMatrix Inv = GetInverse(temp);

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Sauro: mult which takes complex matrix need to be implemented
    DoubleMatrix T3 = mult(Inv, Nr); // Compute ( - Jac)^-1 . Nr

    // Finally include the dependent set as well.
    DoubleMatrix T4 = mult(LinkMatrix, T3); // Compute L (iwI - Jac)^-1 . Nr

    T4.setRowNames(getFloatingSpeciesIds());
    T4.setColNames(getReactionIds());

    //impl->simulateOpt.steps = orig_steps;

    return T4;
}


DoubleMatrix RoadRunner::getScaledConcentrationControlCoefficientMatrix()
{
    get_self();

    check_model();

    DoubleMatrix ucc = getUnscaledConcentrationControlCoefficientMatrix();

    if (ucc.size() > 0 )
    {
        for (int i = 0; i < ucc.RSize(); i++)
        {
            for (int j = 0; j < ucc.CSize(); j++)
            {
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
DoubleMatrix RoadRunner::getUnscaledFluxControlCoefficientMatrix()
{
    get_self();

    check_model();

    DoubleMatrix ucc = getUnscaledConcentrationControlCoefficientMatrix();
    DoubleMatrix uee = getUnscaledElasticityMatrix();

    DoubleMatrix T1 = mult(uee, ucc);

    // Add an identity matrix I to T1, that is add a 1 to every diagonal of T1
    for (int i=0; i<T1.RSize(); i++) {
        T1[i][i] = T1[i][i] + 1;
    }

    T1.setRowNames(getReactionIds());
    T1.setColNames(getReactionIds());

    return T1;
}

// [Help("Compute the matrix of scaled flux control coefficients")]
DoubleMatrix RoadRunner::getScaledFluxControlCoefficientMatrix()
{
    try
    {
        if (!impl->model)
        {
            throw CoreException(gEmptyModelMessage);
        }

        DoubleMatrix ufcc = getUnscaledFluxControlCoefficientMatrix();

        if (ufcc.RSize() > 0)
        {
            for (int i = 0; i < ufcc.RSize(); i++)
            {
                for (int j = 0; j < ufcc.CSize(); j++)
                {
                    double irate = 0;
                    impl->model->getReactionRates(1, &i, &irate);
                    if(irate !=0)
                    {
                        double jrate = 0;
                        impl->model->getReactionRates(1, &j, &jrate);
                        ufcc[i][j] = ufcc[i][j] * jrate / irate;
                    }
                    else
                    {
                        throw(Exception("Dividing with zero"));
                    }
                }
            }
        }
        return ufcc;
    }
    catch (const Exception& e)
    {
        throw CoreException("Unexpected error from getScaledFluxControlCoefficientMatrix(): ", e.Message());
    }
}

static string convertSBMLVersion(const std::string& str, int level, int version) {
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
        doc->setApplicableValidators((unsigned char)Config::getInt(
                Config::SBML_APPLICABLEVALIDATORS));

        versionConverter.setDocument(doc);

        if (versionConverter.convert() != libsbml::LIBSBML_OPERATION_SUCCESS)
        {
            Log(rr::Logger::LOG_ERROR) << "could not change source sbml level or version";

            const libsbml::SBMLErrorLog *log = doc->getErrorLog();
            string errors = log ? log->toString() : string(" NULL SBML Error Log");
            Log(rr::Logger::LOG_ERROR) << "Conversion Errors: " + errors;

            throw std::logic_error("Error version converting sbml: " + errors);
        }

        libsbml::SBMLWriter writer;
        writer.writeSBML(doc, stream);
        delete doc;
    }
    catch(std::exception& exp) {
        delete doc;
        throw;
    }

    return stream.str();
}

string RoadRunner::getSBML(int level, int version)
{
    if (level > 0) {
        return convertSBMLVersion(impl->mCurrentSBML, level, version);
    }
    return impl->mCurrentSBML;
}

string RoadRunner::getCurrentSBML(int level, int version)
{
    check_model();

    get_self();

    libsbml::SBMLReader reader;
    std::stringstream stream;
    libsbml::SBMLDocument *doc = 0;
    libsbml::Model *model = 0;

    try {
        doc = reader.readSBMLFromString(self.mCurrentSBML); // new doc
        model = doc->getModel(); // owned by doc

        vector<string> array = getFloatingSpeciesIds();
        for (int i = 0; i < array.size(); i++)
        {
            double value = 0;
            impl->model->getFloatingSpeciesAmounts(1, &i, &value);
            setSBMLValue(model, array[i], value);
        }

        array = getBoundarySpeciesIds();
        for (int i = 0; i < array.size(); i++)
        {
            double value = 0;
            impl->model->getBoundarySpeciesConcentrations(1, &i, &value);
            setSBMLValue(model, array[i], value);
        }

        array = getCompartmentIds();
        for (int i = 0; i < array.size(); i++)
        {
            double value = 0;
            impl->model->getCompartmentVolumes(1, &i, &value);
            setSBMLValue(model, array[i], value);
        }

        array = getGlobalParameterIds();
        for (int i = 0; i < impl->model->getNumGlobalParameters(); i++)
        {
            double value = 0;
            impl->model->getGlobalParameterValues(1, &i, &value);

            libsbml::Parameter* param = model->getParameter(array[i]);
            if (param != NULL)
            {
                param->setValue(value);
            }
            else
            {
                // sanity check, just make sure that this is a conserved moeity
                if (self.model->getConservedMoietyIndex(array[i]) < 0)
                {
                    throw std::logic_error("The global parameter name "
                            + array[i] + " could not be found in the SBML model, "
                            " and it is not a conserved moiety");
                }
            }
        }

        libsbml::SBMLWriter writer;
        writer.writeSBML(doc, stream);
    }
    catch(std::exception&) {
        delete doc;
        doc = 0;
        throw; // re-throw exception.
    }

    delete doc;


    if (level > 0) {
        return convertSBMLVersion(stream.str(), level, version);
    }
    return stream.str();
}

void RoadRunner::changeInitialConditions(const vector<double>& ic)
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }

    impl->model->setFloatingSpeciesConcentrations(ic.size(), 0, &ic[0]);
    impl->model->setFloatingSpeciesInitConcentrations(ic.size(), 0, &ic[0]);
}

// Help("Returns the current vector of reactions rates")
vector<double> RoadRunner::getReactionRates()
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }

    vector<double> rates(impl->model->getNumReactions());
    impl->model->getReactionRates(rates.size(), 0, &rates[0]);
    return rates;
}

Integrator* RoadRunner::getIntegrator()
{
    //applySimulateOptions();
    return impl->integrator;
}

Integrator* RoadRunner::getIntegratorByName(const std::string& name)
{
    // ensure it exists
    makeIntegrator(name);
    // find it and return
    for (std::vector<Integrator*>::iterator it = impl->integrators.begin(); it != impl->integrators.end(); ++it)
    {
        if ((*it)->getName() == name)
        {
            return *it;
        }
    }
    throw std::runtime_error("No integrator implemented for \"" + name + "\"");
}

SteadyStateSolver* RoadRunner::getSteadyStateSolver()
{
    return impl->steady_state_solver;
}

std::vector<std::string> RoadRunner::getExistingIntegratorNames()
{
    std::vector<std::string> result;
    int i = 0;
    for (std::vector<Integrator*>::iterator it = impl->integrators.begin(); it != impl->integrators.end(); ++it, ++i)
    {
        result.push_back(impl->integrators.at(i)->getName());
    }
    return result;
}

std::vector<std::string> RoadRunner::getRegisteredIntegratorNames()
{
    std::vector<std::string> result;
    for (int n = 0; n<IntegratorFactory::getInstance().getNumIntegrators(); ++n) {
        result.push_back(IntegratorFactory::getInstance().getIntegratorName(n));
    }
    return result;
}

void RoadRunner::ensureSolversRegistered()
{
    // must be run to register integrators at startup
    IntegratorRegistrationMgr::Register();
    // must be run to register solvers at startup
    SolverRegistrationMgr::Register();
}

std::vector<std::string> RoadRunner::getRegisteredSteadyStateSolverNames()
{
    std::vector<std::string> result;
    for (int n = 0; n<SteadyStateSolverFactory::getInstance().getNumSteadyStateSolvers(); ++n) {
        result.push_back(SteadyStateSolverFactory::getInstance().getSteadyStateSolverName(n));
    }
    return result;
}

void RoadRunner::setIntegrator(std::string name)
{
    Log(Logger::LOG_DEBUG) << "Setting integrator to " << name;
    // Try to set integrator from an existing reference.
    if (integratorExists(name))
    {
      int i = 0;
      for (std::vector<Integrator*>::iterator it = impl->integrators.begin(); it != impl->integrators.end(); ++it, ++i)
      {
        if (impl->integrators.at(i)->getName() == name)
        {
            Log(Logger::LOG_DEBUG) << "Using pre-existing integrator for " << name;
            impl->integrator = impl->integrators.at(i);
        }
      }
    }
    // Otherwise, create a new integrator.
    else
    {
        impl->integrator = makeIntegrator(name);
    }
}

Integrator* RoadRunner::makeIntegrator(std::string name)
{
    if (integratorExists(name)) {
        Log(Logger::LOG_DEBUG) << "Integrator \"" << name << "\" already exists";
        return NULL;
    }
    Log(Logger::LOG_DEBUG) << "Creating new integrator for " << name;
    Integrator* result = IntegratorFactory::getInstance().New(name, impl->model);
    impl->integrators.push_back(result);
    return result;
}


bool RoadRunner::integratorExists(std::string name)
{
	int i = 0;
	for (std::vector<Integrator*>::iterator it = impl->integrators.begin(); it != impl->integrators.end(); ++it, ++i)
	{
		if (impl->integrators.at(i)->getName() == name)
		{
			return true;
		}
	}
	return false;
}

void RoadRunner::setSteadyStateSolver(std::string name)
{
    Log(Logger::LOG_DEBUG) << "Setting steady state solver to " << name;
    // Try to set steady_state_solver from an existing reference.
    if (steadyStateSolverExists(name))
    {
        int i = 0;
        for (std::vector<SteadyStateSolver*>::iterator it = impl->steady_state_solvers.begin(); it != impl->steady_state_solvers.end(); ++it, ++i)
        {
            if (impl->steady_state_solvers.at(i)->getName() == name)
            {
                Log(Logger::LOG_DEBUG) << "Using pre-existing steady state solver for " << name;
                impl->steady_state_solver = impl->steady_state_solvers.at(i);
            }
        }
    }
    // Otherwise, create a new steady state solver.
    else
    {
        Log(Logger::LOG_DEBUG) << "Creating new steady state solver for " << name;
        impl->steady_state_solver = SteadyStateSolverFactory::getInstance().New(name, impl->model);
        impl->steady_state_solvers.push_back(impl->steady_state_solver);
    }
}


bool RoadRunner::steadyStateSolverExists(std::string name)
{
    int i = 0;
    for (std::vector<SteadyStateSolver*>::iterator it = impl->steady_state_solvers.begin(); it != impl->steady_state_solvers.end(); ++it, ++i)
    {
        if (impl->steady_state_solvers.at(i)->getName() == name)
        {
            return true;
        }
    }
    return false;
}


void RoadRunner::setValue(const string& sId, double dValue)
{
    check_model();

    impl->model->setValue(sId, dValue);

    SelectionRecord sel(sId);

    if (sel.selectionType == SelectionRecord::INITIAL_FLOATING_AMOUNT ||
            sel.selectionType == SelectionRecord::INITIAL_FLOATING_CONCENTRATION)
    {
        reset();
    }
}


double RoadRunner::getValue(const std::string& sel)
{
    return getValue(createSelection(sel));
}


vector<double> RoadRunner::getSelectedValues()
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }

    vector<double> result;
    result.resize(impl->mSelectionList.size());

    for (int i = 0; i < impl->mSelectionList.size(); i++)
    {
        result[i] = getNthSelectedOutput(i, impl->model->getTime());
    }
    return result;
}


static std::vector<std::string> createSelectionList(const SimulateOptions& o)
{
    //read from settings the variables found in the amounts and concentrations lists
    std::vector<std::string> result;

    // have an implied time
    result.push_back("time");

    // need to add them in order
    for(vector<string>::const_iterator var = o.variables.begin();
            var != o.variables.end(); ++var)
    {
        if (find(o.concentrations.begin(), o.concentrations.end(), *var)
                != o.concentrations.end())
        {
            result.push_back("[" + *var + "]");
        }

        else
        {
            result.push_back(*var);
        }

    }

    return result;
}

SelectionRecord RoadRunner::createSelection(const std::string& str)
{
    if (!impl->model)
    {
        throw Exception("Can not create selection without a model");
    }

    SelectionRecord sel(str);

    if (sel.selectionType == SelectionRecord::UNKNOWN)
    {
        throw Exception("invalid selection string " + str);
    }

    // check to see that we have valid selection ids
    switch(sel.selectionType)
    {
    case SelectionRecord::TIME:
        // time is always fine...
        break;
    case SelectionRecord::UNKNOWN_ELEMENT:
        // check for sbml element types

        if ((sel.index = impl->model->getFloatingSpeciesIndex(sel.p1)) >= 0)
        {
            sel.selectionType = SelectionRecord::FLOATING_AMOUNT;
            break;
        }
        else if ((sel.index = impl->model->getBoundarySpeciesIndex(sel.p1)) >= 0)
        {
            sel.selectionType = SelectionRecord::BOUNDARY_AMOUNT;
            break;
        }
        else if ((sel.index = impl->model->getCompartmentIndex(sel.p1)) >= 0)
        {
            sel.selectionType = SelectionRecord::COMPARTMENT;
            break;
        }
        else if ((sel.index = impl->model->getGlobalParameterIndex(sel.p1)) >= 0)
        {
            sel.selectionType = SelectionRecord::GLOBAL_PARAMETER;
            break;
        }
        else if ((sel.index = impl->model->getReactionIndex(sel.p1)) >= 0)
        {
            sel.selectionType = SelectionRecord::REACTION_RATE;
            break;
        }
        else
        {
            throw Exception("No sbml element exists for symbol '" + str + "'");
            break;
        }
    case SelectionRecord::UNKNOWN_CONCENTRATION:
        if ((sel.index = impl->model->getFloatingSpeciesIndex(sel.p1)) >= 0)
        {
            sel.selectionType = SelectionRecord::FLOATING_CONCENTRATION;
            break;
        }
        else if ((sel.index = impl->model->getBoundarySpeciesIndex(sel.p1)) >= 0)
        {
            sel.selectionType = SelectionRecord::BOUNDARY_CONCENTRATION;
            break;
        }
        else
        {
            string msg = "No sbml element exists for concentration selection '" + str + "'";
            Log(Logger::LOG_ERROR) << msg;
            throw Exception(msg);
            break;
        }
    case SelectionRecord::FLOATING_AMOUNT_RATE:
        if ((sel.index = impl->model->getFloatingSpeciesIndex(sel.p1)) >= 0)
        {
            break;
        }
        else
        {
            throw Exception("Invalid id '" + str + "' for floating amount rate");
            break;
        }
    case SelectionRecord::ELASTICITY:
    case SelectionRecord::UNSCALED_ELASTICITY:
    case SelectionRecord::CONTROL:
    case SelectionRecord::UNSCALED_CONTROL:
        // check that the control coef args are valid
        if (impl->model->getReactionIndex(sel.p1) >= 0 ||
                impl->model->getFloatingSpeciesIndex(sel.p1) >= 0)

        {
            if (impl->model->getGlobalParameterIndex(sel.p2) >= 0 ||
                    impl->model->getBoundarySpeciesIndex(sel.p2) >= 0 ||
                    impl->model->getConservedMoietyIndex(sel.p2) >= 0)
            {
                Log(Logger::LOG_INFORMATION) <<
                        "Valid metabolic control selection: " << sel.to_repr();
            }
            else
            {
                throw Exception("The second argument to a metabolic control "
                        "coefficient selection, " + sel.p2 + ", must be either "
                        "a global parameter, boundary species or conserved sum");
            }
        }
        else
        {
            throw Exception("The first argument to a metabolic control "
                    "coefficient selection, " + sel.p1 + ", must be either "
                    "a reaction, floating species or conserved sum");
        }

        break;
    case SelectionRecord::EIGENVALUE_REAL:
    case SelectionRecord::EIGENVALUE_IMAG:
    case SelectionRecord::EIGENVALUE_COMPLEX:
        if ((sel.index = impl->model->getFloatingSpeciesIndex(sel.p1)) >= 0)
        {
            break;
        }
        else
        {
            throw Exception("eigen argument '" + sel.p1 + "' is not a floating species.");
        }
        break;
    case SelectionRecord::STOICHIOMETRY:
        if (impl->model->getFloatingSpeciesIndex(sel.p1) >= 0)
        {
            if (impl->model->getReactionIndex(sel.p2) >= 0)
            {
                break;
            }
            else
            {
                throw Exception("second argument to stoich '" + sel.p2 + "' is not a reaction id.");
            }
        }
        else
        {
            throw Exception("first argument to stoich '" + sel.p1 + "' is not a floating species id.");
        }
        break;
    case SelectionRecord::INITIAL_CONCENTRATION:
        if ((sel.index = impl->model->getFloatingSpeciesIndex(sel.p1)) >= 0)
        {
            break;
        }
        else
        {
            throw Exception("Invalid id '" + sel.p1 + "' for floating initial concentration");
            break;
        }
    case SelectionRecord::INITIAL_AMOUNT:
        if ((sel.index = impl->model->getFloatingSpeciesIndex(sel.p1)) >= 0)
        {
            break;
        }
        else if ((sel.index = impl->model->getGlobalParameterIndex(sel.p1)) >= 0)
        {
            sel.selectionType = SelectionRecord::INITIAL_GLOBAL_PARAMETER;
            break;
        }
        else if ((sel.index = impl->model->getCompartmentIndex(sel.p1)) >= 0)
        {
            sel.selectionType = SelectionRecord::INITIAL_COMPARTMENT;
            break;
        }
        else
        {
            throw Exception("Invalid id '" + sel.p1 + "' for floating initial amount");
            break;
        }
    default:
        Log(Logger::LOG_ERROR) << "A new SelectionRecord should not have this value: "
            << sel.to_repr();
        break;
    }

    return sel;
}

void RoadRunner::setSelections(const vector<string>& _selList)
{
    impl->mSelectionList.clear();

    for(int i = 0; i < _selList.size(); ++i)
    {
        impl->mSelectionList.push_back(createSelection(_selList[i]));
    }

    vector<string> selstr = vector<string>(impl->mSelectionList.size());
    for(int i = 0; i < selstr.size(); ++i)
    {
        selstr[i] = impl->mSelectionList[i].to_string();
    }
    impl->simulationResult.setColNames(selstr.begin(), selstr.end());
}

void RoadRunner::setSelections(const std::vector<rr::SelectionRecord>& ss)
{
    impl->mSelectionList = ss;
}

void RoadRunner::setSteadyStateSelections(const vector<string>& ss)
{
    impl->mSteadyStateSelection.clear();

    for(int i = 0; i < ss.size(); ++i)
    {
        impl->mSteadyStateSelection.push_back(createSelection(ss[i]));
    }
}

void RoadRunner::setSteadyStateSelections(const std::vector<rr::SelectionRecord>& ss)
{
    impl->mSteadyStateSelection = ss;
}

//Compute the frequency response, startW, Number Of Decades, Number of Points, parameterName, variableName
Matrix<double> RoadRunner::getFrequencyResponse(double startFrequency,
        int numberOfDecades, int numberOfPoints,
        const std::string& parameterName, const std::string& variableName,
        bool useDB, bool useHz)
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }

    try
    {
        vector<string> reactionNames = getReactionIds();
        vector<string> speciesNames = getFloatingSpeciesIds();

        // Prepare the dv/dp array
        Matrix< Complex > dvdp(reactionNames.size(), 1);

        //Guess we don't need to simulate here?? (TK)
        //        SimulateOptions opt;
        //        opt.start = 0;
        //        opt.duration = 50.0;
        //        opt.steps = 1;
        //        simulate(&opt);
        if(steadyState() > 1E-2)
        {
            throw Exception("Unable to locate steady state during frequency response computation");
        }

        ComplexMatrix uelast(getUnscaledElasticityMatrix());
        ComplexMatrix Nr = getNrMatrix();
        ComplexMatrix LinkMatrix = getLinkMatrix();

        // Compute dv/dp
        for (int j = 0; j < reactionNames.size(); j++)
        {
            double val = getUnscaledParameterElasticity(reactionNames[j], parameterName);
            dvdp(j, 0) = Complex(val, 0.0);
            Log(lDebug)<<"dv/dp: " << dvdp(j, 0);
        }

        // Compute the Jacobian first
        ComplexMatrix T1(Nr.RSize(), uelast.CSize());
        T1 = mult(Nr, uelast);
        Log(lInfo)<<T1;

        ComplexMatrix Jac(Nr.RSize(), LinkMatrix.CSize());
        Jac = mult(T1, LinkMatrix);

        ComplexMatrix Id(Nr.RSize(), Nr.RSize());

        //Create Identity matrix
        for(int i = 0; i < Id.RSize(); i++)
        {
            Id(i,i) = Complex(1,0);
        }

        ComplexMatrix T2(Nr.RSize(), LinkMatrix.CSize());   // Stores iwI - Jac  and (iwI - Jac)^-1
        ComplexMatrix T3(LinkMatrix.RSize(), 1);            // Stores (iwI - Jac)^-1 . Nr
        ComplexMatrix T4(Nr.RSize(), Nr.CSize());
        ComplexMatrix T5(LinkMatrix.RSize(), 1);
        ComplexMatrix Inv(Nr.RSize(), LinkMatrix.CSize());

        // resultArray stores frequency, gain and phase
        Matrix<double> resultArray(numberOfPoints, 3);

        // Main loop, generate log spaced frequency numbers and compute the gain and phase at each frequency
        vector<double> w(logspace(startFrequency, numberOfDecades, numberOfPoints));

        for (int i = 0; i < numberOfPoints; i++)
        {
            Complex diagVal(0.0, w[i]);

            T1 = multDiag(Id, diagVal); // Compute iwI
            T2 = subtract(T1, Jac);     // Compute iwI - Jac //// Compute (iwI - Jac)^-1
            Inv = GetInverse(T2);
            T3  = mult(Inv, Nr);        // Compute (iwI - Jac)^-1 . Nr

            for (int j = 0; j < reactionNames.size(); j++)
            {
                double realPart = getUnscaledParameterElasticity(reactionNames[j], parameterName);
                dvdp(j, 0) = ls::Complex(realPart, 0.0);
            }

            T4 = mult(T3, dvdp);    // Compute(iwI - Jac)^-1 . Nr . dvdp

            // Finally include the dependent set as well.
            T5 = mult(LinkMatrix, T4); // Compute L (iwI - Jac)^-1 . Nr . dvdp

            for (int j = 0; j < speciesNames.size(); j++)
            {
                if (speciesNames[j] == variableName)
                {
                    double dw = abs(T5(j, 0));
                    if (useDB)
                    {
                        dw = 20.0*log10(dw);
                    }
                    resultArray(i,1) = dw;
                    Complex val(T5(j, 0));

                    double phase = (180.0/M_PI) * rr::phase(val) + getAdjustment(val);
                    resultArray(i,2) = phase;
                    break;
                }
            }
            if (useHz)
            {
                // Store frequency, convert to Hz by dividing by 2Pi
                resultArray[i][0] = w[i]/(2.*M_PI);
            }
            else
            {
                // Store frequency, leave as rad/sec
                resultArray[i][0] = w[i];
            }
        }
        return resultArray;
    }
    catch(const Exception& e)
    {
      throw Exception("Unexpected error in getFrequencyResponse(): " +  e.Message());
    }
}



double RoadRunner::getUnscaledParameterElasticity(const string& reactionName, const string& parameterName)
{
    int parameterIndex;

    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }
    try
    {
        int reactionIndex = impl->model->getReactionIndex(reactionName);

        if(reactionIndex == -1)
        {
            throw CoreException("Internal Error: unable to locate reaction name while computing unscaled elasticity");
        }

        // Look for the parameter name, check local parameters first, then global
        ParameterType parameterType;

        if(impl->model->getGlobalParameterIndex(parameterName) != -1)
        {
            parameterIndex = impl->model->getGlobalParameterIndex(parameterName);
            parameterType  = ptGlobalParameter;
        }
        else if(impl->model->getBoundarySpeciesIndex(parameterName) != -1)
        {
            parameterIndex = impl->model->getBoundarySpeciesIndex(parameterName);
            parameterType  = ptBoundaryParameter;
        }
        else if(impl->model->getConservedMoietyIndex(parameterName) != -1)
        {
            parameterIndex = impl->model->getConservedMoietyIndex(parameterName);
            parameterType = ptConservationParameter;
        }
        else
        {
            return 0.0;
        }

        double originalParameterValue = 0.0;
        double result = 0;
        switch (parameterType)
        {
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
                throw(Exception("This parameterType is not supported in getUnscaledParameterElasticity"));
        }

        double hstep = impl->mDiffStepSize*originalParameterValue;
        if (fabs(hstep) < 1E-12)
        {
            hstep = impl->mDiffStepSize;
        }

        double f1, f2, fi, fi2, fd, fd2;
        impl->changeParameter(parameterType, reactionIndex, parameterIndex, originalParameterValue, hstep);
        impl->model->getReactionRates(1, &reactionIndex, &fi);

        impl->changeParameter(parameterType, reactionIndex, parameterIndex, originalParameterValue, 2.0*hstep);
        impl->model->getReactionRates(1, &reactionIndex, &fi2);

        impl->changeParameter(parameterType, reactionIndex, parameterIndex, originalParameterValue, -hstep);
        impl->model->getReactionRates(1, &reactionIndex, &fd);

        impl->changeParameter(parameterType, reactionIndex, parameterIndex, originalParameterValue, -2.0*hstep);
        impl->model->getReactionRates(1, &reactionIndex, &fd2);

        // Use instead the 5th order approximation double unscaledElasticity = (0.5/hstep)*(fi-fd);
        // The following separated lines avoid small amounts of roundoff error
        f1 = fd2 + 8.0*fi;
        f2 = -(8.0*fd + fi2);

        // What ever happens, make sure we restore the species level
        impl->changeParameter(parameterType, reactionIndex, parameterIndex, originalParameterValue, 0);
        return 1.0/(12.0*hstep)*(f1 + f2);
    }
    catch(const Exception& ex)
    {
        throw(Exception("Problem in getUnscaledParameterElasticity():" + ex.getMessage()));
    }
}



vector<double> logspace(const double& startW, const double& d2, const int& n)
{
    double d1 = 0;
    vector<double> y(n);
    for (int i = 0; i <= n - 2; i++)
    {
        y[i] = i*(d2 - d1);
        y[i] = y[i]/(n - 1);
        y[i] = d1 + y[i];
        y[i] = pow(10, y[i]) * startW;
    }
    y[n - 1] = pow(10, d2)*startW;
    return y;
}

double phase(Complex& val)
{
    if ((std::real(val) == 0.0) && (imag(val) == 0.0))
    {
        return 0.0;
    }
    else
    {
        return atan2(imag(val), std::real(val));
    }
}

//This function is used in roadrunners freq analysis code...
double getAdjustment(Complex& z)
{
    double adjustment;
    if (std::real(z) >= 0 && imag(z) >= 0)
    {
        adjustment = 0;
    }
    else if (std::real(z) >= 0 && imag(z) < 0)
    {
        adjustment = 360;
    }
    else if (std::real(z) < 0 && imag(z) >= 0)
    {
        adjustment = 0;
    }
    else
    {
        adjustment = 360;
    }
    return adjustment;
}


/************************ Selection Ids Species Section ***********************/
#if (1) /**********************************************************************/
/******************************************************************************/

void RoadRunner::getIds(int types, std::list<std::string>& ids)
{
    if (impl->model)
    {
        impl->model->getIds(types, ids);

        if (types & SelectionRecord::EIGENVALUE_REAL)
        {
            std::list<std::string> eigen;
            impl->model->getIds(SelectionRecord::FLOATING_AMOUNT, eigen);

            for (std::list<std::string>::const_iterator i =
                    eigen.begin(); i != eigen.end(); ++i)
            {
                ids.push_back("eigen(" + *i + ")");
                ids.push_back("eigenReal(" + *i + ")");
                ids.push_back("eigenImag(" + *i + ")");
            }
        }
    }
}

vector<string> RoadRunner::getIndependentFloatingSpeciesIds()
{
    return getLibStruct()->getIndependentSpecies();
}

vector<string> RoadRunner::getDependentFloatingSpeciesIds()
{
    return getLibStruct()->getDependentSpecies();
}

int RoadRunner::getSupportedIdTypes()
{
    int types = impl->model ? impl->model->getSupportedIdTypes() : 0;

    return types | SelectionRecord::ELASTICITY |
            SelectionRecord::UNSCALED_ELASTICITY |
            SelectionRecord::CONTROL |
            SelectionRecord::UNSCALED_CONTROL |
            SelectionRecord::EIGENVALUE_REAL;
}

vector<string> RoadRunner::getRateOfChangeIds()
{
    std::list<std::string> list;

    if (impl->model) {
        impl->model->getIds(SelectionRecord::FLOATING_AMOUNT_RATE, list);
    }

    return std::vector<std::string>(list.begin(), list.end());
}

vector<string> RoadRunner::getCompartmentIds()
{
    std::list<std::string> list;

    if (impl->model) {
        impl->model->getIds(SelectionRecord::COMPARTMENT, list);
    }

    return std::vector<std::string>(list.begin(), list.end());
}

vector<string> RoadRunner::getGlobalParameterIds()
{
    std::list<std::string> list;

    if (impl->model) {
        impl->model->getIds(SelectionRecord::GLOBAL_PARAMETER, list);
    }

    return std::vector<std::string>(list.begin(), list.end());
}

vector<string> RoadRunner::getBoundarySpeciesIds()
{
    std::list<std::string> list;

    if (impl->model) {
        impl->model->getIds(SelectionRecord::BOUNDARY_AMOUNT, list);
    }

    return std::vector<std::string>(list.begin(), list.end());
}

vector<string> RoadRunner::getBoundarySpeciesConcentrationIds()
{
    std::list<std::string> list;

    if (impl->model) {
        impl->model->getIds(SelectionRecord::BOUNDARY_CONCENTRATION, list);
    }

    return std::vector<std::string>(list.begin(), list.end());
}

vector<string> RoadRunner::getConservedMoietyIds()
{
    return createModelStringList(impl->model, &ExecutableModel::getNumConservedMoieties,
            &ExecutableModel::getConservedMoietyId);
}

vector<string> RoadRunner::getFloatingSpeciesIds()
{
    std::list<std::string> list;

    if (impl->model) {
        impl->model->getIds(SelectionRecord::FLOATING_AMOUNT, list);
    }

    return std::vector<std::string>(list.begin(), list.end());
}

vector<string> RoadRunner::getFloatingSpeciesConcentrationIds()
{
    std::list<std::string> list;

    if (impl->model) {
        impl->model->getIds(SelectionRecord::FLOATING_CONCENTRATION, list);
    }

    return std::vector<std::string>(list.begin(), list.end());
}

vector<string> RoadRunner::getFloatingSpeciesInitialConditionIds()
{
    std::list<std::string> list;

    if (impl->model) {
        impl->model->getIds(SelectionRecord::INITIAL_FLOATING_CONCENTRATION, list);
    }

    return std::vector<std::string>(list.begin(), list.end());
}

vector<string> RoadRunner::getFloatingSpeciesInitialConcentrationIds()
{
    std::list<std::string> list;

    if (impl->model) {
        impl->model->getIds(SelectionRecord::INITIAL_FLOATING_CONCENTRATION, list);
    }

    return std::vector<std::string>(list.begin(), list.end());
}

vector<string> RoadRunner::getReactionIds()
{
    std::list<std::string> list;

    if (impl->model) {
        impl->model->getIds(SelectionRecord::REACTION_RATE, list);
    }

    return std::vector<std::string>(list.begin(), list.end());
}

vector<string> RoadRunner::getEigenValueIds()
{
    std::list<std::string> list;

    getIds(SelectionRecord::EIGENVALUE_REAL, list);

    return std::vector<std::string>(list.begin(), list.end());
}


/************************ End Selection Ids Species Section *******************/
#endif /***********************************************************************/
/******************************************************************************/

RoadRunnerOptions& RoadRunner::getOptions()
{
    return impl->roadRunnerOptions;
}

const DoubleMatrix* RoadRunner::getSimulationData() const
{
    return &impl->simulationResult;
}

void RoadRunner::applySimulateOptions()
{
    get_self();

    if (self.simulateOpt.duration < 0 || self.simulateOpt.start < 0
            || self.simulateOpt.steps < 0 )
    {
        throw std::invalid_argument("duration, startTime and steps must be non-negative");
    }

    // This one creates the list of what we will look at in the result
    // uses values (potentially) from simulate options.
    createTimeCourseSelectionList();

    if (self.simulateOpt.reset_model)
    {
        reset(); // reset back to initial conditions
		self.simulateOpt.reset_model = false;
    }
}


static void metabolicControlCheck(ExecutableModel *model)
{
    static const char* e1 = "Metabolic control analysis only valid "
            "for primitive reaction kinetics models. ";
    /*if (model->getNumRateRules() > 0)
    {
        throw std::invalid_argument(string(e1) + "This model has rate rules");
    }*/

    if (model->getNumEvents() > 0 && !Config::getBool(Config::ALLOW_EVENTS_IN_STEADY_STATE_CALCULATIONS))
    {
        throw std::invalid_argument(string(e1) + "This model has events. Set ALLOW_EVENTS_IN_STEADY_STATE_CALCULATIONS to True to override. "
        "To override, run 'Config.setValue(Config.ALLOW_EVENTS_IN_STEADY_STATE_CALCULATIONS, True)'.");
    }
}

/*
*  Saves this roadrunner instance to a file so it can be reloaded later
*
*  
*/
void RoadRunner::saveState(std::string filename)
{
	std::ofstream out(filename, iostream::binary);
	if (!out)
	{
		throw std::invalid_argument("Error opening file " + filename + ": " + std::string(strerror(errno)));
	}
	rr::saveBinary(out, fileMagicNumber);
	rr::saveBinary(out, dataVersionNumber);
	//Save all of roadrunner's data to the file
	rr::saveBinary(out, impl->mInstanceID);
	rr::saveBinary(out, impl->mDiffStepSize);
	rr::saveBinary(out, impl->mSteadyStateThreshold);
    
	saveSelectionVector(out, impl->mSelectionList);
	
	rr::saveBinary(out, impl->loadOpt.version);
	rr::saveBinary(out, impl->loadOpt.size);
	rr::saveBinary(out, impl->loadOpt.modelGeneratorOpt);
	rr::saveBinary(out, impl->loadOpt.loadFlags);
    
	rr::saveBinary(out, impl->loadOpt.getKeys().size());

	for (std::string k : impl->loadOpt.getKeys())
	{
		rr::saveBinary(out, k);
		rr::saveBinary(out, impl->loadOpt.getItem(k));
	}

	saveSelectionVector(out, impl->mSteadyStateSelection);

	rr::saveBinary(out, impl->simulationResult.getColNames());
	rr::saveBinary(out, impl->simulationResult.getRowNames());

	rr::saveBinary(out, impl->simulateOpt.reset_model);
	rr::saveBinary(out, impl->simulateOpt.structured_result);
	rr::saveBinary(out, impl->simulateOpt.copy_result);
	rr::saveBinary(out, impl->simulateOpt.steps);
	rr::saveBinary(out, impl->simulateOpt.start);
	rr::saveBinary(out, impl->simulateOpt.duration);
	rr::saveBinary(out, impl->simulateOpt.variables);
	rr::saveBinary(out, impl->simulateOpt.amounts);
	rr::saveBinary(out, impl->simulateOpt.concentrations);

	rr::saveBinary(out, impl->simulateOpt.getKeys().size());

	for (std::string k : impl->simulateOpt.getKeys())
	{
		rr::saveBinary(out, k);
		rr::saveBinary(out, impl->simulateOpt.getItem(k));
	}

	rr::saveBinary(out, impl->roadRunnerOptions.flags);
	rr::saveBinary(out, impl->roadRunnerOptions.jacobianStepSize);

	rr::saveBinary(out, impl->configurationXML);
	//Save the model (which saves the model data symbols and model resources)
	impl->model->saveState(out);
	
	rr::saveBinary(out, impl->integrator->getName());
	rr::saveBinary(out, impl->integrator->getNumParams());

	for (std::string k : impl->integrator->getSettings())
	{
		rr::saveBinary(out, k);
		rr::saveBinary(out, impl->integrator->getValue(k));
	}

	rr::saveBinary(out, impl->steady_state_solver->getName());
	rr::saveBinary(out, impl->steady_state_solver->getNumParams());

	for (std::string k : impl->steady_state_solver->getSettings())
	{
		rr::saveBinary(out, k);
		rr::saveBinary(out, impl->steady_state_solver->getValue(k));
	}
    //Currently I save and reload the SBML that was used to create the model
	//It is not parsed however, unless a instance of LibStructural needs to be
	//created
	//It might also be possible to construct LibStructural without SBML, but I'm not familiar with it
	//If this implementation is too slow we can change that
	rr::saveBinary(out, impl->mCurrentSBML);
}

void RoadRunner::saveSelectionVector(std::ostream& out, std::vector<SelectionRecord>& v)
{
	rr::saveBinary(out, v.size());
	for (SelectionRecord sr : v)
	{
		rr::saveBinary(out, sr.index);
		rr::saveBinary(out, sr.p1);
		rr::saveBinary(out, sr.p2);
		rr::saveBinary(out, sr.selectionType);
	}
}


void RoadRunner::loadState(std::string filename)
{
	std::ifstream in(filename, iostream::binary);
	if (!in)
	{
		throw std::invalid_argument("Error opening file " + filename + ": " + std::string(strerror(errno)));
	}
	int inMagicNumber;
	rr::loadBinary(in, inMagicNumber);
	if (inMagicNumber != fileMagicNumber)
	{
		throw std::invalid_argument("The file " + filename + " has the wrong magic number. Are you sure it is a roadrunner save state?");
	}

	int inVersionNumber;
	rr::loadBinary(in, inVersionNumber);
	if (inVersionNumber != dataVersionNumber)
	{
		throw std::invalid_argument("The file " + filename + " was saved with an unrecognized version of roadrunner");
	}
   //load roadrunner's data in the same order saveState saves it in
	rr::loadBinary(in, impl->mInstanceID);
	rr::loadBinary(in, impl->mDiffStepSize);
	rr::loadBinary(in, impl->mSteadyStateThreshold);
    
	loadSelectionVector(in, impl->mSelectionList);
	
	rr::loadBinary(in, impl->loadOpt.version);
	rr::loadBinary(in, impl->loadOpt.size);
	rr::loadBinary(in, impl->loadOpt.modelGeneratorOpt);
	rr::loadBinary(in, impl->loadOpt.loadFlags);
    
	size_t loadOptSize;
	rr::loadBinary(in, loadOptSize);

	for (int i = 0; i < loadOptSize; i++)
	{
		std::string k;
		rr::loadBinary(in, k);
		rr::Variant v;
		rr::loadBinary(in, v);
		impl->loadOpt.setItem(k, v);
	}


	loadSelectionVector(in, impl->mSteadyStateSelection);

	std::vector<std::string> colNames;
	rr::loadBinary(in, colNames);
	impl->simulationResult.setColNames(colNames.begin(), colNames.end());

	std::vector<std::string> rowNames;
	rr::loadBinary(in, rowNames);
	impl->simulationResult.setRowNames(rowNames.begin(), rowNames.end());

	rr::loadBinary(in, impl->simulateOpt.reset_model);
	rr::loadBinary(in, impl->simulateOpt.structured_result);
	rr::loadBinary(in, impl->simulateOpt.copy_result);
	rr::loadBinary(in, impl->simulateOpt.steps);
	rr::loadBinary(in, impl->simulateOpt.start);
	rr::loadBinary(in, impl->simulateOpt.duration);
	rr::loadBinary(in, impl->simulateOpt.variables);
	rr::loadBinary(in, impl->simulateOpt.amounts);
	rr::loadBinary(in, impl->simulateOpt.concentrations);

	size_t simulateOptSize;
	rr::loadBinary(in, simulateOptSize);

	for (int i = 0; i < simulateOptSize; i++)
	{
		std::string k;
		rr::loadBinary(in, k);
		rr::Variant v;
		rr::loadBinary(in, v);
		impl->simulateOpt.setItem(k, v);
	}

	rr::loadBinary(in, impl->roadRunnerOptions.flags);
	rr::loadBinary(in, impl->roadRunnerOptions.jacobianStepSize);

	rr::loadBinary(in, impl->configurationXML);
	if(impl->model)
		delete impl->model;
	//Create a new model from the stream
	//impl->model = new rrllvm::LLVMExecutableModel(in, impl->loadOpt.modelGeneratorOpt);
	impl->model = ExecutableModelFactory::createModel(in, impl->loadOpt.modelGeneratorOpt);
    impl->syncAllSolversWithModel(impl->model);

	if (impl->mLS)
		delete impl->mLS;
	std::string integratorName;
	rr::loadBinary(in, integratorName);
	setIntegrator(integratorName);

	unsigned long integratorNumParams;
	rr::loadBinary(in, integratorNumParams);

	for (int i = 0; i < integratorNumParams; i++)
	{
		std::string k;
		rr::loadBinary(in, k);
		rr::Variant v;
		rr::loadBinary(in, v);
		impl->integrator->setValue(k, v);
	}

	std::string steadyStateSolverName;
	rr::loadBinary(in, steadyStateSolverName);
	setSteadyStateSolver(steadyStateSolverName);

	unsigned long solverNumParams;
	rr::loadBinary(in, solverNumParams);
	
	for (int i = 0; i < solverNumParams; i++) 
	{
		std::string k;
		rr::loadBinary(in, k);
		rr::Variant v;
		rr::loadBinary(in, v);
		impl->steady_state_solver->setValue(k, v);
	}

	//Currently the SBML is saved with the binary data, see saveState above
	rr::loadBinary(in, impl->mCurrentSBML);
	//Restart the integrator and reset the model
	//This will need to change if we decide to add pausing
	// impl->integrator->restart(0.0);
	// reset();
}

void RoadRunner::loadSelectionVector(std::istream& in, std::vector<SelectionRecord>& v)
{
	size_t vsize;
	rr::loadBinary(in, vsize);
	v.clear();
	for (int i = 0; i < vsize; i++)
	{
		SelectionRecord sr;
		rr::loadBinary(in, sr.index);
		rr::loadBinary(in, sr.p1);
		rr::loadBinary(in, sr.p2);
		rr::loadBinary(in, sr.selectionType);
		v.push_back(sr);
	}
}

void RoadRunner::removeReaction(const std::string& rid)
{

	libsbml::Reaction* toDelete = impl->document->getModel()->removeReaction(rid);
	if (toDelete == NULL) 
	{
		throw std::invalid_argument("Roadrunner::removeReaction failed, no reaction with ID " + rid + "existed in the model");
	}

	delete toDelete;

	regenerate();
}

void RoadRunner::addSpecies(std::string name, std::string id, std::string compartment, double initAmount, std::string substanceUnits)
{

	libsbml::Species *newSpecies = impl->document->getModel()->createSpecies();
	newSpecies->setName(name);
	newSpecies->setId(id);
	newSpecies->setCompartment(compartment);
	newSpecies->setInitialAmount(initAmount);
	newSpecies->setSubstanceUnits(substanceUnits);
   
	regenerate();
}

void RoadRunner::addReaction(const std::string& sbmlRep)
{
	libsbml::Reaction *newReaction = impl->document->getModel()->createReaction();
	newReaction->read(libsbml::XMLInputStream(sbmlRep.c_str(), false));
	//newReaction->addReactant(impl->document->getModel()->getSpecies("S2"));
	//newReaction->addProduct(impl->document->getModel()->getSpecies("S1"));
	//newReaction->setName("reaction2");
	//newReaction->setId("reaction2");
	//libsbml::KineticLaw *kLaw = newReaction->createKineticLaw();
	//kLaw->addCVTerm()
	//kLaw->setMath(libsbml::SBML_parseFormula("k1*S2"));
    
	regenerate();
}

void RoadRunner::addReaction(const string& rid, boolean reversible, vector<string> reactants, vector<string> products, vector<string> modifiers, const string& kineticLaw)
{
	using namespace libsbml;
	Model* sbmlModel = impl->document->getModel();
	libsbml::Reaction* newReaction = sbmlModel->createReaction();
	
	newReaction->setId(rid);
	newReaction->setReversible(reversible);
	for (int i = 0; i < reactants.size(); i++) 
	{
		Species* s = sbmlModel->getSpecies(reactants[i]);
		if (s == NULL)
		{
			throw std::invalid_argument("Roadrunner::addReaction failed, no species with ID " + reactants[0] + " existed in the model");
		}
		newReaction->addReactant(s);
	}

	for (int i = 0; i < products.size(); i++)
	{
		Species* s = sbmlModel->getSpecies(products[i]);
		if (s == NULL)
		{
			throw std::invalid_argument("Roadrunner::addReaction failed, no species with ID " + products[0] + " existed in the model");
		}
		newReaction->addProduct(s);
	}

	for (int i = 0; i < modifiers.size(); i++)
	{
		Species* s = sbmlModel->getSpecies(modifiers[i]);
		if (s == NULL)
		{
			throw std::invalid_argument("Roadrunner::addReaction failed, no species with ID " + modifiers[0] + " existed in the model");
		}
		newReaction->addModifier(s);
	}

	
	libsbml::KineticLaw* kLaw = newReaction->createKineticLaw();
	ASTNode_t* math = libsbml::SBML_parseFormula(&kineticLaw[0]);
	if (math == NULL)
	{
		throw std::invalid_argument("Roadrunner::addReaction failed, an error occurred in parsing the math formula");
	}
	kLaw->setMath(math);
	
}

void RoadRunner::removeSpecies(const std::string& sid) 
{
	using namespace libsbml;
	Model* sbmlModel = impl->document->getModel();
	Species* s = sbmlModel->removeSpecies(sid);
	if (s == NULL)
	{
		throw std::invalid_argument("Roadrunner::removeSpecies failed, no species with ID " + sid + "existed in the model");
	}
	delete s;
	// delete all related reactions as well
	int index = 0;
	int numReaction = sbmlModel->getNumReactions();
	for (uint i = 0; i < numReaction; i++)
	{
		Reaction* reaction = sbmlModel->getListOfReactions()->get(index);
		Reaction* toDelete = nullptr;
		const ListOfSpeciesReferences* reactants = reaction->getListOfReactants();
		for (uint j = 0; j < reactants->size(); j++)
		{
			if (reactants->get(j)->getSpecies() == sid)
			{
				toDelete = sbmlModel->removeReaction(reaction->getName());
				break;
			}
		}

		if (toDelete != nullptr) 
		{
			delete toDelete;
			// no need to check anymore
			continue;
		}

		const libsbml::ListOfSpeciesReferences* products = reaction->getListOfProducts();
		for (uint j = 0; j < products->size(); j++)
		{
			if (products->get(j)->getSpecies() == sid)
			{
				toDelete = sbmlModel->removeReaction(reaction->getName());
				break;
			}
		}

		if (toDelete != nullptr) 
		{
			delete toDelete;
			// no need to check anymore
			continue;
		}

		const libsbml::ListOfSpeciesReferences* modifiers = reaction->getListOfModifiers();
		for (uint j = 0; j < modifiers->size(); j++)
		{
			if (modifiers->get(j)->getSpecies() == sid)
			{
				toDelete = sbmlModel->removeReaction(reaction->getName());
				break;
			}
		}

		if (toDelete != nullptr) 
		{
			delete toDelete;
			// no need to check anymore
			continue;
		}

		// this reaction is not related to the deleted species 
		index++;
	}
	regenerate();
}

void RoadRunner::regenerate()
{
	ExecutableModel* newModel = ExecutableModelFactory::regenerateModel(impl->model, impl->document, impl->loadOpt.modelGeneratorOpt);
	delete impl->model;
	impl->model = newModel;
	impl->syncAllSolversWithModel(impl->model);
	resetSelectionLists();
}

} //namespace

#if defined(_WIN32)
#pragma comment(lib, "IPHLPAPI.lib") //Becuase of poco needing this
#endif
