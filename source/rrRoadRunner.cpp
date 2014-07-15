#pragma hdrstop
// on Windows, this needs to go first to get M_PI
#define _USE_MATH_DEFINES

#include "rrOSSpecifics.h"

// TODO will clean these up in the future
#undef RR_DEPRECATED
#define RR_DEPRECATED(func) func

#include "rrRoadRunner.h"
#include "rrException.h"
#include "ModelGenerator.h"
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
#include "rrSteadyStateSolver.h"
#include "rrSBMLReader.h"
#include "rrConfig.h"

#include <sbml/conversion/SBMLLocalParameterConverter.h>

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
 * implemention class, hide all details here.
 */
class RoadRunnerImpl {
public:

    int mInstanceID;
    const double mDiffStepSize;

    const double mSteadyStateThreshold;
    ls::DoubleMatrix simulationResult;
    RoadRunnerData mRoadRunnerData;


    /**
     * Points to the current integrator. This is a pointer into the
     * integtators array.
     */
    Integrator *integrator;

    std::vector<SelectionRecord> mSelectionList;

    /**
     * ModelGenerator obtained from the factory
     */
    ModelGenerator *mModelGenerator;

    /**
     * read from the Config, duplicates loadsbmloptions
     *
     * TODO: this needs to be cleaned up.
     */
    bool conservedMoietyAnalysis;

    std::vector<SelectionRecord> mSteadyStateSelection;

    ExecutableModel* model;

    std::string mCurrentSBML;

    std::string mFilename;

    /**
     * structural analysis library.
     */
    LibStructural* mLS;

    /**
     * options that are specific to the simulation
     */
    SimulateOptions simulateOpt;

    /**
     * copy of simualte opt which is returned to clients.
     *
     * used so we can determine if the options have changed.
     */
    SimulateOptions copySimulateOpt;

    /**
     * The sim options may be requested via getSimulateOptions. In this
     * case, the caller may modify it, so we assume its dirty.
     *
     * These options are re-loaded into the integrator in simulate and
     * oneStep.
     */
    bool dirtySimulateOptions;

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
     * store the integrators in a map. When the integrator is switched,
     * this way it saves the previous state. Usefull for correct
     * stream of random numbers for stochastic integrators.
     *
     * This is an array of pointers which are allocated by createIntegrator(),
     * these are freed in the dtor, but kept around for the lifetime of
     * this object.
     */
    Integrator*  integrators[SimulateOptions::GILLESPIE+1];

    /**
     * TODO get rid of this garbage
     */
    friend class aFinalizer;


    RoadRunnerImpl(const std::string& uriOrSBML,
            const LoadSBMLOptions* options) :
                mDiffStepSize(0.05),
                mSteadyStateThreshold(1.E-2),
                simulationResult(),
                mRoadRunnerData(),
                integrator(0),
                mSelectionList(),
                mModelGenerator(0),
                conservedMoietyAnalysis(Config::getBool(Config::LOADSBMLOPTIONS_CONSERVED_MOIETIES)),
                mSteadyStateSelection(),
                model(0),
                mCurrentSBML(),
                mFilename(),
                mLS(0),
                simulateOpt(),
                mInstanceID(0),
                dirtySimulateOptions(true)
    {
        // have to init integrators the hard way in c++98
        memset((void*)integrators, 0, sizeof(integrators)/sizeof(char));
    }


    RoadRunnerImpl(const string& _compiler, const string& _tempDir,
            const string& _supportCodeDir) :
                mDiffStepSize(0.05),
                mSteadyStateThreshold(1.E-2),
                simulationResult(),
                mRoadRunnerData(),
                integrator(0),
                mSelectionList(),
                mModelGenerator(0),
                conservedMoietyAnalysis(Config::getBool(Config::LOADSBMLOPTIONS_CONSERVED_MOIETIES)),
                mSteadyStateSelection(),
                model(0),
                mCurrentSBML(),
                mLS(0),
                simulateOpt(),
                mInstanceID(0),
                dirtySimulateOptions(true)
    {
        // have to init integrators the hard way in c++98
        memset((void*)integrators, 0, sizeof(integrators)/sizeof(char));
    }

    ~RoadRunnerImpl()
    {
        Log(Logger::LOG_DEBUG) << __FUNC__ << ", global instance count: " << mInstanceCount;

        delete mModelGenerator;
        delete model;
        delete mLS;

        deleteIntegrators();

        mInstanceCount--;
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


    void deleteIntegrators()
    {
        for (int i = 0; i < SimulateOptions::INTEGRATOR_END; ++i)
        {
            delete integrators[i];
            integrators[i] = 0;
        }
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

RoadRunner::RoadRunner(const std::string& uriOrSBML,
        const LoadSBMLOptions* options) :
            impl(new RoadRunnerImpl(uriOrSBML, options))
{

#if defined(BUILD_LLVM)
    string compiler =  "LLVM" ;
#else
    string compiler = gDefaultCompiler;
#endif

    // for now, dump out who we are
    Log(Logger::LOG_DEBUG) << __FUNC__ << "compiler: " << compiler <<
            ", tempDir:" << gDefaultTempFolder << ", supportCodeDir: " <<
            gDefaultSupportCodeFolder;

    impl->mModelGenerator = ModelGenerator::New(compiler,
            gDefaultTempFolder, gDefaultSupportCodeFolder);

    setTempDir(gDefaultTempFolder);

    if (!uriOrSBML.empty()) {
        load(uriOrSBML, options);
    }

    //Increase instance count..
    mInstanceCount++;
    impl->mInstanceID = mInstanceCount;
}


RoadRunner::RoadRunner(const string& _compiler, const string& _tempDir,
        const string& _supportCodeDir) :
        impl(new RoadRunnerImpl(_compiler, _tempDir, _supportCodeDir))
{


#if defined(BUILD_LLVM)
    string compiler = _compiler.empty() ? "LLVM" : _compiler;
#else
    string compiler = _compiler.empty() ? gDefaultCompiler : _compiler;
#endif

    string tempDir = _tempDir.empty() ? gDefaultTempFolder : _tempDir;
    string supportCodeDir = _supportCodeDir.empty() ?
            gDefaultSupportCodeFolder : _supportCodeDir;

    // for now, dump out who we are
    Log(Logger::LOG_DEBUG) << __FUNC__ << "compiler: " << compiler <<
            ", tempDir:" << tempDir << ", supportCodeDir: " <<
            supportCodeDir;

    impl->mModelGenerator = ModelGenerator::New(compiler,
            tempDir, supportCodeDir);

    setTempDir(tempDir);

    //Increase instance count..
    mInstanceCount++;
    impl->mInstanceID = mInstanceCount;
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
    stringstream ss;
    ss << "<roadrunner.RoadRunner() { " << std::endl;
    ss << "this: " << (void*)(this) << std::endl;
    ss << "modelLoaded : " << (impl->model == NULL ? "false" : "true") << std::endl;
    if(impl->model)
    {
        ss << "modelName: " <<  impl->model->getModelName() << std::endl;
    }

    ss << impl->simulateOpt.toString();

    ss << "libSBMLVersion: " << getVersionStr(VERSIONSTR_LIBSBML) << std::endl;
    ss << "jacobianStepSize: " << impl->roadRunnerOptions.jacobianStepSize << std::endl;
    ss << "conservedMoietyAnalysis: " << rr::toString(impl->conservedMoietyAnalysis) << std::endl;

#if defined(BUILD_LEGACY_C)
    ss<<"Temporary folder: "        <<    getTempDir()<<endl;
    ss<<"Compiler location: "       <<    getCompiler()->getCompilerLocation() << endl;
    ss<<"Support Code Folder: "     <<    getCompiler()->getSupportCodeFolder() << endl;
    ss<<"Working Directory: "       <<    getCWD() << endl;
#endif

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



LibStructural* RoadRunner::getLibStruct()
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
        throw Exception("could not create structural analysis with no loaded sbml");
        return 0;
    }
}

Compiler* RoadRunner::getCompiler()
{
    return impl->mModelGenerator ? impl->mModelGenerator->getCompiler() : 0;
}


void RoadRunner::setCompiler(const string& compiler)
{
    impl->mModelGenerator ? impl->mModelGenerator->setCompiler(compiler) : false;
}

bool RoadRunner::isModelLoaded()
{
    return impl->model ? true : false;
}

void RoadRunner::setSimulateOptions(const SimulateOptions& settings)
{
    impl->copySimulateOpt = settings;

    // copies the copy over to the actual one.
    _setSimulateOptions(0);
}

SimulateOptions& RoadRunner::getSimulateOptions()
{
    return impl->copySimulateOpt;
}

bool RoadRunner::getConservedMoietyAnalysis()
{
    return impl->conservedMoietyAnalysis;
}

void RoadRunner::setTempDir(const string& folder)
{
    impl->mModelGenerator ? impl->mModelGenerator->setTemporaryDirectory(folder) : false;
}

string RoadRunner::getTempDir()
{
    return impl->mModelGenerator ? impl->mModelGenerator->getTemporaryDirectory() : string("");
}

int RoadRunner::createDefaultTimeCourseSelectionList()
{
    vector<string> theList;
    vector<string> oFloating  = getFloatingSpeciesIds();

    theList.push_back("time");
    for(int i = 0; i < oFloating.size(); i++)
    {
        theList.push_back("[" + oFloating[i] + "]");
    }

    setSelections(theList);

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

ModelGenerator* RoadRunner::getModelGenerator()
{
    return impl->mModelGenerator;
}

string RoadRunner::getParamPromotedSBML(const string& sbml)
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

/**
 * RoadRunner keeps all the created integrators around. If the requested integrator
 * has not been created, this method creates one, and sets self.integrator
 * to point to it.
 */
void RoadRunner::updateIntegrator()
{
    get_self();

    if(self.model)
    {
        // check if valid range
        if (self.simulateOpt.integrator >= SimulateOptions::INTEGRATOR_END)
        {
            std::stringstream ss;
            ss << "Invalid integrator of " << self.simulateOpt.integrator
                    << ", integrator must be >= 0 and < "
                    << SimulateOptions::INTEGRATOR_END;
            throw std::invalid_argument(ss.str());
        }

        if (self.integrators[self.simulateOpt.integrator] == 0)
        {
            self.integrators[self.simulateOpt.integrator]
                    = Integrator::New(&self.simulateOpt, self.model);
        }

        self.integrator = self.integrators[self.simulateOpt.integrator];
    }
}

RoadRunnerData *RoadRunner::getSimulationResult()
{
    // set the data into the RoadRunnerData struct
    populateResult();

    return &impl->mRoadRunnerData;
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
        impl->model->computeAllRatesOfChange();
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

    case SelectionRecord::EIGENVALUE:
    {
        string species = record.p1;
        int index = impl->model->getFloatingSpeciesIndex(species);

        DoubleMatrix mat;
        if (impl->conservedMoietyAnalysis)
        {
            mat = getReducedJacobian();
        }
        else
        {
            mat = getFullJacobian();
        }

        vector<Complex> oComplex = ls::getEigenValues(mat);

        if(impl->mSelectionList.size() == 0)
        {
            throw("Tried to access record in empty impl->mSelectionList in getValue function: eigen");
        }

        if (oComplex.size() > index) //Becuase first one is time !?
        {
            return std::real(oComplex[index]);
        }
        return std::numeric_limits<double>::quiet_NaN();
    }
    break;
    case SelectionRecord::INITIAL_CONCENTRATION:
        impl->model->getFloatingSpeciesInitConcentrations(1, &record.index, &dResult);
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

void RoadRunner::load(const string& uriOrSbml, const LoadSBMLOptions *options)
{
    Mutex::ScopedLock lock(roadRunnerMutex);

    get_self();

    impl->mCurrentSBML = SBMLReader::read(uriOrSbml, impl->mFilename);

    //clear temp folder of roadrunner generated files, only if roadRunner instance == 1
    Log(lDebug)<<"Loading SBML into simulator";
    if (!impl->mCurrentSBML.size())
    {
        throw(CoreException("SBML string is empty!"));
    }

    delete impl->model;
    impl->model = 0;

    self.deleteIntegrators();

    if (options)
    {
        impl->conservedMoietyAnalysis = options->modelGeneratorOpt
                & LoadSBMLOptions::CONSERVED_MOIETIES;
        impl->model = impl->mModelGenerator->createModel(impl->mCurrentSBML, options->modelGeneratorOpt, impl->mFilename);
    }
    else
    {
        LoadSBMLOptions opt;
        opt.modelGeneratorOpt = getConservedMoietyAnalysis() ?
                opt.modelGeneratorOpt | LoadSBMLOptions::CONSERVED_MOIETIES :
                opt.modelGeneratorOpt & ~LoadSBMLOptions::CONSERVED_MOIETIES;
        impl->model = impl->mModelGenerator->createModel(impl->mCurrentSBML, opt.modelGeneratorOpt, impl->mFilename);
    }

    updateIntegrator();

    reset();

    if (!options || !(options->loadFlags & LoadSBMLOptions::NO_DEFAULT_SELECTIONS))
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

bool RoadRunner::unLoadModel()
{
    // The model owns the shared library (if it exists), when the model is deleted,
    // its dtor unloads the shared lib.
    if(impl->model)
    {
        delete impl->model;
        impl->model = NULL;
        return true;
    }
    return false;
}

//Reset the simulator back to the initial conditions specified in the SBML model
void RoadRunner::reset()
{
    if (impl->model)
    {
        // model gets set to before time = 0
        impl->model->reset();

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

bool RoadRunner::populateResult()
{
    vector<string> list(impl->mSelectionList.size());

    for(int i = 0; i < impl->mSelectionList.size(); ++i)
    {
        list[i] = impl->mSelectionList[i].to_string();
    }

    impl->mRoadRunnerData.setColumnNames(list);
    impl->mRoadRunnerData.setData(impl->simulationResult);
    return true;
}


std::vector<rr::SelectionRecord>& RoadRunner::getSelections()
{
    return impl->mSelectionList;
}


double RoadRunner::steadyState()
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }

    if (!impl->conservedMoietyAnalysis &&
            (Config::getBool(Config::ROADRUNNER_DISABLE_WARNINGS)
                & Config::ROADRUNNER_DISABLE_WARNINGS_STEADYSTATE) == 0)
    {
        Log(Logger::LOG_WARNING) << "Conserved Moiety Analysis is not enabled, steady state may fail with singular Jacobian";
        Log(Logger::LOG_WARNING) << "Conserved Moiety Analysis may be enabled via the conservedMoeityAnalysis property or "
                                    "via the configuration file or the Config class setValue, see roadrunner documentation";
        Log(Logger::LOG_WARNING) << "to remove this warning, set ROADRUNNER_DISABLE_WARNINGS to 1 or 3 in the config file";
    }

    SteadyStateSolver *steadyStateSolver = SteadyStateSolver::New(0, impl->model);

    if (impl->configurationXML.length() > 0)
    {
        Configurable::loadXmlConfig(impl->configurationXML, steadyStateSolver);
    }

    //Get a std vector for the solver
    vector<double> someAmounts(impl->model->getNumIndFloatingSpecies(), 0);
    impl->model->getFloatingSpeciesAmounts(someAmounts.size(), 0, &someAmounts[0]);

    double ss = steadyStateSolver->solve(someAmounts);
    if(ss < 0)
    {
        Log(Logger::LOG_ERROR)<<"Steady State solver failed...";
    }


    delete steadyStateSolver;

    return ss;
}




void RoadRunner::setConservedMoietyAnalysis(bool bValue)
{
    if(bValue == impl->conservedMoietyAnalysis)
    {
        Log(lDebug)<<"The compute and assign conservation laws flag already set to : "<<toString(bValue);
    }

    impl->conservedMoietyAnalysis = bValue;

    if(impl->model != NULL)
    {
        LoadSBMLOptions opt;

        // set the comput and assign cons flag
        opt.modelGeneratorOpt = bValue ?
                opt.modelGeneratorOpt | LoadSBMLOptions::CONSERVED_MOIETIES :
                opt.modelGeneratorOpt & ~LoadSBMLOptions::CONSERVED_MOIETIES;

        // have to reload
        opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::RECOMPILE;

        load(impl->mCurrentSBML, &opt);
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

    impl->model->evalReactionRates();

    double variableValue = 0;
    impl->model->getReactionRates(1, &reactionIndex, &variableValue);
    double parameterValue = impl->getParameterValue(parameterType, parameterIndex);
    if (variableValue == 0)
    {
        variableValue = 1e-12;
    }
    return getuEE(reactionName, parameterName, computeSteadyState) * parameterValue / variableValue;
}


double RoadRunner::getuEE(const string& reactionName, const string& parameterName)
{
    return getuEE(reactionName, parameterName, true);
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

        impl->model->convertToConcentrations();
        impl->model->evalReactionRates();

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

        impl->model->convertToConcentrations();

        impl->setParameterValue(parameterType, parameterIndex, originalParameterValue + hstep);
        impl->model->evalReactionRates();
        double fi = 0;
        impl->model->getReactionRates(1, &reactionIndex, &fi);

        impl->setParameterValue(parameterType, parameterIndex, originalParameterValue + 2*hstep);
        impl->model->evalReactionRates();
        double fi2 = 0;
        impl->model->getReactionRates(1, &reactionIndex, &fi2);

        impl->setParameterValue(parameterType, parameterIndex, originalParameterValue - hstep);
        impl->model->evalReactionRates();
        double fd = 0;
        impl->model->getReactionRates(1, &reactionIndex, &fd);

        impl->setParameterValue(parameterType, parameterIndex, originalParameterValue - 2*hstep);
        impl->model->evalReactionRates();
        double fd2 = 0;
        impl->model->getReactionRates(1, &reactionIndex, &fd2);

        // restore original value
        impl->setParameterValue(parameterType, parameterIndex, originalParameterValue);

        // Use instead the 5th order approximation double unscaledValue = (0.5/hstep)*(fi-fd);
        // The following separated lines avoid small amounts of roundoff error
        double f1 = fd2 + 8*fi;
        double f2 = -(8*fd + fi2);

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
    impl->model->convertToAmounts();
    impl->model->getStateVectorRate(impl->model->getTime(), 0);
}


const DoubleMatrix* RoadRunner::simulate(const SimulateOptions* opt)
{
    get_self();

    if (!self.model) {
        throw std::logic_error(gEmptyModelMessage);
    }

    _setSimulateOptions(opt);

    const double timeEnd = self.simulateOpt.duration + self.simulateOpt.start;
    const double timeStart = self.simulateOpt.start;

    // evalute the model with its current state
    self.model->getStateVectorRate(timeStart, 0, 0);

    // Variable Time Step Integration
    if (self.simulateOpt.integratorFlags & SimulateOptions::VARIABLE_STEP )
    {
        Log(Logger::LOG_NOTICE) << "Performing variable step integration";

        DoubleVectorList results;
        std::vector<double> row(self.mSelectionList.size());

        try
        {
            // add current state as first row
            getSelectedValues(row, timeStart);
            results.push_back(row);

            self.integrator->restart(timeStart);

            double tout = timeStart;


            while(tout < timeEnd)
            {
                Log(Logger::LOG_DEBUG) << "variable step, start: " << tout
                        << ", end: " << timeEnd;
                tout = self.integrator->integrate(tout, timeEnd);
                if (!isfinite(tout))
                {
                    // time step is at infinity so bail, but get the last value
                    getSelectedValues(row, timeEnd);
                    results.push_back(row);
                    break;
                }
                getSelectedValues(row, tout);
                results.push_back(row);
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
    else if(SimulateOptions::getIntegratorType(self.simulateOpt.integrator) ==
            SimulateOptions::STOCHASTIC)
    {
        Log(Logger::LOG_NOTICE)
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

    Log(Logger::LOG_DEBUG) << "Simulation done..";

    return &self.simulationResult;
}


double RoadRunner::integrate(double t0, double tf, const SimulateOptions* o)
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }

    _setSimulateOptions(o);

    try
    {
        impl->model->setTime(t0);
        return impl->integrator->integrate(t0, tf);
    }
    catch (EventListenerException& e)
    {
        Log(Logger::LOG_NOTICE) << e.what();
        return impl->model->getTime();
    }
}


double RoadRunner::oneStep(const double currentTime, const double stepSize, const bool reset)
{
    get_self();

    if (!self.model)
    {
        throw CoreException(gEmptyModelMessage);
    }

    self.copySimulateOpt.flags = reset ?
            self.copySimulateOpt.flags | SimulateOptions::RESET_MODEL :
            self.copySimulateOpt.flags & ~SimulateOptions::RESET_MODEL;

    _setSimulateOptions(0);

    try
    {
        return self.integrator->integrate(currentTime, stepSize);
    }
    catch (EventListenerException& e)
    {
        Log(Logger::LOG_NOTICE) << e.what();
        return self.model->getTime();
    }
}


DoubleMatrix RoadRunner::getEigenvalues()
{
    try
    {
        if (!impl->model)
        {
            throw CoreException(gEmptyModelMessage);
        }

        vector<Complex> vals = getEigenvaluesCpx();

        DoubleMatrix result(vals.size(), 2);

        for (int i = 0; i < vals.size(); i++)
        {
            result[i][0] = std::real(vals[i]);
            result[i][1] = imag(vals[i]);
        }
        return result;
    }
    catch (const Exception& e)
    {
        throw CoreException("Unexpected error from getEigenvalues()", e.Message());
    }
}



vector< Complex > RoadRunner::getEigenvaluesCpx()
{
    try
    {
        if (!impl->model)
        {
            throw CoreException(gEmptyModelMessage);
        }

        DoubleMatrix mat;
        if (impl->conservedMoietyAnalysis)
        {
           mat = getReducedJacobian();
        }
        else
        {
           mat = getFullJacobian();
        }
        return ls::getEigenValues(mat);
    }
    catch (const Exception& e)
    {
        throw CoreException("Unexpected error from getEigenvalues()", e.Message());
    }
}

DoubleMatrix RoadRunner::getFullJacobian()
{
    try
    {
        if (!impl->model)
        {
            throw CoreException(gEmptyModelMessage);
        }
        DoubleMatrix uelast = getUnscaledElasticityMatrix();

        // ptr to libstruct owned obj.
        DoubleMatrix *rsm;
        LibStructural *ls = getLibStruct();
        if (impl->conservedMoietyAnalysis)
        {
            rsm = ls->getReorderedStoichiometryMatrix();
        }
        else
        {
            rsm = ls->getStoichiometryMatrix();
        }
       return ls::mult(*rsm, uelast);

    }
    catch (const Exception& e)
    {
        throw CoreException("Unexpected error from fullJacobian()", e.Message());
    }
}

DoubleMatrix RoadRunner::getFullReorderedJacobian()
{
    try
    {
        if (impl->model)
        {
            LibStructural *ls = getLibStruct();
            DoubleMatrix uelast = getUnscaledElasticityMatrix();
            DoubleMatrix *rsm = ls->getStoichiometryMatrix();
            return mult(*rsm, uelast);
        }
        throw CoreException(gEmptyModelMessage);
    }
    catch (const Exception& e)
    {
        throw CoreException("Unexpected error from fullJacobian()", e.Message());
    }
}

DoubleMatrix RoadRunner::getReducedJacobian(double h)
{
    get_self();

    if (!self.model)
    {
        throw CoreException(gEmptyModelMessage);
    }

    if (h <= 0)
    {
        h = self.roadRunnerOptions.jacobianStepSize;
    }

    // need 2h for for central difference
    double inv2h = 1.0 / (2.0 * h);

    int svSize = self.model->getStateVector(0);
    int nIndSpecies = self.model->getNumIndFloatingSpecies();
    int nRates = self.model->getNumRules();

    // need 3 buffers, state vector and 2 for state vector
    // rate for central difference.
    std::vector<double> yv(svSize);
    std::vector<double> dy0v(svSize);
    std::vector<double> dy1v(svSize);

    double* y = &yv[0];
    double* dy0 = &dy0v[0];
    double* dy1 = &dy1v[0];



    // grab the current state vector
    self.model->getStateVector(y);

    // state vector is packed such that first elements are
    // rate rules, final elements are species.
    const int speciesZero = svSize - nIndSpecies;

    DoubleMatrix jac(nIndSpecies, nIndSpecies);

    double time = self.model->getTime();

    for (int i = speciesZero; i < nIndSpecies; ++i)
    {
        double savedVal = y[i];

        y[i] = savedVal + h;

        self.model->getStateVectorRate(time, y, dy0);

        y[i] = savedVal - h;

        self.model->getStateVectorRate(time, y, dy1);

        y[i] = savedVal;

        // matrix is row-major, so have to copy by elements
       for (int j = 0; j < nIndSpecies; ++j)
       {
           jac(j, i - speciesZero) =
                   (dy0[speciesZero + j] - dy1[speciesZero + j]) / (2.0*h) ;
       }
    }
    return jac;
}

DoubleMatrix RoadRunner::getLinkMatrix()
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }
    // pointer to owned matrix
    return *getLibStruct()->getLinkMatrix();
}

DoubleMatrix RoadRunner::getReducedStoichiometryMatrix()
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }
    // pointer to owned matrix
    return *getLibStruct()->getNrMatrix();
}

DoubleMatrix RoadRunner::getNrMatrix()
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }
    // pointer to owned matrix
    return *getLibStruct()->getNrMatrix();
}

DoubleMatrix RoadRunner::getFullStoichiometryMatrix()
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }
    // pointer to owned matrix
    return *getLibStruct()->getStoichiometryMatrix();
}

DoubleMatrix RoadRunner::getL0Matrix()
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }
    // the libstruct getL0Matrix returns a NEW matrix,
    // nice consistent API yes?!?!?
    DoubleMatrix *tmp = getLibStruct()->getL0Matrix();
    DoubleMatrix result = *tmp;
    delete tmp;
    return result;
}


DoubleMatrix RoadRunner::getConservationMatrix()
{
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
            }
            return mat;

       }
       throw CoreException(gEmptyModelMessage);
    }
    catch (const Exception& e)
    {
         throw CoreException("Unexpected error from getConservationMatrix()", e.Message());
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
        throw CoreException("Unexpected error from getNumberOfDependentSpecies()", e.Message());
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
        throw CoreException("Unexpected error from getNumberOfIndependentSpecies()", e.Message());
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
    // default should be species only ...
    vector<string> floatingSpecies = getFloatingSpeciesIds();
    impl->mSteadyStateSelection.resize(floatingSpecies.size());
    for (int i = 0; i < floatingSpecies.size(); i++)
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

    libsbml::Parameter* oParameter = model->getParameter(id);
    if (oParameter != NULL)
    {
        oParameter->setValue(value);
        return;
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


string RoadRunner::getCurrentSBML()
{
    libsbml::SBMLReader reader;
    std::stringstream stream;
    libsbml::SBMLDocument *doc = 0;
    libsbml::Model *model = 0;

    try {
        doc = reader.readSBMLFromString(impl->mCurrentSBML); // new doc
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
            setSBMLValue(model, array[i], value);
        }

        libsbml::SBMLWriter writer;
        writer.writeSBML(doc, stream);
    }
    catch(std::exception& e) {
        delete doc;
        doc = 0;
        throw(e);
    }

    delete doc;
    return stream.str();
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
        impl->model->convertToConcentrations();
        impl->model->evalReactionRates();
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
        impl->model->computeAllRatesOfChange();
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

// Help("Returns an array of boundary species concentrations")
vector<double> RoadRunner::getBoundarySpeciesConcentrations()
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }

    impl->model->convertToConcentrations();
    vector<double> result(impl->model->getNumBoundarySpecies(), 0);
    impl->model->getBoundarySpeciesConcentrations(result.size(), 0, &result[0]);
    return result;
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
        if (!impl->model->getConservedSumChanged())
        {
            impl->model->computeConservedTotals();
        }
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
vector<double> RoadRunner::getFloatingSpeciesConcentrations()
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }


    vector<double> result(impl->model->getNumFloatingSpecies(), 0);

    impl->model->convertToConcentrations();
    impl->model->getFloatingSpeciesConcentrations(result.size(), 0, &result[0]);
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
    impl->model->convertToAmounts();

    if (!impl->model->getConservedSumChanged()) impl->model->computeConservedTotals();
}

// Help("Set the concentrations for all floating species in the model")
void RoadRunner::setBoundarySpeciesConcentrations(const vector<double>& values)
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }

    impl->model->setBoundarySpeciesConcentrations(values.size(), 0, &values[0]);

    // TODO: No idea whatsoever why boundary species concentrations was also
    // setting the floating species concentrations???
    //
    //    for (int i = 0; i < values.size(); i++)
    //    {
    //        impl->mModel->setConcentration(i, values[i]);
    //        if ((impl->mModel->getModelData().numBoundarySpecies) > i)
    //        {
    //            impl->mModel->getModelData().boundarySpeciesConcentrations[i] = values[i];
    //        }
    //    }
    impl->model->convertToAmounts();
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

    if ((impl->model->getNumGlobalParameters()) &&
            (index < impl->model->getNumGlobalParameters() +
                    impl->model->getNumDepFloatingSpecies()))
    {
        impl->model->setConservedSumChanged(true);
    }
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
        double* data = new double[arraySize];

        impl->model->getGlobalParameterValues(impl->model->getNumGlobalParameters(), 0, data);

        impl->model->getConservedMoietyValues(impl->model->getNumDepFloatingSpecies(), 0, data + impl->model->getNumGlobalParameters());

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

        ParameterType parameterType;
        VariableType variableType;
        double originalParameterValue;
        int variableIndex;
        int parameterIndex;

        impl->model->convertToConcentrations();
        impl->model->evalReactionRates();

        // Check the variable name
        if ((variableIndex = impl->model->getReactionIndex(variableName)) >= 0)
        {
            variableType = vtFlux;
        }
        else if ((variableIndex = impl->model->getFloatingSpeciesIndex(variableName)) >= 0)
        {
            variableType = vtSpecies;
        }
        else
        {
            throw CoreException("Unable to locate variable: [" + variableName + "]");
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
            impl->model->convertToConcentrations();

           impl->setParameterValue(parameterType, parameterIndex, originalParameterValue + hstep);
            steadyState();
            impl->model->evalReactionRates();
            double fi = getVariableValue(variableType, variableIndex);

           impl->setParameterValue(parameterType, parameterIndex, originalParameterValue + 2*hstep);
            steadyState();
            impl->model->evalReactionRates();
            double fi2 = getVariableValue(variableType, variableIndex);

           impl->setParameterValue(parameterType, parameterIndex, originalParameterValue - hstep);
            steadyState();
            impl->model->evalReactionRates();
            double fd = getVariableValue(variableType, variableIndex);

           impl->setParameterValue(parameterType, parameterIndex, originalParameterValue - 2*hstep);
            steadyState();
            impl->model->evalReactionRates();
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
            steadyState();
            throw;
        }
    }
    catch (const Exception& e)
    {
        throw CoreException("Unexpected error from getuCC ()", e.Message());
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

    // Check the variable name
    if ((variableIndex = impl->model->getReactionIndex(variableName)) >= 0)
    {
        variableType = vtFlux;
    }
    else if ((variableIndex = impl->model->getFloatingSpeciesIndex(variableName)) >= 0)
    {
        variableType = vtSpecies;
    }
    else
    {
        throw CoreException("Unable to locate variable: [" + variableName + "]");
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

    steadyState();

    double variableValue = getVariableValue(variableType, variableIndex);
    double parameterValue = impl->getParameterValue(parameterType, parameterIndex);
    return getuCC(variableName, parameterName)*parameterValue/variableValue;
}


double RoadRunner::getUnscaledSpeciesElasticity(int reactionId, int speciesIndex)
{
    // TODO this will not work corrrectly where there are rate rules
    get_self();
    double value;
    double originalConc = 0;
    double result = std::numeric_limits<double>::quiet_NaN();

    // note setting init values auotmatically sets the current values to the
    // init values

    // this causes a reset, so need to save the current amounts to set them back
    // as init conditions.
    std::vector<double> conc(self.model->getNumFloatingSpecies());
    self.model->getFloatingSpeciesConcentrations(conc.size(), 0, &conc[0]);

    // save the original init values
    std::vector<double> initConc(self.model->getNumFloatingSpecies());
    self.model->getFloatingSpeciesInitConcentrations(initConc.size(), 0, &initConc[0]);

    // get the original value
    self.model->getFloatingSpeciesConcentrations(1, &speciesIndex, &originalConc);

    // now we start changing things
    try
    {
        // set init amounts to current amounts, restore them later.
        // have to do this as this is only way to set conserved moiety values
        self.model->setFloatingSpeciesInitConcentrations(conc.size(), 0, &conc[0]);

        // sanity check
        assert(originalConc == conc[speciesIndex]);
        double tmp = 0;
        self.model->getFloatingSpeciesInitConcentrations(1, &speciesIndex, &tmp);
        assert(originalConc == tmp);
        self.model->getFloatingSpeciesConcentrations(1, &speciesIndex, &tmp);
        assert(originalConc == tmp);

        // things check out, start fiddling...

        double hstep = self.mDiffStepSize*originalConc;
        if (fabs(hstep) < 1E-12)
        {
            hstep = self.mDiffStepSize;
        }

        value = originalConc + hstep;
        self.model->setFloatingSpeciesInitConcentrations(1, &speciesIndex, &value);

        double fi = 0;
        self.model->getReactionRates(1, &reactionId, &fi);

        value = originalConc + 2*hstep;
        self.model->setFloatingSpeciesInitConcentrations(1, &speciesIndex, &value);
        double fi2 = 0;
        self.model->getReactionRates(1, &reactionId, &fi2);

        value = originalConc - hstep;
        self.model->setFloatingSpeciesInitConcentrations(1, &speciesIndex, &value);
        double fd = 0;
        self.model->getReactionRates(1, &reactionId, &fd);

        value = originalConc - 2*hstep;
        self.model->setFloatingSpeciesInitConcentrations(1, &speciesIndex, &value);
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
        self.model->setFloatingSpeciesInitConcentrations(
                initConc.size(), 0, &initConc[0]);

        // only set the indep species, setting dep species is not permitted.
        self.model->setFloatingSpeciesConcentrations(
                self.model->getNumIndFloatingSpecies(), 0, &conc[0]);

        throw e;
    }

    // What ever happens, make sure we restore the species level
    self.model->setFloatingSpeciesInitConcentrations(
            initConc.size(), 0, &initConc[0]);

    // only set the indep species, setting dep species is not permitted.
    self.model->setFloatingSpeciesConcentrations(
            self.model->getNumIndFloatingSpecies(), 0, &conc[0]);

    return result;
}



DoubleMatrix RoadRunner::getUnscaledElasticityMatrix()
{
    get_self();

    if (!self.model)
    {
        throw CoreException(gEmptyModelMessage);
    }

    DoubleMatrix uElastMatrix(self.model->getNumReactions(), self.model->getNumFloatingSpecies());

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
    try
    {
        if (!impl->model)
        {
            throw CoreException(gEmptyModelMessage);
        }

        DoubleMatrix uelast = getUnscaledElasticityMatrix();

        DoubleMatrix result(uelast.RSize(), uelast.CSize());// = new double[uelast.Length][];
        impl->model->convertToConcentrations();
        impl->model->evalReactionRates();
        vector<double> rates(impl->model->getNumReactions());
        impl->model->getReactionRates(rates.size(), 0, &rates[0]);

        for (int i = 0; i < uelast.RSize(); i++)
        {
            // Rows are rates
            if (impl->model->getNumReactions() == 0 || rates[i] == 0)
            {
                string name;
                if(impl->mModelGenerator && impl->model->getNumReactions())
                {
                    name = impl->model->getReactionId(i);
                }
                else
                {
                    name = "none";
                }

                throw CoreException("Unable to compute elasticity, reaction rate [" + name + "] set to zero");
            }

            for (int j = 0; j < uelast.CSize(); j++) // Columns are species
            {
                double concentration = 0;
                impl->model->getFloatingSpeciesConcentrations(1, &j, &concentration);

                result[i][j] = uelast[i][j]*concentration/rates[i];
            }
        }
        return result;
    }
    catch (const Exception& e)
    {
        throw CoreException("Unexpected error from scaledElasticityMatrix()", e.Message());
    }
}

//        [Help("Compute the scaled elasticity for a given reaction and given species")]
double RoadRunner::getScaledFloatingSpeciesElasticity(const string& reactionName, const string& speciesName)
{
    try
    {
        if (!impl->model)
        {
            throw CoreException(gEmptyModelMessage);
        }
        int speciesIndex = 0;
        int reactionIndex = 0;

        impl->model->convertToConcentrations();
        impl->model->evalReactionRates();

        if ((speciesIndex = impl->model->getFloatingSpeciesIndex(speciesName)) < 0)
        {
            throw CoreException("Internal Error: unable to locate species name while computing unscaled elasticity");
        }
        if ((reactionIndex = impl->model->getReactionIndex(reactionName)) < 0)
        {
            throw CoreException("Internal Error: unable to locate reaction name while computing unscaled elasticity");
        }

        double concentration = 0;
        impl->model->getFloatingSpeciesConcentrations(1, &speciesIndex, &concentration);

        double reactionRate = 0;
        impl->model->getReactionRates(1, &reactionIndex, &reactionRate);
        return getUnscaledSpeciesElasticity(reactionIndex, speciesIndex) *
            concentration / reactionRate;

    }
    catch (const Exception& e)
    {
        throw CoreException("Unexpected error from scaledElasticityMatrix()", e.Message());
    }
}


// Use the formula: ucc = -L Jac^-1 Nr
// [Help("Compute the matrix of unscaled concentration control coefficients")]
DoubleMatrix RoadRunner::getUnscaledConcentrationControlCoefficientMatrix()
{
    try
    {
        if (!impl->model)
        {
            throw CoreException(gEmptyModelMessage);
        }

        impl->simulateOpt.start = 0;
        impl->simulateOpt.duration = 50.0;
        impl->simulateOpt.steps = 1;

        simulate(); //This will crash, because numpoints == 1, not anymore, numPoints = 2 if numPoints <= 1
        if (steadyState() > impl->mSteadyStateThreshold)
        {
            if (steadyState() > 1E-2)
            {
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
        return T4;
    }
    catch (const Exception& e)
    {
        throw CoreException("Unexpected error from getUnscaledConcentrationControlCoefficientMatrix()", e.Message());
    }
}

// [Help("Compute the matrix of scaled concentration control coefficients")]
DoubleMatrix RoadRunner::getScaledConcentrationControlCoefficientMatrix()
{
    try
    {
        if (impl->model)
        {
            DoubleMatrix ucc = getUnscaledConcentrationControlCoefficientMatrix();

            if (ucc.size() > 0 )
            {
                impl->model->convertToConcentrations();
                impl->model->evalReactionRates();
                for (int i = 0; i < ucc.RSize(); i++)
                {
                    for (int j = 0; j < ucc.CSize(); j++)
                    {
                        double conc = 0;
                        impl->model->getFloatingSpeciesConcentrations(1, &i, &conc);
                        if(conc != 0.0)
                        {
                            double rate = 0;
                            impl->model->getReactionRates(1, &j, &rate);
                            ucc[i][j] = ucc[i][j] * rate / conc;
                        }
                        else
                        {
                            throw(Exception("Dividing with zero"));
                        }
                    }
                }
            }
            return ucc;
        }
        else
        {
            throw CoreException(gEmptyModelMessage);
        }
    }
    catch (const Exception& e)
    {
        throw CoreException("Unexpected error from getScaledConcentrationControlCoefficientMatrix()", e.Message());
    }
}

// Use the formula: ucc = elast CS + I
// [Help("Compute the matrix of unscaled flux control coefficients")]
DoubleMatrix RoadRunner::getUnscaledFluxControlCoefficientMatrix()
{
    try
    {
        if (impl->model)
        {
            DoubleMatrix ucc = getUnscaledConcentrationControlCoefficientMatrix();
            DoubleMatrix uee = getUnscaledElasticityMatrix();

            DoubleMatrix T1 = mult(uee, ucc);

            // Add an identity matrix I to T1, that is add a 1 to every diagonal of T1
            for (int i=0; i<T1.RSize(); i++)
                T1[i][i] = T1[i][i] + 1;
            return T1;//Matrix.convertToDouble(T1);
        }
        else throw CoreException(gEmptyModelMessage);
    }
    catch (CoreException&)
    {
        throw;
    }
    catch (const Exception& e)
    {
        throw CoreException("Unexpected error from getUnscaledFluxControlCoefficientMatrix()", e.Message());
    }
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
            impl->model->convertToConcentrations();
            impl->model->evalReactionRates();
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
        throw CoreException("Unexpected error from getScaledFluxControlCoefficientMatrix()", e.Message());
    }
}

// Help("Returns the initially loaded model as SBML")
string RoadRunner::getSBML()
{
    return impl->mCurrentSBML;
}

// Help(
//            "Change the initial conditions to another concentration vector (changes only initial conditions for floating Species)")
void RoadRunner::changeInitialConditions(const vector<double>& ic)
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }

    impl->model->setFloatingSpeciesConcentrations(ic.size(), 0, &ic[0]);
    impl->model->setFloatingSpeciesInitConcentrations(ic.size(), 0, &ic[0]);
    impl->model->convertToAmounts();
    impl->model->computeConservedTotals();
}

// Help("Returns the current vector of reactions rates")
vector<double> RoadRunner::getReactionRates()
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }
    impl->model->convertToConcentrations();
    impl->model->evalReactionRates();

    vector<double> rates(impl->model->getNumReactions());
    impl->model->getReactionRates(rates.size(), 0, &rates[0]);
    return rates;
}

Integrator* RoadRunner::getIntegrator()
{
    return impl->integrator;
}


void RoadRunner::setValue(const string& sId, double dValue)
{
    if (!impl->model)
    {
        throw CoreException(gEmptyModelMessage);
    }

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
    std::vector<std::string> theList;
    SelectionRecord record;

    theList.push_back("time");

    int nrOfVars = o.variables.size();

    for(int i = 0; i < o.amounts.size(); i++)
    {
        theList.push_back(o.amounts[i]);        //In the setSelection list below, the [] selects the correct 'type'
    }

    for(int i = 0; i < o.concentrations.size(); i++)
    {
        theList.push_back("[" + o.concentrations[i] + "]");
    }

    //We may have variables
    //A variable 'exists' only in "variables", not in the amount or concentration section

    for(int i = 0; i < o.variables.size(); i++)
    {
        string aVar = o.variables[i];

        if ((find(o.amounts.begin(), o.amounts.end(), aVar) == o.amounts.end()) &&
                (find(o.concentrations.begin(), o.concentrations.end(), aVar)
                        == o.concentrations.end()))
        {
            theList.push_back(o.variables[i]);
        }
    }

    return theList;
}


_xmlNode *RoadRunner::createConfigNode()
{
    // the top level capabilities
    _xmlNode *capies = Configurable::createCapabilitiesNode("RoadRunner",
            "RoadRunner Capabilities");

    // capability for this class
    _xmlNode *caps = Configurable::createCapabilityNode("RoadRunner Core", "",
            "Core RoadRunner Capability");

    Configurable::addChild(caps, Configurable::createParameterNode("Conservation",
                    "enables (=true) or disables (=false) the conservation analysis of models for timecourse simulations.",
                    (int) impl->conservedMoietyAnalysis));

    Configurable::addChild(capies, caps);

    // capability for child objects
    Configurable *intConf = dynamic_cast<Configurable*>(impl->integrator);
    if (intConf)
    {
        Configurable::addChild(capies, intConf->createConfigNode());
    }


    // nleq only exists during steadyState, so we need to create a tmp
    // one and load it with the xml if we given.
    SteadyStateSolver *ss = SteadyStateSolver::New(0, 0);

    if (impl->configurationXML.length() > 0)
    {
        Configurable::loadXmlConfig(impl->configurationXML, ss);
    }

    Configurable::addChild(capies, ss->createConfigNode());

    delete ss;

    return capies;
}

void RoadRunner::loadConfig(const _xmlDoc* doc)
{
    Configurable *intConf = dynamic_cast<Configurable*>(impl->integrator);
    if (intConf)
    {
        intConf->loadConfig(doc);
    }
}

std::string RoadRunner::getConfigurationXML()
{
    return Configurable::xmlFromConfigNode(createConfigNode());
}

void RoadRunner::setConfigurationXML(const std::string& xml)
{
    impl->configurationXML = xml;
    Configurable::loadXmlConfig(xml, this);
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
    case SelectionRecord::EIGENVALUE:
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
        const string& parameterName, const string& variableName,
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
        impl->model->convertToConcentrations();
        impl->model->evalReactionRates();

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
        impl->model->convertToConcentrations();
        impl->model->evalReactionRates();
        impl->model->getReactionRates(1, &reactionIndex, &fi);

        impl->changeParameter(parameterType, reactionIndex, parameterIndex, originalParameterValue, 2.0*hstep);
        impl->model->evalReactionRates();
        impl->model->getReactionRates(1, &reactionIndex, &fi2);

        impl->changeParameter(parameterType, reactionIndex, parameterIndex, originalParameterValue, -hstep);
        impl->model->evalReactionRates();
        impl->model->getReactionRates(1, &reactionIndex, &fd);

        impl->changeParameter(parameterType, reactionIndex, parameterIndex, originalParameterValue, -2.0*hstep);
        impl->model->evalReactionRates();
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

        if (types & SelectionRecord::EIGENVALUE)
        {
            std::list<std::string> eigen;
            impl->model->getIds(SelectionRecord::FLOATING_AMOUNT, eigen);

            for (std::list<std::string>::const_iterator i =
                    eigen.begin(); i != eigen.end(); ++i)
            {
                ids.push_back("eigen(" + *i + ")");
            }
        }
    }
}

int RoadRunner::getSupportedIdTypes()
{
    int types = impl->model ? impl->model->getSupportedIdTypes() : 0;

    return types | SelectionRecord::ELASTICITY |
            SelectionRecord::UNSCALED_ELASTICITY |
            SelectionRecord::CONTROL |
            SelectionRecord::UNSCALED_CONTROL |
            SelectionRecord::EIGENVALUE;
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

vector<string> RoadRunner::getFloatingSpeciesInitialConditionIds()
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

vector<string> RoadRunner::getEigenvalueIds()
{
    std::list<std::string> list;

    getIds(SelectionRecord::EIGENVALUE, list);

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

void RoadRunner::_setSimulateOptions(const SimulateOptions* opt)
{
    get_self();

    if (opt && opt != &self.copySimulateOpt)
    {
        self.copySimulateOpt = *opt;
    }

    if (self.copySimulateOpt.duration < 0 || self.copySimulateOpt.start < 0
            || self.copySimulateOpt.steps <= 0 )
    {
        throw CoreException("duration, startTime and steps must be positive");
    }


    self.simulateOpt = self.copySimulateOpt;

    // This one creates the list of what we will look at in the result
    // uses values (potentially) from simulate options.
    createTimeCourseSelectionList();

    // updates the integrator to what was specified by simulateOptions,
    // no effect if already using this integrator.
    updateIntegrator();

    self.integrator->setSimulateOptions(&self.simulateOpt);

    if (self.simulateOpt.flags & SimulateOptions::RESET_MODEL)
    {
        reset(); // reset back to initial conditions
    }

    // set how the result should be returned to python
    self.mRoadRunnerData.structuredResult =
            self.simulateOpt.flags & SimulateOptions::STRUCTURED_RESULT;
}

} //namespace



#if defined(_WIN32)
#pragma comment(lib, "IPHLPAPI.lib") //Becuase of poco needing this
#endif

