#pragma hdrstop
#include "rrCompiledExecutableModel.h"
#include "rrCModelGenerator.h"
#include "rrUtils.h"
#include "rrCompiledModelState.h"
#include "rrCModelDataUtil.h"
#include "rrSelectionRecord.h"
#include "rrException.h"
#include "rrLogger.h"

#include <limits>
#include <iostream>

/**
 * checks if the bitfield value has all the required flags
 * from type
 */
inline bool checkExact(uint32_t type, uint32_t value) {
    return (value & type) == type;
}

typedef string (rr::ExecutableModel::*getNamePtr)(int);
typedef int (rr::ExecutableModel::*getNumPtr)();

// make this static here, hide our implementation...
static void addIds(rr::ExecutableModel *model,
        getNumPtr numFunc, getNamePtr nameFunc,
        std::list<std::string>& ids)
{
    const int num = (model->*numFunc)();

    for(int i = 0; i < num; i++)
    {
        const std::string& name  = (model->*nameFunc)(i);
        ids.push_back(name);
    }
}

using namespace std;
namespace rr
{

CompiledExecutableModel::CompiledExecutableModel(const ModelSymbols& symbols, ModelSharedLibrary* dll) :
mDummyInt(0),
mDummyDouble(0),
mDummyDoubleArray(new double[1]),
mData(),
ms(symbols),
mIsInitialized(false),
mConservedSumChanged(false),
mDLL(dll)
{
    //Zero data structure..
    initModelData(mData);
    mDummyDoubleArray[0] = 1;

    if(mDLL->isLoaded())
    {
        setupDLLFunctions();
        setupModelData();
    }
    else
    {
        Log(Logger::LOG_ERROR)<<"The Model DLL is not loaded in CompiledExecutableModel ctor..";
    }
}

CompiledExecutableModel::~CompiledExecutableModel()
{
    Log(lDebug3) << "in " << __FUNC__ << "\n";
    // only free buffers, mData is stack allocated.
    freeModelDataBuffers(mData);
    delete [] mDummyDoubleArray;
    if(mDLL)
    {
        mDLL->unload();
    }
    delete mDLL;

    while (!modelStates.empty())
    {
        CompiledModelState *state = modelStates.top();
        modelStates.pop();
        delete state;
    }
}

string CompiledExecutableModel::getModelName()
{
    return mData.modelName;
}

void CompiledExecutableModel::setTime(double _time)
{
    mData.time = _time;
}

double CompiledExecutableModel::getTime()
{
    return mData.time;
}

/////////////////// The following used to be in IModel
int CompiledExecutableModel::getNumIndFloatingSpecies()
{
    return mData.numIndependentSpecies;
}

int CompiledExecutableModel::getNumDepFloatingSpecies()
{
    return mData.numDependentSpecies;
}

int CompiledExecutableModel::getNumFloatingSpecies()
{
    return mData.numFloatingSpecies;
}

int CompiledExecutableModel::getNumBoundarySpecies()
{
    return mData.numBoundarySpecies;
}

int CompiledExecutableModel::getNumGlobalParameters()
{
    return mData.numGlobalParameters;
}

int CompiledExecutableModel::getNumCompartments()
{
    return mData.numCompartments;
}

int CompiledExecutableModel::getNumReactions()
{
    return mData.numReactions;
}

int CompiledExecutableModel::getNumEvents()
{
    return mData.numEvents;
}

double CompiledExecutableModel::getAmount(const int i)
{
    return (mData.floatingSpeciesAmounts ) ? mData.floatingSpeciesAmounts[i] : -1;
}

int CompiledExecutableModel::getReactionIndex(const std::string& reactionName)
{
    int result = -1;
    return ms.mReactionList.find(reactionName, result) ? result : -1;
}

std::string CompiledExecutableModel::getReactionId(int index)
{
    return ms.mReactionList[index].name;
}

int CompiledExecutableModel::getGlobalParameterIndex(const std::string& name)
{
    int result = -1;
    return ms.mGlobalParameterList.find(name, result) ? result : -1;
}

string CompiledExecutableModel::getGlobalParameterId(int index)
{
    return ms.mGlobalParameterList[index].name;
}

int CompiledExecutableModel::getBoundarySpeciesIndex(const string& name)
{
    int result = -1;
    return ms.mBoundarySpeciesList.find(name, result) ? result : -1;
}

string CompiledExecutableModel::getBoundarySpeciesId(int index)
{
    return ms.mBoundarySpeciesList[index].name;
}

int CompiledExecutableModel::getBoundarySpeciesCompartmentIndex(int index)
{
    // mModel->getCompartments().find(mModel->getBoundarySpecies()[record.index].compartmentName, nIndex))
    string compartmentName = ms.mBoundarySpeciesList[index].compartmentName;
    int compartmentIndex = -1;
    return ms.mCompartmentList.find(compartmentName, compartmentIndex) ? compartmentIndex : -1;
}

int CompiledExecutableModel::getCompartmentIndex(const string& name)
{
    int result = -1;
    return ms.mCompartmentList.find(name, result) ? result : -1;
}

string CompiledExecutableModel::getCompartmentId(int index)
{
    return ms.mCompartmentList[index].name;
}

int CompiledExecutableModel::getFloatingSpeciesIndex(const string& name)
{
    int result = -1;
    return ms.mFloatingSpeciesConcentrationList.find(name, result) ? result : -1;
}

string CompiledExecutableModel::getFloatingSpeciesId(int index)
{
    return ms.mFloatingSpeciesConcentrationList[index].name;
}

int CompiledExecutableModel::pushState(unsigned options)
{
    CompiledModelState *state = new CompiledModelState(*this);
    modelStates.push(state);
    return modelStates.size();
}

int CompiledExecutableModel::popState(unsigned options)
{
    if (modelStates.size() > 0)
    {
        CompiledModelState *state = modelStates.top();
        modelStates.pop();
        if (!(options && PopDiscard))
        {
            state->AssignToModel(*this);
        }
        delete state;
    }
    return modelStates.size();
}

void CompiledExecutableModel::evalInitialConditions()
{
    setCompartmentVolumes();
    initializeInitialConditions();
    setParameterValues();
    setCompartmentVolumes();
    setBoundaryConditions();
    setInitialConditions();
    convertToAmounts();
    evalInitialAssignments();

    computeRules();
    convertToAmounts();

    if (ms.mComputeAndAssignConsevationLaws)
    {
        computeConservedTotals();
    }
}

bool CompiledExecutableModel::getConservedSumChanged()
{
    return mConservedSumChanged;
}

void CompiledExecutableModel::setConservedSumChanged(bool val)
{
    mConservedSumChanged = val;
}

int CompiledExecutableModel::getStateVector(double* stateVector)
{
    if (stateVector == 0)
    {
        return mData.numRateRules + mData.numIndependentSpecies;
    }

    vector<double> dTemp(mData.numRateRules, 0);
    getRateRuleValues(mData.numRateRules==0 ? NULL : &dTemp[0]);

    for (int i = 0; i < mData.numRateRules; i++)
    {
        stateVector[i] = dTemp[i];
    }

    for (int i = 0; i < mData.numIndependentSpecies; i++)
    {
        stateVector[i + mData.numRateRules] = getAmount(i);
    }

    return mData.numRateRules + mData.numIndependentSpecies;
}

int CompiledExecutableModel::setStateVector(const double* stateVector)
{
    const double *floatingSpeciesAmounts = stateVector + mData.numRateRules;

    updateDependentSpeciesValues();

    for (int i = 0; i < mData.numIndependentSpecies; i++)
    {
        mData.floatingSpeciesAmounts[i] = floatingSpeciesAmounts[i];
    }

    computeRules();

    setRateRuleValues(stateVector);

    computeAllRatesOfChange();

    return mData.numRateRules + mData.numRateRules;
}

bool CompiledExecutableModel::setupDLLFunctions()
{
    //Exported functions in the dll need to be assigned to a function pointer here..
    if(!mDLL->isLoaded())
    {
        Log(Logger::LOG_ERROR)<<"DLL handle not valid in SetupModel function";
        return false;
    }

    //Load functions..
    cInitModel                          = (c_int_MDS)                      mDLL->getSymbol("InitModel");
    cInitModelData                      = (c_int_MDS)                      mDLL->getSymbol("InitModelData");
    cinitializeInitialConditions        = (c_void_MDS)                     mDLL->getSymbol("initializeInitialConditions");
    csetParameterValues                 = (c_void_MDS)                     mDLL->getSymbol("setParameterValues");
    csetCompartmentVolumes              = (c_void_MDS)                     mDLL->getSymbol("setCompartmentVolumes");
    cgetNumLocalParameters              = (c_int_MDS_int)                  mDLL->getSymbol("getNumLocalParameters");
    csetBoundaryConditions              = (c_void_MDS)                     mDLL->getSymbol("setBoundaryConditions");
    csetInitialConditions               = (c_void_MDS)                     mDLL->getSymbol("setInitialConditions");
    cevalInitialAssignments             = (c_void_MDS)                     mDLL->getSymbol("evalInitialAssignments");
    ccomputeRules                       = (c_void_MDS)                     mDLL->getSymbol("computeRules");
    cconvertToAmounts                   = (c_void_MDS)                     mDLL->getSymbol("convertToAmounts");
    ccomputeConservedTotals             = (c_void_MDS)                     mDLL->getSymbol("computeConservedTotals");
    cgetConcentration                   = (c_double_MDS_int)               mDLL->getSymbol("getConcentration");
    cGetCurrentValues                   = (c_doubleStar_MDS)               mDLL->getSymbol("GetCurrentValues");
    cevalModel                          = (c_void_MDS_double_doubleStar)   mDLL->getSymbol("__evalModel");
    cconvertToConcentrations            = (c_void_MDS)                     mDLL->getSymbol("convertToConcentrations");
    cevalEvents                         = (c_void_MDS_double_doubleStar)   mDLL->getSymbol("evalEvents");
    cupdateDependentSpeciesValues       = (c_void_MDS)                     mDLL->getSymbol("updateDependentSpeciesValues");
    ccomputeAllRatesOfChange            = (c_void_MDS)                     mDLL->getSymbol("computeAllRatesOfChange");
    cAssignRates_a                      = (c_void_MDS)                     mDLL->getSymbol("AssignRatesA");
    cAssignRates_b                      = (c_void_MDS_doubleStar)          mDLL->getSymbol("AssignRatesB");
    ctestConstraints                    = (c_void_MDS)                     mDLL->getSymbol("testConstraints");
    cresetEvents                        = (c_void_MDS)                     mDLL->getSymbol("resetEvents");
    cInitializeRateRuleSymbols          = (c_void_MDS)                     mDLL->getSymbol("InitializeRateRuleSymbols");
    cInitializeRates                    = (c_void_MDS)                     mDLL->getSymbol("InitializeRates");
    csetConcentration                   = (c_void_MDS_int_double)          mDLL->getSymbol("setConcentration");
    cComputeReactionRates               = (c_void_MDS)                     mDLL->getSymbol("computeReactionRates");
    ccomputeEventPriorities             = (c_void_MDS)                     mDLL->getSymbol("computeEventPriorities");
    return true;
}

bool CompiledExecutableModel::setupModelData()
{
    // zero out the structure
    initModelData(mData);

    // set the buffer sizes
    mData.numIndependentSpecies         = ms.mNumIndependentSpecies;
    mData.numDependentSpecies           = ms.mNumDependentSpecies;
    mData.numGlobalParameters           = ms.mGlobalParameterList.size();
    mData.numReactions                  = ms.mReactionList.size();
    mData.numEvents                     = ms.mNumEvents;
    mData.numFloatingSpecies            = ms.mFloatingSpeciesConcentrationList.size();
    mData.numRateRules                  = ms.mRateRules.size();
    mData.numCompartments               = ms.mCompartmentList.size();
    mData.numBoundarySpecies            = ms.mNumBoundarySpecies;
    mData.srSize                        = ms.mNumModifiableSpeciesReferences;
    mData.eventPrioritiesSize           = ms.mNumEvents;
    mData.eventStatusArraySize          = ms.mNumEvents;
    mData.previousEventStatusArraySize  = ms.mNumEvents;
    mData.eventPersistentTypeSize       = ms.mNumEvents;
    mData.eventTestsSize                = ms.mNumEvents;
    mData.eventTypeSize                 = ms.mNumEvents;

    // allocate the data buffers
    string test = ms.mModelName;
    allocModelDataBuffers(mData, test);

    if(cInitModel)
    {
        cInitModel(&mData);
    }
    return true;
}

void CompiledExecutableModel::setCompartmentVolumes()
{
    if(!csetCompartmentVolumes)
    {
        Log(Logger::LOG_ERROR)<<"Tried to call NULL function in "<<__FUNCTION__;
        return;
    }

    csetCompartmentVolumes(&mData);
}

void  CompiledExecutableModel::setConcentration(int index, double value)
{
    if(!csetConcentration)
    {
        Log(Logger::LOG_ERROR)<<"Tried to call NULL function in "<<__FUNCTION__;
        return;
    }

    csetConcentration(&mData, index, value);
}

void  CompiledExecutableModel::evalReactionRates ()
{
    if(!cComputeReactionRates)
    {
        Log(Logger::LOG_ERROR)<<"Tried to call NULL function in "<<__FUNCTION__;
        return;
    }

    cComputeReactionRates(&mData);
}

void CompiledExecutableModel::getRateRuleValues(double *rateRuleValues)
{
    vector<double> vals;
    if(!cGetCurrentValues)
    {
        Log(Logger::LOG_ERROR)<<"Tried to call NULL function in "<<__FUNCTION__;
    }

    // in CModelGenerator::writeComputeRules, in effect, the following
    // line is writen in the generated code:
    // dResult = (double*) calloc( numAdditionalRates() sizeof(double))
    double* values = cGetCurrentValues(&mData);

    int count = ms.mRateRules.size();
    if(values)
    {
        for(int i = 0; i < count; i++)
        {
            rateRuleValues[i] = values[i];
        }
    }

    // allocated in C, free'd here
    // for now, we'll just leak in Windows.
#if defined (__unix__) || defined(__APPLE__)
    free(values);
#endif

}

double CompiledExecutableModel::getFloatingSpeciesConcentration(int index)
{
    if(!cgetConcentration)
    {
        Log(Logger::LOG_ERROR)<<"Tried to call NULL function in "<<__FUNCTION__;
        return 0;
    }

    return cgetConcentration(&mData, index);
}

int CompiledExecutableModel::getNumLocalParameters(int reactionId)
{
    if(!cgetNumLocalParameters)
    {
        Log(Logger::LOG_ERROR)<<"Tried to call NULL function in "<<__FUNCTION__;
        return 0;
    }

    return cgetNumLocalParameters(&mData, reactionId);
}

void CompiledExecutableModel::initializeInitialConditions()
{
    if(!cinitializeInitialConditions)
    {
        Log(Logger::LOG_ERROR)<<"Tried to call NULL function in "<<__FUNCTION__;
        return;
    }
    cinitializeInitialConditions(&mData);
}

//void CompiledExecutableModel::setInitialConditions(){}
void CompiledExecutableModel::setParameterValues()
{
    if(!csetParameterValues)
    {
        Log(Logger::LOG_ERROR)<<"Tried to call NULL function in "<<__FUNCTION__;
        return;
    }
    csetParameterValues(&mData);
}

void CompiledExecutableModel::setBoundaryConditions()
{
    if(!csetBoundaryConditions)
    {
        Log(Logger::LOG_ERROR)<<"Tried to call NULL function in "<<__FUNCTION__;
        return;
    }
    csetBoundaryConditions(&mData);
}

void CompiledExecutableModel::initializeRates()
{
    if(!cInitializeRates)
    {
        Log(Logger::LOG_ERROR)<<"Tried to call NULL function in "<<__FUNCTION__;
        return;
    }
    cInitializeRates(&mData);
}


void CompiledExecutableModel::setRateRuleValues(const double *_rates)
{
    if(!cAssignRates_b)
    {
        Log(Logger::LOG_ERROR)<<"Tried to call NULL function in "<<__FUNCTION__;
        return;
    }
    cAssignRates_b(&mData, _rates);
}

void CompiledExecutableModel::computeConservedTotals()
{
    if(!ccomputeConservedTotals)
    {
        Log(Logger::LOG_ERROR)<<"Tried to call NULL function in "<<__FUNCTION__;
        return;
    }
    ccomputeConservedTotals(&mData);
}

void CompiledExecutableModel::computeEventPriorites()
{
    if(!ccomputeEventPriorities)
    {
        Log(Logger::LOG_ERROR)<<"Tried to call NULL function in "<<__FUNCTION__;
        return;
    }
    ccomputeEventPriorities(&mData);
}

void CompiledExecutableModel::convertToAmounts()
{
    if(!cconvertToAmounts)
    {
        Log(Logger::LOG_ERROR)<<"Tried to call NULL function in "<<__FUNCTION__;
        return;
    }
    cconvertToAmounts(&mData);
}

void CompiledExecutableModel::convertToConcentrations()
{
    if(!cconvertToConcentrations)
    {
        Log(Logger::LOG_ERROR)<<"Tried to call NULL function in "<<__FUNCTION__;
        return;
    }
    cconvertToConcentrations(&mData);
}

void CompiledExecutableModel::updateDependentSpeciesValues()
{
    if(!cupdateDependentSpeciesValues)
    {
        Log(Logger::LOG_ERROR)<<"Tried to call NULL function in "<<__FUNCTION__;
        return;
    }

    cupdateDependentSpeciesValues(&mData);
}


void CompiledExecutableModel::computeRules()
{
    if(!ccomputeRules)
    {
        Log(Logger::LOG_ERROR)<<"Tried to call NULL function in "<<__FUNCTION__;
        return;
    }

    ccomputeRules(&mData);
}

void CompiledExecutableModel::setInitialConditions()
{
    if(!csetInitialConditions)
    {
        Log(Logger::LOG_ERROR)<<"Tried to call NULL function in "<<__FUNCTION__;
        return;
    }

    csetInitialConditions(&mData);
}

void CompiledExecutableModel::computeAllRatesOfChange()
{
    if(!ccomputeAllRatesOfChange)
    {
        Log(Logger::LOG_ERROR)<<"Tried to call NULL function in "<<__FUNCTION__;
        return;
    }
    ccomputeAllRatesOfChange(&mData);
}

void CompiledExecutableModel::getStateVectorRate(double timein, const double *y, double *dydt)
{
    if(!cevalModel)
    {
        Log(Logger::LOG_ERROR)<<"Tried to call NULL function in "<<__FUNCTION__;
        return;
    }

    if (y == 0)
    {
        // use current state
        int sv = getStateVector(0);
        vector<double> currentState(sv, 0.0);
        getStateVector(sv==0 ? NULL : &currentState[0]);
        cevalModel(&mData, timein, sv==0 ? NULL : &currentState[0]);
    }
    else
    {
        // use the given state
        cevalModel(&mData, timein, y);
    }

    if (dydt)
    {
        memcpy(dydt, mData.rateRuleRates, mData.numRateRules * sizeof(double));

        memcpy(&dydt[mData.numRateRules], mData.floatingSpeciesAmountRates,
                mData.numIndependentSpecies * sizeof(double));
    }
}

void CompiledExecutableModel::evalEvents(const double timeIn, const double*y)
{
    if(!cevalEvents)
    {
        Log(Logger::LOG_ERROR)<<"Tried to call NULL function in "<<__FUNCTION__;
        return;
    }

    if (y == 0)
    {
        // use current state
        int sv = getStateVector(0);
        vector<double> currentState(sv, 0.0);
        getStateVector(sv==0 ? NULL : &currentState[0]);
        cevalEvents(&mData, timeIn, sv==0 ? NULL : &currentState[0]);
    }
    else
    {
        cevalEvents(&mData, timeIn, y);
    }
}

void CompiledExecutableModel::resetEvents()
{
    if(!cresetEvents)
    {
        Log(Logger::LOG_ERROR)<<"Tried to call NULL function in "<<__FUNCTION__;
        return;
    }

    cresetEvents(&mData);
}

void CompiledExecutableModel::evalInitialAssignments()
{
    if(!cevalInitialAssignments)
    {
        Log(Logger::LOG_ERROR)<<"Tried to call NULL function in "<<__FUNCTION__;
        return;
    }

    cevalInitialAssignments(&mData);
}

void CompiledExecutableModel::testConstraints()
{
    if(!ctestConstraints)
    {
        Log(Logger::LOG_ERROR)<<"Tried to call NULL function in "<<__FUNCTION__;
        throw(Exception("Problem in testConstraints"));
    }

    ctestConstraints(&mData);
}

void CompiledExecutableModel::initializeRateRuleSymbols()
{
    if(!cInitializeRateRuleSymbols)
    {
        Log(Logger::LOG_ERROR)<<"Tried to call NULL function in "<<__FUNCTION__;
        return;
    }

    cInitializeRateRuleSymbols(&mData);
}

string CompiledExecutableModel::getInfo()
{
    stringstream info;
    info << "CompiledExecutableModel" << endl;
    info << "ModelName: "             <<  getModelName()<<endl;
    info << "Model DLL Loaded: "      << (mDLL->isLoaded() ? "true" : "false")    <<endl;
    info << "Initialized: "           << (mIsInitialized ? "true" : "false")    <<endl;
    return info.str();
}

void CompiledExecutableModel::reset(int options)
{
    reset();
}


void CompiledExecutableModel::reset()
{
    setTime(0.0);

    // Reset the event flags
    resetEvents();
    setCompartmentVolumes();
    setInitialConditions();
    convertToAmounts();

    // in case we have ODE rules we should assign those as initial values
    initializeRateRuleSymbols();
    initializeRates();

    // and of course initial assignments should override anything
    evalInitialAssignments();
    convertToAmounts();

    // also we might need to set some initial assignment rules.
    convertToConcentrations();
    computeRules();
    initializeRates();
    initializeRateRuleSymbols();
    evalInitialAssignments();
    computeRules();

    convertToAmounts();

    mAssignments.clear();
}

void CompiledExecutableModel::print(std::ostream &stream)
{
    stream << "CompiledExecutableModel" << endl;
    stream << mData;
}

int CompiledExecutableModel::getNumRateRules()
{
    return ms.mNumRules;
}

int CompiledExecutableModel::getFloatingSpeciesAmounts(int len, const int* indx,
        double* values)
{
    for (int i = 0; i < len; ++i)
    {
        int j = indx ? indx[i] : i;
        values[i] = mData.floatingSpeciesAmounts[j];
    }
    return len;
}

int CompiledExecutableModel::getFloatingSpeciesConcentrations(int len,
        const int* indx, double* values)
{
    for (int i = 0; i < len; ++i)
    {
        int j = indx ? indx[i] : i;
        values[i] = mData.floatingSpeciesConcentrations[j];
    }
    return len;
}

int CompiledExecutableModel::setFloatingSpeciesConcentrations(int len,
        const int* indx, const double* values)
{
    for (int i = 0; i < len; ++i)
    {
        int j = indx ? indx[i] : i;
        setConcentration(j, values[i]);
    }
    convertToAmounts();
    return len;
}

int CompiledExecutableModel::getBoundarySpeciesAmounts(int len, const int* indx,
        double* values)
{
    for (int i = 0; i < len; ++i)
    {
        int j = indx ? indx[i] : i;
        int nIndex;
        if ((nIndex = getBoundarySpeciesCompartmentIndex(j)) >= 0)
        {
            values[i] = mData.boundarySpeciesConcentrations[j] *
                    mData.compartmentVolumes[nIndex];
        }
    }
    return len;
}

int CompiledExecutableModel::getBoundarySpeciesConcentrations(int len,
        const int* indx, double* values)
{
    for (int i = 0; i < len; ++i)
    {
        int j = indx ? indx[i] : i;
        if (j < mData.numBoundarySpecies)
        {
            values[i] = mData.boundarySpeciesConcentrations[j];
        }
        else
        {
            throw Exception("index out of range");
        }
    }
    return len;
}

int CompiledExecutableModel::setBoundarySpeciesConcentrations(int len,
        const int* indx, const double* values)
{
    for (int i = 0; i < len; ++i)
    {
        int j = indx ? indx[i] : i;
        if (j < mData.numBoundarySpecies)
        {
            mData.boundarySpeciesConcentrations[j] = values[i];
        }
        else
        {
            throw Exception("index out of range");
        }
    }
    return len;
}

int CompiledExecutableModel::getGlobalParameterValues(int len, const int* indx,
        double* values)
{
    for (int i = 0; i < len; ++i)
    {
        int j = indx ? indx[i] : i;
        if (j < mData.numGlobalParameters)
        {
            values[i] = mData.globalParameters[j];
        }
        else
        {
            throw Exception("index out of range");
        }
    }
    return len;
}

int CompiledExecutableModel::setGlobalParameterValues(int len, const int* indx,
        const double* values)
{
    for (int i = 0; i < len; ++i)
    {
        int j = indx ? indx[i] : i;
        if (j < mData.numGlobalParameters)
        {
            mData.globalParameters[j] = values[i];
        }
        else
        {
            throw Exception("index out of range");
        }
    }
    return len;
}

int CompiledExecutableModel::getCompartmentVolumes(int len, const int* indx,
        double* values)
{
    for (int i = 0; i < len; ++i)
    {
        int j = indx ? indx[i] : i;
        if (j < mData.numCompartments)
        {
            values[i] = mData.compartmentVolumes[j];
        }
        else
        {
            throw Exception("index out of range");
        }
    }
    return len;
}

int CompiledExecutableModel::getEventDelays(int len, const int* indx,
        double* values)
{
    for (int i = 0; i < len; ++i)
    {
        int j = indx ? indx[i] : i;
        if (j < mData.numEvents)
        {
            values[i] = mData.eventDelays[j](&mData);
        }
        else
        {
            throw Exception("index out of range");
        }
    }
    return len;
}

int CompiledExecutableModel::getEventPriorities(int len, const int* indx,
        double* values)
{
    return 0;
}

void CompiledExecutableModel::eventAssignment(int eventId)
{
    if (eventId < mData.numEvents)
    {
        mData.eventAssignments[eventId]();
    }
    else
    {
        throw Exception("index out of range");
    }
}

double* CompiledExecutableModel::evalEventAssignment(int eventId)
{
    return 0;
}

void CompiledExecutableModel::applyEventAssignment(int eventId, double* values)
{
}

int CompiledExecutableModel::getEventTriggers(int len, const int *indx,
        unsigned char *values)
{
    if (len <= 0)
    {
        return mData.numEvents;
    }
    else
    {
        if (mData.time > 0)
        {
            evalEvents(mData.time, 0);
        }

        for (int i = 0; i < len; ++i)
        {
            int j = indx ? indx[i] : i;
            if (j < mData.numEvents)
            {
                values[i] = mData.eventStatusArray[j];
            }
            else
            {
                throw Exception("index out of range");
            }
        }
        return len;
    }
}


vector<int> CompiledExecutableModel::retestEvents(const double& timeEnd, const vector<int>& handledEvents, vector<int>& removeEvents)
{
    return retestEvents(timeEnd, handledEvents, false, removeEvents);
}

vector<int> CompiledExecutableModel::retestEvents(const double& timeEnd, vector<int>& handledEvents, const bool& assignOldState)
{
    vector<int> removeEvents;
    return retestEvents(timeEnd, handledEvents, assignOldState, removeEvents);
}

vector<int> CompiledExecutableModel::retestEvents(const double& timeEnd, const vector<int>& handledEvents,
        const bool& assignOldState, vector<int>& removeEvents)
{
    vector<int> result;
//    vector<int> removeEvents;// = new vector<int>();    //Todo: this code was like this originally.. which removeEvents to use???

    if (this->getConservedSumChanged())
    {
        this->computeConservedTotals();
    }

    this->convertToAmounts();
    this->getStateVectorRate(timeEnd, 0, 0);

    // copy original evenStatusArray
    vector<bool> eventStatusArray(mData.eventStatusArray,
            mData.eventStatusArray +
            mData.numEvents);

    this->pushState();

    this->evalEvents(timeEnd, 0);

    for (int i = 0; i < this->getNumEvents(); i++)
    {
        bool containsI = (std::find(handledEvents.begin(), handledEvents.end(), i) != handledEvents.end()) ? true : false;
        if (mData.eventStatusArray[i] == true && eventStatusArray[i] == false && !containsI)
        {

            result.push_back(i);
        }

        if (mData.eventStatusArray[i] == false && eventStatusArray[i] == true && !mData.eventPersistentType[i])
        {
            removeEvents.push_back(i);
        }
    }

    this->popState(assignOldState ? 0 : PopDiscard);

    return result;
}

int CompiledExecutableModel::applyPendingEvents(const double *stateVector,
        double timeEnd,  double tout)
{
    int handled = 0;
    for (int i = (int) mAssignments.size() - 1; i >= 0; i--)
    {
        if (timeEnd >= mAssignments[i].getTime())
        {
            this->setTime(tout);
            this->setStateVector(stateVector);
            this->convertToConcentrations();
            this->updateDependentSpeciesValues();
            mAssignments[i].eval();

            if (this->getConservedSumChanged())
            {
                this->computeConservedTotals();
            }

            this->convertToAmounts();
            this->getStateVectorRate(timeEnd, 0, 0);
            mAssignments.erase(mAssignments.begin() + i);

            handled++;
        }
    }
    return handled;
}

void CompiledExecutableModel::applyEvents(double timeEnd,
        const unsigned char* previousEventStatus, const double *initialState,
        double* finalState)
{
    this->setStateVector(initialState);
    this->convertToConcentrations();
    this->updateDependentSpeciesValues();
    this->evalEvents(timeEnd, 0);

    vector<int> firedEvents;
    map<int, double* > preComputedAssignments;


    for (int i = 0; i < this->getNumEvents(); i++)
    {
        // We only fire an event if we transition from false to true
        if (mData.eventStatusArray[i] && !previousEventStatus[i])
        {
            firedEvents.push_back(i);
            if (mData.eventType[i])
            {
                preComputedAssignments[i] = mData.computeEventAssignments[i](&(mData));
            }
        }

        else
        {
            // if the trigger condition is not supposed to be persistent,
            // remove the event from the firedEvents list;
            if (!mData.eventPersistentType[i])
            {
                removePendingAssignmentForIndex(i);
            }
        }
    }

    vector<int> handled;
    while (firedEvents.size() > 0)
    {
        sortEventsByPriority(firedEvents);
        // Call event assignment if the eventstatus flag for the particular event is false
        for (u_int i = 0; i < firedEvents.size(); i++)
        {
            int currentEvent = firedEvents[i];
            double eventDelay = 0;
            this->getEventDelays(1, &currentEvent, &eventDelay);

            if (eventDelay == 0)
            {
                if (mData.eventType[currentEvent] && preComputedAssignments.count(currentEvent) > 0)
                {
                    mData.performEventAssignments[currentEvent](&(mData), preComputedAssignments[currentEvent]);
                }
                else
                {
                    this->eventAssignment(currentEvent);
                }

                handled.push_back(currentEvent);
                vector<int> removeEvents;
                vector<int> additionalEvents = retestEvents(timeEnd, handled, removeEvents);

                firedEvents.insert(firedEvents.end(), additionalEvents.begin(), additionalEvents.end());

                for (int j = 0; j < additionalEvents.size(); j++)
                {
                    int newEvent = additionalEvents[j];
                    if (mData.eventType[newEvent])
                    {
                        preComputedAssignments[newEvent] = mData.computeEventAssignments[newEvent](&(mData));
                    }
                }

                mData.eventStatusArray[currentEvent] = false;
                Log(lDebug3)<<"Fired Event with ID:"<<currentEvent;
                firedEvents.erase(firedEvents.begin() + i);

                for (int i = 0; i < removeEvents.size(); i++)
                {
                    int item = removeEvents[i];
                    if (find(firedEvents.begin(), firedEvents.end(), item) != firedEvents.end())
                    {
                        firedEvents.erase(find(firedEvents.begin(), firedEvents.end(), item));
                        removePendingAssignmentForIndex(item);
                    }
                }

                break;
            }
            else
            {
                if (find(mAssignmentTimes.begin(), mAssignmentTimes.end(), timeEnd + eventDelay) == mAssignmentTimes.end())
                {
                    mAssignmentTimes.push_back(timeEnd + eventDelay);
                }

                double *preComputedValues =
                    (mData.eventType[currentEvent] &&
                        preComputedAssignments.count(currentEvent) == 1) ?
                            preComputedAssignments[currentEvent] : 0;

                PendingAssignment pending( &(mData),
                        timeEnd + eventDelay,
                        mData.computeEventAssignments[currentEvent],
                        mData.performEventAssignments[currentEvent],
                        mData.eventType[currentEvent],
                        currentEvent,
                        preComputedValues);

                mAssignments.push_back(pending);

                mData.eventStatusArray[currentEvent] = false;
                firedEvents.erase(firedEvents.begin() + i);
                break;
            }
        }
    }

    if (this->getConservedSumChanged())
    {
        this->computeConservedTotals();
    }
    this->convertToAmounts();

    this->getStateVectorRate(timeEnd, 0, 0);

    this->getStateVector(finalState);

    sort(mAssignmentTimes.begin(), mAssignmentTimes.end());
}

void CompiledExecutableModel::removePendingAssignmentForIndex(int eventIndex)
{
    for (int j = (int) mAssignments.size() - 1; j >= 0; j--)
    {
        if (mAssignments[j].getIndex() == eventIndex)
        {
            mAssignments.erase(mAssignments.begin() + j);
        }
    }
}

void CompiledExecutableModel::sortEventsByPriority(vector<rr::Event>& firedEvents)
{
    if ((firedEvents.size() > 1))
    {
        Log(lDebug3)<<"Sorting event priorities";
        for(int i = 0; i < firedEvents.size(); i++)
        {
            firedEvents[i].SetPriority(mData.eventPriorities[firedEvents[i].GetID()]);
            Log(lDebug3)<<firedEvents[i];
        }
        sort(firedEvents.begin(), firedEvents.end(), SortByPriority());

        Log(lDebug3)<<"After sorting event priorities";
        for(int i = 0; i < firedEvents.size(); i++)
        {
            Log(lDebug3)<<firedEvents[i];
        }
    }
}

void CompiledExecutableModel::sortEventsByPriority(vector<int>& firedEvents)
{
    if (firedEvents.size() > 1)
    {
        this->computeEventPriorites();
        vector<rr::Event> dummy;
        for(int i = 0; i < firedEvents.size(); i++)
        {
            Event event(firedEvents[i]);
            dummy.push_back(event);
        }

        Log(lDebug3)<<"Sorting event priorities";
        for(int i = 0; i < firedEvents.size(); i++)
        {
            Event &event = dummy[i];
            event.SetPriority(mData.eventPriorities[event.GetID()]);
            Log(lDebug3) << event;
        }
        sort(dummy.begin(), dummy.end(), SortByPriority());

        for(int i = 0; i < firedEvents.size(); i++)
        {
            firedEvents[i] = dummy[i].GetID();
        }

        Log(lDebug3)<<"After sorting event priorities";
        for(int i = 0; i < firedEvents.size(); i++)
        {
            Log(lDebug3)<<firedEvents[i];
        }
    }
}

void CompiledExecutableModel::getEventRoots(double time,
        const double *stateVector, double* gdot)
{

    this->pushState();

    this->getStateVectorRate(time, 0, 0);
    this->setStateVector(stateVector);

    this->evalEvents(time, 0);

    for(int i = 0; i < this->getNumEvents(); i++)
    {
        gdot[i] = mData.eventStatusArray[i] ? 1.0 : -1.0;
    }

    this->popState();
}

double CompiledExecutableModel::getNextPendingEventTime(bool pop)
{
    double result = mAssignmentTimes[0];
    if (pop)
    {
        mAssignmentTimes.erase(mAssignmentTimes.begin());
    }
    return result;
}

int CompiledExecutableModel::getPendingEventSize()
{
    return mAssignmentTimes.size();
}

int CompiledExecutableModel::getReactionRates(int len, const int* indx,
        double* values)
{
    for (int i = 0; i < len; ++i)
    {
        int j = indx ? indx[i] : i;
        if (j < mData.numReactions)
        {
            values[i] = mData.reactionRates[j];
        }
        else
        {
            throw Exception("index out of range");
        }
    }
    return len;
}

int CompiledExecutableModel::getNumConservedMoieties()
{
    return ms.mConservationList.size();
}

int CompiledExecutableModel::getConservedMoietyIndex(const string& name)
{
    int result = -1;
    return ms.mConservationList.find(name, result) ? result : -1;
}

string CompiledExecutableModel::getConservedMoietyId(int index)
{
    return ms.mConservationList[index].name;
}

int CompiledExecutableModel::getConservedMoietyValues(int len, const int* indx,
        double* values)
{
    for (int i = 0; i < len; ++i)
    {
        int j = indx ? indx[i] : i;
        if (j < mData.numDependentSpecies)
        {
            values[i] = mData.dependentSpeciesConservedSums[j];
        }
        else
        {
            throw Exception("index out of range");
        }
    }
    return len;
}

int CompiledExecutableModel::setConservedMoietyValues(int len, const int* indx,
        const double* values)
{
    for (int i = 0; i < len; ++i)
    {
        int j = indx ? indx[i] : i;
        if (j < mData.numDependentSpecies)
        {
            mData.dependentSpeciesConservedSums[j] = values[i];
        }
        else
        {
            throw Exception("index out of range");
        }
    }
    return len;
}

int CompiledExecutableModel::getFloatingSpeciesAmountRates(int len,
        int const *indx, double *values)
{
    for (int i = 0; i < len; ++i)
    {
        int j = indx ? indx[i] : i;
        if (j < mData.numFloatingSpecies)
        {
            values[i] = mData.floatingSpeciesAmountRates[j];
        }
        else
        {
            throw Exception("index out of range");
        }
    }
    return len;
}

int CompiledExecutableModel::setFloatingSpeciesAmounts(int len, int const *indx,
        const double *values)
{
    for (int i = 0; i < len; ++i)
    {
        int j = indx ? indx[i] : i;
        mData.floatingSpeciesAmounts[j] = values[i];
    }
    return len;
}


int CompiledExecutableModel::setCompartmentVolumes(int len, const int* indx,
        const double* values)
{
    for (int i = 0; i < len; ++i)
    {
        int j = indx ? indx[i] : i;
        mData.compartmentVolumes[j] = values[i];
    }
    return len;
}

int CompiledExecutableModel::setFloatingSpeciesInitConcentrations(int len,
        const int* indx, const double* values)
{
    for (int i = 0; i < len; ++i)
    {
        int j = indx ? indx[i] : i;
        mData.floatingSpeciesInitConcentrations[j] = values[i];
    }
    return len;
}

int CompiledExecutableModel::getFloatingSpeciesInitConcentrations(int len,
        const int* indx, double* values)
{
    for (int i = 0; i < len; ++i)
    {
        int j = indx ? indx[i] : i;
        values[i] = mData.floatingSpeciesInitConcentrations[j];
    }
    return len;
}

double CompiledExecutableModel::getStoichiometry(int speciesIndex, int reactionIndex)
{
    if (speciesIndex >= 0 && speciesIndex < stoichiometryMatrix.numRows()
            && reactionIndex >= 0 && reactionIndex < stoichiometryMatrix.numCols())
    {
        return stoichiometryMatrix(speciesIndex, reactionIndex);
    }
    else
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

}


int CompiledExecutableModel::getStoichiometryMatrix(int* p_rows, int* p_cols, double** p_data)
{
    int rows = stoichiometryMatrix.numRows();
    int cols = stoichiometryMatrix.numCols();
    double *data = (double*)malloc(sizeof(double)*rows*cols);
    memcpy(data, stoichiometryMatrix.getArray(), sizeof(double)*rows*cols);

    *p_rows = rows;
    *p_cols = cols;
    *p_data = data;

    return rows*cols;
}


int CompiledExecutableModel::setFloatingSpeciesInitAmounts(int len, int const *indx,
            double const *values)
{
    throw rr::Exception(std::string(__FUNC__) + " not supported with legacy C back end");
}

int CompiledExecutableModel::getFloatingSpeciesInitAmounts(int len, int const *indx,
                double *values)
{
    throw rr::Exception(std::string(__FUNC__) + " not supported with legacy C back end");
}

int CompiledExecutableModel::setCompartmentInitVolumes(int len, int const *indx,
            double const *values)
{
    throw rr::Exception(std::string(__FUNC__) + " not supported with legacy C back end");
}

int CompiledExecutableModel::getCompartmentInitVolumes(int len, int const *indx,
                double *values)
{
    throw rr::Exception(std::string(__FUNC__) + " not supported with legacy C back end");
}

void CompiledExecutableModel::getIds(int types, std::list<std::string> &ids)
{
    if (checkExact(SelectionRecord::FLOATING_AMOUNT, types)) {
        addIds(this, &rr::ExecutableModel::getNumFloatingSpecies,
                &rr::ExecutableModel::getFloatingSpeciesId, ids);
    }

    if (checkExact(SelectionRecord::BOUNDARY_AMOUNT, types)) {
        addIds(this, &rr::ExecutableModel::getNumBoundarySpecies,
                &rr::ExecutableModel::getBoundarySpeciesId, ids);
    }

    if (checkExact(SelectionRecord::COMPARTMENT, types)) {
        addIds(this, &rr::ExecutableModel::getNumCompartments,
                &rr::ExecutableModel::getCompartmentId, ids);
    }

    if (checkExact(SelectionRecord::GLOBAL_PARAMETER, types)) {
        addIds(this, &rr::ExecutableModel::getNumGlobalParameters,
                &rr::ExecutableModel::getGlobalParameterId, ids);
    }

    if (checkExact(SelectionRecord::REACTION_RATE, types)) {
        addIds(this, &rr::ExecutableModel::getNumReactions,
                &rr::ExecutableModel::getReactionId, ids);
    }

    if (checkExact(SelectionRecord::INITIAL_FLOATING_CONCENTRATION, types)) {
        for (int i = 0; i < getNumFloatingSpecies(); ++i) {
            ids.push_back("init([" + this->getFloatingSpeciesId(i) + "])");
        }
    }

    if (checkExact(SelectionRecord::FLOATING_AMOUNT_RATE, types)) {
        for (int i = 0; i < getNumFloatingSpecies(); ++i) {
            ids.push_back(this->getFloatingSpeciesId(i) + "'");
        }
    }
}

int CompiledExecutableModel::getSupportedIdTypes()
{
    return SelectionRecord::TIME |
        SelectionRecord::BOUNDARY_CONCENTRATION |
        SelectionRecord::FLOATING_CONCENTRATION |
        SelectionRecord::REACTION_RATE |
        SelectionRecord::FLOATING_AMOUNT_RATE |
        SelectionRecord::COMPARTMENT |
        SelectionRecord::GLOBAL_PARAMETER |
        SelectionRecord::FLOATING_AMOUNT |
        SelectionRecord::BOUNDARY_AMOUNT |
        SelectionRecord::INITIAL_FLOATING_CONCENTRATION |
        SelectionRecord::CONSREVED_MOIETY;
}


double CompiledExecutableModel::getValue(const std::string& id)
{
    SelectionRecord sel(id);

    int index = -1;
    double result = 0;

    if (sel.selectionType == SelectionRecord::UNKNOWN)
    {
        throw Exception("invalid selection string " + id);
    }

    // check to see that we have valid selection ids
    switch(sel.selectionType)
    {
    case SelectionRecord::TIME:
        result = getTime();
        break;
    case SelectionRecord::UNKNOWN_ELEMENT:
        // check for sbml element types

        if ((index = getFloatingSpeciesIndex(sel.p1)) >= 0)
        {
            getFloatingSpeciesAmounts(1, &index, &result);
            break;
        }
        else if ((index = getBoundarySpeciesIndex(sel.p1)) >= 0)
        {
            getBoundarySpeciesAmounts(1, &index, &result);
            break;
        }
        else if ((index = getCompartmentIndex(sel.p1)) >= 0)
        {
            getCompartmentVolumes(1, &index, &result);
            break;
        }
        else if ((index = getGlobalParameterIndex(sel.p1)) >= 0)
        {
            getGlobalParameterValues(1, &index, &result);
            break;
        }
        else if ((index = getReactionIndex(sel.p1)) >= 0)
        {
            getReactionRates(1, &index, &result);
            break;
        }
        else
        {
            throw Exception("No sbml element exists for symbol '" + id + "'");
            break;
        }
    case SelectionRecord::UNKNOWN_CONCENTRATION:
        if ((index = getFloatingSpeciesIndex(sel.p1)) >= 0)
        {
            getFloatingSpeciesConcentrations(1, &index, &result);
            break;
        }
        else if ((index = getBoundarySpeciesIndex(sel.p1)) >= 0)
        {
            getBoundarySpeciesConcentrations(1, &index, &result);
            break;
        }
        else
        {
            string msg = "No sbml element exists for concentration selection '" + id + "'";
            Log(Logger::LOG_ERROR) << msg;
            throw Exception(msg);
            break;
        }
    case SelectionRecord::FLOATING_AMOUNT_RATE:
        if ((index = getFloatingSpeciesIndex(sel.p1)) >= 0)
        {
            getFloatingSpeciesAmountRates(1, &index, &result);
            break;
        }
        else
        {
            throw Exception("Invalid id '" + id + "' for floating amount rate");
            break;
        }

    case SelectionRecord::INITIAL_FLOATING_AMOUNT:
        if ((index = getFloatingSpeciesIndex(sel.p1)) >= 0)
        {
            getFloatingSpeciesInitAmounts(1, &index, &result);
            break;
        }
        else if ((index = getCompartmentIndex(sel.p1)) >= 0)
        {
            getCompartmentInitVolumes(1, &index, &result);
            break;
        }
        else
        {
            throw Exception("Invalid id '" + id + "' for floating amount rate");
            break;
        }
    case SelectionRecord::INITIAL_FLOATING_CONCENTRATION:
        if ((index = getFloatingSpeciesIndex(sel.p1)) >= 0)
        {
            getFloatingSpeciesInitConcentrations(1, &index, &result);
            break;
        }
        else
        {
            throw Exception("Invalid id '" + id + "' for floating species");
            break;
        }


    default:
        Log(Logger::LOG_ERROR) << "A new SelectionRecord should not have this value: "
        << sel.to_repr();
        throw Exception("Invalid selection '" + id + "' for setting value");
        break;
    }

    return result;
}

void CompiledExecutableModel::setValue(const std::string& id, double value)
{
    ExecutableModel* p = this;

    SelectionRecord sel(id);

    int index = -1;

    if (sel.selectionType == SelectionRecord::UNKNOWN)
    {
        throw Exception("invalid selection string " + id);
    }

    // check to see that we have valid selection ids
    switch(sel.selectionType)
    {
    case SelectionRecord::TIME:
        setTime(value);
        break;
    case SelectionRecord::UNKNOWN_ELEMENT:
        // check for sbml element types

        if ((index = getFloatingSpeciesIndex(sel.p1)) >= 0)
        {
            setFloatingSpeciesAmounts(1, &index, &value);
            break;
        }
        else if ((index = getCompartmentIndex(sel.p1)) >= 0)
        {
            setCompartmentVolumes(1, &index, &value);
            break;
        }
        else if ((index = getGlobalParameterIndex(sel.p1)) >= 0)
        {
            setGlobalParameterValues(1, &index, &value);
            break;
        }
        else
        {
            throw Exception("Invalid or non-existant sbml id  '" + id + "' for set value");
            break;
        }
    case SelectionRecord::UNKNOWN_CONCENTRATION:
        if ((index = getFloatingSpeciesIndex(sel.p1)) >= 0)
        {
            setFloatingSpeciesConcentrations(1, &index, &value);
            break;
        }
        else if ((index = getBoundarySpeciesIndex(sel.p1)) >= 0)
        {
            setBoundarySpeciesConcentrations(1, &index, &value);
            break;
        }
        else
        {
            string msg = "No sbml element exists for concentration selection '" + id + "'";
            Log(Logger::LOG_ERROR) << msg;
            throw Exception(msg);
            break;
        }

    case SelectionRecord::INITIAL_FLOATING_AMOUNT:
        if ((index = getFloatingSpeciesIndex(sel.p1)) >= 0)
        {
            setFloatingSpeciesInitAmounts(1, &index, &value);
            break;
        }
        else if ((index = getCompartmentIndex(sel.p1)) >= 0)
        {
            setCompartmentInitVolumes(1, &index, &value);
            break;
        }
        else
        {
            throw Exception("Invalid id '" + id + "' for floating amount rate");
            break;
        }
    case SelectionRecord::INITIAL_FLOATING_CONCENTRATION:
        if ((index = getFloatingSpeciesIndex(sel.p1)) >= 0)
        {
            setFloatingSpeciesInitConcentrations(1, &index, &value);
            break;
        }
        else
        {
            throw Exception("Invalid id '" + id + "' for floating species");
            break;
        }



    default:
        Log(Logger::LOG_ERROR) << "Invalid selection '" + sel.to_string() + "' for setting value";
        throw Exception("Invalid selection '" + sel.to_string() + "' for setting value");
        break;
    }
}

int CompiledExecutableModel::getFloatingSpeciesConcentrationRates(int len, int const *indx,
        double *values)
{
    throw rr::Exception(std::string(__FUNC__) + " not supported with legacy C back end");
}

std::string CompiledExecutableModel::getStateVectorId(int)
{
    throw rr::Exception(std::string(__FUNC__) + " not supported with legacy C back end");
}

int CompiledExecutableModel::getEventIndex(const std::string&)
{
    throw rr::Exception(std::string(__FUNC__) + " not supported with legacy C back end");
}

std::string CompiledExecutableModel::getEventId(int)
{
    throw rr::Exception(std::string(__FUNC__) + " not supported with legacy C back end");
}

void CompiledExecutableModel::setEventListener(int, EventListenerPtr)
{
    throw rr::Exception(std::string(__FUNC__) + " not supported with legacy C back end");
}

EventListenerPtr CompiledExecutableModel::getEventListener(int)
{
    throw rr::Exception(std::string(__FUNC__) + " not supported with legacy C back end");
}

double CompiledExecutableModel::getFloatingSpeciesAmountRate(int index,
           const double *reactionRates)
{
    throw rr::Exception(std::string(__FUNC__) + " not supported with legacy C back end");
}



} //Namespace rr


