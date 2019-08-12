/*
 * LLVMExecutableModel.cpp
 *
 * Author: Andy Somogyi,
 *     email decode: V1 = "."; V2 = "@"; V3 = V1;
 *     andy V1 somogyi V2 gmail V3 com
 */
#pragma hdrstop
#include "LLVMExecutableModel.h"
#include "ModelResources.h"
#include "LLVMIncludes.h"
#include "rrSparse.h"
#include "rrLogger.h"
#include "rrException.h"
#include "LLVMException.h"
#include "rrStringUtils.h"
#include "rrRoadRunnerOptions.h"
#include "LLVMModelGenerator.h"
#include "rrConfig.h"
#include <iomanip>
#include <cstdlib>
#include <iostream>

using rr::Logger;
using rr::getLogger;
using rr::LoggingBuffer;
using rr::SelectionRecord;
using rr::EventListener;
using rr::EventListenerPtr;
using rr::EventListenerException;
using rr::Config;

#if defined (_WIN32)
#define isnan _isnan
#else
#define isnan std::isnan
#endif

using namespace std;

// static assertion, taken from
// http://msdn.microsoft.com/en-us/library/windows/desktop/ms679289(v=vs.85).aspx
#ifndef C_ASSERT
#define C_ASSERT(e) typedef char __C_ASSERT__[(e)?1:-1]
#endif

template <typename numeric_type>
static void dump_array(std::ostream &os, int n, const numeric_type *p)
{
    if (p)
    {
        os << setiosflags(ios::floatfield) << setprecision(8);
        os << '[';
        for (int i = 0; i < n; ++i)
        {
            os << p[i];
            if (i < n - 1)
            {
                os << ", ";
            }
        }
        os << ']' << endl;
    }
    else
    {
        os << "NULL" << endl;
    }
}

typedef string (rr::ExecutableModel::*getNamePtr)(int);
typedef int (rr::ExecutableModel::*getNumPtr)();

/**
 * checks if the bitfield value has all the required flags
 * from type
 */
inline bool checkExact(uint32_t type, uint32_t value) {
    return (value & type) == type;
}

// make this static here, hide our implementation...
static void addIds(rr::ExecutableModel *model, int start, int end,
        getNamePtr nameFunc, std::list<std::string>& ids)
{
    for(int i = start; i < end; i++)
    {
        const std::string& name  = (model->*nameFunc)(i);
        ids.push_back(name);
    }
}

// make this static here, hide our implementation...
static void addConcIds(rr::ExecutableModel *model, int start, int end,
        getNamePtr nameFunc, std::list<std::string>& ids)
{
    for(int i = start; i < end; i++)
    {
        const std::string& name  = "[" + (model->*nameFunc)(i) + "]";
        ids.push_back(name);
    }
}

static uint32_t defaultFlags() {
    uint32_t result = 0;

    if(Config::getBool(Config::OPTIMIZE_REACTION_RATE_SELECTION)) {
        result |= rr::ExecutableModel::OPTIMIZE_REACTION_RATE_SELECTION;
    }

    return result;
}


namespace rrllvm
{

int LLVMExecutableModel::getValues(double (*funcPtr)(LLVMModelData*, int),
        int len, const int *indx, double *values)
{
    double value;
    for (int i = 0; i < len; ++i)
    {
        int j = indx ? indx[i] : i;
        value = funcPtr(modelData, j);

        if (isnan(value))
        {
            Log(Logger::LOG_WARNING) << "error getting value for index " << j
                                   << ", probably out of range";
        }

        values[i] = value;
    }
    return len;
}

int LLVMExecutableModel::setValues(bool (*funcPtr)(LLVMModelData*, int, double),
        GetNameFuncPtr getNameFuncPtr, int len, const int *indx, const double *values)
{
    for (int i = 0; i < len; ++i)
    {
        int j = indx ? indx[i] : i;
        bool result =  funcPtr(modelData, j, values[i]);

        if (!result)
        {
            std::stringstream s;
            string id = (this->*getNameFuncPtr)(j);
            s << "could not set value for " << id;

            if (symbols->hasAssignmentRule(id))
            {
                s << ", it is defined by an assignment rule, can not be set independently.";
            }
            else if (symbols->hasInitialAssignmentRule(id))
            {
                s << ", it is defined by an initial assignment rule and can not be set independently.";
            }
            else if (symbols->hasRateRule(id))
            {
                s << ", it is defined by a rate rule and can not be set independently.";
            }

            throw_llvm_exception(s.str());
        }
    }
    return len;
}

LLVMExecutableModel::LLVMExecutableModel() :
    symbols(0),
    modelData(0),
    conversionFactor(1.0),
    evalInitialConditionsPtr(0),
    evalReactionRatesPtr(0),
    getBoundarySpeciesAmountPtr(0),
    getFloatingSpeciesAmountPtr(0),
    getBoundarySpeciesConcentrationPtr(0),
    getFloatingSpeciesConcentrationPtr(0),
    getGlobalParameterPtr(0),
    getCompartmentVolumePtr(0),
    evalRateRuleRatesPtr(0),
    getEventTriggerPtr(0),
    getEventPriorityPtr(0),
    getEventDelayPtr(0),
    eventTriggerPtr(0),
    eventAssignPtr(0),
    evalVolatileStoichPtr(0),
    evalConversionFactorPtr(0),
    setBoundarySpeciesAmountPtr(0),
    setFloatingSpeciesAmountPtr(0),
    setBoundarySpeciesConcentrationPtr(0),
    setFloatingSpeciesConcentrationPtr(0),
    setCompartmentVolumePtr(0),
    setGlobalParameterPtr(0),
    getFloatingSpeciesInitConcentrationsPtr(0),
    setFloatingSpeciesInitConcentrationsPtr(0),
    getFloatingSpeciesInitAmountsPtr(0),
    setFloatingSpeciesInitAmountsPtr(0),
    getCompartmentInitVolumesPtr(0),
    setCompartmentInitVolumesPtr(0),
    getGlobalParameterInitValuePtr(0),
    setGlobalParameterInitValuePtr(0),
    dirty(0),
    flags(defaultFlags())
{
    std::srand((unsigned)std::time(0));
}

LLVMExecutableModel::LLVMExecutableModel(
    const cxx11_ns::shared_ptr<ModelResources>& rc, LLVMModelData* modelData) :
    resources(rc),
    symbols(rc->symbols),
    modelData(modelData),
    conversionFactor(1.0),
    evalInitialConditionsPtr(rc->evalInitialConditionsPtr),
    evalReactionRatesPtr(rc->evalReactionRatesPtr),
    getBoundarySpeciesAmountPtr(rc->getBoundarySpeciesAmountPtr),
    getFloatingSpeciesAmountPtr(rc->getFloatingSpeciesAmountPtr),
    getBoundarySpeciesConcentrationPtr(rc->getBoundarySpeciesConcentrationPtr),
    getFloatingSpeciesConcentrationPtr(rc->getFloatingSpeciesConcentrationPtr),
    getCompartmentVolumePtr(rc->getCompartmentVolumePtr),
    getGlobalParameterPtr(rc->getGlobalParameterPtr),
    evalRateRuleRatesPtr(rc->evalRateRuleRatesPtr),
    getEventTriggerPtr(rc->getEventTriggerPtr),
    getEventPriorityPtr(rc->getEventPriorityPtr),
    getEventDelayPtr(rc->getEventDelayPtr),
    eventTriggerPtr(rc->eventTriggerPtr),
    eventAssignPtr(rc->eventAssignPtr),
    evalVolatileStoichPtr(rc->evalVolatileStoichPtr),
    evalConversionFactorPtr(rc->evalConversionFactorPtr),
    setBoundarySpeciesAmountPtr(rc->setBoundarySpeciesAmountPtr),
    setFloatingSpeciesAmountPtr(rc->setFloatingSpeciesAmountPtr),
    setBoundarySpeciesConcentrationPtr(rc->setBoundarySpeciesConcentrationPtr),
    setFloatingSpeciesConcentrationPtr(rc->setFloatingSpeciesConcentrationPtr),
    setCompartmentVolumePtr(rc->setCompartmentVolumePtr),
    setGlobalParameterPtr(rc->setGlobalParameterPtr),
    getFloatingSpeciesInitConcentrationsPtr(rc->getFloatingSpeciesInitConcentrationsPtr),
    setFloatingSpeciesInitConcentrationsPtr(rc->setFloatingSpeciesInitConcentrationsPtr),
    getFloatingSpeciesInitAmountsPtr(rc->getFloatingSpeciesInitAmountsPtr),
    setFloatingSpeciesInitAmountsPtr(rc->setFloatingSpeciesInitAmountsPtr),
    getCompartmentInitVolumesPtr(rc->getCompartmentInitVolumesPtr),
    setCompartmentInitVolumesPtr(rc->setCompartmentInitVolumesPtr),
    getGlobalParameterInitValuePtr(rc->getGlobalParameterInitValuePtr),
    setGlobalParameterInitValuePtr(rc->setGlobalParameterInitValuePtr),
    eventListeners(modelData->numEvents, EventListenerPtr()), // init eventHandlers vector
    dirty(0),
    flags(defaultFlags())
{

    modelData->time = -1.0; // time is initially before simulation starts

    std::srand((unsigned)std::time(0));

    eventAssignTimes.resize(modelData->numEvents);

    reset(SelectionRecord::ALL);
}

LLVMExecutableModel::LLVMExecutableModel(std::istream& in, uint modelGeneratorOpt) :
	resources(new ModelResources()),
	dirty(0),
	conversionFactor(1.0),
	flags(defaultFlags())
{
	modelData = LLVMModelData_from_save(in);
	resources->loadState(in, modelGeneratorOpt);
	symbols = resources->symbols;
	eventListeners = std::vector<EventListenerPtr>(modelData->numEvents, EventListenerPtr());
	eventAssignTimes.resize(modelData->numEvents);
	

	evalInitialConditionsPtr = resources->evalInitialConditionsPtr;
	evalReactionRatesPtr = resources->evalReactionRatesPtr;
	getBoundarySpeciesAmountPtr = resources->getBoundarySpeciesAmountPtr;
    getFloatingSpeciesAmountPtr = resources->getFloatingSpeciesAmountPtr;
    getBoundarySpeciesConcentrationPtr = resources->getBoundarySpeciesConcentrationPtr;
    getFloatingSpeciesConcentrationPtr = resources->getFloatingSpeciesConcentrationPtr;
    getCompartmentVolumePtr = resources->getCompartmentVolumePtr;
    getGlobalParameterPtr = resources->getGlobalParameterPtr;
    evalRateRuleRatesPtr = resources->evalRateRuleRatesPtr;
    getEventTriggerPtr = resources->getEventTriggerPtr;
    getEventPriorityPtr = resources->getEventPriorityPtr;
    getEventDelayPtr = resources->getEventDelayPtr;
	eventTriggerPtr = resources->eventTriggerPtr;
    eventAssignPtr = resources->eventAssignPtr;
    evalVolatileStoichPtr = resources->evalVolatileStoichPtr;
    evalConversionFactorPtr = resources->evalConversionFactorPtr;
    setBoundarySpeciesAmountPtr = resources->setBoundarySpeciesAmountPtr;
    setFloatingSpeciesAmountPtr = resources->setFloatingSpeciesAmountPtr;
    setBoundarySpeciesConcentrationPtr = resources->setBoundarySpeciesConcentrationPtr;
    setFloatingSpeciesConcentrationPtr = resources->setFloatingSpeciesConcentrationPtr;
    setCompartmentVolumePtr = resources->setCompartmentVolumePtr;
    setGlobalParameterPtr = resources->setGlobalParameterPtr;
    getFloatingSpeciesInitConcentrationsPtr = resources->getFloatingSpeciesInitConcentrationsPtr;
    setFloatingSpeciesInitConcentrationsPtr = resources->setFloatingSpeciesInitConcentrationsPtr;
    getFloatingSpeciesInitAmountsPtr = resources->getFloatingSpeciesInitAmountsPtr;
    setFloatingSpeciesInitAmountsPtr = resources->setFloatingSpeciesInitAmountsPtr;
    getCompartmentInitVolumesPtr = resources->getCompartmentInitVolumesPtr;
    setCompartmentInitVolumesPtr = resources->setCompartmentInitVolumesPtr;
    getGlobalParameterInitValuePtr = resources->getGlobalParameterInitValuePtr;
    setGlobalParameterInitValuePtr = resources->setGlobalParameterInitValuePtr;

    pendingEvents.loadState(in, *this);
    rr::loadBinary(in, eventAssignTimes);
    rr::loadBinary(in, tieBreakMap);
}

LLVMExecutableModel::~LLVMExecutableModel()
{
    // smart ptr takes care of freeing resources

    LLVMModelData_free(modelData);

    Log(Logger::LOG_DEBUG) << __FUNC__;
}

string LLVMExecutableModel::getModelName()
{
    return symbols->getModelName();
}

void LLVMExecutableModel::setTime(double time)
{
    modelData->time = time;
}

double LLVMExecutableModel::getTime()
{
    return modelData->time;
}


int LLVMExecutableModel::getNumIndFloatingSpecies()
{
    return modelData->numIndFloatingSpecies;
}


int LLVMExecutableModel::getNumDepFloatingSpecies()
{
    return symbols->getFloatingSpeciesSize() - modelData->numIndFloatingSpecies;
}

int LLVMExecutableModel::getNumFloatingSpecies()
{
    return symbols->getFloatingSpeciesSize();
}

int LLVMExecutableModel::getNumBoundarySpecies()
{
    return symbols->getBoundarySpeciesSize();
}

int LLVMExecutableModel::getNumGlobalParameters()
{
    return symbols->getGlobalParametersSize();
}

int LLVMExecutableModel::getNumCompartments()
{
    return symbols->getCompartmentsSize();
}

int LLVMExecutableModel::getCompartmentIndexForFloatingSpecies(int index) 
{
	return symbols->getCompartmentIndexForFloatingSpecies(index);
}

int LLVMExecutableModel::getNumReactions()
{
    return modelData->numReactions;
}

int LLVMExecutableModel::getFloatingSpeciesConcentrations(int len, int const *indx,
        double *values)
{
    return getValues(getFloatingSpeciesConcentrationPtr, len, indx, values);
}

void LLVMExecutableModel::getRateRuleValues(double *rateRuleValues)
{
    memcpy(rateRuleValues, modelData->rateRuleValuesAlias, modelData->numRateRules * sizeof(double));
}

void LLVMExecutableModel::getStateVectorRate(double time, const double *y, double *dydt)
{
    modelData->time = time;

    if (y && dydt)
    {
        // save and assign state vector
        double *savedRateRules = modelData->rateRuleValuesAlias;
        double *savedFloatingSpeciesAmounts = modelData->floatingSpeciesAmountsAlias;

        modelData->rateRuleValuesAlias = const_cast<double*>(y);
        modelData->floatingSpeciesAmountsAlias = const_cast<double*>(y + modelData->numRateRules);
        evalVolatileStoichPtr(modelData);

        // not setting state vector, react rates get dirty
        conversionFactor = evalReactionRatesPtr(modelData);
        dirty |= DIRTY_REACTION_RATES;

        // floatingSpeciesAmountRates only valid for the following two
        // functions, this will move to a parameter shortly...

        modelData->floatingSpeciesAmountRates = dydt + modelData->numRateRules;

        csr_matrix_dgemv(conversionFactor, modelData->stoichiometry,
                modelData->reactionRatesAlias, 0.0, modelData->floatingSpeciesAmountRates);

        evalConversionFactorPtr(modelData);

        modelData->floatingSpeciesAmountRates = 0;

        // this will also move to a parameter for the evalRateRules func...
        modelData->rateRuleRates = dydt;
        evalRateRuleRatesPtr(modelData);
        modelData->rateRuleRates = 0;

        // restore original pointers for state vector
        modelData->rateRuleValuesAlias = savedRateRules;
        modelData->floatingSpeciesAmountsAlias = savedFloatingSpeciesAmounts;
    }
    else if (y && !dydt)
    {
        setStateVector(y);
    }
    else if (!y && dydt)
    {
        // evaluate dydt using current state

        evalVolatileStoichPtr(modelData);

        conversionFactor = evalReactionRatesPtr(modelData);
        dirty &= ~DIRTY_REACTION_RATES;

        // floatingSpeciesAmountRates only valid for the following two
        // functions, this will move to a parameter shortly...

        modelData->floatingSpeciesAmountRates = dydt + modelData->numRateRules;

        csr_matrix_dgemv(conversionFactor, modelData->stoichiometry,
                modelData->reactionRatesAlias, 0.0, modelData->floatingSpeciesAmountRates);

        evalConversionFactorPtr(modelData);

        modelData->floatingSpeciesAmountRates = 0;

        // this will also move to a parameter for the evalRateRules func...
        modelData->rateRuleRates = dydt;
        evalRateRuleRatesPtr(modelData);
        modelData->rateRuleRates = 0;
    }

    /*
	if (Logger::LOG_PRIO_TRACE <= rr::Logger::LOG_GetLogLevel()) {
		LoggingBuffer log(Logger::LOG_PRIO_TRACE, __FILE__, __LINE__);
		log.stream() << __FUNC__ << endl;
		log.stream() << "y: ";
		if (y) {
			dump_array(log.stream(), modelData->numRateRules + modelData->numFloatingSpecies, y);
		}
		else {
			log.stream() << "null";
		}
		log.stream() << endl << "dydt: ";
		if (dydt) {
			dump_array(log.stream(), modelData->numRateRules + modelData->numFloatingSpecies, dydt);
		}
		else {
			log.stream() << "null";
		}
		log.stream() << endl << "Model: " << endl << this;
	}
	*/
}

double LLVMExecutableModel::getFloatingSpeciesAmountRate(int index,
           const double *reactionRates)
{
    if (index >= modelData->stoichiometry->m)
    {
        throw_llvm_exception("index out of range");
    }

    return csr_matrix_ddot(index, modelData->stoichiometry, reactionRates);
}

void LLVMExecutableModel::testConstraints()
{
}

string LLVMExecutableModel::getInfo()
{
    stringstream stream;

    double *tmp;

    int nFloat = getNumFloatingSpecies();
    int nBound = getNumBoundarySpecies();
    int nComp = getNumCompartments();
    int nGlobalParam = getNumGlobalParameters();
    int nEvents = getNumEvents();
    int nReactions = getNumReactions();

    stream << "* Calculated Values *" << endl;

    tmp = new double[nFloat];
    getFloatingSpeciesAmounts(nFloat, 0, tmp);
    stream << "FloatingSpeciesAmounts:" << endl;
    dump_array(stream, nFloat, tmp);

    getFloatingSpeciesConcentrations(nFloat, 0, tmp);
    stream << "FloatingSpeciesConcentrations:" << endl;
    dump_array(stream, nFloat, tmp);

    this->getFloatingSpeciesInitConcentrations(nFloat, 0, tmp);
    stream << "FloatingSpeciesInitConcentrations:" << endl;
    dump_array(stream, nFloat, tmp);
    delete[] tmp;

    tmp = new double[nReactions];
    getReactionRates(nReactions, 0, tmp);
    stream << "Reaction Rates:" << endl;
    dump_array(stream, nReactions, tmp);
    delete[] tmp;

    tmp = new double[nBound];
    getBoundarySpeciesAmounts(nBound, 0, tmp);
    stream << "BoundarySpeciesAmounts:" << endl;
    dump_array(stream, nBound, tmp);

    getBoundarySpeciesConcentrations(nBound, 0, tmp);
    stream << "BoundarySpeciesConcentrations:" << endl;
    dump_array(stream, nBound, tmp);
    delete[] tmp;

    tmp = new double[nComp];
    getCompartmentVolumes(nComp, 0, tmp);
    stream << "CompartmentVolumes:" << endl;
    dump_array(stream, nComp, tmp);

    this->getCompartmentInitVolumes(nComp, 0, tmp);
    stream << "CompartmentInitVolumes:" << endl;
    dump_array(stream, nComp, tmp);
    delete[] tmp;

    tmp = new double[nGlobalParam];
    getGlobalParameterValues(nGlobalParam, 0, tmp);
    stream << "GlobalParameters:" << endl;
    dump_array(stream, nGlobalParam, tmp);
    delete[] tmp;

    tmp = new double[nGlobalParam];
    getGlobalParameterValues(nGlobalParam, 0, tmp);
    stream << "GlobalParameters:" << endl;
    dump_array(stream, nGlobalParam, tmp);
    delete[] tmp;

    unsigned char *tmpEvents = new unsigned char[nEvents];
    getEventTriggers(nEvents, 0, tmpEvents);
    stream << "Events Trigger Status:" << endl;
    dump_array(stream, nEvents, (bool*)tmpEvents);
    delete[] tmpEvents;

    stream << *modelData;

    return stream.str();
}

int LLVMExecutableModel::getFloatingSpeciesIndex(const string& id)
{
    try
    {
        return symbols->getFloatingSpeciesIndex(id, false);
    }
    catch (LLVMException&)
    {
        return -1;
    }
}

string LLVMExecutableModel::getFloatingSpeciesId(int index)
{
    return symbols->getFloatingSpeciesId(index);
}

int LLVMExecutableModel::getBoundarySpeciesIndex(const string& id)
{
    try
    {
        return symbols->getBoundarySpeciesIndex(id);
    }
    catch (LLVMException&)
    {
        return -1;
    }
}

string LLVMExecutableModel::getBoundarySpeciesId(int indx)
{
    vector<string> ids = symbols->getBoundarySpeciesIds();
    if (indx < ids.size())
    {
        return ids[indx];
    }
    else
    {
        throw_llvm_exception("index out of range");
        return "";
    }
}

int LLVMExecutableModel::getGlobalParameterIndex(const string& id)
{
    try
    {
        return symbols->getGlobalParameterIndex(id);
    }
    catch(LLVMException&)
    {
        return -1;
    }
}

string LLVMExecutableModel::getGlobalParameterId(int id)
{
    return symbols->getGlobalParameterId(id);
}

int LLVMExecutableModel::getCompartmentIndex(const string& id)
{
    try
    {
        return symbols->getCompartmentIndex(id);
    }
    catch (LLVMException&)
    {
        return -1;
    }
}

string LLVMExecutableModel::getCompartmentId(int id)
{
    vector<string> ids = symbols->getCompartmentIds();
    if (id < ids.size())
    {
        return ids[id];
    }
    else
    {
        throw_llvm_exception("index out of range");
        return "";
    }
}

int LLVMExecutableModel::getReactionIndex(const string& id)
{
    try
    {
        return symbols->getReactionIndex(id);
    }
    catch (LLVMException&)
    {
        return -1;
    }
}

string LLVMExecutableModel::getReactionId(int id)
{
    vector<string> ids = symbols->getReactionIds();
    if (id < ids.size())
    {
        return ids[id];
    }
    else
    {
        throw_llvm_exception("index out of range");
        return "";
    }
}

void LLVMExecutableModel::evalInitialConditions(uint32_t flags)
{
    evalInitialConditionsPtr(modelData, flags);
}

void LLVMExecutableModel::reset()
{
    uint opt = rr::Config::getInt(rr::Config::MODEL_RESET);
    Log(Logger::LOG_DEBUG) << "calling reset with default values: " << opt;
    reset(opt);
}

void LLVMExecutableModel::reset(int opt)
{
    // initializes the the model the init values specifie in the sbml, and
    // copies these to the initial initial conditions (not a typo),
    // sets the 'init(...)' values to the sbml specified init values.
    if (opt & SelectionRecord::SBML_INITIALIZE)
    {
        Log(Logger::LOG_INFORMATION) << "resetting init conditions";
        evalInitialConditions();
    }

    // eval the initial conditions and rates
    if (opt & SelectionRecord::TIME)
    {
        Log(Logger::LOG_INFORMATION) << "resetting time";
        setTime(0.0);
    }

    if (getCompartmentInitVolumesPtr && getFloatingSpeciesInitAmountsPtr
            && getGlobalParameterInitValuePtr)
    {
        // have to set compartments first, these are used to
        // convert between concentrations and amounts.
        unsigned size = max(modelData->numIndCompartments,
                modelData->numIndFloatingSpecies);

        // need at least 1 for global params
        size = max(size, 1u);

        double *buffer = new double[size];

        if (checkExact(SelectionRecord::COMPARTMENT, opt))
        {
            Log(Logger::LOG_INFORMATION) << "resetting compartment volumes";
            getCompartmentInitVolumes(modelData->numIndCompartments, 0, buffer);
            setCompartmentVolumes(modelData->numIndCompartments, 0, buffer);
        }

        if(opt & SelectionRecord::FLOATING)
        {
            if(opt & SelectionRecord::CONCENTRATION) {
                Log(Logger::LOG_INFORMATION) << "resetting floating species concentrations";
                getFloatingSpeciesInitConcentrations(modelData->numIndFloatingSpecies, 0, buffer);
                setFloatingSpeciesConcentrations(modelData->numIndFloatingSpecies, 0, buffer);
            } else {
                Log(Logger::LOG_INFORMATION) << "resetting floating species amounts";
                getFloatingSpeciesInitAmounts(modelData->numIndFloatingSpecies, 0, buffer);
                setFloatingSpeciesAmounts(modelData->numIndFloatingSpecies, 0, buffer);
            }
        }


        // were we forced to reset cms.
        bool reset_cm = false;

        // did someone change a cm, note: setGlobalParameterValues sets this bit,
        // so we have to save it before changing any GPs in reset.
        const bool dirty_cm = dirty & DIRTY_CONSERVED_MOIETIES;
        const bool dirty_init = dirty & DIRTY_INIT_SPECIES;

        // needed because conserved moiety global parameters depend on
        // float species init conditions.
        for (int gid = 0; gid < modelData->numIndGlobalParameters; ++gid)
        {
            bool cm = symbols->isConservedMoietyParameter(gid);
            bool depInit = !symbols->isIndependentInitGlobalParameter(gid);

            // reset gp if options say so
            if (checkExact(SelectionRecord::GLOBAL_PARAMETER, opt)
                    // or if opt say to reset cms and its a cm
                    || ((opt & SelectionRecord::CONSERVED_MOIETY) && cm)
                    // or if init conds have changes and its a cm (cm depends on init cond)
                    || (dirty_init && cm)
                    // or reseting global params which have init assignment rules
                    || (checkExact(SelectionRecord::DEPENDENT_INITIAL_GLOBAL_PARAMETER, opt) && depInit))
            {
                Log(Logger::LOG_DEBUG) << "!resetting global parameter, "
                        << gid << ", GLOBAL_PARAMETER: "
                        << checkExact(opt, SelectionRecord::GLOBAL_PARAMETER)
                        << ", CONSERVED_MOIETY: "
                        << ((opt & SelectionRecord::CONSERVED_MOIETY) && cm)
                        << "DEPENDENT_INITIAL_GLOBAL_PARAMETER: " <<
                            (checkExact(SelectionRecord::DEPENDENT_INITIAL_GLOBAL_PARAMETER, opt) && depInit);
                reset_cm |= cm;
                getGlobalParameterInitValues(1, &gid, buffer);
		Log(Logger::LOG_DEBUG) << "read global param init values";
                setGlobalParameterValues(1, &gid, buffer);
		Log(Logger::LOG_DEBUG) << "set global param current values";
            }
        }

        if (reset_cm)
        {
            // warn if we were forced to reset CMs
            if (dirty_cm)
            {
                Log(Logger::LOG_ERROR) << "Both initial conditions and "
                        "conserved moieties were user modified. As conserved moieties "
                        "are defined in terms of initial conditions, the conserved "
                        "moiety values were forcibly reset in terms of the species "
                        "initial conditions.";
            }

            // we've reset CMs. clear the dirty bit.
            dirty &= ~DIRTY_CONSERVED_MOIETIES;

            // the DIRTY_INIT_SPECIES bit is alwasy cleared at the
            // end of this func.
        }

        if(opt & SelectionRecord::RATE)
        {
            Log(Logger::LOG_INFORMATION) << "resetting rate rule values";

            for (int gid = modelData->numIndGlobalParameters;
                    gid < symbols->getGlobalParametersSize(); ++gid)
            {
                if(symbols->isRateRuleGlobalParameter(gid))
                {
                    getGlobalParameterInitValues(1, &gid, buffer);
                    setGlobalParameterValues(1, &gid, buffer);
                }
            }
        }

        delete[] buffer;
    }


    // this sets up the event system to pull the initial value
    // before the simulation starts.
    setTime(-1.0);

    // we've reset the species to their init values.
    dirty &= ~DIRTY_INIT_SPECIES;

    Log(Logger::LOG_DEBUG) << __FUNC__ << *modelData;
}

int LLVMExecutableModel::getStateVector(double* stateVector)
{
    if (stateVector == 0)
    {
        Log(Logger::LOG_TRACE) << __FUNC__ << ", stateVector: null, returning "
                << modelData->numRateRules + modelData->numIndFloatingSpecies;
        return modelData->numRateRules + modelData->numIndFloatingSpecies;
    }

    getRateRuleValues(stateVector);

    memcpy(stateVector + modelData->numRateRules,
            modelData->floatingSpeciesAmountsAlias,
            modelData->numIndFloatingSpecies * sizeof(double));


    if (Logger::LOG_TRACE <= rr::Logger::getLevel()) {

        LoggingBuffer log(Logger::LOG_TRACE, __FILE__, __LINE__);

        log.stream() << __FUNC__ << ",  out stateVector: ";
        if (stateVector) {
            dump_array(log.stream(), modelData->numRateRules + modelData->numIndFloatingSpecies, stateVector);
        } else {
            log.stream() << "null";
        }

        log.stream() << endl << __FUNC__ <<  ", Model: " << endl << this;
    }

    return modelData->numRateRules + modelData->numIndFloatingSpecies;
}

int LLVMExecutableModel::setStateVector(const double* stateVector)
{
    if (stateVector == 0)
    {
        return -1;
    }

    memcpy(modelData->rateRuleValuesAlias, stateVector, modelData->numRateRules * sizeof(double));

    memcpy(modelData->floatingSpeciesAmountsAlias,
            stateVector + modelData->numRateRules,
            modelData->numIndFloatingSpecies * sizeof(double));

    evalVolatileStoichPtr(modelData);

    dirty |= DIRTY_REACTION_RATES;

    return modelData->numRateRules + modelData->numIndFloatingSpecies;
}

void LLVMExecutableModel::print(std::ostream &stream)
{
    stream << "LLVMExecutableModel" << endl;
    stream << getInfo();
}

void LLVMExecutableModel::getIds(int types, std::list<std::string> &ids)
{
    if (checkExact(SelectionRecord::FLOATING | SelectionRecord::AMOUNT |
            SelectionRecord::CURRENT, types)
            && (SelectionRecord::INDEPENDENT & types)) {
        addIds(this, 0, symbols->getIndependentFloatingSpeciesSize(),
                &rr::ExecutableModel::getFloatingSpeciesId, ids);
    }

    if (checkExact(SelectionRecord::FLOATING | SelectionRecord::AMOUNT |
            SelectionRecord::CURRENT, types)
            && (SelectionRecord::DEPENDENT & types)) {
        addIds(this, symbols->getIndependentFloatingSpeciesSize(),
                symbols->getFloatingSpeciesSize(),
                &rr::ExecutableModel::getFloatingSpeciesId, ids);
    }

    if (checkExact(SelectionRecord::BOUNDARY | SelectionRecord::AMOUNT |
            SelectionRecord::CURRENT, types)
            && (SelectionRecord::INDEPENDENT & types)) {
        addIds(this, 0, symbols->getIndependentBoundarySpeciesSize(),
                &rr::ExecutableModel::getBoundarySpeciesId, ids);
    }

    if (checkExact(SelectionRecord::BOUNDARY | SelectionRecord::AMOUNT |
            SelectionRecord::CURRENT, types)
            && (SelectionRecord::DEPENDENT & types)) {
        addIds(this, symbols->getIndependentBoundarySpeciesSize(),
                symbols->getBoundarySpeciesSize(),
                &rr::ExecutableModel::getBoundarySpeciesId, ids);
    }

    if (checkExact(SelectionRecord::FLOATING | SelectionRecord::CONCENTRATION |
            SelectionRecord::CURRENT, types)
            && (SelectionRecord::INDEPENDENT & types)) {
        addConcIds(this, 0, symbols->getIndependentFloatingSpeciesSize(),
                &rr::ExecutableModel::getFloatingSpeciesId, ids);
    }

    if (checkExact(SelectionRecord::FLOATING | SelectionRecord::CONCENTRATION |
            SelectionRecord::CURRENT, types)
            && (SelectionRecord::DEPENDENT & types)) {
        addConcIds(this, symbols->getIndependentFloatingSpeciesSize(),
                symbols->getFloatingSpeciesSize(),
                &rr::ExecutableModel::getFloatingSpeciesId, ids);
    }

    if (checkExact(SelectionRecord::BOUNDARY | SelectionRecord::CONCENTRATION |
            SelectionRecord::CURRENT, types)
            && (SelectionRecord::INDEPENDENT & types)) {
        addConcIds(this, 0, symbols->getIndependentBoundarySpeciesSize(),
                &rr::ExecutableModel::getBoundarySpeciesId, ids);
    }

    if (checkExact(SelectionRecord::BOUNDARY | SelectionRecord::CONCENTRATION |
            SelectionRecord::CURRENT, types)
            && (SelectionRecord::DEPENDENT & types)) {
        addConcIds(this, symbols->getIndependentBoundarySpeciesSize(),
                symbols->getBoundarySpeciesSize(),
                &rr::ExecutableModel::getBoundarySpeciesId, ids);
    }


    if (checkExact(SelectionRecord::_COMPARTMENT | SelectionRecord::CURRENT, types)
            && (SelectionRecord::INDEPENDENT & types)) {
        addIds(this, 0, symbols->getIndependentCompartmentSize(),
                &rr::ExecutableModel::getCompartmentId, ids);
    }

    if (checkExact(SelectionRecord::_COMPARTMENT | SelectionRecord::CURRENT, types)
            && (SelectionRecord::DEPENDENT & types)) {
        addIds(this, symbols->getIndependentCompartmentSize(),
                symbols->getCompartmentsSize(),
                &rr::ExecutableModel::getCompartmentId, ids);
    }

    if (checkExact(SelectionRecord::_GLOBAL_PARAMETER | SelectionRecord::CURRENT, types)
            && (SelectionRecord::INDEPENDENT & types)) {
        addIds(this, 0, symbols->getIndependentGlobalParameterSize(),
                &rr::ExecutableModel::getGlobalParameterId, ids);
    }

    if (checkExact(SelectionRecord::_GLOBAL_PARAMETER | SelectionRecord::CURRENT, types)
            && (SelectionRecord::DEPENDENT & types)) {
        addIds(this, symbols->getIndependentGlobalParameterSize(),
                symbols->getGlobalParametersSize(),
                &rr::ExecutableModel::getGlobalParameterId, ids);
    }

    // only get init values indepndent values, dep, with assignment rules
    // always have the same value.
    //if (checkExact(SelectionRecord::_GLOBAL_PARAMETER | SelectionRecord::INITIAL, types)
    //        && (SelectionRecord::INDEPENDENT & types)) {
    //    for(int i = 0; i < symbols->getIndependentGlobalParameterSize(); ++i) {
    //        std::string gs = symbols->getGlobalParameterId(i);
    //        if(symbols->isIndependentGlobalParameter(gs)) {
    //            ids.push_back("init(" + gs + ")");
    //        }
    //    }
    //}

    //if (checkExact(SelectionRecord::_GLOBAL_PARAMETER | SelectionRecord::INITIAL, types)
    //        && (SelectionRecord::DEPENDENT & types)) {
    //    for(int i = symbols->getIndependentGlobalParameterSize();
    //            i < symbols->getGlobalParametersSize(); ++i) {
    //        std::string gs = symbols->getGlobalParameterId(i);
    //        if(symbols->isIndependentGlobalParameter(gs)) {
    //            ids.push_back("init(" + gs + ")");
    //        }
    //    }
    //}

    if (checkExact(SelectionRecord::REACTION_RATE, types)) {
        addIds(this, 0, symbols->getReactionSize(),
                &rr::ExecutableModel::getReactionId, ids);
    }

    if (checkExact(SelectionRecord::INITIAL | SelectionRecord::FLOATING |
            SelectionRecord::CONCENTRATION, types) &&
            (SelectionRecord::INDEPENDENT & types)) {
        for (int i = 0; i < symbols->getInitFloatingSpeciesSize(); ++i) {
            ids.push_back("init([" + this->getFloatingSpeciesId(i) + "])");
        }
    }

    if (checkExact(SelectionRecord::INITIAL | SelectionRecord::FLOATING |
            SelectionRecord::CONCENTRATION, types) &&
            (SelectionRecord::DEPENDENT & types)) {
        for (int i = symbols->getInitFloatingSpeciesSize();
                i < symbols->getFloatingSpeciesSize(); ++i) {
            ids.push_back("init([" + this->getFloatingSpeciesId(i) + "])");
        }
    }

    if (checkExact(SelectionRecord::INITIAL | SelectionRecord::FLOATING |
            SelectionRecord::AMOUNT, types) &&
            (SelectionRecord::INDEPENDENT & types)) {
        for (int i = 0; i < symbols->getInitFloatingSpeciesSize(); ++i) {
            ids.push_back("init(" + this->getFloatingSpeciesId(i) + ")");
        }
    }

    if (checkExact(SelectionRecord::INITIAL | SelectionRecord::FLOATING |
            SelectionRecord::AMOUNT, types) &&
            (SelectionRecord::DEPENDENT & types)) {
        for (int i = symbols->getInitFloatingSpeciesSize();
                i < symbols->getFloatingSpeciesSize(); ++i) {
            ids.push_back("init(" + this->getFloatingSpeciesId(i) + ")");
        }
    }

    if (checkExact(SelectionRecord::FLOATING_AMOUNT_RATE, types)) {
        for (int i = 0; i < getNumIndFloatingSpecies(); ++i) {
            ids.push_back(this->getFloatingSpeciesId(i) + "'");
        }
    }

    if (checkExact(SelectionRecord::GLOBAL_PARAMETER_RATE, types)) {

        for (int i = 0; i < symbols->getRateRuleSize(); ++i) {
            ids.push_back(symbols->getRateRuleId(i) + "'");
        }
    }

    if (SelectionRecord::STATE_VECTOR == types) {
        int stateVectorSize = getStateVector(0);
        for (int i = 0; i < stateVectorSize; ++i) {
            ids.push_back(getStateVectorId(i));
        }
    }

    //if (SelectionRecord::EVENT & types)
    //{
    //    std::vector<std::string> eventIds = symbols->getEventIds();
    //    std::copy( eventIds.begin(), eventIds.end(), std::back_inserter(ids));
    //}

    // These are also displayed with global parameters, so
    // only add them if explicity asked for.
    if (SelectionRecord::CONSERVED_MOIETY == types)
    {
        for(uint i = 0; i < symbols->getConservedMoietySize(); ++i)
        {
            ids.push_back(symbols->getConservedMoietyId(i));
        }
    }
}

int LLVMExecutableModel::getSupportedIdTypes()
{
    return SelectionRecord::TIME |
        SelectionRecord::BOUNDARY_CONCENTRATION |
        SelectionRecord::FLOATING_CONCENTRATION |
        SelectionRecord::REACTION_RATE |
        SelectionRecord::FLOATING_AMOUNT_RATE |
        SelectionRecord::FLOATING_CONCENTRATION_RATE |
        SelectionRecord::COMPARTMENT |
        SelectionRecord::GLOBAL_PARAMETER |
        SelectionRecord::FLOATING_AMOUNT |
        SelectionRecord::BOUNDARY_AMOUNT |
        SelectionRecord::INITIAL_AMOUNT |
        SelectionRecord::INITIAL_CONCENTRATION |
        SelectionRecord::STOICHIOMETRY;
}

double LLVMExecutableModel::getValue(const std::string& id)
{
    const SelectionRecord &sel = getSelection(id);
    int index = sel.index;
    double result = 0;

    // check to see that we have valid selection ids
    switch(sel.selectionType)
    {
    case SelectionRecord::TIME:
        result = getTime();
        break;
    case SelectionRecord::FLOATING_AMOUNT:
        getFloatingSpeciesAmounts(1, &index, &result);
        break;
    case SelectionRecord::BOUNDARY_AMOUNT:
        getBoundarySpeciesAmounts(1, &index, &result);
        break;
    case SelectionRecord::COMPARTMENT:
        getCompartmentVolumes(1, &index, &result);
        break;
    case SelectionRecord::GLOBAL_PARAMETER:
        getGlobalParameterValues(1, &index, &result);
        break;
    case SelectionRecord::REACTION_RATE:
        getReactionRates(1, &index, &result);
        break;
    case SelectionRecord::FLOATING_CONCENTRATION:
        getFloatingSpeciesConcentrations(1, &index, &result);
        break;
    case SelectionRecord::BOUNDARY_CONCENTRATION:
        getBoundarySpeciesConcentrations(1, &index, &result);
        break;
    case SelectionRecord::FLOATING_AMOUNT_RATE:
        getFloatingSpeciesAmountRates(1, &index, &result);
        break;
    case SelectionRecord::GLOBAL_PARAMETER_RATE:
        getRateRueRates(1, &index, &result);
        break;
    case SelectionRecord::INITIAL_FLOATING_AMOUNT:
        getFloatingSpeciesInitAmounts(1, &index, &result);
        break;
    case SelectionRecord::INITIAL_COMPARTMENT:
        getCompartmentInitVolumes(1, &index, &result);	
        break;
    case SelectionRecord::INITIAL_FLOATING_CONCENTRATION:
        getFloatingSpeciesInitConcentrations(1, &index, &result);
        break;
    case SelectionRecord::INITIAL_GLOBAL_PARAMETER:
        getGlobalParameterInitValues(1, &index, &result);
        break;
    case SelectionRecord::EVENT:
    {
            bool trigger = getEventTrigger(index);
            result = trigger ? 1.0 : -1.0;
        }
        break;
    default:
        Log(Logger::LOG_ERROR) << "A new SelectionRecord should not have this value: "
        << sel.to_repr();
        throw LLVMException("Invalid selection '" + id + "' for setting value");
        break;
    }

    return result;
}


const rr::SelectionRecord& LLVMExecutableModel::getSelection(const std::string& str)
{
    SelectionMap::const_iterator i = selectionRecordCache.find(str);

    if (i == selectionRecordCache.end())
    {
        SelectionRecord sel(str);
        int index = -1;

        if (sel.selectionType == SelectionRecord::UNKNOWN)
        {
            throw LLVMException("invalid selection string " + str);
        }

        // check to see that we have valid selection ids
        switch(sel.selectionType)
        {
        case SelectionRecord::TIME:
            break;
        case SelectionRecord::UNKNOWN_ELEMENT:
            // check for sbml element types

            switch(symbols->getSymbolIndex(sel.p1, index))
            {
            case LLVMModelDataSymbols::FLOATING_SPECIES:
                sel.selectionType = SelectionRecord::FLOATING_AMOUNT;
                sel.index = index;
                break;
            case LLVMModelDataSymbols::BOUNDARY_SPECIES:
                sel.selectionType = SelectionRecord::BOUNDARY_AMOUNT;
                sel.index = index;
                break;
            case LLVMModelDataSymbols::COMPARTMENT:
                sel.selectionType = SelectionRecord::COMPARTMENT;
                sel.index = index;
                break;
            case LLVMModelDataSymbols::GLOBAL_PARAMETER:
                sel.selectionType = SelectionRecord::GLOBAL_PARAMETER;
                sel.index = index;
                break;
            case LLVMModelDataSymbols::REACTION:
                sel.selectionType = SelectionRecord::REACTION_RATE;
                sel.index = index;
                break;
            case LLVMModelDataSymbols::EVENT:
                sel.selectionType = SelectionRecord::EVENT;
                sel.index = index;
                break;
            default:
                throw LLVMException("No sbml element exists for symbol '" + str + "'");
                break;
            }
            break;

        case SelectionRecord::UNKNOWN_CONCENTRATION:
            switch(symbols->getSymbolIndex(sel.p1, index))
            {
            case LLVMModelDataSymbols::FLOATING_SPECIES:
                sel.selectionType = SelectionRecord::FLOATING_CONCENTRATION;
                sel.index = index;
                break;
            case LLVMModelDataSymbols::BOUNDARY_SPECIES:
                sel.selectionType = SelectionRecord::BOUNDARY_CONCENTRATION;
                sel.index = index;
                break;
            default:
                string msg = "No sbml element exists for concentration selection '" + str + "'";
                throw LLVMException(msg);
            }
            break;

        case SelectionRecord::FLOATING_AMOUNT_RATE:
            if (symbols->getSymbolIndex(sel.p1, index) == LLVMModelDataSymbols::FLOATING_SPECIES)
            {
                sel.selectionType = SelectionRecord::FLOATING_AMOUNT_RATE;
                sel.index = index;
            }
            else
            {
                // this will throw if there is no rate rule for the symbol.
                sel.index = symbols->getRateRuleIndex(sel.p1);
                sel.selectionType = SelectionRecord::GLOBAL_PARAMETER_RATE;
            }
            break;

        case SelectionRecord::INITIAL_AMOUNT:
            switch(symbols->getSymbolIndex(sel.p1, index))
            {
            case LLVMModelDataSymbols::FLOATING_SPECIES:
                sel.selectionType = SelectionRecord::INITIAL_FLOATING_AMOUNT;
                sel.index = index;
                break;
            case LLVMModelDataSymbols::COMPARTMENT:
                sel.selectionType = SelectionRecord::INITIAL_COMPARTMENT;
                sel.index = index;
                break;
            case LLVMModelDataSymbols::GLOBAL_PARAMETER:
                sel.selectionType = SelectionRecord::INITIAL_GLOBAL_PARAMETER;
                sel.index = index;
                break;
            default:
                string msg = "Invalid Id for initial value: '" + str + "'";
                throw LLVMException(msg);
            }
            assert(sel.index >= 0);
            break;

        case SelectionRecord::INITIAL_CONCENTRATION:
            if (symbols->getSymbolIndex(sel.p1, index) == LLVMModelDataSymbols::FLOATING_SPECIES)
            {
                sel.selectionType = SelectionRecord::INITIAL_FLOATING_CONCENTRATION;
                sel.index = index;
                break;
            }
            else
            {
                throw LLVMException("Invalid id '" + str + "' for inital floating species concentration");
                break;
            }
            break;

        default:
            Log(Logger::LOG_ERROR) << "A new SelectionRecord should not have this value: "
                << sel.to_repr();
            throw LLVMException("Invalid selection '" + str + "' for setting value");
            break;
        }

        // if we get here, should have a valid and set sel record.
        selectionRecordCache[str] = sel;

        Log(Logger::LOG_DEBUG) << "caching selection record stirng " << str;

        i = selectionRecordCache.find(str);
        assert(i != selectionRecordCache.end());
    }

    return i->second;
}

void LLVMExecutableModel::setValue(const std::string& id, double value)
{
    const SelectionRecord &sel = getSelection(id);
    int index = sel.index;
    double result = 0;

    switch(sel.selectionType)
    {
    case SelectionRecord::TIME:
        setTime(value);
        break;
    case SelectionRecord::FLOATING_AMOUNT:
        setFloatingSpeciesAmounts(1, &index, &value);
        break;
    case SelectionRecord::COMPARTMENT:
        setCompartmentVolumes(1, &index, &value);
        break;
    case SelectionRecord::GLOBAL_PARAMETER:
        setGlobalParameterValues(1, &index, &value);
        for(uint cm = 0; cm < symbols->getConservedMoietySize(); ++cm) {
            // is this the index of the cm we are setting?
            if (symbols->getConservedMoietyId(cm) == id) {
                // eliminate loop?
                uint d = symbols->getDepSpeciesIndexForConservedMoietyId(id);
                int i_dep = modelData->numIndFloatingSpecies+d;
                double amt;
                getFloatingSpeciesAmounts(1,&i_dep,&amt);
                // if the dependent species went negative, we need to correct for it
                if (amt < 0) {
                    const std::vector<uint>& ind = symbols->getIndSpeciesIndexForConservedMoietyId(id);
                    std::vector<double> ind_amts(ind.size());
                    double zero=0;

                    uint k=0;
                    for(std::vector<uint>::const_iterator it = ind.begin(); it != ind.end(); ++it, ++k) {
                        int i = *it;
                        getFloatingSpeciesAmounts(1,&i,&ind_amts.at(k));
                        setFloatingSpeciesAmounts(1,&i,&zero);
                    }

                    // double amt2;
                    // getFloatingSpeciesAmounts(1,&i_dep,&amt2);
                    // if (amt2 < 0)
                    //     throw LLVMException("Conserved total exceeded allowable bounds, cannot compute dependent species.");
                    // double alpha=-amt/(amt2-amt);
                    //
                    // k=0;
                    // for(std::vector<uint>::const_iterator it = ind.begin(); it != ind.end(); ++it, ++k) {
                    //     int i = *it;
                    //     double v=ind_amts.at(k)*(1.-alpha);
                    //     setFloatingSpeciesAmounts(1,&i,&v);
                    // }
                }
            }
        }
        break;
    case SelectionRecord::BOUNDARY_AMOUNT:
        setBoundarySpeciesAmounts(1, &index, &value);
        break;
    case SelectionRecord::REACTION_RATE:
        throw LLVMException("The sbml id '" + id +
                "' is for a reaction, reaction rates can not be set externally");
        break;
    case SelectionRecord::FLOATING_CONCENTRATION:
        setFloatingSpeciesConcentrations(1, &index, &value);
        break;
    case SelectionRecord::BOUNDARY_CONCENTRATION:
        setBoundarySpeciesConcentrations(1, &index, &value);
        break;
    case SelectionRecord::INITIAL_FLOATING_AMOUNT:
        setFloatingSpeciesInitAmounts(1, &index, &value);
        break;
    case SelectionRecord::INITIAL_COMPARTMENT:
        setCompartmentInitVolumes(1, &index, &value);
        break;
    case SelectionRecord::INITIAL_FLOATING_CONCENTRATION:
        setFloatingSpeciesInitConcentrations(1, &index, &value);
        break;
    case SelectionRecord::INITIAL_GLOBAL_PARAMETER:
        setGlobalParameterInitValues(1, &index, &value);
        break;
    default:
        throw LLVMException("Invalid selection '" + sel.to_string() + "' for setting value");
        break;
    }
}

int LLVMExecutableModel::getFloatingSpeciesConcentrationRates(int len,
        const int* indx, double* values)
{
    double* dydt = 0;
    // buffer for volumes
    double* volumes = 0;

    try
    {
        uint dydtSize = modelData->numRateRules + modelData->numIndFloatingSpecies;
        uint ncomp = getNumCompartments();

        dydt = (double*)calloc(dydtSize, sizeof(double));
        volumes = (double*)calloc(ncomp, sizeof(double));

        getCompartmentVolumes(ncomp, 0, volumes);

        // state vector is packed such that first numRateRules are the rate rule rates,
        // and the last numIndFloatingSpecies are the number of independent species.
        getStateVectorRate(this->getTime(), 0, dydt);

        double* amountRates = dydt + modelData->numRateRules;

        for (uint i = 0; i < len; ++i)
        {
            uint j = indx ? indx[i] : i;
            if (j >= modelData->numIndFloatingSpecies)
            {
                throw std::out_of_range(std::string("index out of range in ")
                                        + __FUNC__);
            }

            Log(Logger::LOG_DEBUG) << "i: " << i << ", j: " << j
                    << ", comp index: "
                    << symbols->getCompartmentIndexForFloatingSpecies(j)
                    << ", vol: "
                    << volumes[symbols->getCompartmentIndexForFloatingSpecies(j)]
                    << endl;
            values[i] = amountRates[j] /
                    volumes[symbols->getCompartmentIndexForFloatingSpecies(j)];
        }
    }
    catch(...)
    {
        free(dydt);
        free(volumes);
        throw;
    }

    free(dydt);
    free(volumes);
    return len;
}

int LLVMExecutableModel::setBoundarySpeciesAmounts(int len, const int* indx,
        const double* values)
{
    int result = -1;
    if (setBoundarySpeciesAmountPtr)
    {
        result = setValues(setBoundarySpeciesAmountPtr,
                &LLVMExecutableModel::getBoundarySpeciesId, len, indx, values);
    }
    return result;
}

std::string LLVMExecutableModel::getStateVectorId(int index)
{
    if (index < modelData->numRateRules )
    {
        return symbols->getRateRuleId(index);
    }
    else
    {
        return symbols->getFloatingSpeciesId(index - modelData->numRateRules);
    }
}

int LLVMExecutableModel::getEventIndex(const std::string& eventId)
{
    try
    {
        return symbols->getEventIndex(eventId);
    }
    catch(LLVMException&)
    {
        return -1;
    }
}

std::string LLVMExecutableModel::getEventId(int indx)
{
    vector<string> ids = symbols->getEventIds();
    if (indx < ids.size())
    {
        return ids[indx];
    }
    else
    {
        throw_llvm_exception("index out of range");
        return "";
    }
}

void LLVMExecutableModel::setEventListener(int index,
        rr::EventListenerPtr eventHandler)
{
    if (index < modelData->numEvents)
    {
        Log(Logger::LOG_DEBUG) << "setting event handler " << index << " to " << eventHandler;
        eventListeners[index] = eventHandler;
    }
    else
    {
        throw_llvm_exception("index " + rr::toString(index) + " out of range");
    }
}

rr::EventListenerPtr LLVMExecutableModel::getEventListener(int index)
{
    if (index < modelData->numEvents)
    {
        return eventListeners[index];
    }
    else
    {
        throw_llvm_exception("index " + rr::toString(index) + " out of range");
        return EventListenerPtr();
    }
}

LLVMExecutableModel* LLVMExecutableModel::dummy()
{
    return new LLVMExecutableModel();
}

int LLVMExecutableModel::getNumRateRules()
{
    return modelData->numRateRules;
}

std::vector<std::string> LLVMExecutableModel::getRateRuleSymbols() const {
    std::vector<std::string> result;

    for (uint k = 0; k < symbols->getRateRuleSize(); ++k)
        result.push_back(symbols->getRateRuleId(k));

    return result;
}

int LLVMExecutableModel::getFloatingSpeciesAmounts(int len, const int* indx,
        double* values)
{
    return getValues(getFloatingSpeciesAmountPtr, len, indx, values);
}

int LLVMExecutableModel::setFloatingSpeciesAmounts(int len, int const *indx,
        const double *values)
{
    for (int i = 0; i < len; ++i)
    {
        int j = indx ? indx[i] : i;
        bool result =  setFloatingSpeciesAmountPtr(modelData, j, values[i]);
        if (!result)
        {
            // check if a conserved moiety
            uint cmIndex = 0;
            if (symbols->isConservedMoietySpecies(j, cmIndex))
            {
                int gpIndex = symbols->getConservedMoietyGlobalParameterIndex(cmIndex);

                double currAmt;
                getFloatingSpeciesAmounts(1, &j, &currAmt);

                double diff = values[i] - currAmt;

                double currCmVal;
                getGlobalParameterValues(1, &gpIndex, &currCmVal);

                double newCmVal = currCmVal + diff;

                Log(Logger::LOG_INFORMATION) << "updating CM "
                        << symbols->getConservedMoietyId(cmIndex)
                        << " for conserved species "
                        << symbols->getFloatingSpeciesId(j)
                        << ", setting CM to " << newCmVal
                        << ", was " << currCmVal;

                setGlobalParameterValues(1, &gpIndex, &newCmVal);
            }
            else
            {
            std::stringstream s;
            string id = symbols->getFloatingSpeciesId(j);
            s << "could not set value for NON conserved moiety floating species " << id;

            if (symbols->hasAssignmentRule(id))
            {
                s << ", it is defined by an assignment rule, can not be set independently.";
            }
            else if (symbols->hasRateRule(id))
            {
                s << ", it is defined by a rate rule and can not be set independently.";
            }

            throw_llvm_exception(s.str());
            }
        }
    }
    return len;
}

int LLVMExecutableModel::setFloatingSpeciesConcentrations(int len,
        const int* indx, const double* values)
{
    for (int i = 0; i < len; ++i)
    {
        int j = indx ? indx[i] : i;
        bool result = setFloatingSpeciesConcentrationPtr(modelData, j, values[i]);

        if (!result)
        {
            // check if a conserved moiety
            uint cmIndex = 0;
            if (symbols->isConservedMoietySpecies(j, cmIndex))
            {
                int gpIndex = symbols->getConservedMoietyGlobalParameterIndex(cmIndex);

                double currAmt;
                getFloatingSpeciesAmounts(1, &j, &currAmt);

                int volIndex = symbols->getCompartmentIndexForFloatingSpecies(j);
                double volume;
                this->getCompartmentVolumes(1, &volIndex, &volume);

                double diff = (values[i] * volume) - currAmt;

                double currCmVal;
                getGlobalParameterValues(1, &gpIndex, &currCmVal);

                double newCmVal = currCmVal + diff;

                Log(Logger::LOG_INFORMATION) << "updating CM "
                        << symbols->getConservedMoietyId(cmIndex)
                        << " for conserved species "
                        << symbols->getFloatingSpeciesId(j)
                        << ", setting CM to " << newCmVal
                        << ", was " << currCmVal;

                setGlobalParameterValues(1, &gpIndex, &newCmVal);
            }
            else
            {
                std::stringstream s;
                string id = symbols->getFloatingSpeciesId(j);
                s << "could not set value for NON conserved moiety floating species " << id;

                if (symbols->hasAssignmentRule(id))
                {
                    s << ", it is defined by an assignment rule, can not be set independently.";
                }
                else if (symbols->hasRateRule(id))
                {
                    s << ", it is defined by a rate rule and can not be set independently.";
                }

                throw_llvm_exception(s.str());
            }
        }
    }
    return len;
}

int LLVMExecutableModel::getBoundarySpeciesAmounts(int len, const int* indx,
        double* values)
{
    return getValues(getBoundarySpeciesAmountPtr, len, indx, values);
}

int LLVMExecutableModel::getBoundarySpeciesConcentrations(int len,
        const int* indx, double* values)
{
    return getValues(getBoundarySpeciesConcentrationPtr, len, indx, values);
}

int LLVMExecutableModel::setBoundarySpeciesConcentrations(int len,
        const int* indx, const double* values)
{
    int result = -1;
    if (setBoundarySpeciesConcentrationPtr)
    {
        result = setValues(setBoundarySpeciesConcentrationPtr,
                &LLVMExecutableModel::getBoundarySpeciesId, len, indx, values);
    }
    return result;
}

int LLVMExecutableModel::getGlobalParameterValues(int len, const int* indx,
        double* values)
{
    return getValues(getGlobalParameterPtr, len, indx, values);
}

int LLVMExecutableModel::setGlobalParameterValues(int len, const int* indx,
        const double* values)
{
    int result = -1;
    if (setGlobalParameterPtr)
    {
        result = setValues(setGlobalParameterPtr,
                &LLVMExecutableModel::getGlobalParameterId, len, indx, values);

        for (int i = 0; i < len; ++i)
        {
            int j = indx ? indx[i] : i;
            if (symbols->isConservedMoietyParameter(j))
            {
                dirty |= DIRTY_CONSERVED_MOIETIES;

                // clear the dirty init flag, user explicity set a cm, so they don't care
                // about init any more.
                dirty &= ~DIRTY_INIT_SPECIES;
            }
        }
    }
    return result;
}

int LLVMExecutableModel::getCompartmentVolumes(int len, const int* indx,
        double* values)
{
    return getValues(getCompartmentVolumePtr, len, indx, values);
}

int LLVMExecutableModel::getReactionRates(int len, const int* indx,
        double* values)
{
    // the reaction rates are a function of the model state, so someone
    // could have changed some parameter, so we need to re-evaluate.

    // this is an optimization to not re-calcualte the reaction rates
    // when selecting reaction rates in a simulations.
    // its possible to extend this approach to check whenever any value
    // is set, save that for a future optimization.
    if((flags & OPTIMIZE_REACTION_RATE_SELECTION) == 0
            || (flags & INTEGRATION) == 0 || (dirty & DIRTY_REACTION_RATES) != 0) {
        conversionFactor = evalReactionRatesPtr(modelData);
        dirty &= ~DIRTY_REACTION_RATES;
    }


    if (indx)
    {
        for (int i = 0; i < len; ++i)
        {
            int j = indx ? indx[i] : i;
            if (j < modelData->numReactions)
            {
                values[i] = modelData->reactionRatesAlias[j];
            }
            else
            {
                throw LLVMException("index out of range");
            }
        }
    }
    else
    {
        if (len > modelData->numReactions)
        {
            throw_llvm_exception("invalid length, length must be <= numReactions");
        }
        std::memcpy(values, modelData->reactionRatesAlias, len * sizeof(double));
    }
    return len;
}

int LLVMExecutableModel::getNumConservedMoieties()
{
    return symbols->getConservedMoietySize();
}

int LLVMExecutableModel::getConservedMoietyIndex(const string& name)
{
    try
    {
        return symbols->getConservedMoietyIndex(name);
    }
    catch(std::exception& e)
    {
        Log(Logger::LOG_DEBUG) << __FUNC__ << ", " << e.what();
    }
    return -1;
}

string LLVMExecutableModel::getConservedMoietyId(int index)
{
    return symbols->getConservedMoietyId(index);
}

int LLVMExecutableModel::getConservedMoietyValues(int len, const int* indx,
        double* values)
{
    int result = 0;
    for(int i = 0; i < len; ++i)
    {
        int j = indx ? indx[i] : i;
        int gpIndex = symbols->getConservedMoietyGlobalParameterIndex(j);
        result += getGlobalParameterValues(1, &gpIndex, &values[i]);
    }
    return result;
}

int LLVMExecutableModel::setConservedMoietyValues(int len, const int* indx,
        const double* values)
{
    int result = 0;
    for(int i = 0; i < len; ++i)
    {
        int j = indx ? indx[i] : i;
        int gpIndex = symbols->getConservedMoietyGlobalParameterIndex(j);
        result += setGlobalParameterValues(1, &gpIndex, &values[i]);
    }
    return result;
}

int LLVMExecutableModel::getFloatingSpeciesAmountRates(int len,
        int const *indx, double *values)
{
    uint dydtSize = modelData->numRateRules + modelData->numIndFloatingSpecies;

    double* dydt = (double*)calloc(dydtSize, sizeof(double));

    // state vector is packed such that first numRateRules are the rate rule rates,
    // and the last numIndFloatingSpecies are the number of independent species.
    this->getStateVectorRate(this->getTime(), 0, dydt);

    double* amountRates = dydt + modelData->numRateRules;

    for (uint i = 0; i < len; ++i)
    {
        uint j = indx ? indx[i] : i;
        if (j >= modelData->numIndFloatingSpecies)
        {
            free(dydt);
            throw std::out_of_range(std::string("index out of range in") + __FUNC__);
        }
        values[i] = amountRates[j];
    }

    free(dydt);
    return len;
}


int LLVMExecutableModel::getRateRueRates(int len,
        int const *indx, double *values)
{
    uint dydtSize = modelData->numRateRules;
    double* dydt = (double*)calloc(dydtSize, sizeof(double));

    // this will also move to a parameter for the evalRateRules func...
    modelData->rateRuleRates = dydt;
    evalRateRuleRatesPtr(modelData);
    modelData->rateRuleRates = 0;

    for (uint i = 0; i < len; ++i)
    {
        uint j = indx ? indx[i] : i;

        if (j > modelData->numRateRules) {
            std::stringstream ss;
            ss << "index " << j << " out of range";
            free(dydt);
            throw std::out_of_range(ss.str());
        }
        values[i] = dydt[j];
    }

    free(dydt);
    return len;
}


int LLVMExecutableModel::setCompartmentVolumes(int len, const int* indx,
        const double* values)
{
    int result = -1;
    if (setCompartmentVolumePtr)
    {
        result = setValues(setCompartmentVolumePtr,
                &LLVMExecutableModel::getCompartmentId, len, indx, values);
    }
    return result;
}



double LLVMExecutableModel::getStoichiometry(int speciesIndex, int reactionIndex)
{
    double result = csr_matrix_get_nz(modelData->stoichiometry, speciesIndex, reactionIndex);
    return isnan(result) ? 0 : result;
}

int LLVMExecutableModel::getStoichiometryMatrix(int* pRows, int* pCols,
        double** pData)
{
    // asking for matrix size
    if (pRows && pCols && pData == 0)
    {
        *pRows = modelData->stoichiometry->m;
        *pCols = modelData->stoichiometry->n;
        return modelData->stoichiometry->m * modelData->stoichiometry->n;
    }

    // allocate data
    if (pRows && pCols && pData && *pData == 0)
    {
        double* data = (double*)malloc(modelData->stoichiometry->m *
                modelData->stoichiometry->n * sizeof(double));
        *pRows = modelData->stoichiometry->m;
        *pCols = modelData->stoichiometry->n;
        *pData = data;
        csr_matrix_fill_dense(modelData->stoichiometry, data);

        return modelData->stoichiometry->m * modelData->stoichiometry->n;
    }

    // use user data
    if (pRows && *pRows == modelData->stoichiometry->m &&
            pCols && *pCols == modelData->stoichiometry->n && pData && *pData)
    {
        double* data = *pData;
        csr_matrix_fill_dense(modelData->stoichiometry, data);

        return modelData->stoichiometry->m * modelData->stoichiometry->n;
    }


    throw_llvm_exception("invalid args");
}



/******************************* Events Section *******************************/
#if (1) /**********************************************************************/
/******************************************************************************/

int LLVMExecutableModel::getNumEvents()
{
    return modelData->numEvents;
}

int LLVMExecutableModel::getEventTriggers(int len, const int *indx, unsigned char *values)
{
    if (len <= 0)
    {
        return modelData->numEvents;
    }
    else
    {
        for (int i = 0; i < len; ++i)
        {
            int j = indx ? indx[i] : i;
            if (j < modelData->numEvents)
            {
                values[j] = getEventTrigger(j);
            }
            else
            {
                throw LLVMException("index out of range");
            }
        }
        return len;
    }
}

int LLVMExecutableModel::applyEvents(double timeEnd,
        const unsigned char* previousEventStatus, const double *initialState,
        double* finalState)
{
    int assignedEvents = 0;
    modelData->time = timeEnd;

    if(initialState) {
        setStateVector(initialState);
    }

    if(modelData->numEvents) {
        // copy event status into local vars, these
        // are modified via applyEvents
		// Uses pointers as iterators
        vector<unsigned char> prevEventState(previousEventStatus,
                previousEventStatus + modelData->numEvents);

        vector<unsigned char> currEventStatus(modelData->numEvents);

        unsigned char *p1 = &prevEventState[0];
        unsigned char *p2 = &currEventStatus[0];

        pendingEvents.eraseExpiredEvents();

        while(applyEvents(p1, p2))
        {
            assignedEvents++;
			// Update previous events
            std::swap(p1, p2);
        }
    }

    if(finalState) {
        getStateVector(finalState);
    }

    return assignedEvents;
}


void  LLVMExecutableModel::getEventRoots(double time, const double* y, double* gdot)
{
    modelData->time = time;

    double *savedRateRules = modelData->rateRuleValuesAlias;
    double *savedFloatingSpeciesAmounts = modelData->floatingSpeciesAmountsAlias;

    if (y)
    {
        //memcpy(modelData->rateRuleValues, y,
        //        modelData->numRateRules * sizeof(double));

        //memcpy(modelData->floatingSpeciesAmounts, y + modelData->numRateRules,
        //        modelData->numIndFloatingSpecies * sizeof(double));

        modelData->rateRuleValuesAlias = const_cast<double*>(y);
        modelData->floatingSpeciesAmountsAlias = const_cast<double*>(y + modelData->numRateRules);

        evalVolatileStoichPtr(modelData);
    }

    for (uint i = 0; i < modelData->numEvents; ++i)
    {
        unsigned char triggered = getEventTriggerPtr(modelData, i);

        gdot[i] = triggered ? 1.0 : -1.0;
    }

    modelData->rateRuleValuesAlias = savedRateRules;
    modelData->floatingSpeciesAmountsAlias = savedFloatingSpeciesAmounts;

    return;
}

double LLVMExecutableModel::getNextPendingEventTime(bool pop)
{
    return pendingEvents.getNextPendingEventTime();
}

int LLVMExecutableModel::getPendingEventSize()
{
    return pendingEvents.size();
}

void LLVMExecutableModel::resetEvents()
{
}

bool LLVMExecutableModel::applyEvents(unsigned char* prevEventState,
        unsigned char* currEventState)
{
    for (uint i = 0; i < modelData->numEvents; ++i)
    {
        bool c = getEventTrigger(i);
        currEventState[i] = c;

        Log(Logger::LOG_DEBUG) << "event " << i << ", previous state: " <<
                (bool)prevEventState[i] << ", current state: " << (bool)c;
		/*std::cerr << "event " << i << ", previous state: " <<
			(bool)prevEventState[i] << ", current state: " << (bool)c << std::endl;*/

        // transition from non-triggered to triggered
        if (c && !prevEventState[i])
        {
            const EventListenerPtr &handler = eventListeners[i];
            if(handler)
            {
                uint result = handler->onTrigger(this, i, symbols->getEventId(i));

                if(result & EventListener::HALT_SIMULATION) {
                    throw EventListenerException(result);
                }
            }
            pendingEvents.push(rrllvm::Event(*this, i));
        }
    }

    // fire the highest priority event, this causes state change
    // return true if we incured a state change
    return pendingEvents.applyEvents();
}

bool LLVMExecutableModel::getEventTieBreak(uint eventA, uint eventB)
{
    /*
    C_ASSERT(sizeof(TieBreakKey) == 8 && sizeof(uint) == 4);

    bool result;
    TieBreakKey keyA = eventA;
    TieBreakKey keyB = eventB;
    TieBreakKey key = keyA << 32 | keyB;
    TieBreakMap::iterator i = tieBreakMap.find(key);
    if (i != tieBreakMap.end())
    {
        result = !i->second;
        i->second = result;
    }
    else
    {
        result = true;
        tieBreakMap.insert(std::pair<TieBreakKey,bool>(key, result));
    }
    */

    bool result = (std::rand() % 2) == 0;



    Log(Logger::LOG_DEBUG) << "tie break, a: " << eventA << ", b: " <<
            eventB << ", result: " << result;

    return result;
}


/******************************* Events Section *******************************/
#endif /***********************************************************************/
/******************************************************************************/

/******************************* Initial Conditions Section *******************/
#if (1) /**********************************************************************/
/******************************************************************************/


int LLVMExecutableModel::setFloatingSpeciesInitConcentrations(int len,
        const int* indx, const double* values)
{
    int result = -1;
    if (setFloatingSpeciesInitConcentrationsPtr)
    {
        result = setValues(setFloatingSpeciesInitConcentrationsPtr,
                &LLVMExecutableModel::getFloatingSpeciesId, len, indx, values);
    }

    dirty |= DIRTY_INIT_SPECIES;

    // as a convienice to users, this resets the amounts and whatever depends
    // on them.
    reset(SelectionRecord::TIME | SelectionRecord::FLOATING);

    return result;
}

int LLVMExecutableModel::getFloatingSpeciesInitConcentrations(int len,
        const int* indx, double* values)
{
    int result = -1;
    if (getFloatingSpeciesInitConcentrationsPtr)
    {
        result = getValues(getFloatingSpeciesInitConcentrationsPtr, len, indx, values);
    }
    return result;
}

int LLVMExecutableModel::setFloatingSpeciesInitAmounts(int len, int const *indx,
            double const *values)
{
    int result = -1;
    if (setFloatingSpeciesInitAmountsPtr)
    {
        result = setValues(setFloatingSpeciesInitAmountsPtr,
                &LLVMExecutableModel::getFloatingSpeciesId, len, indx, values);
    }

    dirty |= DIRTY_INIT_SPECIES;

    // as a convienice to users, this resets the amounts and whatever depends
    // on them.
    reset(SelectionRecord::TIME | SelectionRecord::FLOATING);
    return result;
}

int LLVMExecutableModel::getFloatingSpeciesInitAmounts(int len, int const *indx,
                double *values)
{
    int result = -1;
    if (getFloatingSpeciesInitAmountsPtr)
    {
        result = getValues(getFloatingSpeciesInitAmountsPtr, len, indx, values);
    }
    return result;
}

int LLVMExecutableModel::setCompartmentInitVolumes(int len, const int *indx,
            double const *values)
{
    int result = -1;
    if (setCompartmentInitVolumesPtr)
    {
        result = setValues(setCompartmentInitVolumesPtr,
                &LLVMExecutableModel::getCompartmentId, len, indx, values);
    }
    return result;
}

int LLVMExecutableModel::getCompartmentInitVolumes(int len, const int *indx,
                double *values)
{
    int result = -1;
    if (getCompartmentInitVolumesPtr)
    {
        result = getValues(getCompartmentInitVolumesPtr, len, indx, values);
    }
    return result;
}

int LLVMExecutableModel::setGlobalParameterInitValues(int len, const int* indx,
        const double* values)
{
    int result = -1;
    if (setGlobalParameterInitValuePtr)
    {
        result = setValues(setGlobalParameterInitValuePtr,
                &LLVMExecutableModel::getGlobalParameterId, len, indx, values);
    }
    return result;
}

int LLVMExecutableModel::getGlobalParameterInitValues(int len, const int *indx,
                double *values)
{
    int result = -1;
    if (getGlobalParameterInitValuePtr)
    {
        result = getValues(getGlobalParameterInitValuePtr, len, indx, values);
    }
    return result;
}

/******************************* End Initial Conditions Section ***************/
#endif /***********************************************************************/
/******************************************************************************/



/******************************* Random Section *******************************/
#if (1) /**********************************************************************/
/******************************************************************************/

/**
 * set the seed used by the random number generator. This will by definition
 * reset the RNG.
 */
void LLVMExecutableModel::setRandomSeed(int64_t seed)
{
    // if this does not exist, can create it, this will be freed
    // by LLVMModelData_free
    if (modelData->random == 0)
    {
        modelData->random = new Random();
    }
    modelData->random->setRandomSeed(seed);

}

/**
 * get the seed used by the RNG.
 */
int64_t LLVMExecutableModel::getRandomSeed()
{
    // if this does not exist, can create it, this will be freed
    // by LLVMModelData_free
    if (modelData->random == 0)
    {
        modelData->random = new Random();
    }
    return modelData->random->getRandomSeed();
}

/**
 * Get a uniform random number between 0 and 1 created by the RNG.
 *
 * The type of RNG can be specified in the config file with the
 * RANDOM_GENERATOR_TYPE key.
 */
double LLVMExecutableModel::getRandom()
{
    // if this does not exist, can create it, this will be freed
    // by LLVMModelData_free
    if (modelData->random == 0)
    {
        modelData->random = new Random();
    }
    return (*modelData->random)();
}

void LLVMExecutableModel::saveState(std::ostream& out)
{
	LLVMModelData_save(modelData, out);
	resources->saveState(out);
	
	pendingEvents.saveState(out);
	rr::saveBinary(out, eventAssignTimes);
	rr::saveBinary(out, tieBreakMap);
}


/******************************* End Random Section ***************************/
#endif  /**********************************************************************/
/******************************************************************************/


} /* namespace rr */
