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
#include <algorithm>
#include <map>
#include "sbml/SBMLTypeCodes.h"


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
        os << std::setiosflags(std::ios::floatfield) << std::setprecision(8);
        os << '[';
        for (int i = 0; i < n; ++i)
        {
            os << std::fixed << p[i];
            if (i < n - 1)
            {
                os << ", ";
            }
        }
        os << ']' << std::endl;
    }
    else
    {
        os << "NULL" << std::endl;
    }
}

typedef std::string (rr::ExecutableModel::*getNamePtr)(size_t);
typedef int (rr::ExecutableModel::*getNumPtr)();

/**
 * checks if the bitfield value has all the required flags
 * from type
 */
inline bool checkExact(uint32_t type, uint32_t value) {
    return (value & type) == type;
}

// make this static here, hide our implementation...
static void addIds(rr::ExecutableModel *model, size_t start, size_t end,
        getNamePtr nameFunc, std::list<std::string>& ids)
{
    for(size_t i = start; i < end; i++)
    {
        const std::string& name  = (model->*nameFunc)(i);
        ids.push_back(name);
    }
}

// make this static here, hide our implementation...
static void addConcIds(rr::ExecutableModel *model, size_t start, size_t end,
        getNamePtr nameFunc, std::list<std::string>& ids)
{
    for(size_t i = start; i < end; i++)
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

int LLVMExecutableModel::getValues(double (*funcPtr)(LLVMModelData*, size_t),
        size_t len, const int *indx, double *values)
{
    double value;
    for (size_t i = 0; i < len; ++i)
    {
        size_t j = indx ? indx[i] : i;
        value = funcPtr(modelData, j);

        if (isnan(value))
        {
            rrLog(Logger::LOG_WARNING) << "NaN value for index " << j
                                   << ".  Could be out of range or illegal computation.";
        }

        values[i] = value;
    }
    return static_cast<int>(len);
}

int LLVMExecutableModel::setValues(bool (*funcPtr)(LLVMModelData*, int, double),
    GetNameFuncPtr getNameFuncPtr, size_t len, const int* indx, const double* values, bool strict)
{
    for (size_t i = 0; i < len; ++i)
    {
        size_t j = indx ? indx[i] : i;
        bool result =  funcPtr(modelData, static_cast<int>(j), values[i]);

        if (!result && strict)
        {
            std::stringstream s;
            std::string id = (this->*getNameFuncPtr)(j);
            s << "Could not set value for " << id;

            if (symbols->hasAssignmentRule(id))
            {
                s << ", as it is defined by an assignment rule, and can not be set independently.";
            }

            throw_llvm_exception(s.str());
        }
    }
    return static_cast<int>(len);
}

LLVMExecutableModel::LLVMExecutableModel() :
    ExecutableModel(),
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
    getPiecewiseTriggerPtr(0),
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
    getBoundarySpeciesInitConcentrationsPtr(0),
    setBoundarySpeciesInitConcentrationsPtr(0),
    getBoundarySpeciesInitAmountsPtr(0),
    setBoundarySpeciesInitAmountsPtr(0),
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
    const std::shared_ptr<ModelResources>& modelResources, LLVMModelData* modelData) :
    ExecutableModel(),
    resources(modelResources),
    symbols(modelResources->symbols),
    modelData(modelData),
    conversionFactor(1.0),
    evalInitialConditionsPtr(modelResources->evalInitialConditionsPtr),
    evalReactionRatesPtr(modelResources->evalReactionRatesPtr),
    getBoundarySpeciesAmountPtr(modelResources->getBoundarySpeciesAmountPtr),
    getFloatingSpeciesAmountPtr(modelResources->getFloatingSpeciesAmountPtr),
    getBoundarySpeciesConcentrationPtr(modelResources->getBoundarySpeciesConcentrationPtr),
    getFloatingSpeciesConcentrationPtr(modelResources->getFloatingSpeciesConcentrationPtr),
    getCompartmentVolumePtr(modelResources->getCompartmentVolumePtr),
    getGlobalParameterPtr(modelResources->getGlobalParameterPtr),
    evalRateRuleRatesPtr(modelResources->evalRateRuleRatesPtr),
    getEventTriggerPtr(modelResources->getEventTriggerPtr),
    getEventPriorityPtr(modelResources->getEventPriorityPtr),
    getEventDelayPtr(modelResources->getEventDelayPtr),
    eventTriggerPtr(modelResources->eventTriggerPtr),
    eventAssignPtr(modelResources->eventAssignPtr),
    getPiecewiseTriggerPtr(modelResources->getPiecewiseTriggerPtr),
    evalVolatileStoichPtr(modelResources->evalVolatileStoichPtr),
    evalConversionFactorPtr(modelResources->evalConversionFactorPtr),
    setBoundarySpeciesAmountPtr(modelResources->setBoundarySpeciesAmountPtr),
    setFloatingSpeciesAmountPtr(modelResources->setFloatingSpeciesAmountPtr),
    setBoundarySpeciesConcentrationPtr(modelResources->setBoundarySpeciesConcentrationPtr),
    setFloatingSpeciesConcentrationPtr(modelResources->setFloatingSpeciesConcentrationPtr),
    setCompartmentVolumePtr(modelResources->setCompartmentVolumePtr),
    setGlobalParameterPtr(modelResources->setGlobalParameterPtr),
    getFloatingSpeciesInitConcentrationsPtr(modelResources->getFloatingSpeciesInitConcentrationsPtr),
    setFloatingSpeciesInitConcentrationsPtr(modelResources->setFloatingSpeciesInitConcentrationsPtr),
    getFloatingSpeciesInitAmountsPtr(modelResources->getFloatingSpeciesInitAmountsPtr),
    setFloatingSpeciesInitAmountsPtr(modelResources->setFloatingSpeciesInitAmountsPtr),
    getBoundarySpeciesInitConcentrationsPtr(modelResources->getBoundarySpeciesInitConcentrationsPtr),
    setBoundarySpeciesInitConcentrationsPtr(modelResources->setBoundarySpeciesInitConcentrationsPtr),
    getBoundarySpeciesInitAmountsPtr(modelResources->getBoundarySpeciesInitAmountsPtr),
    setBoundarySpeciesInitAmountsPtr(modelResources->setBoundarySpeciesInitAmountsPtr),
    getCompartmentInitVolumesPtr(modelResources->getCompartmentInitVolumesPtr),
    setCompartmentInitVolumesPtr(modelResources->setCompartmentInitVolumesPtr),
    getGlobalParameterInitValuePtr(modelResources->getGlobalParameterInitValuePtr),
    setGlobalParameterInitValuePtr(modelResources->setGlobalParameterInitValuePtr),
    eventListeners(modelData->numEvents, EventListenerPtr()), // init eventHandlers std::vector
    dirty(0),
    flags(defaultFlags())
{

    //The key here is that 'time' has to be less than 'mIntegrationStartTime' for events to work.  Not sure whether it actually has to be -inf, but it can't hurt?  See T0EventFiring tests, particularly 1120 in the SBML test suite (test_semantic_STS).  The actual integration start time is set in 'simulate'. --LS
    modelData->time = -std::numeric_limits<double>::infinity(); // time is initially before simulation starts
    // mIntegrationStartTime defaults to zero in constructor.

    std::srand((unsigned)std::time(0));

    eventAssignTimes.resize(modelData->numEvents);

    reset(int(SelectionRecord::ALL));
}

LLVMExecutableModel::LLVMExecutableModel(std::istream& in, uint modelGeneratorOpt) :
    ExecutableModel(),
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
    getPiecewiseTriggerPtr = resources->getPiecewiseTriggerPtr;
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
    getBoundarySpeciesInitConcentrationsPtr = resources->getBoundarySpeciesInitConcentrationsPtr;
    setBoundarySpeciesInitConcentrationsPtr = resources->setBoundarySpeciesInitConcentrationsPtr;
    getBoundarySpeciesInitAmountsPtr = resources->getBoundarySpeciesInitAmountsPtr;
    setBoundarySpeciesInitAmountsPtr = resources->setBoundarySpeciesInitAmountsPtr;
    getCompartmentInitVolumesPtr = resources->getCompartmentInitVolumesPtr;
    setCompartmentInitVolumesPtr = resources->setCompartmentInitVolumesPtr;
    getGlobalParameterInitValuePtr = resources->getGlobalParameterInitValuePtr;
    setGlobalParameterInitValuePtr = resources->setGlobalParameterInitValuePtr;

    pendingEvents.loadState(in, *this);
    rr::loadBinary(in, eventAssignTimes);
    rr::loadBinary(in, tieBreakMap);
    rr::loadBinary(in, mIntegrationStartTime);

}

LLVMExecutableModel::~LLVMExecutableModel()
{
    // smart ptr takes care of freeing resources

    LLVMModelData_free(modelData);

    rrLog(Logger::LOG_DEBUG) << __FUNC__;
}

std::string LLVMExecutableModel::getModelName()
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
    return static_cast<int>(modelData->numIndFloatingSpecies);
}

int LLVMExecutableModel::getNumDepFloatingSpecies()
{
    return static_cast<int>(symbols->getFloatingSpeciesSize() - modelData->numIndFloatingSpecies);
}

int LLVMExecutableModel::getNumFloatingSpecies()
{
    return static_cast<int>(symbols->getFloatingSpeciesSize());
}

int LLVMExecutableModel::getNumBoundarySpecies()
{
    return static_cast<int>(symbols->getBoundarySpeciesSize());
}

int LLVMExecutableModel::getNumGlobalParameters()
{
    return static_cast<int>(symbols->getGlobalParametersSize());
}

int LLVMExecutableModel::getNumCompartments()
{
    return static_cast<int>(symbols->getCompartmentsSize());
}

int LLVMExecutableModel::getCompartmentIndexForFloatingSpecies(size_t index) 
{
	return symbols->getCompartmentIndexForFloatingSpecies(index);
}

int LLVMExecutableModel::getCompartmentIndexForBoundarySpecies(size_t index)
{
    return symbols->getCompartmentIndexForBoundarySpecies(index);
}

int LLVMExecutableModel::getNumReactions()
{
    return static_cast<int>(modelData->numReactions);
}

int LLVMExecutableModel::getFloatingSpeciesConcentrations(size_t len, int const *indx,
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
        // save and assign state std::vector
        double *savedRateRules = modelData->rateRuleValuesAlias;
        double *savedFloatingSpeciesAmounts = modelData->floatingSpeciesAmountsAlias;

        modelData->rateRuleValuesAlias = const_cast<double*>(y);
        modelData->floatingSpeciesAmountsAlias = const_cast<double*>(y + modelData->numRateRules);
        evalVolatileStoichPtr(modelData);

        // not setting state std::vector, react rates get dirty
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

        // restore original pointers for state std::vector
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
		log.stream() << __FUNC__ << std::endl;
		log.stream() << "y: ";
		if (y) {
			dump_array(log.stream(), modelData->numRateRules + modelData->numFloatingSpecies, y);
		}
		else {
			log.stream() << "null";
		}
		log.stream() << std::endl << "dydt: ";
		if (dydt) {
			dump_array(log.stream(), modelData->numRateRules + modelData->numFloatingSpecies, dydt);
		}
		else {
			log.stream() << "null";
		}
		log.stream() << std::endl << "Model: " << std::endl << this;
	}
	*/
}

double LLVMExecutableModel::getFloatingSpeciesAmountRate(size_t index,
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

std::string LLVMExecutableModel::getInfo()
{
    std::stringstream stream;

    double *tmp;

    int nFloat = getNumFloatingSpecies();
    int nBound = getNumBoundarySpecies();
    int nComp = getNumCompartments();
    int nGlobalParam = getNumGlobalParameters();
    int nEvents = getNumEvents();
    int nReactions = getNumReactions();

    stream << "* Calculated Values *" << std::endl;

    tmp = new double[nFloat];
    getFloatingSpeciesAmounts(nFloat, 0, tmp);
    stream << "FloatingSpeciesAmounts:" << std::endl;
    dump_array(stream, nFloat, tmp);

    getFloatingSpeciesConcentrations(nFloat, 0, tmp);
    stream << "FloatingSpeciesConcentrations:" << std::endl;
    dump_array(stream, nFloat, tmp);

    this->getFloatingSpeciesInitConcentrations(nFloat, 0, tmp);
    stream << "FloatingSpeciesInitConcentrations:" << std::endl;
    dump_array(stream, nFloat, tmp);

    this->getFloatingSpeciesInitAmounts(nFloat, 0, tmp);
    stream << "FloatingSpeciesInitAmounts:" << std::endl;
    dump_array(stream, nFloat, tmp);
    delete[] tmp;

    tmp = new double[nReactions];
    getReactionRates(nReactions, 0, tmp);
    stream << "Reaction Rates:" << std::endl;
    dump_array(stream, nReactions, tmp);
    delete[] tmp;

    tmp = new double[nBound];
    getBoundarySpeciesAmounts(nBound, 0, tmp);
    stream << "BoundarySpeciesAmounts:" << std::endl;
    dump_array(stream, nBound, tmp);

    getBoundarySpeciesConcentrations(nBound, 0, tmp);
    stream << "BoundarySpeciesConcentrations:" << std::endl;
    dump_array(stream, nBound, tmp);

    this->getBoundarySpeciesInitAmounts(nBound, 0, tmp);
    stream << "BoundarySpeciesInitAmounts:" << std::endl;
    dump_array(stream, nBound, tmp);

    this->getBoundarySpeciesInitConcentrations(nBound, 0, tmp);
    stream << "BoundarySpeciesInitConcentrations:" << std::endl;
    dump_array(stream, nBound, tmp);
    delete[] tmp;

    tmp = new double[nComp];
    getCompartmentVolumes(nComp, 0, tmp);
    stream << "CompartmentVolumes:" << std::endl;
    dump_array(stream, nComp, tmp);

    this->getCompartmentInitVolumes(nComp, 0, tmp);
    stream << "CompartmentInitVolumes:" << std::endl;
    dump_array(stream, nComp, tmp);
    delete[] tmp;

    tmp = new double[nGlobalParam];
    getGlobalParameterValues(nGlobalParam, 0, tmp);
    stream << "GlobalParameters:" << std::endl;
    dump_array(stream, nGlobalParam, tmp);
    delete[] tmp;

    tmp = new double[nGlobalParam];
    getGlobalParameterInitValues(nGlobalParam, 0, tmp);
    stream << "Init GlobalParameters:" << std::endl;
    dump_array(stream, nGlobalParam, tmp);
    delete[] tmp;

    unsigned char *tmpEvents = new unsigned char[nEvents];
    getEventTriggers(nEvents, 0, tmpEvents);
    stream << "Events Trigger Status:" << std::endl;
    dump_array(stream, nEvents, (bool*)tmpEvents);
    delete[] tmpEvents;

    stream << std::endl;
    stream << *modelData;

    return stream.str();
}

int LLVMExecutableModel::getFloatingSpeciesIndex(const std::string& id)
{
    return symbols->getFloatingSpeciesIndex(id, false);
}

std::string LLVMExecutableModel::getFloatingSpeciesId(size_t index)
{
    return symbols->getFloatingSpeciesId(index);
}

int LLVMExecutableModel::getBoundarySpeciesIndex(const std::string& id)
{
    return symbols->getBoundarySpeciesIndex(id);
}

std::string LLVMExecutableModel::getBoundarySpeciesId(size_t index)
{
    std::vector<std::string> ids = symbols->getBoundarySpeciesIds();
    if (index < ids.size())
    {
        return ids[index];
    }
    else
    {
        throw_llvm_exception("index out of range");
        return "";
    }
}

int LLVMExecutableModel::getGlobalParameterIndex(const std::string& id)
{
    return symbols->getGlobalParameterIndex(id);
}

std::string LLVMExecutableModel::getGlobalParameterId(size_t id)
{
    return symbols->getGlobalParameterId(id);
}

int LLVMExecutableModel::getCompartmentIndex(const std::string& id)
{
    return symbols->getCompartmentIndex(id);
}

std::string LLVMExecutableModel::getCompartmentId(size_t id)
{
    std::vector<std::string> ids = symbols->getCompartmentIds();
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

int LLVMExecutableModel::getReactionIndex(const std::string& id)
{
    return symbols->getReactionIndex(id);
}

std::string LLVMExecutableModel::getReactionId(size_t id)
{
    std::vector<std::string> ids = symbols->getReactionIds();
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

int LLVMExecutableModel::getStoichiometryIndex(const std::string& id)
{
    return symbols->getStoichiometryIndex(id);
}

int LLVMExecutableModel::getStoichiometryIndex(const std::string& speciesId, const std::string& reactionId)
{
    return symbols->getStoichiometryIndex(speciesId, reactionId);
}

std::string LLVMExecutableModel::getStoichiometryId(size_t id)
{
    std::vector<std::string> ids = symbols->getStoichiometryIds();
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

//Resets a single type of model element:  compartment, floating, boundary, or global parameter
void LLVMExecutableModel::resetOneType(
        int& opt, int thistype, int independents, int total,
        int (LLVMExecutableModel::*getInit)(size_t, const int*, double*),
        int (LLVMExecutableModel::*setCurrent)(size_t, const int*, const double*),
        std::string (LLVMModelDataSymbols::*getTypeId)(size_t) const,
        double* buffer, std::map<std::string, int>& inits,
        std::map<std::string, double>& initvals)
{
    int s = independents;
    //LLVMModelDataSymbols* varsymbols = const_cast<LLVMModelDataSymbols*>(symbols);

    if (opt & thistype)
    {
        rrLog(Logger::LOG_INFORMATION) << "resetting model type " << thistype;
        (this->*getInit)(independents, 0, buffer);
        (this->*setCurrent)(independents, 0, buffer);
        //Need to save any values we just changed in initial assignment list.
        s = 0;
    }
    if (opt & thistype || opt & SelectionRecord::RATE)
    {
        for (; s < total; s++)
        {
            string sid = (symbols->*getTypeId)(s);
            if (s < independents)
            {
                if (symbols->hasInitialAssignmentRule(sid))
                {
                    inits[sid] = thistype;
                    initvals[sid] = buffer[s];
                }
            }
            else
            {
                if (!symbols->hasAssignmentRule(sid))
                {
                    if (!(opt & thistype) || symbols->hasRateRule(sid))
                    {
                        (this->*getInit)(1, &s, buffer);
                        (this->*setCurrent)(1, &s, buffer);
                        if (symbols->hasInitialAssignmentRule(sid))
                        {
                            inits[sid] = thistype;
                            initvals[sid] = buffer[0];
                        }
                    }
                }
            }
        }
    }
}

void LLVMExecutableModel::reset()
{
    uint opt = rr::Config::getInt(rr::Config::MODEL_RESET);
    rrLog(Logger::LOG_DEBUG) << "calling reset with default values: " << opt;
    reset(opt);
}

void LLVMExecutableModel::reset(int opt)
{
    using std::max; // weird linux vs windows thing

    // initializes the model to the init values specified in the sbml, and
    // copies these to the initial initial conditions (not a typo),
    // sets the 'init(...)' values to the sbml specified init values.
    if (opt & SelectionRecord::SBML_INITIALIZE)
    {
        rrLog(Logger::LOG_INFORMATION) << "resetting init conditions";
        evalInitialConditions();
    }

    // eval the initial conditions and rates
    if (opt & SelectionRecord::TIME)
    {
        rrLog(Logger::LOG_INFORMATION) << "resetting time";
        setTime(0.0);
    }

    if (getCompartmentInitVolumesPtr && getGlobalParameterInitValuePtr
        && getFloatingSpeciesInitAmountsPtr && getBoundarySpeciesInitAmountsPtr
        && getFloatingSpeciesInitConcentrationsPtr && getBoundarySpeciesInitConcentrationsPtr)
    {
        //Save a list of all values that have initial assignments, since we may need to re-do them.
        std::map<std::string, int> inits;
        std::map<std::string, double> initvals;
        // need at least 1 for global params
        unsigned size = max(1u, modelData->numIndCompartments);
        size = max(size, modelData->numIndFloatingSpecies);
        size = max(size, modelData->numIndBoundarySpecies);
        size = max(size, modelData->numIndGlobalParameters);

        double* buffer = new double[size];

        //std::cout << this;
        // Reset compartments first, as these are used to
        // convert between concentrations and amounts.
        resetOneType(opt, SelectionRecord::_COMPARTMENT, modelData->numIndCompartments, getNumCompartments(), &LLVMExecutableModel::getCompartmentInitVolumes, &LLVMExecutableModel::setCompartmentVolumes, &LLVMModelDataSymbols::getCompartmentId, buffer, inits, initvals);
        //std::cout << this;

        // Reset floating species
        resetOneType(opt, SelectionRecord::FLOATING, modelData->numIndFloatingSpecies, getNumFloatingSpecies(), &LLVMExecutableModel::getFloatingSpeciesInitAmounts, &LLVMExecutableModel::setFloatingSpeciesAmounts, &LLVMModelDataSymbols::getFloatingSpeciesId, buffer, inits, initvals);
        //std::cout << this;

        // Reset boundary species
        resetOneType(opt, SelectionRecord::BOUNDARY, modelData->numIndBoundarySpecies, getNumBoundarySpecies(), &LLVMExecutableModel::getBoundarySpeciesInitAmounts, &LLVMExecutableModel::setBoundarySpeciesAmounts, &LLVMModelDataSymbols::getBoundarySpeciesId, buffer, inits, initvals);
        //std::cout << this;

        // Save whether someone changed a cm.  Note: setGlobalParameterValues sets this bit,
        // so we have to save it before changing any GPs in reset.
        const bool dirty_cm = dirty & DIRTY_CONSERVED_MOIETIES;
        const bool dirty_init = dirty & DIRTY_INIT_SPECIES;

        // Reset parameters
        resetOneType(opt, SelectionRecord::_GLOBAL_PARAMETER, modelData->numIndGlobalParameters, getNumGlobalParameters(), &LLVMExecutableModel::getGlobalParameterInitValues, &LLVMExecutableModel::setGlobalParameterValues, &LLVMModelDataSymbols::getGlobalParameterId, buffer, inits, initvals);
        //std::cout << this;

        // Whether were we forced to reset cms:
        bool reset_cm = false;

        if (!(opt & SelectionRecord::_GLOBAL_PARAMETER))
        {
            //We didn't reset the global parameters in general, but we
            // might need to reset the conserved moieties, which depend
            // on the floating species initial values.

            for (int gid = 0; gid < modelData->numIndGlobalParameters; ++gid)
            {
                if (symbols->isConservedMoietyParameter(gid))
                {
                    // reset if asked to reset conserved moieties specifically, or if asked to reset floating species, or if floating species initial values were set, even if not reset explicitly.  (The last is because conserved moieties depend on the initial species values.)
                    if (opt & SelectionRecord::CONSERVED_MOIETY || 
                        opt & SelectionRecord::FLOATING  ||
                        dirty_init)
                    {
                        rrLog(Logger::LOG_DEBUG) << "Resetting conserved moiety "
                            << gid << ".";
                        reset_cm = true;
                        getGlobalParameterInitValues(1, &gid, buffer);
                        setGlobalParameterValues(1, &gid, buffer);
                    }
                }
            }

            if (reset_cm)
            {
                // warn if we were forced to reset CMs
                if (dirty_cm && dirty_init)
                {
                    rrLog(Logger::LOG_WARNING) << "Both initial conditions and "
                        "conserved moieties were user modified. As conserved moieties "
                        "are defined in terms of initial conditions, the conserved "
                        "moiety values were forcibly reset in terms of the species "
                        "initial conditions.";
                }

                // we've reset CMs, so clear the dirty bit.
                dirty &= ~DIRTY_CONSERVED_MOIETIES;

                // the DIRTY_INIT_SPECIES bit is always cleared at the
                // end of this func.
            }
        }

        //We now need to loop through and reset initial values if they were set by initial assignments, since we might not have reset the values those assignments depend on in time.
        bool changed = false;
        int loops = 0; //So we don't loop indefinitely.
        //std::cout << this;
        do
        {
            loops++;
            changed = false;
            for (std::map<std::string, int>::iterator init = inits.begin(); init != inits.end(); init++)
            {
                string id = init->first;
                double val = initvals[id];
                int index = 0;
                switch (init->second)
                {
                case SelectionRecord::_COMPARTMENT:
                    index = symbols->getCompartmentIndex(id);
                    getCompartmentInitVolumes(1, &index, buffer);
                    if (buffer[0] != val) {
                        changed = true;
                        setCompartmentVolumes(1, &index, buffer);
                        initvals[id] = buffer[0];
                    }
                    break;
                case SelectionRecord::FLOATING:
                    index = symbols->getFloatingSpeciesIndex(id, false);
                    getFloatingSpeciesInitAmounts(1, &index, buffer);
                    if (buffer[0] != val) {
                        changed = true;
                        setFloatingSpeciesAmounts(1, &index, buffer);
                        initvals[id] = buffer[0];
                    }
                    break;
                case SelectionRecord::BOUNDARY:
                    index = symbols->getBoundarySpeciesIndex(id);
                    getBoundarySpeciesInitAmounts(1, &index, buffer);
                    if (buffer[0] != val) {
                        changed = true;
                        setBoundarySpeciesAmounts(1, &index, buffer);
                        initvals[id] = buffer[0];
                    }
                    break;
                case SelectionRecord::_GLOBAL_PARAMETER:
                    index = symbols->getGlobalParameterIndex(id);
                    getGlobalParameterInitValues(1, &index, buffer);
                    if (buffer[0] != val) {
                        changed = true;
                        setGlobalParameterValues(1, &index, buffer);
                        initvals[id] = buffer[0];
                    }
                    break;
                default:
                    assert(false);
                    break;
                }
            }
        } while (changed == true && loops < 15);
        //std::cout << this;

        delete[] buffer;
    }


    // this sets up the event system to pull the initial value
    // before the simulation starts.
    setTime(-std::numeric_limits<double>::infinity());

    // we've reset the species to their init values.
    dirty &= ~DIRTY_INIT_SPECIES;

    rrLog(Logger::LOG_DEBUG) << __FUNC__ << *modelData;
}

int LLVMExecutableModel::getStateVector(double* stateVector)
{
    if (stateVector == 0)
    {
        rrLog(Logger::LOG_TRACE) << __FUNC__ << ", stateVector: null, returning "
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

        log.stream() << std::endl << __FUNC__ <<  ", Model: " << std::endl << this;
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
    stream << "LLVMExecutableModel" << std::endl;
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

    if (checkExact(SelectionRecord::REACTION_RATE, types)) {
        addIds(this, 0, symbols->getReactionSize(),
                &rr::ExecutableModel::getReactionId, ids);
    }

    if (checkExact(SelectionRecord::INITIAL | SelectionRecord::FLOATING |
            SelectionRecord::CONCENTRATION, types)) {
        for (size_t i = 0; i < symbols->getInitFloatingSpeciesSize(); ++i) {
            ids.push_back("init([" + this->getFloatingSpeciesId(i) + "])");
        }
    }

    if (checkExact(SelectionRecord::INITIAL | SelectionRecord::FLOATING |
            SelectionRecord::CONCENTRATION, types)) {
        for (size_t i = symbols->getInitFloatingSpeciesSize();
                i < symbols->getFloatingSpeciesSize(); ++i) {
            ids.push_back("init([" + this->getFloatingSpeciesId(i) + "])");
        }
    }

    if (checkExact(SelectionRecord::INITIAL | SelectionRecord::FLOATING |
            SelectionRecord::AMOUNT, types)) {
        for (size_t i = 0; i < symbols->getInitFloatingSpeciesSize(); ++i) {
            ids.push_back("init(" + this->getFloatingSpeciesId(i) + ")");
        }
    }

    if (checkExact(SelectionRecord::INITIAL | SelectionRecord::FLOATING |
            SelectionRecord::AMOUNT, types)) {
        for (size_t i = symbols->getInitFloatingSpeciesSize();
                i < symbols->getFloatingSpeciesSize(); ++i) {
            ids.push_back("init(" + this->getFloatingSpeciesId(i) + ")");
        }
    }

    if (checkExact(SelectionRecord::INITIAL | SelectionRecord::BOUNDARY |
        SelectionRecord::CONCENTRATION, types)) {
        for (size_t i = 0; i < symbols->getInitBoundarySpeciesSize(); ++i) {
            ids.push_back("init([" + this->getBoundarySpeciesId(i) + "])");
        }
    }

    if (checkExact(SelectionRecord::INITIAL | SelectionRecord::BOUNDARY |
        SelectionRecord::CONCENTRATION, types)) {
        for (size_t i = symbols->getInitBoundarySpeciesSize();
            i < symbols->getBoundarySpeciesSize(); ++i) {
            ids.push_back("init([" + this->getBoundarySpeciesId(i) + "])");
        }
    }

    if (checkExact(SelectionRecord::INITIAL | SelectionRecord::BOUNDARY |
        SelectionRecord::AMOUNT, types)) {
        for (size_t i = 0; i < symbols->getInitBoundarySpeciesSize(); ++i) {
            ids.push_back("init(" + this->getBoundarySpeciesId(i) + ")");
        }
    }

    if (checkExact(SelectionRecord::INITIAL | SelectionRecord::BOUNDARY |
        SelectionRecord::AMOUNT, types)) {
        for (size_t i = symbols->getInitBoundarySpeciesSize();
            i < symbols->getBoundarySpeciesSize(); ++i) {
            ids.push_back("init(" + this->getBoundarySpeciesId(i) + ")");
        }
    }

    if (checkExact(SelectionRecord::INITIAL | SelectionRecord::_COMPARTMENT, types)) {
        for (size_t i = 0; i < symbols->getCompartmentsSize(); ++i) {
            ids.push_back("init(" + this->getCompartmentId(i) + ")");
        }
    }

    if (checkExact(SelectionRecord::INITIAL | SelectionRecord::GLOBAL_PARAMETER, types)) {
        for (size_t i = 0; i < symbols->getGlobalParametersSize(); ++i) {
            ids.push_back("init(" + this->getGlobalParameterId(i) + ")");
        }
    }

    //Do the rate rules first because that's the way they're ordered in the 
    // underlying data model.
    if (checkExact(SelectionRecord::RATE, types)) {

        for (size_t i = 0; i < symbols->getRateRuleSize(); ++i) {
            string rid = symbols->getRateRuleId(i);
            if (symbols->getBoundarySpeciesIndex(rid) >= 0 &&
                (checkExact(SelectionRecord::BOUNDARY_AMOUNT_RATE, types) ||
                    SelectionRecord::RATE == types))
            {
                ids.push_back(rid + "'");
            }
            if (symbols->getFloatingSpeciesIndex(rid, false) >= 0 &&
                (checkExact(SelectionRecord::FLOATING_AMOUNT_RATE, types) ||
                    SelectionRecord::RATE == types))
            {
                ids.push_back(rid + "'");
            }
            if (symbols->getCompartmentIndex(rid) >= 0 &&
                (checkExact(SelectionRecord::COMPARTMENT_RATE, types) ||
                    SelectionRecord::RATE == types))
            {
                ids.push_back(rid + "'");
            }
            if (symbols->getGlobalParameterIndex(rid) >= 0 &&
                (checkExact(SelectionRecord::GLOBAL_PARAMETER_RATE, types) ||
                    SelectionRecord::RATE == types))
            {
                ids.push_back(rid + "'");
            }
        }
    }

    if (checkExact(SelectionRecord::FLOATING_AMOUNT_RATE, types)) {
        for (size_t i = 0; i < getNumFloatingSpecies(); ++i) {
            string sid = this->getFloatingSpeciesId(i);
            if (!symbols->hasAssignmentRule(sid) &&
                !symbols->hasRateRule(sid))
            {
                ids.push_back(sid + "'");
            }
        }
    }

    if (checkExact(SelectionRecord::FLOATING_CONCENTRATION_RATE, types)) {
        for (size_t i = 0; i < getNumFloatingSpecies(); ++i) {
            string sid = this->getFloatingSpeciesId(i);
            if (!symbols->hasAssignmentRule(sid))
            {
                ids.push_back("[" + sid + "]'");
            }
        }
    }

    if (checkExact(SelectionRecord::BOUNDARY_CONCENTRATION_RATE, types)) {
        for (size_t i = 0; i < getNumBoundarySpecies(); ++i) {
            string sid = this->getBoundarySpeciesId(i);
            if (!symbols->hasAssignmentRule(sid))
            {
                ids.push_back("[" + sid + "]'");
            }
        }
    }

    if (checkExact(SelectionRecord::STOICHIOMETRY, types)) {
        for (size_t s = 0; s < getNumFloatingSpecies(); ++s) {
            string sid = getFloatingSpeciesId(s);
            for (size_t r = 0; r < getNumReactions(); ++r) {
                if (getStoichiometry(s, r) != 0)
                {
                    string rid = getReactionId(r);
                    ids.push_back("stoich(" + sid + ", " + rid + ")");
                }
            }
        }
    }

    if (checkExact(SelectionRecord::UNSCALED_ELASTICITY, types)) {
        for (size_t r = 0; r < getNumReactions(); ++r) {
            string rid = getReactionId(r);
            for (size_t fs = 0; fs < getNumFloatingSpecies(); ++fs) {
                string fsid = getFloatingSpeciesId(fs);
                ids.push_back("uec(" + rid + ", " + fsid + ")");
            }
            for (size_t bs = 0; bs < getNumBoundarySpecies(); bs++) {
                string bsid = getBoundarySpeciesId(bs);
                //Remove when https://github.com/sys-bio/roadrunner/issues/914 addressed.
                if (!symbols->hasAssignmentRule(bsid))
                {
                    ids.push_back("uec(" + rid + ", " + bsid + ")");
                }
            }
            for (size_t g = 0; g < getNumGlobalParameters(); g++) {
                string gid = getGlobalParameterId(g);
                //Remove when https://github.com/sys-bio/roadrunner/issues/914 addressed.
                if (!symbols->hasAssignmentRule(gid) && symbols->getConservedMoietyIndex(gid) == -1)
                {
                    ids.push_back("uec(" + rid + ", " + gid + ")");
                }
            }
        }
    }

    if (checkExact(SelectionRecord::ELASTICITY, types)) {
        for (size_t r = 0; r < getNumReactions(); ++r) {
            string rid = getReactionId(r);
            for (size_t fs = 0; fs < getNumFloatingSpecies(); ++fs) {
                string fsid = getFloatingSpeciesId(fs);
                ids.push_back("ec(" + rid + ", " + fsid + ")");
            }
            for (size_t bs = 0; bs < getNumBoundarySpecies(); bs++) {
                string bsid = getBoundarySpeciesId(bs);
                //Remove when https://github.com/sys-bio/roadrunner/issues/914 addressed.
                if (!symbols->hasAssignmentRule(bsid))
                {
                    ids.push_back("ec(" + rid + ", " + bsid + ")");
                }
            }
            for (size_t g = 0; g < getNumGlobalParameters(); g++) {
                string gid = getGlobalParameterId(g);
                //Remove when https://github.com/sys-bio/roadrunner/issues/914 addressed.
                if (!symbols->hasAssignmentRule(gid) && symbols->getConservedMoietyIndex(gid) == -1)
                {
                    ids.push_back("ec(" + rid + ", " + gid + ")");
                }
            }
        }
    }

    if (SelectionRecord::STATE_VECTOR == types) {
        int stateVectorSize = getStateVector(0);
        for (int i = 0; i < stateVectorSize; ++i) {
            ids.push_back(getStateVectorId(i));
        }
    }

    // These are also displayed with global parameters, so
    // only add them if explicity asked for.
    if (SelectionRecord::CONSERVED_MOIETY == types)
    {
        for(size_t i = 0; i < symbols->getConservedMoietySize(); ++i)
        {
            ids.push_back(symbols->getConservedMoietyId(i));
        }
    }

    if (checkExact(SelectionRecord::UNSCALED_CONTROL, types)) {
        for (size_t r = 0; r < getNumReactions(); ++r) {
            string rid = getReactionId(r);
            for (size_t bs = 0; bs < getNumBoundarySpecies(); bs++) {
                string bsid = getBoundarySpeciesId(bs);
                //Remove when https://github.com/sys-bio/roadrunner/issues/914 addressed.
                if (!symbols->hasAssignmentRule(bsid))
                {
                    ids.push_back("ucc(" + rid + ", " + bsid + ")");
                }
            }
            for (size_t g = 0; g < getNumGlobalParameters(); g++) {
                string gid = getGlobalParameterId(g);
                //Remove when https://github.com/sys-bio/roadrunner/issues/914 addressed.
                if (!symbols->hasAssignmentRule(gid))
                {
                    ids.push_back("ucc(" + rid + ", " + gid + ")");
                }
            }
        }
        for (size_t fs = 0; fs < getNumFloatingSpecies(); ++fs) {
            string fsid = getFloatingSpeciesId(fs);
            for (size_t bs = 0; bs < getNumBoundarySpecies(); bs++) {
                string bsid = getBoundarySpeciesId(bs);
                //Remove when https://github.com/sys-bio/roadrunner/issues/914 addressed.
                if (!symbols->hasAssignmentRule(bsid))
                {
                    ids.push_back("ucc(" + fsid + ", " + bsid + ")");
                }
            }
            for (size_t g = 0; g < getNumGlobalParameters(); g++) {
                string gid = getGlobalParameterId(g);
                //Remove when https://github.com/sys-bio/roadrunner/issues/914 addressed.
                if (!symbols->hasAssignmentRule(gid))
                {
                    ids.push_back("ucc(" + fsid + ", " + gid + ")");
                }
            }
        }
    }

    if (checkExact(SelectionRecord::CONTROL, types)) {
        for (size_t r = 0; r < getNumReactions(); ++r) {
            string rid = getReactionId(r);
            for (size_t bs = 0; bs < getNumBoundarySpecies(); bs++) {
                string bsid = getBoundarySpeciesId(bs);
                //Remove when https://github.com/sys-bio/roadrunner/issues/914 addressed.
                if (!symbols->hasAssignmentRule(bsid))
                {
                    ids.push_back("cc(" + rid + ", " + bsid + ")");
                }
            }
            for (size_t g = 0; g < getNumGlobalParameters(); g++) {
                string gid = getGlobalParameterId(g);
                //Remove when https://github.com/sys-bio/roadrunner/issues/914 addressed.
                if (!symbols->hasAssignmentRule(gid))
                {
                    ids.push_back("cc(" + rid + ", " + gid + ")");
                }
            }
        }
        for (size_t fs = 0; fs < getNumFloatingSpecies(); ++fs) {
            string fsid = getFloatingSpeciesId(fs);
            for (size_t bs = 0; bs < getNumBoundarySpecies(); bs++) {
                string bsid = getBoundarySpeciesId(bs);
                //Remove when https://github.com/sys-bio/roadrunner/issues/914 addressed.
                if (!symbols->hasAssignmentRule(bsid))
                {
                    ids.push_back("cc(" + fsid + ", " + bsid + ")");
                }
            }
            for (size_t g = 0; g < getNumGlobalParameters(); g++) {
                string gid = getGlobalParameterId(g);
                //Remove when https://github.com/sys-bio/roadrunner/issues/914 addressed.
                if (!symbols->hasAssignmentRule(gid))
                {
                    ids.push_back("cc(" + fsid + ", " + gid + ")");
                }
            }
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
        getRateRuleRates(1, &index, &result);
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
    case SelectionRecord::INITIAL_BOUNDARY_CONCENTRATION:
        getBoundarySpeciesInitConcentrations(1, &index, &result);
        break;
    case SelectionRecord::INITIAL_BOUNDARY_AMOUNT:
        getBoundarySpeciesInitAmounts(1, &index, &result);
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
        rrLog(Logger::LOG_ERROR) << "A new SelectionRecord should not have this value: "
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
            throw LLVMException("invalid selection std::string " + str);
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
            case LLVMModelDataSymbols::STOICHIOMETRY:
                sel.selectionType = SelectionRecord::STOICHIOMETRY;
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
                std::string msg = "No sbml element exists for concentration selection '" + str + "'";
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
            case LLVMModelDataSymbols::BOUNDARY_SPECIES:
                sel.selectionType = SelectionRecord::INITIAL_BOUNDARY_AMOUNT;
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
                std::string msg = "Invalid Id for initial value: '" + str + "'";
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
            else if (symbols->getSymbolIndex(sel.p1, index) == LLVMModelDataSymbols::BOUNDARY_SPECIES)
            {
                sel.selectionType = SelectionRecord::INITIAL_BOUNDARY_CONCENTRATION;
                sel.index = index;
                break;
            }

            {
                throw LLVMException("Invalid id '" + str + "' for inital floating species concentration");
                break;
            }
            break;
        case SelectionRecord::STOICHIOMETRY:
            sel.index = getStoichiometryIndex(sel.p1, sel.p2);
            break;

        default:
            rrLog(Logger::LOG_ERROR) << "A new SelectionRecord should not have this value: "
                << sel.to_repr();
            throw LLVMException("Invalid selection '" + str + "' for setting value");
            break;
        }

        // if we get here, should have a valid and set sel record.
        selectionRecordCache[str] = sel;

        rrLog(Logger::LOG_DEBUG) << "caching selection record stirng " << str;

        i = selectionRecordCache.find(str);
        assert(i != selectionRecordCache.end());
    }

    return i->second;
}

void LLVMExecutableModel::setValue(const std::string& id, double value)
{
    const SelectionRecord &sel = getSelection(id);
    int index = sel.index;

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
        for(size_t cm = 0; cm < symbols->getConservedMoietySize(); ++cm) {
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
    case SelectionRecord::INITIAL_BOUNDARY_AMOUNT:
        setBoundarySpeciesInitAmounts(1, &index, &value);
        break;
    case SelectionRecord::INITIAL_COMPARTMENT:
        setCompartmentInitVolumes(1, &index, &value);
        break;
    case SelectionRecord::INITIAL_FLOATING_CONCENTRATION:
        setFloatingSpeciesInitConcentrations(1, &index, &value);
        break;
    case SelectionRecord::INITIAL_BOUNDARY_CONCENTRATION:
        setBoundarySpeciesInitConcentrations(1, &index, &value);
        break;
    case SelectionRecord::INITIAL_GLOBAL_PARAMETER:
        setGlobalParameterInitValues(1, &index, &value);
        break;
    case SelectionRecord::STOICHIOMETRY:
        setStoichiometries(1, &index, &value);
        break;
    default:
        throw LLVMException("Invalid selection '" + sel.to_string() + "' for setting value");
        break;
    }
}

int LLVMExecutableModel::getFloatingSpeciesConcentrationRates(size_t len,
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

        // state std::vector is packed such that first numRateRules are the rate rule rates,
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

            rrLog(Logger::LOG_DEBUG) << "i: " << i << ", j: " << j
                    << ", comp index: "
                    << symbols->getCompartmentIndexForFloatingSpecies(j)
                    << ", vol: "
                    << volumes[symbols->getCompartmentIndexForFloatingSpecies(j)]
                    << std::endl;
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
    return static_cast<int>(len);
}

int LLVMExecutableModel::setBoundarySpeciesAmounts(size_t len, const int* indx,
    const double* values)
{
    return setBoundarySpeciesAmounts(len, indx, values, true);
}

int LLVMExecutableModel::setBoundarySpeciesAmounts(size_t len, const int* indx,
    const double* values, bool strict)
{
    bool result = false;
    if (setBoundarySpeciesAmountPtr)
    {
        for (int i = 0; i < len; ++i)
        {
            int j = indx ? indx[i] : i;
            result = setBoundarySpeciesAmountPtr(modelData, j, values[i]);
            if (!result && strict)
            {
                std::stringstream s;
                std::string id = symbols->getBoundarySpeciesId(j);
                s << "Could not set value for NON conserved moiety floating species " << id;

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
    return result;
}

std::string LLVMExecutableModel::getStateVectorId(size_t index)
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
    return symbols->getEventIndex(eventId);
}

std::string LLVMExecutableModel::getEventId(size_t index)
{
    std::vector<std::string> ids = symbols->getEventIds();
    if (index < ids.size())
    {
        return ids[index];
    }
    else
    {
        throw_llvm_exception("index out of range");
        return "";
    }
}

void LLVMExecutableModel::getEventIds(std::list<std::string>& out)
{
    std::vector<std::string> eventIds = symbols->getEventIds();
    std::copy(eventIds.begin(), eventIds.end(), std::back_inserter(out));
}

int LLVMExecutableModel::getNumPiecewiseTriggers()
{
    return modelData->numPiecewiseTriggers;
}

void LLVMExecutableModel::getAssignmentRuleIds(std::list<std::string>& out)
{
    std::vector<std::string> arIds = symbols->getAssignmentRuleIds();
    std::copy(arIds.begin(), arIds.end(), std::back_inserter(out));
}

void LLVMExecutableModel::getRateRuleIds(std::list<std::string>& out)
{
    std::vector<std::string> rrIds = symbols->getRateRuleIds();
    std::copy(rrIds.begin(), rrIds.end(), std::back_inserter(out));
}

void LLVMExecutableModel::getInitialAssignmentIds(std::list<std::string>& out)
{
    std::vector<std::string> rrIds = symbols->getInitialAssignmentIds();
    std::copy(rrIds.begin(), rrIds.end(), std::back_inserter(out));
}

void LLVMExecutableModel::setEventListener(size_t index,
        rr::EventListenerPtr eventHandler)
{
    if (index < modelData->numEvents)
    {
        rrLog(Logger::LOG_DEBUG) << "setting event handler " << index << " to " << eventHandler;
        eventListeners[index] = eventHandler;
    }
    else
    {
        throw_llvm_exception("index " + rr::toStringSize(index) + " out of range");
    }
}

rr::EventListenerPtr LLVMExecutableModel::getEventListener(size_t index)
{
    if (index < modelData->numEvents)
    {
        return eventListeners[index];
    }
    else
    {
        throw_llvm_exception("index " + rr::toStringSize(index) + " out of range");
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

int LLVMExecutableModel::getFloatingSpeciesAmounts(size_t len, const int* indx,
        double* values)
{
    return getValues(getFloatingSpeciesAmountPtr, len, indx, values);
}

int LLVMExecutableModel::setFloatingSpeciesAmounts(size_t len, int const* indx,
    const double* values)
{
    return setFloatingSpeciesAmounts(len, indx, values, true);
}

int LLVMExecutableModel::setFloatingSpeciesAmounts(size_t len, int const* indx,
    const double* values, bool strict)
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

                rrLog(Logger::LOG_INFORMATION) << "updating CM "
                        << symbols->getConservedMoietyId(cmIndex)
                        << " for conserved species "
                        << symbols->getFloatingSpeciesId(j)
                        << ", setting CM to " << newCmVal
                        << ", was " << currCmVal;

                setGlobalParameterValues(1, &gpIndex, &newCmVal, strict);
            }
            else if (strict)
            {
                std::stringstream s;
                std::string id = symbols->getFloatingSpeciesId(j);
                s << "Could not set value for NON conserved moiety floating species " << id;

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
    return static_cast<int>(len);
}

int LLVMExecutableModel::setFloatingSpeciesConcentrations(size_t len,
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

                rrLog(Logger::LOG_INFORMATION) << "updating CM "
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
                std::string id = symbols->getFloatingSpeciesId(j);
                s << "Could not set value for NON conserved moiety floating species " << id;

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
    return static_cast<int>(len);
}

int LLVMExecutableModel::getBoundarySpeciesAmounts(size_t len, const int* indx,
        double* values)
{
    return getValues(getBoundarySpeciesAmountPtr, len, indx, values);
}

int LLVMExecutableModel::getBoundarySpeciesConcentrations(size_t len,
        const int* indx, double* values)
{
    return getValues(getBoundarySpeciesConcentrationPtr, len, indx, values);
}

int LLVMExecutableModel::setBoundarySpeciesConcentrations(size_t len,
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

int LLVMExecutableModel::getGlobalParameterValues(size_t len, const int* indx,
        double* values)
{
    return getValues(getGlobalParameterPtr, len, indx, values);
}

int LLVMExecutableModel::setGlobalParameterValues(size_t len, const int* indx,
    const double* values)
{
    return setGlobalParameterValues(len, indx, values, true);
}

int LLVMExecutableModel::setGlobalParameterValues(size_t len, const int* indx,
    const double* values, bool strict)
{
    int result = -1;
    if (setGlobalParameterPtr)
    {
        result = setValues(setGlobalParameterPtr,
                &LLVMExecutableModel::getGlobalParameterId, len, indx, values, strict);

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

int LLVMExecutableModel::getCompartmentVolumes(size_t len, const int* indx,
        double* values)
{
    return getValues(getCompartmentVolumePtr, len, indx, values);
}

int LLVMExecutableModel::getReactionRates(size_t len, const int* indx,
        double* values)
{
    // the reaction rates are a function of the model state, so someone
    // could have changed some parameter, so we need to re-evaluate.

    // this is an optimization to not re-calcualte the reaction rates
    // when selecting reaction rates in a simulations.
    // its possible to extend this approach to check whenever any value
    // is set, save that for a future optimization.

    // this is disabled for now due to a bug identified in model_editing_test
    // ONE_ASSIGNMENT_RULE. It is also present in models with reactions that
    // depend only on time (i.e. it does not recalculate when it should).
    /*
    if((flags & OPTIMIZE_REACTION_RATE_SELECTION) == 0
            || (flags & INTEGRATION) == 0 || (dirty & DIRTY_REACTION_RATES) != 0) {
        conversionFactor = evalReactionRatesPtr(modelData);
        dirty &= ~DIRTY_REACTION_RATES;
    }
    */
    conversionFactor = evalReactionRatesPtr(modelData);
    dirty &= ~DIRTY_REACTION_RATES;


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
    return static_cast<int>(len);
}

int LLVMExecutableModel::getNumConservedMoieties()
{
    return static_cast<int>(symbols->getConservedMoietySize());
}

int LLVMExecutableModel::getConservedMoietyIndex(const std::string& name)
{
    int ret = symbols->getConservedMoietyIndex(name);
    if (ret<0)
    {
        rrLog(Logger::LOG_DEBUG) << __FUNC__ << ", invalid conserved moiety index " << name << ".";
    }
    return ret;
}

std::string LLVMExecutableModel::getConservedMoietyId(size_t index)
{
    return symbols->getConservedMoietyId(index);
}

int LLVMExecutableModel::getConservedMoietyValues(size_t len, const int* indx,
        double* values)
{
    int result = 0;
    for(size_t i = 0; i < len; ++i)
    {
        size_t j = indx ? indx[i] : i;
        int gpIndex = symbols->getConservedMoietyGlobalParameterIndex(static_cast<int>(j));
        result += getGlobalParameterValues(1, &gpIndex, &values[i]);
    }
    return result;
}

int LLVMExecutableModel::setConservedMoietyValues(size_t len, const int* indx,
        const double* values)
{
    int result = 0;
    for(size_t i = 0; i < len; ++i)
    {
        size_t j = indx ? indx[i] : i;
        int gpIndex = symbols->getConservedMoietyGlobalParameterIndex(static_cast<int>(j));
        result += setGlobalParameterValues(1, &gpIndex, &values[i]);
    }
    return result;
}

int LLVMExecutableModel::getFloatingSpeciesAmountRates(size_t len,
        int const *indx, double *values)
{
    uint dydtSize = modelData->numRateRules + modelData->numIndFloatingSpecies;

    double* dydt = (double*)calloc(dydtSize, sizeof(double));

    // state std::vector is packed such that first numRateRules are the rate rule rates,
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
    return static_cast<int>(len);
}


int LLVMExecutableModel::getRateRuleRates(size_t len,
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
    return static_cast<int>(len);
}


int LLVMExecutableModel::setCompartmentVolumes(size_t len, const int* indx,
    const double* values)
{
    return setCompartmentVolumes(len, indx, values, true);
}

int LLVMExecutableModel::setCompartmentVolumes(size_t len, const int* indx,
    const double* values, bool strict)
{
    int result = -1;
    if (setCompartmentVolumePtr)
    {
        result = setValues(setCompartmentVolumePtr,
                &LLVMExecutableModel::getCompartmentId, len, indx, values, strict);
    }
    return result;
}

int LLVMExecutableModel::setStoichiometries(size_t len, const int* indx,
                                               const double* values)
{
    return setStoichiometries(len, indx, values, true);
}

int LLVMExecutableModel::setStoichiometries(size_t len, const int* indx,
                                               const double* values, bool strict)
{
    if (len == 1) {
        int index = *indx;
        double value = *values;
        return setStoichiometry(index, value);
    }

    return -1;
}

int LLVMExecutableModel::setStoichiometry(int index, double value)
{
    if (std::signbit(value))
        throw LLVMException("Invalid stoichiometry value");

    if (symbols->isConservedMoietyAnalysis())
        throw LLVMException("Unable to set stoichiometries when conserved moieties are on");

    std::list <LLVMModelDataSymbols::SpeciesReferenceInfo> stoichiometryIndx = symbols->getStoichiometryList();
    std::list<LLVMModelDataSymbols::SpeciesReferenceInfo>::const_iterator stoichiometry = stoichiometryIndx.begin();
    for (int i = 0; i < index; i++)
        ++stoichiometry;
    if (stoichiometry->type == LLVMModelDataSymbols::SpeciesReferenceType::Product)
        return setStoichiometry(stoichiometry->row, stoichiometry->column, value);
    else if (stoichiometry->type == LLVMModelDataSymbols::SpeciesReferenceType::Reactant)
        return setStoichiometry(stoichiometry->row, stoichiometry->column, -1 * value);
    else if (stoichiometry->type == LLVMModelDataSymbols::SpeciesReferenceType::MultiReactantProduct)
        throw LLVMException("Cannot set stoichiometry for a MultiReactantProduct");
    else
        throw LLVMException("Cannot set stoichiometry for a Modifier");

    return -1;
}

int LLVMExecutableModel::setStoichiometry(int speciesIndex, int reactionIndex, double value)
{
    double result = csr_matrix_set_nz(modelData->stoichiometry, speciesIndex, reactionIndex, value);
    return isnan(result) ? 0 : result;
}

double LLVMExecutableModel::getStoichiometry(int index)
{
    if (symbols->isConservedMoietyAnalysis())
        throw LLVMException("Unable to get stoichiometries when conserved moieties are on");

    if (index < 0)
        throw LLVMException("The stoichiometry index is not valid");
    std::list<LLVMModelDataSymbols::SpeciesReferenceInfo> stoichiometryIndx = symbols->getStoichiometryList();
    std::list<LLVMModelDataSymbols::SpeciesReferenceInfo>::const_iterator stoichiometry = stoichiometryIndx.begin();
    for (int i = 0; i < index; i++)
        ++stoichiometry;
    if (stoichiometry->type == LLVMModelDataSymbols::SpeciesReferenceType::Reactant)
        return -1 * getStoichiometry(stoichiometry->row, stoichiometry->column);
    else if (stoichiometry->type == LLVMModelDataSymbols::SpeciesReferenceType::Product)
        return getStoichiometry(stoichiometry->row, stoichiometry->column);
    else if (stoichiometry->type == LLVMModelDataSymbols::SpeciesReferenceType::MultiReactantProduct)
        throw LLVMException("Cannot return stoichiometry for a MultiReactantProduct");
    else
        throw LLVMException("Cannot return stoichiometry for a Modifier");
}

double LLVMExecutableModel::getStoichiometry(int speciesIndex, int reactionIndex)
{
    double result = csr_matrix_get_nz(modelData->stoichiometry, speciesIndex, reactionIndex);
    return isnan(result) ? 0 : result;
}

/******************************* Events Section *******************************/
#if (1) /**********************************************************************/
/******************************************************************************/

int LLVMExecutableModel::getNumEvents()
{
    return modelData->numEvents;
}

int LLVMExecutableModel::getEventTriggers(size_t len, const int *indx, unsigned char *values)
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
        return static_cast<int>(len);
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
        std::vector<unsigned char> prevEventState(previousEventStatus,
                previousEventStatus + modelData->numEvents);

        std::vector<unsigned char> currEventStatus(modelData->numEvents);

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

void  LLVMExecutableModel::getPiecewiseTriggerRoots(double time, const double* y, double* gdot)
{
    modelData->time = time;

    double* savedRateRules = modelData->rateRuleValuesAlias;
    double* savedFloatingSpeciesAmounts = modelData->floatingSpeciesAmountsAlias;

    if (y)
    {
        modelData->rateRuleValuesAlias = const_cast<double*>(y);
        modelData->floatingSpeciesAmountsAlias = const_cast<double*>(y + modelData->numRateRules);

        evalVolatileStoichPtr(modelData);
    }

    for (uint i = 0; i < modelData->numPiecewiseTriggers; ++i)
    {
        unsigned char triggered = getPiecewiseTriggerPtr(modelData, i);

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

bool LLVMExecutableModel::getEventTrigger(size_t event)
{
    {
        assert(event < symbols->getEventAttributes().size()
            && "event out of bounds");
        if (modelData->time >= mIntegrationStartTime)
        {
            return getEventTriggerPtr(modelData, event);
        }
        else
        {
            return symbols->getEventAttributes()[event] & EventInitialValue
                ? true : false;
        }
    }
}

bool LLVMExecutableModel::applyEvents(unsigned char* prevEventState,
        unsigned char* currEventState)
{
    for (uint i = 0; i < modelData->numEvents; ++i)
    {
        bool c = getEventTrigger(i);
        currEventState[i] = c;

        rrLog(Logger::LOG_DEBUG) << "event " << i << ", previous state: " <<
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

bool LLVMExecutableModel::getEventTieBreak(size_t eventA, size_t eventB)
{
    /*
    C_ASSERT(sizeof(TieBreakKey) == 8 && sizeof(size_t) == 4);

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



    rrLog(Logger::LOG_DEBUG) << "tie break, a: " << eventA << ", b: " <<
            eventB << ", result: " << result;

    return result;
}


/******************************* Events Section *******************************/
#endif /***********************************************************************/
/******************************************************************************/

/******************************* Initial Conditions Section *******************/
#if (1) /**********************************************************************/
/******************************************************************************/


int LLVMExecutableModel::setFloatingSpeciesInitConcentrations(size_t len,
        const int* indx, const double* values)
{
    int result = -1;
    if (setFloatingSpeciesInitConcentrationsPtr)
    {
        result = setValues(setFloatingSpeciesInitConcentrationsPtr,
                &LLVMExecutableModel::getFloatingSpeciesId, len, indx, values);
    }

    dirty |= DIRTY_INIT_SPECIES;

    // as a convenience to users, this resets the amounts and whatever depends
    // on them.
    reset(SelectionRecord::TIME | SelectionRecord::FLOATING);

    return result;
}

int LLVMExecutableModel::setBoundarySpeciesInitConcentrations(size_t len,
    const int* indx, const double* values)
{
    int result = -1;
    if (setBoundarySpeciesInitConcentrationsPtr)
    {
        result = setValues(setBoundarySpeciesInitConcentrationsPtr,
            &LLVMExecutableModel::getBoundarySpeciesId, len, indx, values);
    }

    dirty |= DIRTY_INIT_SPECIES;

    // as a convienice to users, this resets the amounts and whatever depends
    // on them.
    reset(SelectionRecord::TIME | SelectionRecord::FLOATING);

    return result;
}

int LLVMExecutableModel::getFloatingSpeciesInitConcentrations(size_t len,
        const int* indx, double* values)
{
    int result = -1;
    if (getFloatingSpeciesInitConcentrationsPtr)
    {
        result = getValues(getFloatingSpeciesInitConcentrationsPtr, len, indx, values);
    }
    return result;
}

int LLVMExecutableModel::getBoundarySpeciesInitConcentrations(size_t len,
    const int* indx, double* values)
{
    int result = -1;
    if (getBoundarySpeciesInitConcentrationsPtr)
    {
        result = getValues(getBoundarySpeciesInitConcentrationsPtr, len, indx, values);
    }
    return result;
}

int LLVMExecutableModel::setFloatingSpeciesInitAmounts(size_t len, int const *indx,
            double const *values)
{
    int result = -1;
    if (setFloatingSpeciesInitAmountsPtr)
    {
        result = setValues(setFloatingSpeciesInitAmountsPtr,
                &LLVMExecutableModel::getFloatingSpeciesId, len, indx, values);
    }

    dirty |= DIRTY_INIT_SPECIES;

    // as a convenience to users, this resets the amounts and whatever depends
    // on them.
    reset(SelectionRecord::TIME | SelectionRecord::FLOATING);
    return result;
}

int LLVMExecutableModel::setBoundarySpeciesInitAmounts(size_t len, int const* indx,
    double const* values)
{
    int result = -1;
    if (setBoundarySpeciesInitAmountsPtr)
    {
        result = setValues(setBoundarySpeciesInitAmountsPtr,
            &LLVMExecutableModel::getBoundarySpeciesId, len, indx, values);
    }

    dirty |= DIRTY_INIT_SPECIES;

    // as a convienice to users, this resets the amounts and whatever depends
    // on them.
    reset(SelectionRecord::TIME | SelectionRecord::BOUNDARY);
    return result;
}

int LLVMExecutableModel::getFloatingSpeciesInitAmounts(size_t len, int const *indx,
                double *values)
{
    int result = -1;
    if (getFloatingSpeciesInitAmountsPtr)
    {
        result = getValues(getFloatingSpeciesInitAmountsPtr, len, indx, values);
    }
    return result;
}

int LLVMExecutableModel::getBoundarySpeciesInitAmounts(size_t len, int const* indx,
    double* values)
{
    int result = -1;
    if (getBoundarySpeciesInitAmountsPtr)
    {
        result = getValues(getBoundarySpeciesInitAmountsPtr, len, indx, values);
    }
    return result;
}

int LLVMExecutableModel::setCompartmentInitVolumes(size_t len, const int *indx,
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

int LLVMExecutableModel::getCompartmentInitVolumes(size_t len, const int *indx,
                double *values)
{
    int result = -1;
    if (getCompartmentInitVolumesPtr)
    {
        result = getValues(getCompartmentInitVolumesPtr, len, indx, values);
    }
    return result;
}

int LLVMExecutableModel::setGlobalParameterInitValues(size_t len, const int* indx,
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

int LLVMExecutableModel::getGlobalParameterInitValues(size_t len, const int *indx,
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
    rr::saveBinary(out, mIntegrationStartTime);
}


/******************************* End Random Section ***************************/
#endif  /**********************************************************************/
/******************************************************************************/


} /* namespace rr */
