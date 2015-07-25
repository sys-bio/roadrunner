/*
 * CachedModel.cpp
 *
 *  Created on: Aug 28, 2013
 *      Author: andy
 */
#pragma hdrstop
#include "ModelResources.h"
#include "Random.h"

#include <rrLogger.h>
#include <list>

using rr::Logger;
using rr::getLogger;

namespace rrllvm
{

ModelResources::ModelResources() :
    symbols(0), executionEngine(0), context(0), errStr(0), useMCJIT(false),

    evalInitialConditionsPtr(0),
    evalReactionRatesPtr(0),
    getBoundarySpeciesAmountPtr(0),
    getFloatingSpeciesAmountPtr(0),
    getBoundarySpeciesConcentrationPtr(0),
    getFloatingSpeciesConcentrationPtr(0),
    getCompartmentVolumePtr(0),
    getGlobalParameterPtr(0),
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

    // init value accessors
    setFloatingSpeciesInitConcentrationsPtr(0),
    getFloatingSpeciesInitConcentrationsPtr(0),

    setFloatingSpeciesInitAmountsPtr(0),
    getFloatingSpeciesInitAmountsPtr(0),

    getCompartmentInitVolumesPtr(0),
    setCompartmentInitVolumesPtr(0)
{
    // Most of the ivars are assigned by the generator,
    // just make sure they are null.
}

ModelResources::~ModelResources()
{
    Log(Logger::LOG_DEBUG) << __FUNC__;

    if (errStr && errStr->size() > 0)
    {
        Log(Logger::LOG_WARNING) << "Non-empty LLVM ExecutionEngine error string: " << *errStr;
    }

    delete symbols;
    // the exe engine owns all the functions

    if (!useMCJIT) {
        delete executionEngine;
    } else {
        Log(Logger::LOG_NOTICE) << "Leaking LLVM ExecutionEngine " << (const void*)executionEngine;
    }

    delete context;
    delete random;
    delete errStr;
}

} /* namespace rrllvm */
