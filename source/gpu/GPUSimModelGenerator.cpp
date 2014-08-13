/*
 * GPUSimModelGenerator.cpp
 *
 * Created on: Jun 3, 2013
 *
 * Author: Andy Somogyi,
 *     email decode: V1 = "."; V2 = "@"; V3 = V1;
 *     andy V1 somogyi V2 gmail V3 com
 */
#pragma hdrstop
#include "GPUSimModelGenerator.h"
#include "GPUSimExecutableModel.h"
// #include "ModelGeneratorContext.h"
#include "ModelResources.h"
#include "rrUtils.h"
#include "GPUSimException.h"
#include <rrLogger.h>
#include <Poco/Mutex.h>
#include <algorithm>

using rr::Logger;
using rr::getLogger;
using rr::ExecutableModel;
using rr::ModelGenerator;
using rr::Compiler;

namespace rr
{

namespace rrgpu
{

typedef cxx11_ns::weak_ptr<ModelResources> WeakModelPtr;
typedef cxx11_ns::shared_ptr<ModelResources> SharedModelPtr;
typedef cxx11_ns::unordered_map<std::string, WeakModelPtr> ModelPtrMap;

static Poco::Mutex cachedModelsMutex;
static ModelPtrMap cachedModels;


/**
 * copy the cached model fields between a cached model, and a
 * executable model.
 *
 * We don't want to have ExecutableModel inherit from CahcedModel
 * because they do compleltly different things, and have completly
 * differnt deletion semantics
 */
// template <typename a_type, typename b_type>
// void copyCachedModel(a_type* src, b_type* dst)
// {
//     dst->symbols = src->symbols;
//     dst->context = src->context;
//     dst->executionEngine = src->executionEngine;
//     dst->errStr = src->errStr;
//
//     dst->evalInitialConditionsPtr = src->evalInitialConditionsPtr;
//     dst->evalReactionRatesPtr = src->evalReactionRatesPtr;
//     dst->getBoundarySpeciesAmountPtr = src->getBoundarySpeciesAmountPtr;
//     dst->getFloatingSpeciesAmountPtr = src->getFloatingSpeciesAmountPtr;
//     dst->getBoundarySpeciesConcentrationPtr = src->getBoundarySpeciesConcentrationPtr;
//     dst->getFloatingSpeciesConcentrationPtr = src->getFloatingSpeciesConcentrationPtr;
//     dst->getCompartmentVolumePtr = src->getCompartmentVolumePtr;
//     dst->getGlobalParameterPtr = src->getGlobalParameterPtr;
//     dst->evalRateRuleRatesPtr = src->evalRateRuleRatesPtr;
//     dst->getEventTriggerPtr = src->getEventTriggerPtr;
//     dst->getEventPriorityPtr = src->getEventPriorityPtr;
//     dst->getEventDelayPtr = src->getEventDelayPtr;
//     dst->eventTriggerPtr = src->eventTriggerPtr;
//     dst->eventAssignPtr = src->eventAssignPtr;
//     dst->evalVolatileStoichPtr = src->evalVolatileStoichPtr;
//     dst->evalConversionFactorPtr = src->evalConversionFactorPtr;
// }


GPUSimModelGenerator::GPUSimModelGenerator(const std::string &str)
: compilerStr(str)
{
    std::transform(compilerStr.begin(), compilerStr.end(), compilerStr.begin(), toupper);
    Log(Logger::LOG_TRACE) << __FUNC__;
}

GPUSimModelGenerator::~GPUSimModelGenerator()
{
    Log(Logger::LOG_TRACE) << __FUNC__;

}

bool GPUSimModelGenerator::setTemporaryDirectory(const std::string& path)
{
    return true;
}

std::string GPUSimModelGenerator::getTemporaryDirectory()
{
    return "not used";
}

ExecutableModel* GPUSimModelGenerator::createModel(const std::string& sbml,
        uint options)
{
    bool forceReCompile = options & ModelGenerator::RECOMPILE;

    std::string md5;

    if (!forceReCompile)
    {
        // check for a chached copy
        md5 = rr::getMD5(sbml);

        if (options & ModelGenerator::CONSERVED_MOIETIES)
        {
            md5 += "_conserved";
        }

        ModelPtrMap::const_iterator i;

        SharedModelPtr sp;

        cachedModelsMutex.lock();

        if ((i = cachedModels.find(md5)) != cachedModels.end())
        {
            sp = i->second.lock();
        }

        cachedModelsMutex.unlock();

        // we could have recieved a bad ptr, a model could have been deleted,
        // in which case, we should have a bad ptr.

        if (sp)
        {
            Log(Logger::LOG_DEBUG) << "found a cached model for " << md5;
            throw_gpusim_exception("not implemented yet");
	    // must have ability to relocate resources
//             return new GPUSimExecutableModel();
        }
        else
        {
            Log(Logger::LOG_TRACE) << "no cached model found for " << md5
                    << ", creating new one";
        }
    }

    SharedModelPtr rc(new ModelResources());

    if (!forceReCompile)
    {
        // check for a chached copy, another thread could have
        // created one while we were making ours...

        ModelPtrMap::const_iterator i;

        SharedModelPtr sp;

        cachedModelsMutex.lock();

        // whilst we have it locked, clear any expired ptrs
        for (ModelPtrMap::const_iterator j = cachedModels.begin();
                j != cachedModels.end();)
        {
            if (j->second.expired())
            {
                Log(Logger::LOG_DEBUG) <<
                        "removing expired model resource for hash " << md5;

                j = cachedModels.erase(j);
            }
            else
            {
                ++j;
            }
        }

        if ((i = cachedModels.find(md5)) == cachedModels.end())
        {
            Log(Logger::LOG_DEBUG) << "could not find existing cached resource "
                    "resources, for hash " << md5 <<
                    ", inserting new resources into cache";

            cachedModels[md5] = rc;
        }

        cachedModelsMutex.unlock();
    }

    return new GPUSimExecutableModel();
}

Compiler* GPUSimModelGenerator::getCompiler()
{
    return &compiler;
}

bool GPUSimModelGenerator::setCompiler(const std::string& compiler)
{
    return true;
}

} // namespace rrgpu

} // namespace rr
