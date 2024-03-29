/*
 * LLVMModelGenerator.cpp
 *
 * Created on: Jun 3, 2013
 *
 * Author: Andy Somogyi,
 *     email decode: V1 = "."; V2 = "@"; V3 = V1;
 *     andy V1 somogyi V2 gmail V3 com
 */
#pragma hdrstop

#define NOMINMAX

#include "LLVMModelGenerator.h"
#include "LLVMExecutableModel.h"
#include "ModelGeneratorContext.h"
#include "LLVMIncludes.h"
#include "JitFactory.h"
#include "ModelResources.h"
#include "Random.h"
#include <rrLogger.h>
#include <rrUtils.h>
#include <Poco/Mutex.h>

#ifdef _MSC_VER
#pragma warning(disable: 4146)
#pragma warning(disable: 4141)
#pragma warning(disable: 4267)
#pragma warning(disable: 4624)
#endif

#include "llvm/Object/ObjectFile.h"
#include "llvm/LLVMExecutableModel.h"
#include "llvm/MCJit.h"

#if LLVM_VERSION_MAJOR >= 13

#   include "llvm/LLJit.h"

#endif

#ifdef _MSC_VER
#pragma warning(default: 4146)
#pragma warning(default: 4141)
#pragma warning(default: 4267)
#pragma warning(default: 4624)
#endif


using rr::Logger;
using rr::getLogger;
using rr::ExecutableModel;
using rr::LoadSBMLOptions;
using rr::Compiler;
using llvm::Function;

namespace rrllvm {

    typedef std::weak_ptr<ModelResources> WeakModelResourcesPtr;
    typedef std::shared_ptr<ModelResources> SharedModelResourcesPtr;
    typedef std::unordered_map<std::string, WeakModelResourcesPtr> ModelResourcesPtrMap;

    static Poco::Mutex cachedModelsMutex;
    static ModelResourcesPtrMap cachedModelResources;

    inline void codeGeneration(ModelGeneratorContext& context, std::uint32_t options);

    inline ModelGeneratorContext
        createModelGeneratorContext(const libsbml::SBMLDocument* sbml, std::uint32_t options);

    inline std::string getSBMLMD5(const std::string& sbml, const std::uint32_t& options);

    inline LLVMModelData* codeGenAddModuleAndMakeModelData(
        ModelGeneratorContext* modelGeneratorContext,
        std::shared_ptr<ModelResources>& modelResources,
        std::uint32_t options);
    /**
     * copy the cached model fields between a cached model, and a
     * executable model.
     *
     * We don't want to have ExecutableModel inherit from CahcedModel
     * because they do compleltly different things, and have completly
     * differnt deletion semantics
     */
     // this is not used anywhere -- delete??.
    template<typename a_type, typename b_type>
    void copyCachedModel(a_type* src, b_type* dst) {
        dst->symbols = src->symbols;
        dst->context = src->context;
        dst->executionEngine = src->executionEngine;
        dst->errStr = src->errStr;

        dst->evalInitialConditionsPtr = src->evalInitialConditionsPtr;
        dst->evalReactionRatesPtr = src->evalReactionRatesPtr;
        dst->getBoundarySpeciesAmountPtr = src->getBoundarySpeciesAmountPtr;
        dst->getFloatingSpeciesAmountPtr = src->getFloatingSpeciesAmountPtr;
        dst->getBoundarySpeciesConcentrationPtr = src->getBoundarySpeciesConcentrationPtr;
        dst->getFloatingSpeciesConcentrationPtr = src->getFloatingSpeciesConcentrationPtr;
        dst->getCompartmentVolumePtr = src->getCompartmentVolumePtr;
        dst->getGlobalParameterPtr = src->getGlobalParameterPtr;
        dst->evalRateRuleRatesPtr = src->evalRateRuleRatesPtr;
        dst->getEventTriggerPtr = src->getEventTriggerPtr;
        dst->getEventPriorityPtr = src->getEventPriorityPtr;
        dst->getEventDelayPtr = src->getEventDelayPtr;
        dst->eventTriggerPtr = src->eventTriggerPtr;
        dst->eventAssignPtr = src->eventAssignPtr;
        dst->getPiecewiseTriggerPtr = src->getPiecewiseTriggerPtr;
        dst->evalVolatileStoichPtr = src->evalVolatileStoichPtr;
        dst->evalConversionFactorPtr = src->evalConversionFactorPtr;
    }

    inline void codeGeneration(ModelGeneratorContext& context, std::uint32_t options) {
        EvalInitialConditionsCodeGen(context).createFunction();
        EvalReactionRatesCodeGen(context).createFunction();
        GetBoundarySpeciesAmountCodeGen(context).createFunction();
        GetFloatingSpeciesAmountCodeGen(context).createFunction();
        GetBoundarySpeciesConcentrationCodeGen(context).createFunction();
        GetFloatingSpeciesConcentrationCodeGen(context).createFunction();
        GetCompartmentVolumeCodeGen(context).createFunction();
        GetGlobalParameterCodeGen(context).createFunction();
        EvalRateRuleRatesCodeGen(context).createFunction();
        GetEventTriggerCodeGen(context).createFunction();
        GetEventPriorityCodeGen(context).createFunction();
        GetEventDelayCodeGen(context).createFunction();
        EventTriggerCodeGen(context).createFunction();
        EventAssignCodeGen(context).createFunction();
        GetPiecewiseTriggerCodeGen(context).createFunction();
        EvalVolatileStoichCodeGen(context).createFunction();
        EvalConversionFactorCodeGen(context).createFunction();

        Function* setBoundarySpeciesAmountIR = nullptr;
        Function* setBoundarySpeciesConcentrationIR;
        Function* setFloatingSpeciesConcentrationIR = nullptr;
        Function* setCompartmentVolumeIR = nullptr;
        Function* setFloatingSpeciesAmountIR = nullptr;
        Function* setGlobalParameterIR = nullptr;
        Function* getFloatingSpeciesInitConcentrationsIR = nullptr;
        Function* setFloatingSpeciesInitConcentrationsIR = nullptr;
        Function* getFloatingSpeciesInitAmountsIR = nullptr;
        Function* setFloatingSpeciesInitAmountsIR = nullptr;
        Function* getBoundarySpeciesInitConcentrationsIR = nullptr;
        Function* setBoundarySpeciesInitConcentrationsIR = nullptr;
        Function* getBoundarySpeciesInitAmountsIR = nullptr;
        Function* setBoundarySpeciesInitAmountsIR = nullptr;
        Function* getCompartmentInitVolumesIR = nullptr;
        Function* setCompartmentInitVolumesIR = nullptr;
        Function* getGlobalParameterInitValueIR = nullptr;
        Function* setGlobalParameterInitValueIR = nullptr;
        if (options & LoadSBMLOptions::READ_ONLY) {
            setBoundarySpeciesAmountIR = nullptr;
            setBoundarySpeciesConcentrationIR = nullptr;
            setFloatingSpeciesConcentrationIR = nullptr;
            setCompartmentVolumeIR = nullptr;
            setFloatingSpeciesAmountIR = nullptr;
            setGlobalParameterIR = nullptr;
        }
        else {
            setBoundarySpeciesAmountIR =
                SetBoundarySpeciesAmountCodeGen(context).createFunction();

            setBoundarySpeciesConcentrationIR =
                SetBoundarySpeciesConcentrationCodeGen(context).createFunction();

            setFloatingSpeciesConcentrationIR =
                SetFloatingSpeciesConcentrationCodeGen(context).createFunction();

            setCompartmentVolumeIR =
                SetCompartmentVolumeCodeGen(context).createFunction();

            setFloatingSpeciesAmountIR =
                SetFloatingSpeciesAmountCodeGen(context).createFunction();

            setGlobalParameterIR =
                SetGlobalParameterCodeGen(context).createFunction();
        }

        if (options & LoadSBMLOptions::MUTABLE_INITIAL_CONDITIONS) {
            getFloatingSpeciesInitConcentrationsIR =
                GetFloatingSpeciesInitConcentrationCodeGen(context).createFunction();
            setFloatingSpeciesInitConcentrationsIR =
                SetFloatingSpeciesInitConcentrationCodeGen(context).createFunction();

            getFloatingSpeciesInitAmountsIR =
                GetFloatingSpeciesInitAmountCodeGen(context).createFunction();
            setFloatingSpeciesInitAmountsIR =
                SetFloatingSpeciesInitAmountCodeGen(context).createFunction();

            getBoundarySpeciesInitConcentrationsIR =
                GetBoundarySpeciesInitConcentrationCodeGen(context).createFunction();
            setBoundarySpeciesInitConcentrationsIR =
                SetBoundarySpeciesInitConcentrationCodeGen(context).createFunction();

            getBoundarySpeciesInitAmountsIR =
                GetBoundarySpeciesInitAmountCodeGen(context).createFunction();
            setBoundarySpeciesInitAmountsIR =
                SetBoundarySpeciesInitAmountCodeGen(context).createFunction();

            getCompartmentInitVolumesIR =
                GetCompartmentInitVolumeCodeGen(context).createFunction();
            setCompartmentInitVolumesIR =
                SetCompartmentInitVolumeCodeGen(context).createFunction();

            getGlobalParameterInitValueIR =
                GetGlobalParameterInitValueCodeGen(context).createFunction();
            setGlobalParameterInitValueIR =
                SetGlobalParameterInitValueCodeGen(context).createFunction();
        }
        else {
            getFloatingSpeciesInitConcentrationsIR = nullptr;
            setFloatingSpeciesInitConcentrationsIR = nullptr;
            getFloatingSpeciesInitAmountsIR = nullptr;
            setFloatingSpeciesInitAmountsIR = nullptr;
            getBoundarySpeciesInitConcentrationsIR = nullptr;
            setBoundarySpeciesInitConcentrationsIR = nullptr;
            getBoundarySpeciesInitAmountsIR = nullptr;
            setBoundarySpeciesInitAmountsIR = nullptr;
            setCompartmentInitVolumesIR = nullptr;
            getCompartmentInitVolumesIR = nullptr;
            getGlobalParameterInitValueIR = nullptr;
            setGlobalParameterInitValueIR = nullptr;
        }

    }

    std::string getSBMLMD5(const std::string& sbml, const std::uint32_t& options) {
        std::string sbmlMD5;
        sbmlMD5 = rr::getMD5(sbml);

        if (options & LoadSBMLOptions::CONSERVED_MOIETIES) {
            sbmlMD5 += "_conserved";
        }
        return sbmlMD5;
    }

    /**
     * @brief collect a few essential calls for creating a model into a function
     * @details these operations are required in multiple places and so they are
     * factored out into a single function.
     */
    LLVMModelData* codeGenAddModuleAndMakeModelData(
        ModelGeneratorContext* modelGeneratorContext,
        std::shared_ptr<ModelResources>& modelResources,
        std::uint32_t options) {

        // Do all code generation here. This populates the IR module representing
        // this sbml model.
        codeGeneration(*modelGeneratorContext, options);

        /**
         * Adds the module and context which is owned by the Jit
         * to the developing LLVM IR module.
         * In some Jit's (MCJit), this also triggers llvm IR optimization
         */
        modelGeneratorContext->getJitNonOwning()->addModule();

        LLVMModelData* modelData = createModelData(modelGeneratorContext->getModelDataSymbols(),
            modelGeneratorContext->getRandom(), modelGeneratorContext->getNumPiecewiseTriggers());

        uint llvmsize = ModelDataIRBuilder::getModelDataSize(
            modelGeneratorContext->getJitNonOwning()->getModuleNonOwning(),
            modelGeneratorContext->getJitNonOwning()->getDataLayout()
        );

        if (llvmsize != modelData->size) {
            std::stringstream s;

            s << "LLVM Model Data size " << llvmsize << " is different from " <<
                "C++ size of LLVM ModelData, " << modelData->size;

            LLVMModelData_free(modelData);

            rrLog(Logger::LOG_FATAL) << s.str();

            throw_llvm_exception(s.str())
        }

        modelGeneratorContext->getJitNonOwning()->mapLLVMGeneratedFunctionsToSymbols(modelResources.get(), options);


        return modelData;

    }

    ExecutableModel*
        LLVMModelGenerator::regenerateModel(rr::ExecutableModel* oldModel, libsbml::SBMLDocument* doc, uint options) 
    {
        SharedModelResourcesPtr modelResources = std::make_shared<ModelResources>();

        char* docSBML = doc->toSBML();

        ModelGeneratorContext modelGeneratorContext(doc, options, JitFactory::makeJitEngine(options));

        std::string sbmlMD5 = getSBMLMD5(std::string((const char*)docSBML), options);
        if (modelResources->sbmlMD5.empty()) {
            modelResources->sbmlMD5 = sbmlMD5;
        }
        modelGeneratorContext.getJitNonOwning()->setModuleIdentifier(sbmlMD5);

        free(docSBML);

        LLVMModelData* modelData = codeGenAddModuleAndMakeModelData(&modelGeneratorContext, modelResources, options);

        // * MOVE * the bits over from the context to the exe model.
        modelGeneratorContext.transferObjectsToResources(modelResources);
        LLVMExecutableModel* newModel = new LLVMExecutableModel(modelResources, modelData);

        if (oldModel) {
            // the stored SBML document will not keep updated, so we need to
            // copy the old values (e.g, initial values, current values) to new model
            // so that we can start from where we paused

            std::vector<std::string> newSymbols = newModel->getRateRuleSymbols();

            for (int i = 0; i < oldModel->getNumFloatingSpecies(); i++) {
                std::string id = oldModel->getFloatingSpeciesId(i);
                int index = newModel->getFloatingSpeciesIndex(id);

                if (index >= 0 && index < newModel->modelData->numInitFloatingSpecies) {
                    // new model has this species

                    if (!newModel->symbols->hasAssignmentRule(id) && !newModel->symbols->hasRateRule(id)) {
                        if (!newModel->symbols->hasInitialAssignmentRule(id)) {
                            double initValue = 0;
                            oldModel->getFloatingSpeciesInitAmounts(1, &i, &initValue);
                            newModel->modelData->initFloatingSpeciesAmountsAlias[index] = initValue;
                            //newModel->setFloatingSpeciesInitAmounts(1, &index, &initValue);
                        }
                    }
                }
            }


            for (int i = 0; i < oldModel->getNumFloatingSpecies(); i++) {
                std::string id = oldModel->getFloatingSpeciesId(i);
                int index = newModel->getFloatingSpeciesIndex(id);

                if (index >= 0 && index < newModel->modelData->numIndFloatingSpecies) {
                    // new model has this species

                    // TODO: should we change the dirty flag when we call setters?
                    double value = 0;
                    oldModel->getFloatingSpeciesAmounts(1, &i, &value);

                    if (!newModel->symbols->hasAssignmentRule(id) && !newModel->symbols->hasRateRule(id)) {
                        newModel->modelData->floatingSpeciesAmountsAlias[index] = value;
                        //newModel->setFloatingSpeciesAmounts(1, &index, &value);
                    }
                    else if (newModel->symbols->hasRateRule(id)) {
                        // copy to rate rule value data block
                        std::vector<std::string>::iterator it = std::find(newSymbols.begin(), newSymbols.end(), id);
                        if (it != newSymbols.end()) {
                            // found it
                            index = std::distance(newSymbols.begin(), it);
                            newModel->modelData->rateRuleValuesAlias[index] = value;
                        }

                    }
                }
            }


            for (int i = 0; i < oldModel->getNumBoundarySpecies(); i++) {
                std::string id = oldModel->getBoundarySpeciesId(i);
                int index = newModel->getBoundarySpeciesIndex(id);

                // TODO: set concentration or amount?

                if (index >= 0 && index < newModel->modelData->numIndBoundarySpecies) {
                    // new model has this species

                    double value = 0;
                    oldModel->getBoundarySpeciesAmounts(1, &i, &value);

                    if (!newModel->symbols->hasAssignmentRule(id) && !newModel->symbols->hasRateRule(id)) {
                        newModel->modelData->boundarySpeciesAmountsAlias[index] = value;
                        //newModel->setBoundarySpeciesAmounts(1, &index, &value);
                    }
                    else if (newModel->symbols->hasRateRule(id)) {
                        // copy to rate rule value data block
                        std::vector<std::string>::iterator it = std::find(newSymbols.begin(), newSymbols.end(), id);
                        if (it != newSymbols.end()) {
                            // found it
                            index = std::distance(newSymbols.begin(), it);
                            newModel->modelData->rateRuleValuesAlias[index] = value;
                        }
                    }
                    else {
                        if (!newModel->symbols->hasInitialAssignmentRule(id)) {
                            double initValue = 0;
                            oldModel->getBoundarySpeciesInitAmounts(1, &i, &initValue);
                            newModel->modelData->initBoundarySpeciesAmountsAlias[index] = initValue;
                        }
                    }
                }

            }


            for (int i = 0; i < oldModel->getNumCompartments(); i++) {
                std::string id = oldModel->getCompartmentId(i);
                int index = newModel->getCompartmentIndex(id);

                if (index >= 0 && index < newModel->modelData->numInitCompartments) {
                    // new model has this compartment

                    if (!newModel->symbols->hasAssignmentRule(id) && !newModel->symbols->hasRateRule(id)) {
                        if (!newModel->symbols->hasInitialAssignmentRule(id)) {
                            double initValue = 0;
                            oldModel->getCompartmentInitVolumes(1, &i, &initValue);
                            newModel->modelData->initCompartmentVolumesAlias[index] = initValue;
                            //newModel->setCompartmentInitVolumes(1, &index, &initValue);
                        }

                    }

                }

            }


            for (int i = 0; i < oldModel->getNumCompartments(); i++) {
                std::string id = oldModel->getCompartmentId(i);
                int index = newModel->getCompartmentIndex(id);

                if (index >= 0 && index < newModel->modelData->numIndCompartments) {
                    // new model has this compartment
                    double value = 0;
                    oldModel->getCompartmentVolumes(1, &i, &value);

                    if (!newModel->symbols->hasAssignmentRule(id) && !newModel->symbols->hasRateRule(id)) {
                        newModel->modelData->compartmentVolumesAlias[index] = value;
                        //newModel->setCompartmentVolumes(1, &index, &value);
                    }
                    else if (newModel->symbols->hasRateRule(id)) {
                        // copy to rate rule value data block
                        std::vector<std::string>::iterator it = std::find(newSymbols.begin(), newSymbols.end(), id);
                        if (it != newSymbols.end()) {
                            // found it
                            index = std::distance(newSymbols.begin(), it);
                            newModel->modelData->rateRuleValuesAlias[index] = value;
                        }

                    }

                }

            }

            for (int i = 0; i < oldModel->getNumGlobalParameters(); i++) {
                std::string id = oldModel->getGlobalParameterId(i);
                int index = newModel->getGlobalParameterIndex(id);

                if (index >= 0 && index < newModel->modelData->numInitGlobalParameters) {
                    // new model has this parameter

                    if (!newModel->symbols->hasAssignmentRule(id) && !newModel->symbols->hasRateRule(id)) {
                        if (!newModel->symbols->hasInitialAssignmentRule(id)) {
                            double initValue = 0;
                            oldModel->getGlobalParameterInitValues(1, &i, &initValue);
                            newModel->modelData->initGlobalParametersAlias[index] = initValue;
                            //newModel->setGlobalParameterInitValues(1, &index, &initValue);
                        }
                    }
                }
            }


            for (int i = 0; i < oldModel->getNumGlobalParameters(); i++) {
                std::string id = oldModel->getGlobalParameterId(i);
                int index = newModel->getGlobalParameterIndex(id);

                if (index >= 0 && index < newModel->modelData->numIndGlobalParameters) {
                    // new model has this parameter

                    double value = 0;
                    oldModel->getGlobalParameterValues(1, &i, &value);

                    if (!newModel->symbols->hasAssignmentRule(id) && !newModel->symbols->hasRateRule(id)) {
                        newModel->modelData->globalParametersAlias[index] = value;
                        //newModel->setGlobalParameterValues(1, &index, &value);
                    }
                    else if (newModel->symbols->hasRateRule(id)) {
                        // copy to rate rule value data block
                        std::vector<std::string>::iterator it = std::find(newSymbols.begin(), newSymbols.end(), id);
                        if (it != newSymbols.end()) {
                            // found it
                            index = std::distance(newSymbols.begin(), it);
                            newModel->modelData->rateRuleValuesAlias[index] = value;
                        }
                    }
                }
            }
            newModel->setTime(oldModel->getTime());
        }

        return newModel;
    }


    ExecutableModel* LLVMModelGenerator::createModel(const libsbml::SBMLDocument* sbml, const std::string& sbmlMD5, std::uint32_t options) {
        bool forceReCompile = options & LoadSBMLOptions::RECOMPILE;

        // if we force recompile, then we don't need to think
        // about locating a previously compiled model
        if (!forceReCompile) {
            // check for a cached copy

            SharedModelResourcesPtr sp;

            cachedModelsMutex.lock();

            // if count == 1, key is in map. Keys are unique so can only be 1 or 0.
            if (cachedModelResources.count(sbmlMD5) == 1) {
                // if key found, lock the weak pointer into shared pointer
                sp = cachedModelResources.at(sbmlMD5).lock();
            }

            cachedModelsMutex.unlock();

            // we could have recieved a bad ptr, a model could have been deleted,
            // in which case, we should have a bad ptr.

            if (sp) {
                rrLog(Logger::LOG_DEBUG) << "found a cached model for \"" << sbmlMD5 << "\"";
                return new LLVMExecutableModel(sp, createModelData(*sp->symbols, sp->random, 0));
            }
            else {
                rrLog(Logger::LOG_DEBUG) << "no cached model found for " << sbmlMD5
                    << ", creating new one";
            }
        }

        SharedModelResourcesPtr modelResources = std::make_shared<ModelResources>();

        /**
         * The sbml md5 is used as the LLVM module name.
         * This facilitates object caching via rrObjectCache since
         * the LLVM module name is the key in the object map used to
         * cache the objects.
         */
        modelResources->sbmlMD5 = sbmlMD5;

        ModelGeneratorContext modelGeneratorContext(sbml, options, JitFactory::makeJitEngine(options));

        // name the llvm module
        modelGeneratorContext.getJitNonOwning()->setModuleIdentifier(sbmlMD5);

        // todo figure out whether the various bigs of codegen can be threadded?
        //  Or do things need to happen in a certain order?
        //
        LLVMModelData* modelData = codeGenAddModuleAndMakeModelData(
            &modelGeneratorContext, modelResources, options
        );

        // if anything up to this point throws an exception, thats OK, because
        // we have not allocated any memory yet that is not taken care of by
        // something else.
        // Now that everything that could have thrown would have thrown, we
        // can now create the model and set its fields.

        // * MOVE * the bits over from the context to the exe model.
        modelGeneratorContext.transferObjectsToResources(modelResources);

        if (!forceReCompile) {
            // check for a chached copy, another thread could have
            // created one while we were making ours...

            ModelResourcesPtrMap::const_iterator i;

            SharedModelResourcesPtr sp;

            cachedModelsMutex.lock();

            // whilst we have it locked, clear any expired ptrs
            for (ModelResourcesPtrMap::const_iterator j = cachedModelResources.begin();
                j != cachedModelResources.end();) {
                if (j->second.expired()) {
                    rrLog(Logger::LOG_DEBUG) <<
                        "removing expired model resource for hash " << sbmlMD5;

                    j = cachedModelResources.erase(j);
                }
                else {
                    ++j;
                }
            }

            if ((i = cachedModelResources.find(sbmlMD5)) == cachedModelResources.end()) {
                rrLog(Logger::LOG_DEBUG) << "could not find existing cached resource "
                    "for hash " << sbmlMD5 <<
                    ", inserting new resources into cache";

                cachedModelResources[sbmlMD5] = modelResources;
            }

            cachedModelsMutex.unlock();
        }

        return new LLVMExecutableModel(modelResources, modelData);
    }



    /************ LLVM Utility Functions, TODO: Move To Separate File ************/

    /**
     * C++ 11 style to_string for LLVM types
     */
    std::string to_string(const llvm::Value* value) {
        std::string str;
        llvm::raw_string_ostream stream(str);
        value->print(stream);
        return str;
    }

    LLVMModelData* createModelData(const rrllvm::LLVMModelDataSymbols& symbols, const Random* random, uint numPiecewiseTriggers) {
        uint modelDataBaseSize = sizeof(LLVMModelData);

        uint numIndCompartments = static_cast<uint>(symbols.getIndependentCompartmentSize());
        uint numIndFloatingSpecies = static_cast<uint>(symbols.getIndependentFloatingSpeciesSize());
        uint numIndBoundarySpecies = static_cast<uint>(symbols.getIndependentBoundarySpeciesSize());
        uint numIndGlobalParameters = static_cast<uint>(symbols.getIndependentGlobalParameterSize());

        uint numInitCompartments = static_cast<uint>(symbols.getInitCompartmentSize());
        uint numInitFloatingSpecies = static_cast<uint>(symbols.getInitFloatingSpeciesSize());
        uint numInitBoundarySpecies = static_cast<uint>(symbols.getInitBoundarySpeciesSize());
        uint numInitGlobalParameters = static_cast<uint>(symbols.getInitGlobalParameterSize());

        // no initial conditions for these
        uint numRateRules = static_cast<uint>(symbols.getRateRuleSize());
        uint numReactions = static_cast<uint>(symbols.getReactionSize());

        uint modelDataSize = modelDataBaseSize +
            sizeof(double) * (
                numIndCompartments +
                numInitCompartments +
                numInitFloatingSpecies +
                numIndBoundarySpecies +
                numInitBoundarySpecies +
                numIndGlobalParameters +
                numInitGlobalParameters +
                numReactions +
                numRateRules +
                numIndFloatingSpecies
                );

        LLVMModelData* modelData = (LLVMModelData*)calloc(
            modelDataSize, sizeof(unsigned char));

        modelData->size = modelDataSize;
        modelData->numIndCompartments = numIndCompartments;
        modelData->numIndFloatingSpecies = numIndFloatingSpecies;
        modelData->numIndBoundarySpecies = numIndBoundarySpecies;
        modelData->numIndGlobalParameters = numIndGlobalParameters;

        modelData->numInitCompartments = numInitCompartments;
        modelData->numInitFloatingSpecies = numInitFloatingSpecies;
        modelData->numInitBoundarySpecies = numInitBoundarySpecies;
        modelData->numInitGlobalParameters = numInitGlobalParameters;

        modelData->numRateRules = numRateRules;
        modelData->numReactions = numReactions;
        modelData->numEvents = static_cast<uint>(symbols.getEventAttributes().size());
        modelData->numPiecewiseTriggers = numPiecewiseTriggers;

        // set the aliases to the offsets
        uint offset = 0;

        modelData->compartmentVolumesAlias = &modelData->data[offset];
        offset += numIndCompartments;

        modelData->initCompartmentVolumesAlias = &modelData->data[offset];
        offset += numInitCompartments;

        modelData->initFloatingSpeciesAmountsAlias = &modelData->data[offset];
        offset += numInitFloatingSpecies;

        modelData->boundarySpeciesAmountsAlias = &modelData->data[offset];
        offset += numIndBoundarySpecies;

        modelData->initBoundarySpeciesAmountsAlias = &modelData->data[offset];
        offset += numInitBoundarySpecies;

        modelData->globalParametersAlias = &modelData->data[offset];
        offset += numIndGlobalParameters;

        modelData->initGlobalParametersAlias = &modelData->data[offset];
        offset += numInitGlobalParameters;

        modelData->reactionRatesAlias = &modelData->data[offset];
        offset += numReactions;

        modelData->rateRuleValuesAlias = &modelData->data[offset];
        offset += numRateRules;

        modelData->floatingSpeciesAmountsAlias = &modelData->data[offset];
        offset += numIndFloatingSpecies;

        assert(modelDataBaseSize + offset * sizeof(double) == modelDataSize &&
            "LLVMModelData size not equal to base size + data");

        // allocate the stoichiometry matrix
        const std::vector<uint>& stoichRowIndx = symbols.getStoichRowIndx();
        const std::vector<uint>& stoichColIndx = symbols.getStoichColIndx();
        std::vector<double> stoichValues(stoichRowIndx.size(), 0);

        modelData->stoichiometry = rr::csr_matrix_new(numIndFloatingSpecies, numReactions,
            stoichRowIndx, stoichColIndx, stoichValues);

        // make a copy of the random object
        modelData->random = random ? new Random(*random) : 0;

        return modelData;
    }

} /* namespace rrllvm */


