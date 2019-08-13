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
#include "LLVMModelGenerator.h"
#include "LLVMExecutableModel.h"
#include "ModelGeneratorContext.h"
#include "LLVMIncludes.h"
#include "llvm/Object/ObjectFile.h"
#include "ModelResources.h"
#include "Random.h"
#include <rrLogger.h>
#include <rrUtils.h>
#include <Poco/Mutex.h>

using rr::Logger;
using rr::getLogger;
using rr::ExecutableModel;
using rr::LoadSBMLOptions;
using rr::Compiler;
using llvm::Function;

namespace rrllvm
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
template <typename a_type, typename b_type>
void copyCachedModel(a_type* src, b_type* dst)
{
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
    dst->evalVolatileStoichPtr = src->evalVolatileStoichPtr;
    dst->evalConversionFactorPtr = src->evalConversionFactorPtr;
}


ExecutableModel* LLVMModelGenerator::regenerateModel(ExecutableModel* oldModel, libsbml::SBMLDocument* doc, uint options)
{
	// string md5;

	SharedModelPtr rc(new ModelResources());

	ModelGeneratorContext context(doc->toSBML(), options);

	// code generation part

	Function* evalInitialConditionsIR =
		EvalInitialConditionsCodeGen(context).createFunction();

	Function* evalReactionRatesIR =
		EvalReactionRatesCodeGen(context).createFunction();

	Function* getBoundarySpeciesAmountIR =
		GetBoundarySpeciesAmountCodeGen(context).createFunction();

	Function* getFloatingSpeciesAmountIR =
		GetFloatingSpeciesAmountCodeGen(context).createFunction();

	Function* getBoundarySpeciesConcentrationIR =
		GetBoundarySpeciesConcentrationCodeGen(context).createFunction();

	Function* getFloatingSpeciesConcentrationIR =
		GetFloatingSpeciesConcentrationCodeGen(context).createFunction();

	Function* getCompartmentVolumeIR =
		GetCompartmentVolumeCodeGen(context).createFunction();

	Function* getGlobalParameterIR =
		GetGlobalParameterCodeGen(context).createFunction();

	Function* evalRateRuleRatesIR =
		EvalRateRuleRatesCodeGen(context).createFunction();

	Function* getEventTriggerIR =
		GetEventTriggerCodeGen(context).createFunction();

	Function* getEventPriorityIR =
		GetEventPriorityCodeGen(context).createFunction();

	Function* getEventDelayIR =
		GetEventDelayCodeGen(context).createFunction();

	Function* eventTriggerIR =
		EventTriggerCodeGen(context).createFunction();

	Function* eventAssignIR =
		EventAssignCodeGen(context).createFunction();

	Function* evalVolatileStoichIR =
		EvalVolatileStoichCodeGen(context).createFunction();

	Function* evalConversionFactorIR =
		EvalConversionFactorCodeGen(context).createFunction();

	Function* setBoundarySpeciesAmountIR = 0;
	Function* setBoundarySpeciesConcentrationIR;
	Function* setFloatingSpeciesConcentrationIR = 0;
	Function* setCompartmentVolumeIR = 0;
	Function* setFloatingSpeciesAmountIR = 0;
	Function* setGlobalParameterIR = 0;
	Function* getFloatingSpeciesInitConcentrationsIR = 0;
	Function* setFloatingSpeciesInitConcentrationsIR = 0;
	Function* getFloatingSpeciesInitAmountsIR = 0;
	Function* setFloatingSpeciesInitAmountsIR = 0;
	Function* getCompartmentInitVolumesIR = 0;
	Function* setCompartmentInitVolumesIR = 0;
	Function* getGlobalParameterInitValueIR = 0;
	Function* setGlobalParameterInitValueIR = 0;
	if (options & LoadSBMLOptions::READ_ONLY)
	{
		setBoundarySpeciesAmountIR = 0;
		setBoundarySpeciesConcentrationIR = 0;
		setFloatingSpeciesConcentrationIR = 0;
		setCompartmentVolumeIR = 0;
		setFloatingSpeciesAmountIR = 0;
		setGlobalParameterIR = 0;
	}
	else
	{
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

	if (options & LoadSBMLOptions::MUTABLE_INITIAL_CONDITIONS)
	{
		getFloatingSpeciesInitConcentrationsIR =
			GetFloatingSpeciesInitConcentrationCodeGen(context).createFunction();
		setFloatingSpeciesInitConcentrationsIR =
			SetFloatingSpeciesInitConcentrationCodeGen(context).createFunction();

		getFloatingSpeciesInitAmountsIR =
			GetFloatingSpeciesInitAmountCodeGen(context).createFunction();
		setFloatingSpeciesInitAmountsIR =
			SetFloatingSpeciesInitAmountCodeGen(context).createFunction();

		getCompartmentInitVolumesIR =
			GetCompartmentInitVolumeCodeGen(context).createFunction();
		setCompartmentInitVolumesIR =
			SetCompartmentInitVolumeCodeGen(context).createFunction();

		getGlobalParameterInitValueIR =
			GetGlobalParameterInitValueCodeGen(context).createFunction();
		setGlobalParameterInitValueIR =
			SetGlobalParameterInitValueCodeGen(context).createFunction();
	}
	else
	{
		getFloatingSpeciesInitConcentrationsIR = 0;
		setFloatingSpeciesInitConcentrationsIR = 0;
		getFloatingSpeciesInitAmountsIR = 0;
		setFloatingSpeciesInitAmountsIR = 0;
		setCompartmentInitVolumesIR = 0;
		getCompartmentInitVolumesIR = 0;
		getGlobalParameterInitValueIR = 0;
		setGlobalParameterInitValueIR = 0;
	}

	//Currently we save jitted functions in object file format 
	//in save state. Compiling the functions into this format in the first place
	//makes saveState significantly faster than creating the object file when it is called
	//We then load the object file into the jit engine to avoid compiling the functions twice
	auto TargetMachine = context.getExecutionEngine().getTargetMachine();

	llvm::InitializeNativeTarget();

	//Write the object file to modBuffer
	std::error_code EC;
	llvm::SmallVector<char, 10> modBuffer;
	llvm::raw_svector_ostream mStrStream(modBuffer);

	llvm::legacy::PassManager pass;
	auto FileType = TargetMachine->CGFT_ObjectFile;

	if (TargetMachine->addPassesToEmitFile(pass, mStrStream, FileType))
	{
		throw "TargetMachine can't emit a file of type CGFT_ObjectFile";
	}

	pass.run(*context.module);
	
	//Read from modBuffer into our execution engine
	std::string moduleStr(modBuffer.begin(), modBuffer.end());

	auto memBuffer(llvm::MemoryBuffer::getMemBuffer(moduleStr));

	llvm::Expected<std::unique_ptr<llvm::object::ObjectFile> > objectFileExpected =
		llvm::object::ObjectFile::createObjectFile(llvm::MemoryBufferRef(moduleStr, "id"));
    
	std::unique_ptr<llvm::object::ObjectFile> objectFile(std::move(objectFileExpected.get()));
	
	llvm::object::OwningBinary<llvm::object::ObjectFile> owningObject(std::move(objectFile), std::move(memBuffer));

	context.getExecutionEngine().addObjectFile(std::move(owningObject));
	//https://stackoverflow.com/questions/28851646/llvm-jit-windows-8-1
	context.getExecutionEngine().finalizeObject();

	rc->evalInitialConditionsPtr = (EvalInitialConditionsCodeGen::FunctionPtr)
		context.getExecutionEngine().getFunctionAddress("evalInitialConditions");

	rc->evalReactionRatesPtr = (EvalReactionRatesCodeGen::FunctionPtr)
		context.getExecutionEngine().getFunctionAddress("evalReactionRates");

	rc->getBoundarySpeciesAmountPtr = (GetBoundarySpeciesAmountCodeGen::FunctionPtr)
		context.getExecutionEngine().getFunctionAddress("getBoundarySpeciesAmount");

	rc->getFloatingSpeciesAmountPtr = (GetFloatingSpeciesAmountCodeGen::FunctionPtr)
		context.getExecutionEngine().getFunctionAddress("getFloatingSpeciesAmount");

	rc->getBoundarySpeciesConcentrationPtr = (GetBoundarySpeciesConcentrationCodeGen::FunctionPtr)
		context.getExecutionEngine().getFunctionAddress("getBoundarySpeciesConcentration");

	rc->getFloatingSpeciesConcentrationPtr = (GetFloatingSpeciesAmountCodeGen::FunctionPtr)
		context.getExecutionEngine().getFunctionAddress("getFloatingSpeciesConcentration");

	rc->getCompartmentVolumePtr = (GetCompartmentVolumeCodeGen::FunctionPtr)
		context.getExecutionEngine().getFunctionAddress("getCompartmentVolume");

	rc->getGlobalParameterPtr = (GetGlobalParameterCodeGen::FunctionPtr)
		context.getExecutionEngine().getFunctionAddress("getGlobalParameter");

	rc->evalRateRuleRatesPtr = (EvalRateRuleRatesCodeGen::FunctionPtr)
		context.getExecutionEngine().getFunctionAddress("evalRateRuleRates");

	rc->getEventTriggerPtr = (GetEventTriggerCodeGen::FunctionPtr)
		context.getExecutionEngine().getFunctionAddress("getEventTrigger");

	rc->getEventPriorityPtr = (GetEventPriorityCodeGen::FunctionPtr)
		context.getExecutionEngine().getFunctionAddress("getEventPriority");

	rc->getEventDelayPtr = (GetEventDelayCodeGen::FunctionPtr)
		context.getExecutionEngine().getFunctionAddress("getEventDelay");

	rc->eventTriggerPtr = (EventTriggerCodeGen::FunctionPtr)
		context.getExecutionEngine().getFunctionAddress("eventTrigger");

	rc->eventAssignPtr = (EventAssignCodeGen::FunctionPtr)
		context.getExecutionEngine().getFunctionAddress("eventAssign");

	rc->evalVolatileStoichPtr = (EvalVolatileStoichCodeGen::FunctionPtr)
		context.getExecutionEngine().getFunctionAddress("evalVolatileStoich");

	rc->evalConversionFactorPtr = (EvalConversionFactorCodeGen::FunctionPtr)
		context.getExecutionEngine().getFunctionAddress("evalConversionFactor");
	if (options & LoadSBMLOptions::READ_ONLY)
	{
		rc->setBoundarySpeciesAmountPtr = 0;
		rc->setBoundarySpeciesConcentrationPtr = 0;
		rc->setFloatingSpeciesConcentrationPtr = 0;
		rc->setCompartmentVolumePtr = 0;
		rc->setFloatingSpeciesAmountPtr = 0;
		rc->setGlobalParameterPtr = 0;
	}
	else
	{
		rc->setBoundarySpeciesAmountPtr = (SetBoundarySpeciesAmountCodeGen::FunctionPtr)
context.getExecutionEngine().getFunctionAddress("setBoundarySpeciesAmount");

rc->setBoundarySpeciesConcentrationPtr = (SetBoundarySpeciesAmountCodeGen::FunctionPtr)
context.getExecutionEngine().getFunctionAddress("setBoundarySpeciesConcentration");

rc->setFloatingSpeciesConcentrationPtr = (SetBoundarySpeciesAmountCodeGen::FunctionPtr)
context.getExecutionEngine().getFunctionAddress("setFloatingSpeciesConcentration");

rc->setCompartmentVolumePtr = (SetBoundarySpeciesAmountCodeGen::FunctionPtr)
context.getExecutionEngine().getFunctionAddress("setCompartmentVolume");

rc->setFloatingSpeciesAmountPtr = (SetBoundarySpeciesAmountCodeGen::FunctionPtr)
context.getExecutionEngine().getFunctionAddress("setFloatingSpeciesAmount");

rc->setGlobalParameterPtr = (SetGlobalParameterCodeGen::FunctionPtr)
context.getExecutionEngine().getFunctionAddress("setGlobalParameter");
	}

	if (options & LoadSBMLOptions::MUTABLE_INITIAL_CONDITIONS)
	{
		rc->getFloatingSpeciesInitConcentrationsPtr = (GetBoundarySpeciesAmountCodeGen::FunctionPtr)
			context.getExecutionEngine().getFunctionAddress("getFloatingSpeciesInitConcentrations");
		rc->setFloatingSpeciesInitConcentrationsPtr = (SetBoundarySpeciesAmountCodeGen::FunctionPtr)
			context.getExecutionEngine().getFunctionAddress("setFloatingSpeciesInitConcentrations");

		rc->getFloatingSpeciesInitAmountsPtr = (GetBoundarySpeciesAmountCodeGen::FunctionPtr)
			context.getExecutionEngine().getFunctionAddress("getFloatingSpeciesInitAmounts");
		rc->setFloatingSpeciesInitAmountsPtr = (SetBoundarySpeciesAmountCodeGen::FunctionPtr)
			context.getExecutionEngine().getFunctionAddress("setFloatingSpeciesInitAmounts");

		rc->getCompartmentInitVolumesPtr = (GetBoundarySpeciesAmountCodeGen::FunctionPtr)
			context.getExecutionEngine().getFunctionAddress("getCompartmentInitVolumes");
		rc->setCompartmentInitVolumesPtr = (SetBoundarySpeciesAmountCodeGen::FunctionPtr)
			context.getExecutionEngine().getFunctionAddress("setCompartmentInitVolumes");

		rc->getGlobalParameterInitValuePtr = (GetBoundarySpeciesAmountCodeGen::FunctionPtr)
			context.getExecutionEngine().getFunctionAddress("getGlobalParameterInitValue");
		rc->setGlobalParameterInitValuePtr = (SetBoundarySpeciesAmountCodeGen::FunctionPtr)
			context.getExecutionEngine().getFunctionAddress("setGlobalParameterInitValue");
	}
	else
	{
		rc->getFloatingSpeciesInitConcentrationsPtr = 0;
		rc->setFloatingSpeciesInitConcentrationsPtr = 0;

		rc->getFloatingSpeciesInitAmountsPtr = 0;
		rc->setFloatingSpeciesInitAmountsPtr = 0;

		rc->getCompartmentInitVolumesPtr = 0;
		rc->setCompartmentInitVolumesPtr = 0;

		rc->getGlobalParameterInitValuePtr = 0;
		rc->setGlobalParameterInitValuePtr = 0;
	}



	// if anything up to this point throws an exception, thats OK, because	
	// we have not allocated any memory yet that is not taken care of by	
	// something else.	
	// Now that everything that could have thrown would have thrown, we	
	// can now create the model and set its fields.

	LLVMModelData* modelData = createModelData(context.getModelDataSymbols(),
		context.getRandom());

	uint llvmsize = ModelDataIRBuilder::getModelDataSize(context.getModule(),
		&context.getExecutionEngine());

	if (llvmsize != modelData->size)
	{
		std::stringstream s;

		s << "LLVM Model Data size " << llvmsize << " is different from " <<
			"C++ size of LLVM ModelData, " << modelData->size;

		LLVMModelData_free(modelData);

		Log(Logger::LOG_FATAL) << s.str();

		throw_llvm_exception(s.str());
	}

	// * MOVE * the bits over from the context to the exe model.
	context.stealThePeach(&rc->symbols, &rc->context,
		&rc->executionEngine, &rc->random, &rc->errStr);
    rc->moduleStr = moduleStr;

	LLVMExecutableModel* newModel = new LLVMExecutableModel(rc, modelData);


	if (oldModel) {
		// the stored SBML document will not keep updated, so we need to
		// copy the old values (e.g, initial values, current values) to new model
		// so that we can start from where we paused

		std::vector<std::string> newSymbols = newModel->getRateRuleSymbols();

		for (int i = 0; i < oldModel->getNumFloatingSpecies(); i++)
		{
			string id = oldModel->getFloatingSpeciesId(i);
			int index = newModel->getFloatingSpeciesIndex(id);

			if (index >= 0 && index < newModel->modelData->numInitFloatingSpecies)
			{
				// new model has this species

				if (!newModel->symbols->hasAssignmentRule(id) && !newModel->symbols->hasRateRule(id))
				{
					if (!newModel->symbols->hasInitialAssignmentRule(id))
					{
						double initValue = 0;
						oldModel->getFloatingSpeciesInitAmounts(1, &i, &initValue);
						newModel->modelData->initFloatingSpeciesAmountsAlias[index] = initValue;
						//newModel->setFloatingSpeciesInitAmounts(1, &index, &initValue);
					}
				}
			}
		}


		for (int i = 0; i < oldModel->getNumFloatingSpecies(); i++)
		{
			string id = oldModel->getFloatingSpeciesId(i);
			int index = newModel->getFloatingSpeciesIndex(id);

			if (index >= 0 && index < newModel->modelData->numIndFloatingSpecies)
			{
				// new model has this species

				// TODO: should we change the dirty flag when we call setters?
				double value = 0;
				oldModel->getFloatingSpeciesAmounts(1, &i, &value);

				if (!newModel->symbols->hasAssignmentRule(id) && !newModel->symbols->hasRateRule(id))
				{
					newModel->modelData->floatingSpeciesAmountsAlias[index] = value;
					//newModel->setFloatingSpeciesAmounts(1, &index, &value);
				}
				else if (newModel->symbols->hasRateRule(id))
				{
					// copy to rate rule value data block
					std::vector<string>::iterator it = std::find(newSymbols.begin(), newSymbols.end(), id);
					if (it != newSymbols.end())
					{
						// found it
						index = std::distance(newSymbols.begin(), it);
						newModel->modelData->rateRuleValuesAlias[index] = value;
					}

				}
			}
		}




		for (int i = 0; i < oldModel->getNumBoundarySpecies(); i++)
		{
			string id = oldModel->getBoundarySpeciesId(i);
			int index = newModel->getBoundarySpeciesIndex(id);

			// TODO: set concentration or amount?

			if (index >= 0 && index < newModel->modelData->numIndBoundarySpecies)
			{
				// new model has this species

				double value = 0;
				oldModel->getBoundarySpeciesAmounts(1, &i, &value);

				if (!newModel->symbols->hasAssignmentRule(id) && !newModel->symbols->hasRateRule(id))
				{
					newModel->modelData->boundarySpeciesAmountsAlias[index] = value;
					//newModel->setBoundarySpeciesAmounts(1, &index, &value);
				}
				else if (newModel->symbols->hasRateRule(id))
				{
					// copy to rate rule value data block
					std::vector<string>::iterator it = std::find(newSymbols.begin(), newSymbols.end(), id);
					if (it != newSymbols.end())
					{
						// found it
						index = std::distance(newSymbols.begin(), it);
						newModel->modelData->rateRuleValuesAlias[index] = value;
					}
				}
			}
			
		}


		for (int i = 0; i < oldModel->getNumCompartments(); i++)
		{
			string id = oldModel->getCompartmentId(i);
			int index = newModel->getCompartmentIndex(id);

			if (index >= 0 && index < newModel->modelData->numInitCompartments)
			{
				// new model has this compartment

				if (!newModel->symbols->hasAssignmentRule(id) && !newModel->symbols->hasRateRule(id))
				{
					if (!newModel->symbols->hasInitialAssignmentRule(id))
					{
						double initValue = 0;
						oldModel->getCompartmentInitVolumes(1, &i, &initValue);
						newModel->modelData->initCompartmentVolumesAlias[index] = initValue;
						//newModel->setCompartmentInitVolumes(1, &index, &initValue);
					}

				}

			}
			
		}



		for (int i = 0; i < oldModel->getNumCompartments(); i++)
		{
			string id = oldModel->getCompartmentId(i);
			int index = newModel->getCompartmentIndex(id);

			if (index >= 0 && index < newModel->modelData->numIndCompartments)
			{
				// new model has this compartment
				double value = 0;
				oldModel->getCompartmentVolumes(1, &i, &value);

				if (!newModel->symbols->hasAssignmentRule(id) && !newModel->symbols->hasRateRule(id))
				{
					newModel->modelData->compartmentVolumesAlias[index] = value;
					//newModel->setCompartmentVolumes(1, &index, &value);
				}
				else if (newModel->symbols->hasRateRule(id))
				{
					// copy to rate rule value data block
					std::vector<string>::iterator it = std::find(newSymbols.begin(), newSymbols.end(), id);
					if (it != newSymbols.end())
					{
						// found it
						index = std::distance(newSymbols.begin(), it);
						newModel->modelData->rateRuleValuesAlias[index] = value;
					}

				}

			}

		}

		for (int i = 0; i < oldModel->getNumGlobalParameters(); i++)
		{
			string id = oldModel->getGlobalParameterId(i);
			int index = newModel->getGlobalParameterIndex(id);

			if (index >= 0 && index < newModel->modelData->numInitGlobalParameters)
			{
				// new model has this parameter

				if (!newModel->symbols->hasAssignmentRule(id) && !newModel->symbols->hasRateRule(id))
				{
					if (!newModel->symbols->hasInitialAssignmentRule(id))
					{
						double initValue = 0;
						oldModel->getGlobalParameterInitValues(1, &i, &initValue);
						newModel->modelData->initGlobalParametersAlias[index] = initValue;
						//newModel->setGlobalParameterInitValues(1, &index, &initValue);
					}


				}

			}

		}


		for (int i = 0; i < oldModel->getNumGlobalParameters(); i++)
		{
			string id = oldModel->getGlobalParameterId(i);
			int index = newModel->getGlobalParameterIndex(id);

			if (index >= 0 && index < newModel->modelData->numIndGlobalParameters)
			{
				// new model has this parameter

				double value = 0;
				oldModel->getGlobalParameterValues(1, &i, &value);

				if (!newModel->symbols->hasAssignmentRule(id) && !newModel->symbols->hasRateRule(id))
				{
					newModel->modelData->globalParametersAlias[index] = value;
					//newModel->setGlobalParameterValues(1, &index, &value);
				}
		
				else if (newModel->symbols->hasRateRule(id))
				{
					// copy to rate rule value data block
					std::vector<string>::iterator it = std::find(newSymbols.begin(), newSymbols.end(), id);
					if (it != newSymbols.end())
					{
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


ExecutableModel* LLVMModelGenerator::createModel(const std::string& sbml,
        uint options)
{
    bool forceReCompile = options & LoadSBMLOptions::RECOMPILE;

    string md5;

    if (!forceReCompile)
    {
        // check for a cached copy
        md5 = rr::getMD5(sbml);

        if (options & LoadSBMLOptions::CONSERVED_MOIETIES)
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
            return new LLVMExecutableModel(sp, createModelData(*sp->symbols, sp->random));
        }
        else
        {
            Log(Logger::LOG_TRACE) << "no cached model found for " << md5
                    << ", creating new one";
        }
    }

    SharedModelPtr rc(new ModelResources());

    ModelGeneratorContext context(sbml, options);

    Function* evalInitialConditionsIR =
            EvalInitialConditionsCodeGen(context).createFunction();

	Function* evalReactionRatesIR =
            EvalReactionRatesCodeGen(context).createFunction();

	Function* getBoundarySpeciesAmountIR =
            GetBoundarySpeciesAmountCodeGen(context).createFunction();

	Function* getFloatingSpeciesAmountIR =
            GetFloatingSpeciesAmountCodeGen(context).createFunction();

	Function* getBoundarySpeciesConcentrationIR =
            GetBoundarySpeciesConcentrationCodeGen(context).createFunction();

	Function* getFloatingSpeciesConcentrationIR =
            GetFloatingSpeciesConcentrationCodeGen(context).createFunction();

	Function* getCompartmentVolumeIR =
            GetCompartmentVolumeCodeGen(context).createFunction();

	Function* getGlobalParameterIR =
            GetGlobalParameterCodeGen(context).createFunction();

	Function* evalRateRuleRatesIR =
            EvalRateRuleRatesCodeGen(context).createFunction();

	Function* getEventTriggerIR =
            GetEventTriggerCodeGen(context).createFunction();

	Function* getEventPriorityIR =
            GetEventPriorityCodeGen(context).createFunction();

	Function* getEventDelayIR =
            GetEventDelayCodeGen(context).createFunction();

	Function* eventTriggerIR =
            EventTriggerCodeGen(context).createFunction();

	Function* eventAssignIR =
            EventAssignCodeGen(context).createFunction();

	Function* evalVolatileStoichIR =
            EvalVolatileStoichCodeGen(context).createFunction();

	Function* evalConversionFactorIR =
            EvalConversionFactorCodeGen(context).createFunction();

	Function* setBoundarySpeciesAmountIR = 0 ;
	Function* setBoundarySpeciesConcentrationIR;
	Function* setFloatingSpeciesConcentrationIR = 0;
	Function* setCompartmentVolumeIR = 0;
	Function* setFloatingSpeciesAmountIR = 0;
	Function* setGlobalParameterIR = 0;
	Function* getFloatingSpeciesInitConcentrationsIR = 0;
	Function* setFloatingSpeciesInitConcentrationsIR = 0;
	Function* getFloatingSpeciesInitAmountsIR = 0;
	Function* setFloatingSpeciesInitAmountsIR = 0;
	Function* getCompartmentInitVolumesIR = 0;
	Function* setCompartmentInitVolumesIR = 0;
	Function* getGlobalParameterInitValueIR = 0;
	Function* setGlobalParameterInitValueIR = 0;
	if (options & LoadSBMLOptions::READ_ONLY)
	{
		setBoundarySpeciesAmountIR= 0;
		setBoundarySpeciesConcentrationIR = 0;
		setFloatingSpeciesConcentrationIR = 0;
		setCompartmentVolumeIR	= 0;
		setFloatingSpeciesAmountIR= 0;
		setGlobalParameterIR	= 0;
	}
	else
	{
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

	if (options & LoadSBMLOptions::MUTABLE_INITIAL_CONDITIONS)
	{
		getFloatingSpeciesInitConcentrationsIR =
			GetFloatingSpeciesInitConcentrationCodeGen(context).createFunction();
		setFloatingSpeciesInitConcentrationsIR =
			SetFloatingSpeciesInitConcentrationCodeGen(context).createFunction();

		getFloatingSpeciesInitAmountsIR =
			GetFloatingSpeciesInitAmountCodeGen(context).createFunction();
		setFloatingSpeciesInitAmountsIR =
			SetFloatingSpeciesInitAmountCodeGen(context).createFunction();

		getCompartmentInitVolumesIR =
			GetCompartmentInitVolumeCodeGen(context).createFunction();
		setCompartmentInitVolumesIR =
			SetCompartmentInitVolumeCodeGen(context).createFunction();

		getGlobalParameterInitValueIR =
			GetGlobalParameterInitValueCodeGen(context).createFunction();
		setGlobalParameterInitValueIR =
			SetGlobalParameterInitValueCodeGen(context).createFunction();
	}
	else
	{
		getFloatingSpeciesInitConcentrationsIR = 0;
		setFloatingSpeciesInitConcentrationsIR = 0;
		getFloatingSpeciesInitAmountsIR		= 0;
		setFloatingSpeciesInitAmountsIR		= 0;
		setCompartmentInitVolumesIR			= 0;
		getCompartmentInitVolumesIR			= 0;
		getGlobalParameterInitValueIR			= 0;
		setGlobalParameterInitValueIR			= 0;
	}

	//Currently we save jitted functions in object file format 
	//in save state. Compiling the functions into this format in the first place
	//makes saveState significantly faster than creating the object file when it is called
	//We then load the object file into the jit engine to avoid compiling the functions twice
	auto TargetMachine = context.getExecutionEngine().getTargetMachine();

	llvm::InitializeNativeTarget();

	//Write the object file to modBuffer
	std::error_code EC;
	llvm::SmallVector<char, 10> modBuffer;
	llvm::raw_svector_ostream mStrStream(modBuffer);

	llvm::legacy::PassManager pass;
	auto FileType = TargetMachine->CGFT_ObjectFile;

	if (TargetMachine->addPassesToEmitFile(pass, mStrStream, FileType))
	{
		throw "TargetMachine can't emit a file of type CGFT_ObjectFile";
	}

	pass.run(*context.module);
	
	//Read from modBuffer into our execution engine
	std::string moduleStr(modBuffer.begin(), modBuffer.end());

	auto memBuffer(llvm::MemoryBuffer::getMemBuffer(moduleStr));

	llvm::Expected<std::unique_ptr<llvm::object::ObjectFile> > objectFileExpected =
		llvm::object::ObjectFile::createObjectFile(llvm::MemoryBufferRef(moduleStr, "id"));
    
	std::unique_ptr<llvm::object::ObjectFile> objectFile(std::move(objectFileExpected.get()));
	
	llvm::object::OwningBinary<llvm::object::ObjectFile> owningObject(std::move(objectFile), std::move(memBuffer));

	context.getExecutionEngine().addObjectFile(std::move(owningObject));

	//https://stackoverflow.com/questions/28851646/llvm-jit-windows-8-1
	context.getExecutionEngine().finalizeObject();

	rc->evalInitialConditionsPtr = (EvalInitialConditionsCodeGen::FunctionPtr)
        context.getExecutionEngine().getFunctionAddress("evalInitialConditions");

	rc->evalReactionRatesPtr = (EvalReactionRatesCodeGen::FunctionPtr)
        context.getExecutionEngine().getFunctionAddress("evalReactionRates");

	rc->getBoundarySpeciesAmountPtr = (GetBoundarySpeciesAmountCodeGen::FunctionPtr)
        context.getExecutionEngine().getFunctionAddress("getBoundarySpeciesAmount");

	rc->getFloatingSpeciesAmountPtr = (GetFloatingSpeciesAmountCodeGen::FunctionPtr)
        context.getExecutionEngine().getFunctionAddress("getFloatingSpeciesAmount");

	rc->getBoundarySpeciesConcentrationPtr = (GetBoundarySpeciesConcentrationCodeGen::FunctionPtr)
        context.getExecutionEngine().getFunctionAddress("getBoundarySpeciesConcentration");

	rc->getFloatingSpeciesConcentrationPtr = (GetFloatingSpeciesAmountCodeGen::FunctionPtr)
        context.getExecutionEngine().getFunctionAddress("getFloatingSpeciesConcentration");

	rc->getCompartmentVolumePtr = (GetCompartmentVolumeCodeGen::FunctionPtr)
        context.getExecutionEngine().getFunctionAddress("getCompartmentVolume");

	rc->getGlobalParameterPtr = (GetGlobalParameterCodeGen::FunctionPtr)
        context.getExecutionEngine().getFunctionAddress("getGlobalParameter");

	rc->evalRateRuleRatesPtr = (EvalRateRuleRatesCodeGen::FunctionPtr)
        context.getExecutionEngine().getFunctionAddress("evalRateRuleRates");

	rc->getEventTriggerPtr = (GetEventTriggerCodeGen::FunctionPtr)
        context.getExecutionEngine().getFunctionAddress("getEventTrigger");

	rc->getEventPriorityPtr = (GetEventPriorityCodeGen::FunctionPtr)
        context.getExecutionEngine().getFunctionAddress("getEventPriority");

	rc->getEventDelayPtr = (GetEventDelayCodeGen::FunctionPtr)
        context.getExecutionEngine().getFunctionAddress("getEventDelay");

	rc->eventTriggerPtr = (EventTriggerCodeGen::FunctionPtr)
        context.getExecutionEngine().getFunctionAddress("eventTrigger");

	rc->eventAssignPtr = (EventAssignCodeGen::FunctionPtr)
        context.getExecutionEngine().getFunctionAddress("eventAssign");

	rc->evalVolatileStoichPtr = (EvalVolatileStoichCodeGen::FunctionPtr)
        context.getExecutionEngine().getFunctionAddress("evalVolatileStoich");

	rc->evalConversionFactorPtr = (EvalConversionFactorCodeGen::FunctionPtr)
        context.getExecutionEngine().getFunctionAddress("evalConversionFactor");
	if (options & LoadSBMLOptions::READ_ONLY)
	{
		rc->setBoundarySpeciesAmountPtr = 0;
		rc->setBoundarySpeciesConcentrationPtr = 0;
		rc->setFloatingSpeciesConcentrationPtr = 0;
		rc->setCompartmentVolumePtr = 0;
		rc->setFloatingSpeciesAmountPtr = 0;
		rc->setGlobalParameterPtr = 0;
	}
	else
	{
		rc->setBoundarySpeciesAmountPtr = (SetBoundarySpeciesAmountCodeGen::FunctionPtr)
            context.getExecutionEngine().getFunctionAddress("setBoundarySpeciesAmount");

		rc->setBoundarySpeciesConcentrationPtr = (SetBoundarySpeciesAmountCodeGen::FunctionPtr)
			context.getExecutionEngine().getFunctionAddress("setBoundarySpeciesConcentration");

		rc->setFloatingSpeciesConcentrationPtr = (SetBoundarySpeciesAmountCodeGen::FunctionPtr)
			context.getExecutionEngine().getFunctionAddress("setFloatingSpeciesConcentration");

		rc->setCompartmentVolumePtr = (SetBoundarySpeciesAmountCodeGen::FunctionPtr)
			context.getExecutionEngine().getFunctionAddress("setCompartmentVolume");

		rc->setFloatingSpeciesAmountPtr = (SetBoundarySpeciesAmountCodeGen::FunctionPtr)
			context.getExecutionEngine().getFunctionAddress("setFloatingSpeciesAmount");

        rc->setGlobalParameterPtr = (SetGlobalParameterCodeGen::FunctionPtr)
            context.getExecutionEngine().getFunctionAddress("setGlobalParameter");
	}

	if (options & LoadSBMLOptions::MUTABLE_INITIAL_CONDITIONS)
	{
		rc->getFloatingSpeciesInitConcentrationsPtr = (GetBoundarySpeciesAmountCodeGen::FunctionPtr)
            context.getExecutionEngine().getFunctionAddress("getFloatingSpeciesInitConcentrations");
		rc->setFloatingSpeciesInitConcentrationsPtr = (SetBoundarySpeciesAmountCodeGen::FunctionPtr)
            context.getExecutionEngine().getFunctionAddress("setFloatingSpeciesInitConcentrations");

		rc->getFloatingSpeciesInitAmountsPtr = (GetBoundarySpeciesAmountCodeGen::FunctionPtr)
            context.getExecutionEngine().getFunctionAddress("getFloatingSpeciesInitAmounts");
		rc->setFloatingSpeciesInitAmountsPtr = (SetBoundarySpeciesAmountCodeGen::FunctionPtr)
            context.getExecutionEngine().getFunctionAddress("setFloatingSpeciesInitAmounts");

		rc->getCompartmentInitVolumesPtr = (GetBoundarySpeciesAmountCodeGen::FunctionPtr)
            context.getExecutionEngine().getFunctionAddress("getCompartmentInitVolumes");
		rc->setCompartmentInitVolumesPtr = (SetBoundarySpeciesAmountCodeGen::FunctionPtr)
            context.getExecutionEngine().getFunctionAddress("setCompartmentInitVolumes");

		rc->getGlobalParameterInitValuePtr = (GetBoundarySpeciesAmountCodeGen::FunctionPtr)
            context.getExecutionEngine().getFunctionAddress("getGlobalParameterInitValue");
		rc->setGlobalParameterInitValuePtr = (SetBoundarySpeciesAmountCodeGen::FunctionPtr)
            context.getExecutionEngine().getFunctionAddress("setGlobalParameterInitValue");
	}
	else
	{
		rc->getFloatingSpeciesInitConcentrationsPtr = 0;
		rc->setFloatingSpeciesInitConcentrationsPtr = 0;

		rc->getFloatingSpeciesInitAmountsPtr = 0;
		rc->setFloatingSpeciesInitAmountsPtr = 0;

		rc->getCompartmentInitVolumesPtr = 0;
		rc->setCompartmentInitVolumesPtr = 0;

		rc->getGlobalParameterInitValuePtr = 0;
		rc->setGlobalParameterInitValuePtr = 0;
	}






	// if anything up to this point throws an exception, thats OK, because	
	// we have not allocated any memory yet that is not taken care of by	
	// something else.	
	// Now that everything that could have thrown would have thrown, we	
	// can now create the model and set its fields.

    LLVMModelData *modelData = createModelData(context.getModelDataSymbols(),
            context.getRandom());
   
    uint llvmsize = ModelDataIRBuilder::getModelDataSize(context.getModule(),
            &context.getExecutionEngine());

    if (llvmsize != modelData->size)
    {
        std::stringstream s;

        s << "LLVM Model Data size " << llvmsize << " is different from " <<
                "C++ size of LLVM ModelData, " << modelData->size;

        LLVMModelData_free(modelData);

        Log(Logger::LOG_FATAL) << s.str();

        throw_llvm_exception(s.str());
    }

    // * MOVE * the bits over from the context to the exe model.
    context.stealThePeach(&rc->symbols, &rc->context,
            &rc->executionEngine, &rc->random, &rc->errStr);

    //Save the object so we can saveState quickly later
    rc->moduleStr = moduleStr;

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

    return new LLVMExecutableModel(rc, modelData);
}



/************ LLVM Utility Functions, TODO: Move To Separate File ************/

/**
 * C++ 11 style to_string for LLVM types
 */
std::string to_string(const llvm::Value *value)
{
    std::string str;
    llvm::raw_string_ostream stream(str);
    value->print(stream);
    return str;
}

LLVMModelData *createModelData(const rrllvm::LLVMModelDataSymbols &symbols,
        const Random *random)
{
    uint modelDataBaseSize = sizeof(LLVMModelData);

    uint numIndCompartments = symbols.getIndependentCompartmentSize();
    uint numIndFloatingSpecies = symbols.getIndependentFloatingSpeciesSize();
    uint numIndBoundarySpecies = symbols.getIndependentBoundarySpeciesSize();
    uint numIndGlobalParameters = symbols.getIndependentGlobalParameterSize();

    uint numInitCompartments = symbols.getInitCompartmentSize();
    uint numInitFloatingSpecies = symbols.getInitFloatingSpeciesSize();
    uint numInitBoundarySpecies = symbols.getInitBoundarySpeciesSize();
    uint numInitGlobalParameters = symbols.getInitGlobalParameterSize();

    // no initial conditions for these
    uint numRateRules = symbols.getRateRuleSize();
    uint numReactions = symbols.getReactionSize();

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

    LLVMModelData *modelData = (LLVMModelData*)calloc(
            modelDataSize, sizeof(unsigned char));

    modelData->size = modelDataSize;
    modelData->numIndCompartments = numIndCompartments;
    modelData->numIndFloatingSpecies = numIndFloatingSpecies;
    modelData->numIndBoundarySpecies = numIndBoundarySpecies;
    modelData->numIndGlobalParameters = numIndGlobalParameters;

    modelData->numInitCompartments = numInitCompartments;
    modelData->numInitFloatingSpecies = numInitFloatingSpecies;
    modelData->numInitBoundarySpecies = numInitBoundarySpecies;
    modelData->numInitBoundarySpecies = numInitGlobalParameters;

    modelData->numRateRules = numRateRules;
    modelData->numReactions = numReactions;
    modelData->numEvents = symbols.getEventAttributes().size();

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

    assert (modelDataBaseSize + offset * sizeof(double) == modelDataSize  &&
            "LLVMModelData size not equal to base size + data");

    // allocate the stoichiometry matrix
    const std::vector<uint> &stoichRowIndx = symbols.getStoichRowIndx();
    const std::vector<uint> &stoichColIndx = symbols.getStoichColIndx();
    std::vector<double> stoichValues(stoichRowIndx.size(), 0);

    modelData->stoichiometry = rr::csr_matrix_new(numIndFloatingSpecies, numReactions,
            stoichRowIndx, stoichColIndx, stoichValues);

    // make a copy of the random object
    modelData->random = random ? new Random(*random) : 0;

    return modelData;
}

} /* namespace rrllvm */


