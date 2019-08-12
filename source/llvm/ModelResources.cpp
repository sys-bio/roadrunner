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
#include <rrStringUtils.h>
#undef min
#undef max
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Support/DynamicLibrary.h"
#include "source/llvm/SBMLSupportFunctions.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "rrRoadRunnerOptions.h"

using rr::Logger;
using rr::getLogger;

namespace rrllvm
{

ModelResources::ModelResources() :
        symbols(0), executionEngine(0), context(0), random(0), errStr(0)
{
    // the reset of the ivars are assigned by the generator,
    // and in an exception they are not, does not matter as
    // we don't have to delete them.
}

ModelResources::~ModelResources()
{
    Log(Logger::LOG_DEBUG) << __FUNC__;

    if (errStr && errStr->size() > 0)
    {
        Log(Logger::LOG_WARNING) << "Non-empty LLVM ExecutionEngine error string: " << *errStr;
    }

    // delete symbols;

    // the exe engine owns all the functions
    delete executionEngine;
    delete context;
    delete random;
    delete errStr;
}

void ModelResources::saveState(std::ostream& out) const
{
	symbols->saveState(out);
 
	rr::saveBinary(out, moduleStr);
}

void ModelResources::addGlobalMapping(std::string name, void *addr)
{
	llvm::sys::DynamicLibrary::AddSymbol(name, addr);
}

void ModelResources::addGlobalMappings()
{
    using namespace llvm;
    Type *double_type = Type::getDoubleTy(*context);
    Type *int_type = Type::getInt32Ty(*context);
    Type* args_i1[] = { int_type };
    Type* args_d1[] = { double_type };
    Type* args_d2[] = { double_type, double_type };

    llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);

    addGlobalMapping("rr_csr_matrix_set_nz", (void*)rr::csr_matrix_set_nz);

    addGlobalMapping("rr_csr_matrix_get_nz", (void*)rr::csr_matrix_get_nz);

    // AST_FUNCTION_ARCCOT:
	llvm::RTDyldMemoryManager::getSymbolAddressInProcess("arccot");
    addGlobalMapping(
            "arccot",
			        (void*) sbmlsupport::arccot);

    addGlobalMapping(
            "rr_arccot_negzero",
                        (void*) sbmlsupport::arccot_negzero);

    // AST_FUNCTION_ARCCOTH:
    addGlobalMapping(
            "arccoth",
                        (void*) sbmlsupport::arccoth);

    // AST_FUNCTION_ARCCSC:
    addGlobalMapping(
            "arccsc",
                        (void*) sbmlsupport::arccsc);

    // AST_FUNCTION_ARCCSCH:
            "arccsch",
                        (void*) sbmlsupport::arccsch);

    // AST_FUNCTION_ARCSEC:
    addGlobalMapping(
            "arcsec",
                        (void*) sbmlsupport::arcsec);

    // AST_FUNCTION_ARCSECH:
    addGlobalMapping(
            "arcsech",
                        (void*) sbmlsupport::arcsech);

    // AST_FUNCTION_COT:
    addGlobalMapping(
            "cot",
                        (void*) sbmlsupport::cot);

    // AST_FUNCTION_COTH:
    addGlobalMapping(
            "coth",
                        (void*) sbmlsupport::coth);

    // AST_FUNCTION_CSC:
    addGlobalMapping(
            "csc",
                        (void*) sbmlsupport::csc);

    // AST_FUNCTION_CSCH:
    addGlobalMapping(
            "csch",
                        (void*) sbmlsupport::csch);

    // AST_FUNCTION_FACTORIAL:
    addGlobalMapping(
            "rr_factoriali",
                        (void*) sbmlsupport::factoriali);

    addGlobalMapping(
            "rr_factoriald",
                        (void*) sbmlsupport::factoriald);

    // AST_FUNCTION_LOG:
    addGlobalMapping(
            "rr_logd",
                        (void*) sbmlsupport::logd);

    // AST_FUNCTION_ROOT:
    addGlobalMapping(
            "rr_rootd",
                        (void*) sbmlsupport::rootd);

    // AST_FUNCTION_SEC:
    addGlobalMapping(
            "sec",
                        (void*) sbmlsupport::sec);

    // AST_FUNCTION_SECH:
    addGlobalMapping(
            "sech",
                        (void*) sbmlsupport::sech);

    // AST_FUNCTION_ARCCOSH:
    addGlobalMapping(
            "arccosh",
                        (void*)static_cast<double (*)(double)>(acosh));

    // AST_FUNCTION_ARCSINH:
    addGlobalMapping(
            "arcsinh",
                        (void*)static_cast<double (*)(double)>(asinh));

    // AST_FUNCTION_ARCTANH:
    addGlobalMapping(
            "arctanh",
                        (void*)static_cast<double (*)(double)>(atanh));

    // AST_FUNCTION_QUOTIENT:
	addGlobalMapping("quotient", (void*)sbmlsupport::quotient);
    // AST_FUNCTION_MAX:
	addGlobalMapping("rr_max", (void*)sbmlsupport::max);
    // AST_FUNCTION_MIN:
	addGlobalMapping("rr_min", (void*)sbmlsupport::min);
}


void ModelResources::loadState(std::istream& in, uint modelGeneratorOpt) 
{
	if (symbols)
		delete symbols;
	//load the model data symbols from the stream
	symbols = new LLVMModelDataSymbols(in);
	//Get the object file from the input stream
	rr::loadBinary(in, moduleStr);
	//Set up the llvm context 
	if (context)
		delete context;
	context = new llvm::LLVMContext();
	//Set up a buffer to read the object code from
	auto memBuffer(llvm::MemoryBuffer::getMemBuffer(moduleStr));
    
	llvm::Expected<std::unique_ptr<llvm::object::ObjectFile> > objectFileExpected =
		llvm::object::ObjectFile::createObjectFile(llvm::MemoryBufferRef(moduleStr, "id"));
	if (!objectFileExpected)
	{
		throw std::invalid_argument("Failed to load object data");
	}
    
	std::unique_ptr<llvm::object::ObjectFile> objectFile(std::move(objectFileExpected.get()));
	
	llvm::object::OwningBinary<llvm::object::ObjectFile> owningObject(std::move(objectFile), std::move(memBuffer));

	//Not sure why, but engineBuilder.create() crashes here if not initialized with an empty module
	auto emptyModule = std::unique_ptr<llvm::Module>(new llvm::Module("Module test", *context));
	llvm::EngineBuilder engineBuilder(std::move(emptyModule));

	//Set the necessary parameters on the engine builder
	std::string engineBuilderErrStr;
	engineBuilder.setErrorStr(&engineBuilderErrStr)
		.setMCJITMemoryManager(std::unique_ptr<llvm::SectionMemoryManager>(new llvm::SectionMemoryManager()));
	
	//We have to call this function before calling engineBuilder.create()
    llvm::InitializeNativeTarget();

	executionEngine = engineBuilder.create();
	
	//Add mappings to the functions that aren't saved in the object file (like sin, cos, factorial)
	addGlobalMappings();

	//Add the object file we loaded to the execution engine
	executionEngine->addObjectFile(std::move(owningObject));
	//Finalize the engine
	executionEngine->finalizeObject();
	//Get the function pointers we need from the execution engine
	evalInitialConditionsPtr = (EvalInitialConditionsCodeGen::FunctionPtr)
		executionEngine->getFunctionAddress("evalInitialConditions");

	evalReactionRatesPtr = (EvalReactionRatesCodeGen::FunctionPtr)
		executionEngine->getFunctionAddress("evalReactionRates");

	getBoundarySpeciesAmountPtr = (GetBoundarySpeciesAmountCodeGen::FunctionPtr)
		executionEngine->getFunctionAddress("getBoundarySpeciesAmount");

	getFloatingSpeciesAmountPtr = (GetFloatingSpeciesAmountCodeGen::FunctionPtr)
		executionEngine->getFunctionAddress("getFloatingSpeciesAmount");

	getBoundarySpeciesConcentrationPtr = (GetBoundarySpeciesConcentrationCodeGen::FunctionPtr)
		executionEngine->getFunctionAddress("getBoundarySpeciesConcentration");

	getFloatingSpeciesConcentrationPtr = (GetFloatingSpeciesAmountCodeGen::FunctionPtr)
		executionEngine->getFunctionAddress("getFloatingSpeciesConcentration");

	getCompartmentVolumePtr = (GetCompartmentVolumeCodeGen::FunctionPtr)
		executionEngine->getFunctionAddress("getCompartmentVolume");

	getGlobalParameterPtr = (GetGlobalParameterCodeGen::FunctionPtr)
		executionEngine->getFunctionAddress("getGlobalParameter");

	evalRateRuleRatesPtr = (EvalRateRuleRatesCodeGen::FunctionPtr)
		executionEngine->getFunctionAddress("evalRateRuleRates");

	getEventTriggerPtr = (GetEventTriggerCodeGen::FunctionPtr)
		executionEngine->getFunctionAddress("getEventTrigger");

	getEventPriorityPtr = (GetEventPriorityCodeGen::FunctionPtr)
		executionEngine->getFunctionAddress("getEventPriority");

	getEventDelayPtr = (GetEventDelayCodeGen::FunctionPtr)
		executionEngine->getFunctionAddress("getEventDelay");

	eventTriggerPtr = (EventTriggerCodeGen::FunctionPtr)
		executionEngine->getFunctionAddress("eventTrigger");

	eventAssignPtr = (EventAssignCodeGen::FunctionPtr)
		executionEngine->getFunctionAddress("eventAssign");

	evalVolatileStoichPtr = (EvalVolatileStoichCodeGen::FunctionPtr)
		executionEngine->getFunctionAddress("evalVolatileStoich");

	evalConversionFactorPtr = (EvalConversionFactorCodeGen::FunctionPtr)
		executionEngine->getFunctionAddress("evalConversionFactor");
	

	if (modelGeneratorOpt & rr::LoadSBMLOptions::READ_ONLY)
	{
		setBoundarySpeciesAmountPtr = 0;
		setBoundarySpeciesConcentrationPtr = 0;
		setFloatingSpeciesConcentrationPtr = 0;
		setCompartmentVolumePtr = 0;
		setFloatingSpeciesAmountPtr = 0;
		setGlobalParameterPtr = 0;
	} 
	else
	{

		setBoundarySpeciesAmountPtr = (SetBoundarySpeciesAmountCodeGen::FunctionPtr)
			executionEngine->getFunctionAddress("setBoundarySpeciesAmount");

		setBoundarySpeciesConcentrationPtr = (SetBoundarySpeciesAmountCodeGen::FunctionPtr)
			executionEngine->getFunctionAddress("setBoundarySpeciesConcentration");

		setFloatingSpeciesConcentrationPtr = (SetBoundarySpeciesAmountCodeGen::FunctionPtr)
			executionEngine->getFunctionAddress("setFloatingSpeciesConcentration");

		setCompartmentVolumePtr = (SetBoundarySpeciesAmountCodeGen::FunctionPtr)
			executionEngine->getFunctionAddress("setCompartmentVolume");

		setFloatingSpeciesAmountPtr = (SetBoundarySpeciesAmountCodeGen::FunctionPtr)
			executionEngine->getFunctionAddress("setFloatingSpeciesAmount");

		setGlobalParameterPtr = (SetGlobalParameterCodeGen::FunctionPtr)
			executionEngine->getFunctionAddress("setGlobalParameter");
	}
    
	if (modelGeneratorOpt & rr::LoadSBMLOptions::MUTABLE_INITIAL_CONDITIONS)
	{

		getFloatingSpeciesInitConcentrationsPtr = (GetBoundarySpeciesAmountCodeGen::FunctionPtr)
			executionEngine->getFunctionAddress("getFloatingSpeciesInitConcentrations");
		setFloatingSpeciesInitConcentrationsPtr = (SetBoundarySpeciesAmountCodeGen::FunctionPtr)
			executionEngine->getFunctionAddress("setFloatingSpeciesInitConcentrations");

		getFloatingSpeciesInitAmountsPtr = (GetBoundarySpeciesAmountCodeGen::FunctionPtr)
			executionEngine->getFunctionAddress("getFloatingSpeciesInitAmounts");
		setFloatingSpeciesInitAmountsPtr = (SetBoundarySpeciesAmountCodeGen::FunctionPtr)
			executionEngine->getFunctionAddress("setFloatingSpeciesInitAmounts");

		getCompartmentInitVolumesPtr = (GetBoundarySpeciesAmountCodeGen::FunctionPtr)
			executionEngine->getFunctionAddress("getCompartmentInitVolumes");
		setCompartmentInitVolumesPtr = (SetBoundarySpeciesAmountCodeGen::FunctionPtr)
			executionEngine->getFunctionAddress("setCompartmentInitVolumes");

		getGlobalParameterInitValuePtr = (GetBoundarySpeciesAmountCodeGen::FunctionPtr)
			executionEngine->getFunctionAddress("getGlobalParameterInitValue");
		setGlobalParameterInitValuePtr = (SetBoundarySpeciesAmountCodeGen::FunctionPtr)
			executionEngine->getFunctionAddress("setGlobalParameterInitValue");
	}
	else 
	{
		getFloatingSpeciesInitConcentrationsPtr = 0;
		setFloatingSpeciesInitConcentrationsPtr = 0;
		getFloatingSpeciesInitAmountsPtr = 0;
		setFloatingSpeciesInitAmountsPtr = 0;
		getCompartmentInitVolumesPtr = 0;
		setCompartmentInitVolumesPtr = 0;
		getGlobalParameterInitValuePtr = 0;
		setGlobalParameterInitValuePtr = 0;
	}
}


} /* namespace rrllvm */
