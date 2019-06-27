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
#include "llvm/IRReader/IRReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Support/DynamicLibrary.h"
#include "source/llvm/SBMLSupportFunctions.h"
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

//    delete symbols;
    // the exe engine owns all the functions
    delete executionEngine;
    delete context;
    delete random;
    delete errStr;
}

void ModelResources::saveState(std::ostream& out) const
{
	symbols->saveState(out);
	std::string moduleStr;
	llvm::raw_string_ostream moduleSStream(moduleStr);
	//llvm::WriteBitcodeToFile(module.get(), moduleSStream);
	moduleSStream << *module;	
	rr::saveBinary(out, moduleStr);
}

void ModelResources::addGlobalMapping(const llvm::GlobalValue *gv, void *addr)
{
	llvm::sys::DynamicLibrary::AddSymbol(gv->getName(), addr);
	executionEngine->addGlobalMapping(gv, addr);
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

    addGlobalMapping(ModelDataIRBuilder::getCSRMatrixSetNZDecl(module), (void*)rr::csr_matrix_set_nz);
    addGlobalMapping(ModelDataIRBuilder::getCSRMatrixGetNZDecl(module), (void*)rr::csr_matrix_get_nz);
   // addGlobalMapping(LLVMModelDataIRBuilderTesting::getDispIntDecl(module), (void*)rrllvm::dispInt);
    //addGlobalMapping(LLVMModelDataIRBuilderTesting::getDispDoubleDecl(module), (void*)rrllvm::dispDouble);
    //addGlobalMapping(LLVMModelDataIRBuilderTesting::getDispCharDecl(module), (void*)rrllvm:dispChar);

    // AST_FUNCTION_ARCCOT:
	llvm::RTDyldMemoryManager::getSymbolAddressInProcess("arccot");
    addGlobalMapping(
            module->getFunction("arccot"),
			        (void*) sbmlsupport::arccot);

    addGlobalMapping(
            module->getFunction("rr_arccot_negzero"),
                        (void*) sbmlsupport::arccot_negzero);

    // AST_FUNCTION_ARCCOTH:
    addGlobalMapping(
            module->getFunction("arccoth"),
                        (void*) sbmlsupport::arccoth);

    // AST_FUNCTION_ARCCSC:
    addGlobalMapping(
            module->getFunction("arccsc"),
                        (void*) sbmlsupport::arccsc);

    // AST_FUNCTION_ARCCSCH:
    addGlobalMapping(
            module->getFunction("arccsch"),
                        (void*) sbmlsupport::arccsch);

    // AST_FUNCTION_ARCSEC:
    addGlobalMapping(
            module->getFunction("arcsec"),
                        (void*) sbmlsupport::arcsec);

    // AST_FUNCTION_ARCSECH:
    addGlobalMapping(
            module->getFunction("arcsech"),
                        (void*) sbmlsupport::arcsech);

    // AST_FUNCTION_COT:
    addGlobalMapping(
            module->getFunction("cot"),
                        (void*) sbmlsupport::cot);

    // AST_FUNCTION_COTH:
    addGlobalMapping(
            module->getFunction("coth"),
                        (void*) sbmlsupport::coth);

    // AST_FUNCTION_CSC:
    addGlobalMapping(
            module->getFunction("csc"),
                        (void*) sbmlsupport::csc);

    // AST_FUNCTION_CSCH:
    addGlobalMapping(
            module->getFunction("csch"),
                        (void*) sbmlsupport::csch);

    // AST_FUNCTION_FACTORIAL:
    addGlobalMapping(
            module->getFunction("rr_factoriali"),
                        (void*) sbmlsupport::factoriali);

    addGlobalMapping(
            module->getFunction("rr_factoriald"),
                        (void*) sbmlsupport::factoriald);

    // AST_FUNCTION_LOG:
    addGlobalMapping(
            module->getFunction("rr_logd"),
                        (void*) sbmlsupport::logd);

    // AST_FUNCTION_ROOT:
    addGlobalMapping(
            module->getFunction("rr_rootd"),
                        (void*) sbmlsupport::rootd);

    // AST_FUNCTION_SEC:
    addGlobalMapping(
            module->getFunction("sec"),
                        (void*) sbmlsupport::sec);

    // AST_FUNCTION_SECH:
    addGlobalMapping(
            module->getFunction("sech"),
                        (void*) sbmlsupport::sech);

    // AST_FUNCTION_ARCCOSH:
    addGlobalMapping(
            module->getFunction("arccosh"),
                        (void*)static_cast<double (*)(double)>(acosh));

    // AST_FUNCTION_ARCSINH:
    addGlobalMapping(
            module->getFunction("arcsinh"),
                        (void*)static_cast<double (*)(double)>(asinh));

    // AST_FUNCTION_ARCTANH:
    addGlobalMapping(
            module->getFunction("arctanh"),
                        (void*)static_cast<double (*)(double)>(atanh));

    // AST_FUNCTION_QUOTIENT:
    executionEngine->addGlobalMapping(
            module->getFunction("quotient"),
                        (void*)sbmlsupport::quotient);

    // AST_FUNCTION_MAX:
    executionEngine->addGlobalMapping(
        module->getFunction("rr_max"),
            (void*) sbmlsupport::max);

    // AST_FUNCTION_MIN:
    executionEngine->addGlobalMapping(
        module->getFunction("rr_min"),
            (void*) sbmlsupport::min);
}


void ModelResources::loadState(std::istream& in, uint modelGeneratorOpt) 
{
	std::string *engineBuilderErrStr = new std::string();
	symbols = new LLVMModelDataSymbols(in);
	std::string moduleStr;
	rr::loadBinary(in, moduleStr);
	context = new llvm::LLVMContext();
	auto memBuffer(llvm::MemoryBuffer::getMemBuffer(moduleStr));
	//auto expected = llvm::parseBitcodeFile(*memBuffer, *context);
	llvm::SMDiagnostic sm;
	module = llvm::parseIR(*memBuffer, sm, *context).release();
	llvm::Function* f = module->getFunction("evalInitialConditions");
	llvm::StructType *structType = module->getTypeByName("rr_LLVMModelData");
	auto emptyModule = std::unique_ptr<llvm::Module>(new llvm::Module("Module test", *context));
	
	llvm::EngineBuilder engineBuilder(std::move(emptyModule));
	engineBuilder.setErrorStr(engineBuilderErrStr)
		.setMCJITMemoryManager(std::unique_ptr<llvm::SectionMemoryManager>(new llvm::SectionMemoryManager()));
    llvm::InitializeNativeTarget();
	llvm::InitializeNativeTargetAsmPrinter();

	executionEngine = engineBuilder.create();
	addGlobalMappings();
	llvm::Module* tempModule = module;
	executionEngine->addModule(std::unique_ptr<llvm::Module>(tempModule));
	//llvm::Function *f2 = module->getFunction("rr_csr_matrix_set_nz");
	//llvm::sys::DynamicLibrary::AddSymbol(f2->getName(), (void*)rr::csr_matrix_set_nz);
	//executionEngine->addGlobalMapping(f2, (void*)rr::csr_matrix_set_nz);
	auto dataLayout = executionEngine->getDataLayout();
	executionEngine->finalizeObject();
    //register targets ???
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
