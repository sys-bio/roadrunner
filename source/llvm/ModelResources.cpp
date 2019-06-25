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
	module.release();
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

    addGlobalMapping(ModelDataIRBuilder::getCSRMatrixSetNZDecl(module.get()), (void*)rr::csr_matrix_set_nz);
    addGlobalMapping(ModelDataIRBuilder::getCSRMatrixGetNZDecl(module.get()), (void*)rr::csr_matrix_get_nz);
   // addGlobalMapping(LLVMModelDataIRBuilderTesting::getDispIntDecl(module.get()), (void*)rrllvm::dispInt);
    //addGlobalMapping(LLVMModelDataIRBuilderTesting::getDispDoubleDecl(module.get()), (void*)rrllvm::dispDouble);
    //addGlobalMapping(LLVMModelDataIRBuilderTesting::getDispCharDecl(module.get()), (void*)rrllvm:dispChar);

    // AST_FUNCTION_ARCCOT:
	llvm::RTDyldMemoryManager::getSymbolAddressInProcess("arccot");
    addGlobalMapping(
            createGlobalMappingFunction("arccot", 
                FunctionType::get(double_type, args_d1, false), module.get()),
			        (void*) sbmlsupport::arccot);

    addGlobalMapping(
            createGlobalMappingFunction("rr_arccot_negzero",
                    FunctionType::get(double_type, args_d1, false), module.get()),
                        (void*) sbmlsupport::arccot_negzero);

    // AST_FUNCTION_ARCCOTH:
    addGlobalMapping(
            createGlobalMappingFunction("arccoth",
                    FunctionType::get(double_type, args_d1, false), module.get()),
                        (void*) sbmlsupport::arccoth);

    // AST_FUNCTION_ARCCSC:
    addGlobalMapping(
            createGlobalMappingFunction("arccsc",
                    FunctionType::get(double_type, args_d1, false), module.get()),
                        (void*) sbmlsupport::arccsc);

    // AST_FUNCTION_ARCCSCH:
    addGlobalMapping(
            createGlobalMappingFunction("arccsch",
                    FunctionType::get(double_type, args_d1, false), module.get()),
                        (void*) sbmlsupport::arccsch);

    // AST_FUNCTION_ARCSEC:
    addGlobalMapping(
            createGlobalMappingFunction("arcsec",
                    FunctionType::get(double_type, args_d1, false), module.get()),
                        (void*) sbmlsupport::arcsec);

    // AST_FUNCTION_ARCSECH:
    addGlobalMapping(
            createGlobalMappingFunction("arcsech",
                    FunctionType::get(double_type, args_d1, false), module.get()),
                        (void*) sbmlsupport::arcsech);

    // AST_FUNCTION_COT:
    addGlobalMapping(
            createGlobalMappingFunction("cot",
                    FunctionType::get(double_type, args_d1, false), module.get()),
                        (void*) sbmlsupport::cot);

    // AST_FUNCTION_COTH:
    addGlobalMapping(
            createGlobalMappingFunction("coth",
                    FunctionType::get(double_type, args_d1, false), module.get()),
                        (void*) sbmlsupport::coth);

    // AST_FUNCTION_CSC:
    addGlobalMapping(
            createGlobalMappingFunction("csc",
                    FunctionType::get(double_type, args_d1, false), module.get()),
                        (void*) sbmlsupport::csc);

    // AST_FUNCTION_CSCH:
    addGlobalMapping(
            createGlobalMappingFunction("csch",
                    FunctionType::get(double_type, args_d1, false), module.get()),
                        (void*) sbmlsupport::csch);

    // AST_FUNCTION_FACTORIAL:
    addGlobalMapping(
            createGlobalMappingFunction("rr_factoriali",
                    FunctionType::get(int_type, args_i1, false), module.get()),
                        (void*) sbmlsupport::factoriali);

    addGlobalMapping(
            createGlobalMappingFunction("rr_factoriald",
                    FunctionType::get(double_type, args_d1, false), module.get()),
                        (void*) sbmlsupport::factoriald);

    // AST_FUNCTION_LOG:
    addGlobalMapping(
            createGlobalMappingFunction("rr_logd",
                    FunctionType::get(double_type, args_d2, false), module.get()),
                        (void*) sbmlsupport::logd);

    // AST_FUNCTION_ROOT:
    addGlobalMapping(
            createGlobalMappingFunction("rr_rootd",
                    FunctionType::get(double_type, args_d2, false), module.get()),
                        (void*) sbmlsupport::rootd);

    // AST_FUNCTION_SEC:
    addGlobalMapping(
            createGlobalMappingFunction("sec",
                    FunctionType::get(double_type, args_d1, false), module.get()),
                        (void*) sbmlsupport::sec);

    // AST_FUNCTION_SECH:
    addGlobalMapping(
            createGlobalMappingFunction("sech",
                    FunctionType::get(double_type, args_d1, false), module.get()),
                        (void*) sbmlsupport::sech);

    // AST_FUNCTION_ARCCOSH:
    addGlobalMapping(
            createGlobalMappingFunction("arccosh",
                    FunctionType::get(double_type, args_d1, false), module.get()),
                        (void*)static_cast<double (*)(double)>(acosh));

    // AST_FUNCTION_ARCSINH:
    addGlobalMapping(
            createGlobalMappingFunction("arcsinh",
                    FunctionType::get(double_type, args_d1, false), module.get()),
                        (void*)static_cast<double (*)(double)>(asinh));

    // AST_FUNCTION_ARCTANH:
    addGlobalMapping(
            createGlobalMappingFunction("arctanh",
                    FunctionType::get(double_type, args_d1, false), module.get()),
                        (void*)static_cast<double (*)(double)>(atanh));

    // AST_FUNCTION_QUOTIENT:
    executionEngine->addGlobalMapping(
            createGlobalMappingFunction("quotient",
                    FunctionType::get(double_type, args_d2, false), module.get()),
                        (void*)sbmlsupport::quotient);

    // AST_FUNCTION_MAX:
    executionEngine->addGlobalMapping(
        createGlobalMappingFunction("rr_max",
            FunctionType::get(double_type, args_d2, false), module.get()),
            (void*) sbmlsupport::max);

    // AST_FUNCTION_MIN:
    executionEngine->addGlobalMapping(
        createGlobalMappingFunction("rr_min",
            FunctionType::get(double_type, args_d2, false), module.get()),
            (void*) sbmlsupport::min);
}

llvm::Function* ModelResources::createGlobalMappingFunction(const char* funcName,
        llvm::FunctionType *funcType, llvm::Module *module)
{
	return module->getFunction(funcName);
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
	module = llvm::parseIR(*memBuffer, sm, *context);
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
	executionEngine->addModule(std::move(module));
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


} /* namespace rrllvm */
