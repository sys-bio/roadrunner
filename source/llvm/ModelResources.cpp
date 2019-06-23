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

void ModelResources::loadState(std::istream& in) 
{
	std::string *engineBuilderErrStr = new std::string();
	symbols = new LLVMModelDataSymbols(in);
	std::string moduleStr;
	rr::loadBinary(in, moduleStr);
	context = new llvm::LLVMContext();
	auto memBuffer(llvm::MemoryBuffer::getMemBuffer(moduleStr));
	//auto expected = llvm::parseBitcodeFile(*memBuffer, *context);
	llvm::SMDiagnostic sm;
	auto module = llvm::parseIR(*memBuffer, sm, *context);
	std::vector<std::string> fnames;
	for (auto f = module->functions().begin(); f != module->functions().end(); f++)
	{
		fnames.push_back(f->getName());
	}
	std::string fname = module->functions().begin()->getName();
	//auto module = std::unique_ptr<llvm::Module>(new llvm::Module("Module test", *context));
	llvm::EngineBuilder engineBuilder(std::move(module));
	engineBuilder.setErrorStr(engineBuilderErrStr)
		.setMCJITMemoryManager(std::unique_ptr<llvm::SectionMemoryManager>(new llvm::SectionMemoryManager()));
    llvm::InitializeNativeTarget();
	executionEngine = engineBuilder.create();
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
