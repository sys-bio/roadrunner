/*

 * ModelResources.h
 *
 *  Created on: Aug 28, 2013
 *      Author: andy
 */

#ifndef CACHEDMODEL_H_
#define CACHEDMODEL_H_

#include "LLVMExecutableModel.h"

namespace rrllvm
{

class ModelResources
{
public:
    ModelResources();
    ~ModelResources();

	void saveState(std::ostream& out) const;
	void loadState(std::istream& in, uint loadOpt);

    const LLVMModelDataSymbols *symbols;
    llvm::LLVMContext *context;
    llvm::ExecutionEngine *executionEngine;
    llvm::Module *module = 0;
    std::string moduleStr;
    const class Random *random;
    const std::string *errStr;

    EvalInitialConditionsCodeGen::FunctionPtr evalInitialConditionsPtr;
    EvalReactionRatesCodeGen::FunctionPtr evalReactionRatesPtr;
    GetBoundarySpeciesAmountCodeGen::FunctionPtr getBoundarySpeciesAmountPtr;
    GetFloatingSpeciesAmountCodeGen::FunctionPtr getFloatingSpeciesAmountPtr;
    GetBoundarySpeciesConcentrationCodeGen::FunctionPtr getBoundarySpeciesConcentrationPtr;
    GetFloatingSpeciesConcentrationCodeGen::FunctionPtr getFloatingSpeciesConcentrationPtr;
    GetCompartmentVolumeCodeGen::FunctionPtr getCompartmentVolumePtr;
    GetGlobalParameterCodeGen::FunctionPtr getGlobalParameterPtr;
    EvalRateRuleRatesCodeGen::FunctionPtr evalRateRuleRatesPtr;
    GetEventTriggerCodeGen::FunctionPtr getEventTriggerPtr;
    GetEventPriorityCodeGen::FunctionPtr getEventPriorityPtr;
    GetEventDelayCodeGen::FunctionPtr getEventDelayPtr;
    EventTriggerCodeGen::FunctionPtr eventTriggerPtr;
    EventAssignCodeGen::FunctionPtr eventAssignPtr;
    EvalVolatileStoichCodeGen::FunctionPtr evalVolatileStoichPtr;
    EvalConversionFactorCodeGen::FunctionPtr evalConversionFactorPtr;
    SetBoundarySpeciesAmountCodeGen::FunctionPtr setBoundarySpeciesAmountPtr;
    SetFloatingSpeciesAmountCodeGen::FunctionPtr setFloatingSpeciesAmountPtr;
    SetBoundarySpeciesConcentrationCodeGen::FunctionPtr setBoundarySpeciesConcentrationPtr;
    SetFloatingSpeciesConcentrationCodeGen::FunctionPtr setFloatingSpeciesConcentrationPtr;
    SetCompartmentVolumeCodeGen::FunctionPtr setCompartmentVolumePtr;
    SetGlobalParameterCodeGen::FunctionPtr setGlobalParameterPtr;

    // init value accessors
    SetFloatingSpeciesInitConcentrationCodeGen::FunctionPtr setFloatingSpeciesInitConcentrationsPtr;
    GetFloatingSpeciesInitConcentrationCodeGen::FunctionPtr getFloatingSpeciesInitConcentrationsPtr;

    SetFloatingSpeciesInitAmountCodeGen::FunctionPtr setFloatingSpeciesInitAmountsPtr;
    GetFloatingSpeciesInitAmountCodeGen::FunctionPtr getFloatingSpeciesInitAmountsPtr;

    GetCompartmentInitVolumeCodeGen::FunctionPtr getCompartmentInitVolumesPtr;
    SetCompartmentInitVolumeCodeGen::FunctionPtr setCompartmentInitVolumesPtr;

    GetGlobalParameterInitValueCodeGen::FunctionPtr getGlobalParameterInitValuePtr;
    SetGlobalParameterInitValueCodeGen::FunctionPtr setGlobalParameterInitValuePtr;
private:
	void addGlobalMapping(std::string name, void*);
	void addGlobalMappings();
	static llvm::Function* createGlobalMappingFunction(const char* funcName, llvm::FunctionType *funcType, llvm::Module*);
};

} /* namespace rrllvm */
#endif /* CACHEDMODEL_H_ */
