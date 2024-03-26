/*

 * ModelResources.h
 *
 *  Created on: Aug 28, 2013
 *      Author: andy
 */

#ifndef CACHEDMODEL_H_
#define CACHEDMODEL_H_

#include "LLVMExecutableModel.h"
#ifdef _MSC_VER
#pragma warning(disable: 4146)
#pragma warning(disable: 4141)
#pragma warning(disable: 4267)
#pragma warning(disable: 4624)
#endif

#include "llvm/Jit.h"

#ifdef _MSC_VER
#pragma warning(default: 4146)
#pragma warning(default: 4141)
#pragma warning(default: 4267)
#pragma warning(default: 4624)
#endif

namespace rr {
    class ExecutableModel;
}
namespace rrllvm
{

    class Jit;

    /**@cond PRIVATE */

    class ModelResources
    {
    public:

        ModelResources();
        ~ModelResources();

        void saveState(std::ostream& out) const;
        void loadState(std::istream& in, uint loadOpt);


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
        GetPiecewiseTriggerCodeGen::FunctionPtr getPiecewiseTriggerPtr;
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

        SetBoundarySpeciesInitConcentrationCodeGen::FunctionPtr setBoundarySpeciesInitConcentrationsPtr;
        GetBoundarySpeciesInitConcentrationCodeGen::FunctionPtr getBoundarySpeciesInitConcentrationsPtr;

        SetBoundarySpeciesInitAmountCodeGen::FunctionPtr setBoundarySpeciesInitAmountsPtr;
        GetBoundarySpeciesInitAmountCodeGen::FunctionPtr getBoundarySpeciesInitAmountsPtr;

        GetCompartmentInitVolumeCodeGen::FunctionPtr getCompartmentInitVolumesPtr;
        SetCompartmentInitVolumeCodeGen::FunctionPtr setCompartmentInitVolumesPtr;

        GetGlobalParameterInitValueCodeGen::FunctionPtr getGlobalParameterInitValuePtr;
        SetGlobalParameterInitValueCodeGen::FunctionPtr setGlobalParameterInitValuePtr;

        /**
         * Public Member variables
         */

        const LLVMModelDataSymbols* symbols;
        std::unique_ptr<llvm::LLVMContext> context;
        std::unique_ptr<llvm::ExecutionEngine> executionEngine;

        llvm::Module* module = nullptr;

        std::string moduleStr;
        std::string sbmlMD5;
        const class Random* random;
        std::unique_ptr<std::string> errStr;

        std::unique_ptr<Jit> jit;

        //private:
        //	void addGlobalMapping(std::string name, void*);
        //	void addGlobalMappings();
    };

    /**@endcond PRIVATE */
} /* namespace rrllvm */
#endif /* CACHEDMODEL_H_ */
