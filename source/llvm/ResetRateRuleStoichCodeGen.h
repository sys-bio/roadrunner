/*
 * ResetRateRuleStoichCodeGen.h
 */

#ifndef RESETRATERULESTOICHCODEGEN_H_
#define RESETRATERULESTOICHCODEGEN_H_

#include "CodeGenBase.h"
#include "ModelGeneratorContext.h"
#include "ModelDataIRBuilder.h"
#include <sbml/Model.h>

namespace rrllvm
{

    typedef void (*ResetRateRuleStoichCodeGen_FunctionPtr)(LLVMModelData*);

    /** @class ResetRateRuleStoichCodeGen
    * Restores every rate-rule-governed named stoichiometry (and the rate rule's
    * own state) to the value declared in the SBML, without requiring the full
    * evalInitialConditions pass. This is what lets a plain reset() (TIME | RATE
    * | FLOATING) restore rate-rule-controlled stoichiometries the same way it
    * already restores rate-rule-controlled compartments, species, and
    * parameters.
    */
    class ResetRateRuleStoichCodeGen :
        public CodeGenBase<ResetRateRuleStoichCodeGen_FunctionPtr>
    {
    public:
        ResetRateRuleStoichCodeGen(const ModelGeneratorContext& mgc);
        virtual ~ResetRateRuleStoichCodeGen();

        llvm::Value* codeGen();

        static const char* FunctionName;
    };

} /* namespace rrllvm */
#endif /* RESETRATERULESTOICHCODEGEN_H_ */
