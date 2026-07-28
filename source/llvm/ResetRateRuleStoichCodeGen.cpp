/*
 * ResetRateRuleStoichCodeGen.cpp
 */
#pragma hdrstop

#include "ResetRateRuleStoichCodeGen.h"
#include "LLVMException.h"
#include "ASTNodeCodeGen.h"
#include "SBMLInitialValueSymbolResolver.h"
#include "rrLogger.h"

#include <sbml/math/ASTNode.h>
#include <Poco/Logger.h>

namespace rrllvm {
    using namespace rr;
    using namespace llvm;
    using namespace libsbml;


    const char *ResetRateRuleStoichCodeGen::FunctionName = "resetRateRuleStoich";

    ResetRateRuleStoichCodeGen::ResetRateRuleStoichCodeGen(
            const ModelGeneratorContext &mgc) :
            CodeGenBase<ResetRateRuleStoichCodeGen_FunctionPtr>(mgc) {
    }

    ResetRateRuleStoichCodeGen::~ResetRateRuleStoichCodeGen() {
    }

    Value *ResetRateRuleStoichCodeGen::codeGen() {
        Value *modelData = 0;

        codeGenVoidModelDataHeader(FunctionName, modelData);

        // Reads declared/SBML values, same source evalInitialConditions uses,
        // so this stays consistent with whatever the model was last
        // (re)compiled with (e.g. after a setValue("init(...)") call, which
        // mutates the SBML and regenerates the model).
        SBMLInitialValueSymbolResolver initialValueResolver(modelData, modelGenContext);
        ModelDataIRBuilder mdbuilder(modelData, dataSymbols, builder);
        ASTNodeCodeGen astCodeGen(builder, initialValueResolver, modelGenContext, modelData);

        std::list<LLVMModelDataSymbols::SpeciesReferenceInfo> stoichEntries =
                dataSymbols.getStoichiometryList();

        for (std::list<LLVMModelDataSymbols::SpeciesReferenceInfo>::iterator i =
                stoichEntries.begin(); i != stoichEntries.end(); i++)
        {
            const LLVMModelDataSymbols::SpeciesReferenceInfo& nz = *i;

            if (nz.id.empty() || !dataSymbols.hasRateRule(nz.id))
            {
                continue;
            }

            const ASTNode *node = modelSymbols.createStoichiometryNode(nz.row, nz.column);
            Value *stoichValue = astCodeGen.codeGenDouble(node);
            delete node;

            mdbuilder.createRateRuleValueStore(nz.id, stoichValue);
            mdbuilder.createStoichiometryStore(nz.row, nz.column, stoichValue, nz.id);
        }

        builder.CreateRetVoid();

        return verifyFunction();
    }

} /* namespace rrllvm */
