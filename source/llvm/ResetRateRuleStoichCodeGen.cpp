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

        // Reads back the already-computed initStoichiometry matrix (frozen at
        // load time / the last setInitStoichiometry call), the same source
        // codeGenInitStoichiometry copies from -- rather than re-evaluating
        // the SBML stoichiometry expression, which would ignore any
        // subsequent setInitStoichiometry call.
        ModelDataIRBuilder mdbuilder(modelData, dataSymbols, builder);

        Value *initStoichEP = mdbuilder.createGEP(InitStoichiometry);
        Value *initStoich = builder.CreateLoad(initStoichEP->getType()->getPointerElementType(), initStoichEP, "initStoichiometry");

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

            Value *row = ConstantInt::get(Type::getInt32Ty(context), nz.row, true);
            Value *col = ConstantInt::get(Type::getInt32Ty(context), nz.column, true);
            Value *stoichValue = ModelDataIRBuilder::createCSRMatrixGetNZ(builder, initStoich, row, col);

            // stoichValue is the net, CSR-signed value (negative for a
            // reactant); the rate rule slot holds the reference's own
            // magnitude, so undo that sign flip before storing it there.
            Value *rateRuleSeed = stoichValue;
            if (nz.type == LLVMModelDataSymbols::SpeciesReferenceType::Reactant)
            {
                Value *negOne = ConstantFP::get(builder.getContext(), APFloat(-1.0));
                rateRuleSeed = builder.CreateFMul(negOne, stoichValue, "unneg_" + nz.id);
            }

            mdbuilder.createRateRuleValueStore(nz.id, rateRuleSeed);
            mdbuilder.createStoichiometryStore(nz.row, nz.column, stoichValue, nz.id);
        }

        builder.CreateRetVoid();

        return verifyFunction();
    }

} /* namespace rrllvm */
