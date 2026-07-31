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

            // MultiSpeciesReference-typed cells are handled separately below:
            // nz.id/nz.type here are the PRIMARY colliding reference's, but
            // the cell can hold contributions from other references too, so
            // it can't just be overwritten wholesale from a single reference's
            // rate rule.
            if (nz.id.empty() || nz.type == LLVMModelDataSymbols::SpeciesReferenceType::MultiSpeciesReference
                    || !dataSymbols.hasRateRule(nz.id))
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

        // restore each rate-rule-governed MultiSpeciesReference individually,
        // by the delta between its current value and its frozen init value --
        // other references sharing the same cell (rule-governed or not) are
        // left alone.
        {
            Value *stoichEP = mdbuilder.createGEP(Stoichiometry);
            Value *stoich = builder.CreateLoad(stoichEP->getType()->getPointerElementType(), stoichEP, "stoichiometry");

            for (size_t i = 0; i < dataSymbols.getMultiSpeciesReferenceSize(); ++i)
            {
                const LLVMModelDataSymbols::SpeciesReferenceInfo &info =
                        dataSymbols.getMultiSpeciesReferenceInfo(static_cast<int>(i));

                if (!dataSymbols.hasRateRule(info.id))
                {
                    continue;
                }

                Value *aliasGEP = mdbuilder.createGEP(MultiSpeciesReferencesAlias,
                        static_cast<unsigned>(i), info.id);
                Value *oldRaw = builder.CreateLoad(aliasGEP->getType()->getPointerElementType(),
                        aliasGEP, info.id + "_old");

                Value *initAliasGEP = mdbuilder.createGEP(MultiSpeciesReferencesInitAlias,
                        static_cast<unsigned>(i), info.id);
                Value *initRaw = builder.CreateLoad(initAliasGEP->getType()->getPointerElementType(),
                        initAliasGEP, info.id + "_init");

                Value *row = ConstantInt::get(Type::getInt32Ty(context), info.row, true);
                Value *col = ConstantInt::get(Type::getInt32Ty(context), info.column, true);
                Value *oldCell = ModelDataIRBuilder::createCSRMatrixGetNZ(builder, stoich, row, col);

                Value *delta = builder.CreateFSub(initRaw, oldRaw, "reset_delta_" + info.id);

                // info.type has been retroactively overwritten to
                // MultiSpeciesReference on collision, so use the preserved
                // original role instead.
                LLVMModelDataSymbols::SpeciesReferenceType role =
                        dataSymbols.getMultiSpeciesReferenceRole(static_cast<int>(i));
                if (role == LLVMModelDataSymbols::SpeciesReferenceType::Reactant)
                {
                    Value *negOne = ConstantFP::get(builder.getContext(), APFloat(-1.0));
                    delta = builder.CreateFMul(negOne, delta, "neg_reset_delta_" + info.id);
                }

                Value *newCell = builder.CreateFAdd(oldCell, delta, "reset_new_cell_" + info.id);
                ModelDataIRBuilder::createCSRMatrixSetNZ(builder, stoich, row, col, newCell, info.id);

                mdbuilder.createRateRuleValueStore(info.id, initRaw);
                builder.CreateStore(initRaw, aliasGEP);
            }
        }

        builder.CreateRetVoid();

        return verifyFunction();
    }

} /* namespace rrllvm */
