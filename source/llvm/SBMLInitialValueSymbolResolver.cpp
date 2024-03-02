/*
 * SBMLInitialValueSymbolResolver.cpp
 *
 *  Created on: Jul 25, 2013
 *      Author: andy
 */
#pragma hdrstop
#include "SBMLInitialValueSymbolResolver.h"
#include "ASTNodeCodeGen.h"
#include "LLVMException.h"
#include "FunctionResolver.h"
#include <sbml/Model.h>


using namespace libsbml;
using namespace llvm;

namespace rrllvm
{

SBMLInitialValueSymbolResolver::SBMLInitialValueSymbolResolver(
        llvm::Value *modelData,
        const ModelGeneratorContext& ctx) :
        LoadSymbolResolverBase(ctx, modelData), modelData(modelData)
{
}

llvm::Value* SBMLInitialValueSymbolResolver::loadSymbolValue(
        const std::string& symbol,
        const llvm::ArrayRef<llvm::Value*>& args)
{
    /*************************************************************************/
    /* time */
    /*************************************************************************/
    if (symbol.compare(SBML_TIME_SYMBOL) == 0)
    {
        return ConstantFP::get(builder.getContext(), APFloat(0.0));
    }

    /*************************************************************************/
    /* Function */
    /*************************************************************************/
    {
        Value *funcVal =
            FunctionResolver(*this, modelData, modelGenContext).loadSymbolValue(symbol, args);

        if (funcVal)
        {
            return funcVal;
        }
    }

    /*************************************************************************/
    /* AssignmentRule */
    /*************************************************************************/
    if (!modelDataSymbols.isConservedMoietySpecies(symbol))
    {
        SymbolForest::ConstIterator i =
                modelSymbols.getAssigmentRules().find(symbol);
        if (i != modelSymbols.getAssigmentRules().end())
        {
            recursiveSymbolPush(symbol);
            Value* result = ASTNodeCodeGen(builder, *this, modelGenContext, modelData).codeGenDouble(i->second);
            recursiveSymbolPop();
            return result;
        }
    }

    /*************************************************************************/
    /* Initial Value */
    /*************************************************************************/
    {
        SymbolForest::ConstIterator i =
                modelSymbols.getInitialValues().find(symbol);

        if (i != modelSymbols.getInitialValues().end())
        {
            return ASTNodeCodeGen(builder, *this, modelGenContext, modelData).codeGenDouble(i->second);
        }
    }

    /*************************************************************************/
    /* Reaction Rate */
    /*************************************************************************/
    const Reaction* reaction = model->getReaction(symbol);
    if (reaction)
    {
        return loadReactionRate(reaction);
    }

    LLVMModelDataSymbols* modelDataSymbolsPtr = const_cast<LLVMModelDataSymbols*>(&modelDataSymbols);

    if (modelDataSymbols.isNamedSpeciesReference(symbol))
    {
        const LLVMModelDataSymbols::SpeciesReferenceInfo& info =
            modelDataSymbolsPtr->getNamedSpeciesReferenceInfo(symbol);

        ModelDataIRBuilder mdbuilder(modelData, modelDataSymbols, builder);
        Value* value = mdbuilder.createStoichiometryLoad(info.row, info.column, symbol);

        if (info.type == LLVMModelDataSymbols::MultiReactantProduct)
        {
            std::string msg = "Mutable stochiometry for species which appear "
                "multiple times in a single reaction is not currently "
                "supported, species reference id: ";
            msg += symbol;
            throw_llvm_exception(msg);
        }

        if (info.type == LLVMModelDataSymbols::Reactant)
        {
            // it's consumed in the reaction, so has a negative in the stoich
            // matrix
            Value* negOne = ConstantFP::get(builder.getContext(), APFloat(-1.0));
            negOne->setName("neg_one");
            value = builder.CreateFMul(negOne, value, "neg_" + symbol);
        }

        return cacheValue(symbol, args, value);
    }

    std::string msg = "Could not find requested symbol \'";
    msg += symbol;
    msg += "\' in the model";
    throw_llvm_exception(msg);

    return 0;
}

} /* namespace rr */


