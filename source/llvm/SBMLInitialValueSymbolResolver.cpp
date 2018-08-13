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

using namespace std;
using namespace libsbml;
using namespace llvm;

namespace rrllvm
{

SBMLInitialValueSymbolResolver::SBMLInitialValueSymbolResolver(
        llvm::Value *modelData,
        const ModelGeneratorContext& ctx) :
        LoadSymbolResolverBase(ctx), modelData(modelData)
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
            Value* result = ASTNodeCodeGen(builder, *this).codeGen(i->second);
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
            return ASTNodeCodeGen(builder, *this).codeGen(i->second);
        }
    }

    /*************************************************************************/
    /* Reaction Rate */
    /*************************************************************************/
    const Reaction* reaction = model->getReaction(modelDataSymbols.decodeArrayId(symbol));
    if (reaction)
    {
        return loadReactionRate(reaction, symbol);
    }

    string msg = "Could not find requested symbol \'";
    msg += symbol;
    msg += "\' in the model";
    throw_llvm_exception(msg);

    return 0;
}

} /* namespace rr */


