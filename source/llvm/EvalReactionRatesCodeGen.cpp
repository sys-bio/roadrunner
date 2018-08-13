/*
 * EvalReactionRatesCodeGen.cpp
 *
 *  Created on: Jul 15, 2013
 *      Author: andy
 */
#pragma hdrstop
#include "EvalReactionRatesCodeGen.h"
#include "LLVMException.h"
#include "ASTNodeCodeGen.h"
#include "ASTNodeFactory.h"
#include "ModelDataSymbolResolver.h"
#include "KineticLawParameterResolver.h"
#include "rrLogger.h"
#include <sbml/math/ASTNode.h>
#include <sbml/math/FormulaFormatter.h>
#include <Poco/Logger.h>


using namespace libsbml;
using namespace llvm;
using namespace std;


namespace rrllvm
{

const char* EvalReactionRatesCodeGen::FunctionName = "evalReactionRates";

EvalReactionRatesCodeGen::EvalReactionRatesCodeGen(
        const ModelGeneratorContext &mgc) :
        CodeGenBase<EvalReactionRates_FunctionPtr>(mgc)
{
}

EvalReactionRatesCodeGen::~EvalReactionRatesCodeGen()
{
}

Value* EvalReactionRatesCodeGen::codeGen()
{
    // single arg type of LLVMModelData*
    llvm::Type *argTypes[] = {
        llvm::PointerType::get(
            ModelDataIRBuilder::getStructType(module), 0)
    };

    const char *argNames[] = { "modelData" };

    llvm::Value *args[] = { 0 };

    codeGenHeader(FunctionName, llvm::Type::getDoubleTy(context),
                argTypes, argNames, args);

    Value *modelData = args[0];

    ModelDataLoadSymbolResolver resolver(modelData, modelGenContext);
    ModelDataIRBuilder mdbuilder(modelData, dataSymbols, builder);
    ASTNodeFactory nodes;

    // iterate through all of the reaction, and generate code based on thier
    // kinetic rules.

    // currently, our AST does not support assigments, so we do it
    // here.

    const ListOfReactions *reactions = model->getListOfReactions();

    for (int i = 0; i < reactions->size(); ++i)
    {
        const Reaction *r = reactions->get(i);
		set<string> arrayedReactions = dataSymbols.getArrayedElements(r->getId(), 4);
		for (set<string>::const_iterator it = arrayedReactions.begin(), jt = arrayedReactions.end(); it != jt; it++)
		{
			Value *value = resolver.loadReactionRate(r, *it);
			mdbuilder.createReactionRateStore(*it, value);
		}
    }

    Value *conversionFactor = 0;

    if (model->isSetConversionFactor() && model->getConversionFactor().length() > 0)
    {
        conversionFactor = resolver.loadSymbolValue(model->getConversionFactor());
    }
    else
    {
        conversionFactor = ConstantFP::get(Type::getDoubleTy(context), 1.0);
    }

    builder.CreateRet(conversionFactor);

    return verifyFunction();
}


} /* namespace rr */
