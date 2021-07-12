/*
 * EvalConversionFactorCodeGen.cpp
 *
 *  Created on: Aug 25, 2013
 *      Author: andy
 */
#pragma hdrstop
#include "EvalConversionFactorCodeGen.h"

#include "LLVMException.h"
#include "ASTNodeCodeGen.h"
#include "ASTNodeFactory.h"
#include "ModelDataSymbolResolver.h"
#include "rrLogger.h"

#include <vector>
#include <sbml/math/ASTNode.h>
#include <sbml/math/FormulaFormatter.h>
#include <Poco/Logger.h>

namespace rrllvm
{
using namespace rr;
using namespace llvm;
using namespace libsbml;


const char* EvalConversionFactorCodeGen::FunctionName = "evalConversionFactor";

EvalConversionFactorCodeGen::EvalConversionFactorCodeGen(
        const ModelGeneratorContext &mgc) :
        CodeGenBase<EvalConversionFactorCodeGen_FunctionPtr>(mgc)
{
}

EvalConversionFactorCodeGen::~EvalConversionFactorCodeGen()
{
}

Value* EvalConversionFactorCodeGen::codeGen()
{
    Value *modelData = nullptr;

    codeGenVoidModelDataHeader(FunctionName, modelData);

    ModelDataLoadSymbolResolver resolver(modelData, modelGenContext);

    ModelDataIRBuilder mdbuilder(modelData, dataSymbols, builder);

    ASTNodeCodeGen astCodeGen(builder, resolver, modelGenContext, modelData);

    std::string mcfName = model->isSetConversionFactor() ?
            model->getConversionFactor() : "";

    Value *mcfVal = mcfName.empty() ?
            ConstantFP::get(Type::getDoubleTy(context), 1.0) :
            resolver.loadSymbolValue(mcfName);

    const ListOfSpecies *species = model->getListOfSpecies();

    for (uint i = 0; i < species->size(); ++i)
    {
        const Species *s = species->get(i);

        if (dataSymbols.isIndependentFloatingSpecies(s->getId()))
        {
            if (s->isSetConversionFactor())
            {
                const std::string& cf = s->getConversionFactor();
                if (cf.compare(mcfName) != 0)
                {
                    Value *cfv = resolver.loadSymbolValue(cf);
                    Value *cfScale = builder.CreateFDiv(cfv, mcfVal, "scaled_" + cf);
                    Value *amtRate = mdbuilder.createFloatSpeciesAmtRateLoad(s->getId(), s->getId() + "_amtRate");
                    Value *scaledAmtRate = builder.CreateFMul(amtRate, cfScale, s->getId() + "_scaledAmtRate");
                    mdbuilder.createFloatSpeciesAmtRateStore(s->getId(), scaledAmtRate);
                }
            }
        }
    }

    builder.CreateRetVoid();

    return verifyFunction();
}



} /* namespace rrllvm */
