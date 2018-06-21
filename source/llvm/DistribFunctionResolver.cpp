/*
 * DistribFunctionResolver.cpp
 *
 *  Created on: Sep 23, 2014
 *      Author: andy
 */

#include <llvm/DistribFunctionResolver.h>

#ifdef LIBSBML_HAS_PACKAGE_DISTRIB

#include "LLVMException.h"
#include "rrLogger.h"

using namespace llvm;
using namespace rr;
using namespace libsbml;

namespace rrllvm
{

DistribFunctionResolver::DistribFunctionResolver(
        const ModelGeneratorContext& ctx, llvm::Value *modelData) :
                        builder(ctx.getBuilder()),
                        random(ctx.getRandom()),
                        modelData(modelData),
                        modelDataSymbols(ctx.getModelDataSymbols()),
                        module(ctx.getModule())
{
}


llvm::Value* DistribFunctionResolver::loadSymbolValue(
        const libsbml::FunctionDefinition* funcDef,
        const libsbml::DistribFunctionDefinitionPlugin* distribFunc,
        const llvm::ArrayRef<llvm::Value*>& args)
{
    Log(Logger::LOG_NOTICE) << "distrib function: " << funcDef->getId();

    ModelDataIRBuilder mdbuilder(modelData, modelDataSymbols,
            builder);

    // pointer to random field.
    llvm::Value *randomPtr = mdbuilder.createRandomLoad();

    if (args.size() != 2)
    {
        throw_llvm_exception("invalid number of args");
    }

    const DistribDrawFromDistribution* distrib =
            distribFunc->getDistribDrawFromDistribution();

    // const UncertMLNode *uml = distrib->getUncertML();

    if (distrib->getName() == "uniformDistribution")
    {
        llvm::Value *funcArgs[] = {randomPtr, args[0], args[1]};

        llvm::Value *func = module->getFunction("rr_distrib_uniform");

        assert(func && "could not get rr_distrib_uniform");

        return builder.CreateCall(func, funcArgs, "call_rr_distrib_uniform");
    }

    else if (distrib->getName() == "normalDistribution")
    {
        llvm::Value *funcArgs[] = {randomPtr, args[0], args[1]};

        llvm::Value *func = module->getFunction("rr_distrib_normal");

        assert(func && "could not get rr_distrib_normal");

        return builder.CreateCall(func, funcArgs, "call_rr_normal_uniform");
    }

    else
    {
        string name = distrib->getName();
        throw_llvm_exception("Unsupported distribution: " + name);
    }
}




} /* namespace rrllvm */

#endif /* LIBSBML_HAS_PACKAGE_DISTRIB */

