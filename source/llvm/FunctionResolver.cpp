/*
 * FunctionResolver.cpp
 *
 *  Created on: Aug 11, 2013
 *      Author: andy
 */
#pragma hdrstop
#include "FunctionResolver.h"
#include "ASTNodeCodeGen.h"
#include "LLVMException.h"
#include "rrStringUtils.h"
#include <sbml/common/libsbml-version.h>
#include <sbml/common/libsbml-config-packages.h>

#ifdef LIBSBML_HAS_PACKAGE_DISTRIB
#include <sbml/packages/distrib/common/DistribExtensionTypes.h>
#endif


namespace rrllvm
{

using namespace libsbml;

using namespace llvm;

using rr::Logger;
using rr::getLogger;

FunctionResolver::FunctionResolver(LoadSymbolResolver& parentResolver,
        llvm::Value *modelData,
        const ModelGeneratorContext& ctx) :
                modelGenContext(ctx),
                parentResolver(parentResolver),
                model(ctx.getModel()),
                builder(*ctx.getJitNonOwning()->getBuilderNonOwning()),
                modelData(modelData),
                symbols(0)
{
}


llvm::Value* FunctionResolver::loadSymbolValue(const std::string& symbol,
        const llvm::ArrayRef<llvm::Value*>& args)
{
    const FunctionDefinition *funcDef = 0;
    if (symbols)
    {
        std::map<std::string,Value*>::const_iterator i = symbols->find(symbol);
        if (i != symbols->end())
        {
            return i->second;
        }
        else
        {
            return parentResolver.loadSymbolValue(symbol, args);
        }
    }
    else if ((funcDef = model->getListOfFunctionDefinitions()->get(symbol)))
    {

        recursiveSymbolPush(symbol);

        const ASTNode *math = funcDef->getMath();
        const unsigned nchild = math->getNumChildren();

        if (!math->isLambda())
        {
            throw_llvm_exception(symbol + ", math element of function definition must be a lambda");
        }

        // must have at least one trailing apply node
        if (nchild < 1)
        {
            throw_llvm_exception("function node " + symbol + " must have at least one child node, "
                    "this node has " + rr::toString(nchild) +  " child nodes");
        }

        if (nchild - 1 != args.size())
        {
            std::string expected = rr::toString(nchild - 1);
            std::string got = rr::toString((int)args.size());

            throw_llvm_exception(symbol + ", argument count does not match, expected " +
                    expected + ", recieved: " + got);
        }

        symbols = new std::map<std::string, Value*>();

        // first set of child nodes are bvars, these go into new local scope
        for (uint i = 0; i < nchild - 1; ++i)
        {
            const ASTNode *c = math->getChild(i);
#if (LIBSBML_VERSION >= 51000)
            // assert(c && c->getType() == AST_QUALIFIER_BVAR);
#else
            assert(c->isBvar());
#endif
            (*symbols)[c->getName()] = args[i];
        }

        ASTNodeCodeGen astCodeGen(builder, *this, modelGenContext, modelData);

        // the last child should be an apply node
        const ASTNode *apply = math->getChild(nchild - 1);
        Value *result = astCodeGen.codeGenDouble(apply);

        delete symbols;
        symbols = 0;

        recursiveSymbolPop();

        return result;
    }

    return 0;
}

void FunctionResolver::recursiveSymbolPush(const std::string& symbol)
{
    parentResolver.recursiveSymbolPush(symbol);
}

void FunctionResolver::recursiveSymbolPop()
{
    parentResolver.recursiveSymbolPop();
}

bool FunctionResolver::isLocalParameter(const std::string& symbol)
{
    return false;
}

} /* namespace rr */
