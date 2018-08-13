/*
 * LoadSymbolResolverBase.cpp
 *
 *  Created on: May 15, 2014
 *      Author: andy
 */
#include "LoadSymbolResolverBase.h"

#include "KineticLawParameterResolver.h"
#include "ASTNodeCodeGen.h"
#include "LLVMException.h"
#include "rrLogger.h"
#include <sbml/Model.h>

using namespace std;
using namespace libsbml;
using namespace llvm;

using rr::Logger;


namespace rrllvm
{

LoadSymbolResolverBase::LoadSymbolResolverBase(
        const ModelGeneratorContext& ctx) :
        modelGenContext(ctx),
        model(ctx.getModel()),
        modelSymbols(ctx.getModelSymbols()),
        modelDataSymbols(ctx.getModelDataSymbols()),
        builder(ctx.getBuilder())
{
    // start with a single empty cache block
    symbolCache.push_back(ValueMap());
}


llvm::Value* LoadSymbolResolverBase::loadReactionRate(
        const libsbml::Reaction* reaction, const string& id)
{
	const KineticLaw *kinetic = reaction->getKineticLaw();
	const ASTNode *math = modelDataSymbols.getReactionLaw(id);
    KineticLawParameterResolver lpResolver(*this, *kinetic, builder);
    ASTNodeCodeGen astCodeGen(builder, lpResolver);

    ASTNodeCodeGenScalarTicket t(astCodeGen, true);
    return astCodeGen.codeGen(math);
}

void LoadSymbolResolverBase::recursiveSymbolPush(const std::string& symbol)
{
    StringStack::const_iterator prevSymbol =
                std::find(symbolStack.begin(), symbolStack.end(), symbol);
        if (prevSymbol != symbolStack.end())
        {
            std::string msg =
                "recursive assignment rule or function detected, the symbol \'";
            msg += symbol;
            msg += "\' is a parent of itself";
            Log(rr::Logger::LOG_ERROR) << msg;
            throw LLVMException(msg, __FUNC__);
        }
        symbolStack.push_back(symbol);
}

void LoadSymbolResolverBase::recursiveSymbolPop()
{
    assert(symbolStack.size() > 0 && "attempt to pop emtpy symbol stack");
    symbolStack.pop_back();
}

void LoadSymbolResolverBase::flushCache()
{
    symbolCache.clear();
    symbolCache.push_back(ValueMap());
}

unsigned LoadSymbolResolverBase::pushCacheBlock()
{
    symbolCache.push_back(ValueMap());
    return symbolCache.size();
}

unsigned LoadSymbolResolverBase::popCacheBlock()
{
    if(symbolCache.empty()) {
        throw std::logic_error("attempt to pop from an empty symbol cache stack");
    }
    symbolCache.pop_back();
    return symbolCache.size();
}

llvm::Value* LoadSymbolResolverBase::cacheValue(const std::string& symbol,
        const llvm::ArrayRef<llvm::Value*>& args,
        llvm::Value* value)
{
    if(args.size() || !modelGenContext.useSymbolCache()) {
        return value;
    }

    assert(symbolCache.size() && "symbol cache stack empty");

    if(value) {
        ValueMap &values = symbolCache.back();
        values[symbol] = value;
        Log(Logger::LOG_DEBUG) << "caching value for " << symbol;
        return value;
    }

    for(ValueMapStack::const_reverse_iterator i = symbolCache.rbegin();
            i != symbolCache.rend(); ++i) {
        const ValueMap &values = *i;
        ValueMap::const_iterator j = values.find(symbol);
        Value* result = j != values.end() ? j->second : NULL;

        if(result) {
            Log(Logger::LOG_DEBUG) <<  "found cached value for " << symbol;
            return result;
        }
    }

    Log(Logger::LOG_DEBUG) <<  "did not found cached value for " << symbol;
    return NULL;
}

} /* namespace rrllvm */


