/*
 * LoadSymbolResolverBase.h
 *
 *  Created on: May 15, 2014
 *      Author: andy
 */

#ifndef LOADSYMBOLRESOLVERBASE_H_
#define LOADSYMBOLRESOLVERBASE_H_

#include "CodeGen.h"
#include "LLVMIncludes.h"
#include "LLVMModelDataSymbols.h"
#include "LLVMModelSymbols.h"
#include "ModelGeneratorContext.h"

#include <rr_unordered_map>

namespace libsbml
{
class Model;
}

namespace rrllvm
{

/**
 * provide common sbml model functionality to three kinds of resolvers.
 *
 * This has common functionality such as dealing with sbml elements
 * such as reactions, species and so forth.
 */
class LoadSymbolResolverBase: public LoadSymbolResolver
{
public:
    /**
     * shortcut to get a reaction rate, performance wise, quicker than
     * going through all the loadSymbolValue types.
     */
    llvm::Value* loadReactionRate(const libsbml::Reaction*, const string& id);

    virtual void recursiveSymbolPush(const std::string& symbol);

    virtual void recursiveSymbolPop();

    /**
     * Flush the symbol cache. This is required in branches and switch blocks as
     * a symbol used in a previous block can not be re-used in the current block.
     */
    void flushCache();

    /**
     * nested conditionals (or functions?) can push a local cache block, where
     * symbols would be chached. These need to be popped as these symbols are
     * not valid outside of the local conditional or scope block.
     */
    virtual unsigned pushCacheBlock();

    /**
     * Pop a scoped cache block, this clears these values, and any subsequent reads
     * re-evaluate the requested symbol.
     *
     * Will throw an exception if an empty stack pop is attempted.
     */
    virtual unsigned popCacheBlock();


protected:
    LoadSymbolResolverBase(const ModelGeneratorContext &ctx);

    typedef std::list<std::string> StringStack;
    typedef cxx11_ns::unordered_map<std::string, llvm::Value*> ValueMap;
    typedef std::deque<ValueMap> ValueMapStack;

    const ModelGeneratorContext &modelGenContext;
    const libsbml::Model *model;
    const LLVMModelDataSymbols &modelDataSymbols;
    const LLVMModelSymbols &modelSymbols;
    llvm::IRBuilder<> &builder;

    // check for recursive symbol definitions in rule evaluations
    StringStack symbolStack;

    // cache symbols
    ValueMapStack symbolCache;

    /**
     * check in the symbol cache if the symbol exists, if so return it.
     *
     * If the value is not NULL, it is cached and returned.
     *
     * Cached function args not currently supported (just return NULL).
     */
    llvm::Value* cacheValue(const std::string& symbol,
            const llvm::ArrayRef<llvm::Value*>& args,
            llvm::Value* value = NULL);
};


} /* namespace rrllvm */

#endif /* LOADSYMBOLRESOLVERBASE_H_ */
