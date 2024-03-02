/*
 * FunctionResolver.h
 *
 *  Created on: Aug 11, 2013
 *      Author: andy
 */

#ifndef EVENTSYMBOLRESOLVER_H_
#define EVENTSYMBOLRESOLVER_H_

#include "CodeGen.h"
#include "ModelGeneratorContext.h"
#include <map>

namespace rrllvm
{

/**
 * Not thread safe -- but perfectly fine if stack allocated.
 */
class FunctionResolver: public LoadSymbolResolver
{
public:
    FunctionResolver(LoadSymbolResolver& parentResolver,
            llvm::Value *modelData,
            const ModelGeneratorContext& ctx);

    virtual ~FunctionResolver() {};

    virtual llvm::Value *loadSymbolValue(const std::string& symbol,
            const llvm::ArrayRef<llvm::Value*>& args =
                    llvm::ArrayRef<llvm::Value*>());

    virtual void recursiveSymbolPush(const std::string& symbol);

    virtual void recursiveSymbolPop();

    virtual bool isLocalParameter(const std::string& symbol);

private:
    LoadSymbolResolver& parentResolver;
    const ModelGeneratorContext& modelGenContext;
    const libsbml::Model *model;
    llvm::IRBuilder<> &builder;
    llvm::Value *modelData;

    /**
     * only valid during an outer call to loadSymbolValue, otherwise
     * is null.
     */
    std::map<std::string, llvm::Value*> *symbols;
};

} /* namespace rr */
#endif /* EVENTSYMBOLRESOLVER_H_ */
