/*
 * LocalParameterSymbolResolver.cpp
 *
 *  Created on: Aug 14, 2013
 *      Author: andy
 */
#pragma hdrstop
#include "KineticLawParameterResolver.h"
#include "LLVMIncludes.h"
#include "rrLogger.h"
#include <sbml/Reaction.h>

using namespace llvm;
using namespace libsbml;

using rr::Logger;
using rr::getLogger;

namespace rrllvm
{

KineticLawParameterResolver::KineticLawParameterResolver(
        LoadSymbolResolver& parentResolver,
        const libsbml::KineticLaw& kineticLaw, llvm::IRBuilder<>& builder) :
                parentResolver(parentResolver),
                kineticLaw(kineticLaw),
                builder(builder)
{
}


llvm::Value* KineticLawParameterResolver::loadSymbolValue(
        const std::string& symbol,
        const llvm::ArrayRef<llvm::Value*>& args)
{
    const libsbml::Parameter *parameter =
            kineticLaw.getLocalParameter(symbol);

    if (!parameter)
    {
        parameter = kineticLaw.getParameter(symbol);
    }

    if (parameter)
    {
        const Reaction *reaction = (const Reaction*)kineticLaw.getParentSBMLObject();
        rrLog(Logger::LOG_DEBUG) << "reaction id " << reaction->getId() <<
                " found local parameter for symbol " << symbol <<
                ", value: " << parameter->getValue();

        Value *value =  ConstantFP::get(builder.getContext(),
                APFloat(parameter->getValue()));
        value->setName(symbol + "_local");
        return value;
    }

    return parentResolver.loadSymbolValue(symbol, args);
}

void KineticLawParameterResolver::recursiveSymbolPush(const std::string& symbol)
{
    parentResolver.recursiveSymbolPush(symbol);
}

void KineticLawParameterResolver::recursiveSymbolPop()
{
    parentResolver.recursiveSymbolPop();
}

bool KineticLawParameterResolver::isLocalParameter(const std::string& symbol)
{
    if (kineticLaw.getLocalParameter(symbol) != NULL) {
        return true;
    }
    return kineticLaw.getParameter(symbol) != NULL;
}

} /* namespace rr */
