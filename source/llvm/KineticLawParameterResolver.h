/*
 * KineticLawParameterResolver.h
 *
 *  Created on: Aug 14, 2013
 *      Author: andy
 */

#ifndef KineticLawParameterResolver_H_
#define KineticLawParameterResolver_H_

#include "CodeGen.h"
#include <sbml/KineticLaw.h>

namespace rrllvm
{
    /** @class KineticLawParameterResolver
    * Sorts out local parameters in a kinetic law.
    */

    class KineticLawParameterResolver : public LoadSymbolResolver
    {
    public:
        KineticLawParameterResolver(LoadSymbolResolver& parentResolver,
            const libsbml::KineticLaw& kineticLaw, llvm::IRBuilder<>& builder);

        virtual ~KineticLawParameterResolver() {};

        virtual llvm::Value* loadSymbolValue(const std::string& symbol,
            const llvm::ArrayRef<llvm::Value*>& args =
            llvm::ArrayRef<llvm::Value*>());

        virtual void recursiveSymbolPush(const std::string& symbol);

        virtual void recursiveSymbolPop();

        virtual bool isLocalParameter(const std::string& symbol);

    private:
        LoadSymbolResolver& parentResolver;
        const libsbml::KineticLaw& kineticLaw;
        llvm::IRBuilder<>& builder;
    };

} /* namespace rr */
#endif /* KineticLawParameterResolver_H_ */
