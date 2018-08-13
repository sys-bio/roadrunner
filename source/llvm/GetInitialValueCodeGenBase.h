/*
 *
 *  Created on: Jul 29, 2013
 *      Author: andy
 */

#ifndef RRLLVM_GETINITIALVALUECODEGENBASE_H_
#define RRLLVM_GETINITIALVALUECODEGENBASE_H_

#include "CodeGenBase.h"
#include "ModelGeneratorContext.h"
#include "SymbolForest.h"
#include "ASTNodeFactory.h"
#include "ModelDataIRBuilder.h"
#include "ModelInitialValueSymbolResolver.h"
#include "LLVMException.h"
#include "rrLogger.h"
#include <sbml/Model.h>
#include <Poco/Logger.h>
#include <vector>

namespace rrllvm
{

typedef double (*GetInitialValueCodeGenBase_FunctionPtr)(LLVMModelData*, int32_t);

template <typename Derived, bool substanceUnits>
class GetInitialValueCodeGenBase :
        public CodeGenBase<GetInitialValueCodeGenBase_FunctionPtr>
{
public:
    GetInitialValueCodeGenBase(const ModelGeneratorContext &mgc);
    virtual ~GetInitialValueCodeGenBase();

    llvm::Value *codeGen();

    typedef GetInitialValueCodeGenBase_FunctionPtr FunctionPtr;

};

template <typename Derived, bool substanceUnits>
GetInitialValueCodeGenBase<Derived, substanceUnits>::GetInitialValueCodeGenBase(
        const ModelGeneratorContext &mgc) :
        CodeGenBase<GetInitialValueCodeGenBase_FunctionPtr>(mgc)
{
}

template <typename Derived, bool substanceUnits>
GetInitialValueCodeGenBase<Derived, substanceUnits>::~GetInitialValueCodeGenBase()
{
}

template <typename Derived, bool substanceUnits>
llvm::Value* GetInitialValueCodeGenBase<Derived, substanceUnits>::codeGen()
{
    // make the set init value function
    llvm::Type *argTypes[] = {
        llvm::PointerType::get(ModelDataIRBuilder::getStructType(this->module), 0),
        llvm::Type::getInt32Ty(this->context)
    };

    const char *argNames[] = {
        "modelData", Derived::IndexArgName
    };

    llvm::Value *args[] = {0, 0};

    llvm::BasicBlock *entry = this->codeGenHeader(Derived::FunctionName, llvm::Type::getDoubleTy(this->context),
            argTypes, argNames, args);

    std::vector<std::string> ids = static_cast<Derived*>(this)->getIds();

    ModelInitialValueSymbolResolver resolver(args[0], this->modelGenContext);

    // default, return NaN
    llvm::BasicBlock *def = llvm::BasicBlock::Create(this->context, "default", this->function);
    this->builder.SetInsertPoint(def);

    this->builder.CreateRet(llvm::ConstantFP::get(this->context,
            llvm::APFloat::getQNaN(llvm::APFloat::IEEEdouble)));

    // write the switch at the func entry point, the switch is also the
    // entry block terminator
    this->builder.SetInsertPoint(entry);

    llvm::SwitchInst *s = this->builder.CreateSwitch(args[1], def, ids.size());

    for (int i = 0; i < ids.size(); ++i)
    {
        llvm::BasicBlock *block = llvm::BasicBlock::Create(this->context, ids[i] + "_block", this->function);
        this->builder.SetInsertPoint(block);

        // the requested value
        llvm::Value *value = resolver.loadSymbolValue(ids[i]);

        // need to check if we have an amount or concentration and check if we
        // are asked for asked for an amount or concentration and convert accordingly
        const libsbml::Species *species = dynamic_cast<const libsbml::Species*>(
                const_cast<libsbml::Model*>(this->model)->getElementBySId(dataSymbols.decodeArrayId(ids[i])));

        if(species)
        {
            if (species->getHasOnlySubstanceUnits())
            {
                value->setName(ids[i] + "_amt");
                // species is treated as an amount
                if (!substanceUnits)
                {
                    // convert to concentration
                    llvm::Value *comp = resolver.loadSymbolValue(dataSymbols.getSpeciesCompartment(ids[i]));
                    value = this->builder.CreateFDiv(value, comp, ids[i] + "_conc");
                }
            }
            else
            {
                value->setName(ids[i] + "_conc");
                // species is treated as a concentration
                if (substanceUnits)
                {
                    // convert to amount
                    llvm::Value *comp = resolver.loadSymbolValue(dataSymbols.getSpeciesCompartment(ids[i]));
                    value = this->builder.CreateFMul(value, comp, ids[i] + "_amt");
                }
            }
        }
        else
        {
            value->setName(ids[i] + "_value");
        }
        this->builder.CreateRet(value);
        s->addCase(llvm::ConstantInt::get(llvm::Type::getInt32Ty(this->context), i), block);
    }

    return this->verifyFunction();
}


} /* namespace rr */




#endif /* RRLLVM_GETINITIALVALUECODEGENBASE_H_ */
