/*
 *
 *  Created on: Jul 29, 2013
 *      Author: andy
 */

#ifndef RRLLVMGETVALUECODEGENBASE_H_
#define RRLLVMGETVALUECODEGENBASE_H_

#include "CodeGenBase.h"
#include "ModelGeneratorContext.h"
#include "SymbolForest.h"
#include "ASTNodeFactory.h"
#include "ModelDataIRBuilder.h"
#include "ModelDataSymbolResolver.h"
#include "LLVMException.h"
#include "rrLogger.h"
#include <sbml/Model.h>
#include <Poco/Logger.h>
#include <vector>

namespace rrllvm
{

typedef double (*GetValueCodeGenBase_FunctionPtr)(LLVMModelData*, int32_t);

template <typename Derived, bool substanceUnits>
class GetValueCodeGenBase :
        public CodeGenBase<GetValueCodeGenBase_FunctionPtr>
{
public:
    GetValueCodeGenBase(const ModelGeneratorContext &mgc);
    virtual ~GetValueCodeGenBase();

    llvm::Value *codeGen();

    typedef GetValueCodeGenBase_FunctionPtr FunctionPtr;

};

template <typename Derived, bool substanceUnits>
GetValueCodeGenBase<Derived, substanceUnits>::GetValueCodeGenBase(
        const ModelGeneratorContext &mgc) :
        CodeGenBase<GetValueCodeGenBase_FunctionPtr>(mgc)
{
}

template <typename Derived, bool substanceUnits>
GetValueCodeGenBase<Derived, substanceUnits>::~GetValueCodeGenBase()
{
}

template <typename Derived, bool substanceUnits>
llvm::Value* GetValueCodeGenBase<Derived, substanceUnits>::codeGen()
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

    std::vector<string> ids = static_cast<Derived*>(this)->getIds();

    ModelDataLoadSymbolResolver resolver(args[0], this->modelGenContext);
    // default, return NaN
    llvm::BasicBlock *def = llvm::BasicBlock::Create(this->context, "default", this->function);
    this->builder.SetInsertPoint(def);

    this->builder.CreateRet(llvm::ConstantFP::get(this->context,
            llvm::APFloat::getQNaN(llvm::APFloat::IEEEdouble())));

    // write the switch at the func entry point, the switch is also the
    // entry block terminator
    this->builder.SetInsertPoint(entry);

    llvm::SwitchInst *s = this->builder.CreateSwitch(args[1], def, ids.size());

    for (int i = 0; i < ids.size(); ++i)
    {
        llvm::BasicBlock *block = llvm::BasicBlock::Create(this->context, ids[i] + "_block", this->function);
        this->builder.SetInsertPoint(block);
        resolver.flushCache();

        // the requested value
        llvm::Value *value = resolver.loadSymbolValue(ids[i]);

        // need to check if we have an amount or concentration and check if we
        // are asked for asked for an amount or concentration and convert accordingly
        const libsbml::Species *species = dynamic_cast<const libsbml::Species*>(
                const_cast<libsbml::Model*>(this->model)->getElementBySId(ids[i]));

        if(species)
        {
            if (species->getHasOnlySubstanceUnits())
            {
                value->setName(ids[i] + "_amt");
                // species is treated as an amount
                if (!substanceUnits)
                {
                    // convert to concentration
                    llvm::Value *comp = resolver.loadSymbolValue(species->getCompartment());
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
                    llvm::Value *comp = resolver.loadSymbolValue(species->getCompartment());
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




#endif /* RRLLVMGETVALUECODEGENBASE_H_ */
