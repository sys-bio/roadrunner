/*
 *
 *  Created on: Sept 16, 2013
 *      Author: andy
 */

#ifndef RRLLVMSETVALUECODEGENBASE_H_
#define RRLLVMSETVALUECODEGENBASE_H_

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

typedef bool (*SetValueCodeGenBase_FunctionPtr)(LLVMModelData*, int32_t, double);

template <typename Derived, bool substanceUnits>
class SetValueCodeGenBase :
        public CodeGenBase<SetValueCodeGenBase_FunctionPtr>
{
public:
    SetValueCodeGenBase(const ModelGeneratorContext &mgc);
    virtual ~SetValueCodeGenBase();

    llvm::Value *codeGen();

    typedef SetValueCodeGenBase_FunctionPtr FunctionPtr;
};

template <typename Derived, bool substanceUnits>
SetValueCodeGenBase<Derived, substanceUnits>::SetValueCodeGenBase(
        const ModelGeneratorContext &mgc) :
        CodeGenBase<SetValueCodeGenBase_FunctionPtr>(mgc)
{
}

template <typename Derived, bool substanceUnits>
SetValueCodeGenBase<Derived, substanceUnits>::~SetValueCodeGenBase()
{
}

template <typename Derived, bool substanceUnits>
llvm::Value* SetValueCodeGenBase<Derived, substanceUnits>::codeGen()
{
    // make the set init value function
    llvm::Type *argTypes[] = {
        llvm::PointerType::get(ModelDataIRBuilder::getStructType(this->module), 0),
        llvm::Type::getInt32Ty(this->context),
        llvm::Type::getDoubleTy(this->context)
    };

    const char *argNames[] = {
        "modelData", Derived::IndexArgName, "value"
    };

    llvm::Value *args[] = {0, 0, 0};

    llvm::BasicBlock *entry = this->codeGenHeader(Derived::FunctionName, llvm::Type::getInt8Ty(this->context),
            argTypes, argNames, args);

    StringIntVector ids = static_cast<Derived*>(this)->getIds();

    ModelDataLoadSymbolResolver loadResolver(args[0], this->modelGenContext);

    ModelDataStoreSymbolResolver storeResolver(args[0], this->model, this->modelSymbols,
            this->dataSymbols, this->builder, loadResolver);

    // default, return false
    llvm::BasicBlock *def = llvm::BasicBlock::Create(this->context, "default", this->function);
    this->builder.SetInsertPoint(def);

    // false (0) retval
    this->builder.CreateRet(llvm::ConstantInt::get(llvm::Type::getInt8Ty(this->context), 0));

    // write the switch at the func entry point, the switch is also the
    // entry block terminator
    this->builder.SetInsertPoint(entry);

    llvm::SwitchInst *s = this->builder.CreateSwitch(args[1], def, ids.size());

    for (int i = 0; i < ids.size(); ++i)
    {
        llvm::BasicBlock *block = llvm::BasicBlock::Create(this->context, ids[i].first + "_block", this->function);
        this->builder.SetInsertPoint(block);
        loadResolver.flushCache();

        // the value we are attempting to store, make a copy in loop scope
        // because we can modify the copy.
        llvm::Value *value = args[2];

        // need to check if we have an amount or concentration and check if we
        // are asked for asked for an amount or concentration and convert accordingly
        const libsbml::Species *species = dynamic_cast<const libsbml::Species*>(
                const_cast<libsbml::Model*>(this->model)->getElementBySId(dataSymbols.decodeArrayId(ids[i].first)));

        if(species)
        {
            if (species->getHasOnlySubstanceUnits())
            {
                // species is treated as an amount
                if (!substanceUnits)
                {
                    // given a conc, convert to amount
                    llvm::Value *comp = loadResolver.loadSymbolValue(dataSymbols.getSpeciesCompartment(ids[i].first));
                    value = this->builder.CreateFMul(value, comp, ids[i].first + "_amt");
                }
            }
            else
            {
                // species is treated as a concentration
                if (substanceUnits)
                {
                    // given an amount, need to convert to conc
                    llvm::Value *comp = loadResolver.loadSymbolValue(dataSymbols.getSpeciesCompartment(ids[i].first));
                    value = this->builder.CreateFDiv(value, comp, ids[i].first + "_value_conc");
                }
            }
        }

        // load after we've figured the species bits out
        storeResolver.storeSymbolValue(ids[i].first, value);

        // true retval
        this->builder.CreateRet(llvm::ConstantInt::get(llvm::Type::getInt8Ty(this->context), 1));
        s->addCase(llvm::ConstantInt::get(llvm::Type::getInt32Ty(this->context), ids[i].second), block);
    }

    return this->verifyFunction();
}


} /* namespace rrllvm */
#endif /* RRLLVMSETVALUECODEGENBASE_H_ */
