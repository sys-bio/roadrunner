/*
 *
 *  Created on: Sept Oct 31, 2013
 *      Author: andy
 */

#ifndef SETINITIALVALUECODEGENBASE_H_
#define SETINITIALVALUECODEGENBASE_H_

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

typedef bool (*SetInitialValueCodeGenBase_FunctionPtr)(LLVMModelData*, int32_t, double);

template <typename Derived, bool substanceUnits>
class SetInitialValueCodeGenBase :
        public CodeGenBase<SetInitialValueCodeGenBase_FunctionPtr>
{
public:
    SetInitialValueCodeGenBase(const ModelGeneratorContext &mgc);
    virtual ~SetInitialValueCodeGenBase();

    llvm::Value *codeGen();

    typedef SetInitialValueCodeGenBase_FunctionPtr FunctionPtr;

};

template <typename Derived, bool substanceUnits>
SetInitialValueCodeGenBase<Derived, substanceUnits>::SetInitialValueCodeGenBase(
        const ModelGeneratorContext &mgc) :
        CodeGenBase<SetInitialValueCodeGenBase_FunctionPtr>(mgc)
{
}

template <typename Derived, bool substanceUnits>
SetInitialValueCodeGenBase<Derived, substanceUnits>::~SetInitialValueCodeGenBase()
{
}

template <typename Derived, bool substanceUnits>
llvm::Value* SetInitialValueCodeGenBase<Derived, substanceUnits>::codeGen()
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

    // are we doing init symbols or regular symbols.
    bool isInit = static_cast<Derived*>(this)->isInitialValue();

    ModelInitialValueSymbolResolver loadResolver(args[0], this->modelGenContext);

    ModelInitialValueStoreSymbolResolver storeResolver(args[0], this->model, this->modelSymbols,
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

        // the value we are attempting to store, make a copy in loop scope
        // because we can modify the copy.
        llvm::Value *value = args[2];

        std::string element = ids[i].first;

        // need to check if we have an amount or concentration and check if we
        // are asked for asked for an amount or concentration and convert accordingly
        const libsbml::Species *species = dynamic_cast<const libsbml::Species*>(
                const_cast<libsbml::Model*>(this->model)->getElementBySId(dataSymbols.decodeArrayId(element)));

        if(species)
        {
            // get the id of the compartment the species belongs to
            std::string compId = dataSymbols.getSpeciesCompartment(element);

            llvm::Value *comp = loadResolver.loadSymbolValue(compId);

            if (species->getHasOnlySubstanceUnits())
            {
                // species is treated as an amount
                if (!substanceUnits)
                {
                    // given a conc, convert to amount
                    value = this->builder.CreateFMul(value, comp, ids[i].first + "_amt");
                }
            }
            else
            {
                // species is treated as a concentration
                if (substanceUnits)
                {
                    // given an amount, need to convert to conc
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
#endif /* SETINITIALVALUECODEGENBASE_H_ */
