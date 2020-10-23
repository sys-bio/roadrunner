/*
 * EventCodeGenBase.h
 *
 *  Created on: Aug 11, 2013
 *      Author: andy
 */

#ifndef EVENTCODEGENBASE_H_
#define EVENTCODEGENBASE_H_

#include "CodeGenBase.h"
#include "ModelGeneratorContext.h"
#include "SymbolForest.h"
#include "ASTNodeCodeGen.h"
#include "ASTNodeFactory.h"
#include "ModelDataIRBuilder.h"
#include "ModelDataSymbolResolver.h"
#include "LLVMException.h"
#include "rrLogger.h"
#include <sbml/Model.h>
#include <Poco/Logger.h>
#include <vector>
#include <cstdio>

namespace rrllvm
{

typedef void (*EventCodeGenBase_FunctionPtr)(LLVMModelData*, size_t, double*);

template <typename Derived>
class EventCodeGenBase :
        public CodeGenBase<EventCodeGenBase_FunctionPtr>
{
public:
    EventCodeGenBase(const ModelGeneratorContext &mgc) :
            CodeGenBase<EventCodeGenBase_FunctionPtr>(mgc)
    {
    };

    virtual ~EventCodeGenBase() {};

    llvm::Value *codeGen();

    /**
     * derived classes must implement this method to generate the event
     * trigger / assignment code.
     *
     * Derived classes are called with an event and this call is in the middle
     * of the code generation block. So, a derived class should simply begin
     * outputing the instruction so process all of the event assignments /
     * triggers in this call. The base class takes care of generating
     * the return value.
     */
    bool eventCodeGen (llvm::Value *modelData, llvm::Value *data,
            const libsbml::Event *event)
    {
        return false;
    };

    typedef EventCodeGenBase_FunctionPtr FunctionPtr;
};

template <typename Derived>
llvm::Value *EventCodeGenBase<Derived>::codeGen()
{
    // make the set init value function
    llvm::Type *argTypes[] = {
        llvm::PointerType::get(ModelDataIRBuilder::getStructType(this->module), 0),
        llvm::Type::getInt32Ty(this->context),
        llvm::Type::getDoublePtrTy(this->context)
    };

    const char *argNames[] = {
        "modelData", "eventIndx", "data"
    };

    llvm::Value *args[] = {0, 0, 0};

    llvm::Type *retType = llvm::Type::getVoidTy(context);

    llvm::BasicBlock *entry = this->codeGenHeader(Derived::FunctionName, retType,
            argTypes, argNames, args);

    const libsbml::ListOfEvents *events = this->model->getListOfEvents();

    // default, return NaN
    llvm::BasicBlock *def = llvm::BasicBlock::Create(this->context, "default", this->function);
    this->builder.SetInsertPoint(def);

    this->builder.CreateRetVoid();

    // write the switch at the func entry point, the switch is also the
    // entry block terminator
    this->builder.SetInsertPoint(entry);

    llvm::SwitchInst *s = this->builder.CreateSwitch(args[1], def, events->size());

    for (uint i = 0; i < events->size(); ++i)
    {
        char block_name[64];
        std::sprintf(block_name, "event_%i_block", i);
        llvm::BasicBlock *block = llvm::BasicBlock::Create(this->context, block_name, this->function);
        this->builder.SetInsertPoint(block);

        const libsbml::Event *event = events->get(i);

        bool cont = static_cast<Derived*>(this)->eventCodeGen(args[0], args[2], event);

        this->builder.CreateRetVoid();
        s->addCase(llvm::ConstantInt::get(llvm::Type::getInt32Ty(this->context), i), block);

        if (!cont) break;
    }

    return this->verifyFunction();
}

}

#endif /* EVENTCODEGENBASE_H_ */
