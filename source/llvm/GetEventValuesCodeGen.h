/*
 * GetEventValuesCodeGenBas.h
 *
 *  Created on: Aug 10, 2013
 *      Author: andy
 */

#ifndef RRLLVMGETEVENTVALUECODEGENBASE_H_
#define RRLLVMGETEVENTVALUECODEGENBASE_H_

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

typedef double (*GetEventValueCodeGenBase_FunctionPtr)(LLVMModelData*, size_t);

template <typename Derived, typename
    FunctionPtrType=GetEventValueCodeGenBase_FunctionPtr>
class GetEventValueCodeGenBase :
        public CodeGenBase<FunctionPtrType>
{
public:
    GetEventValueCodeGenBase(const ModelGeneratorContext &mgc) :
            CodeGenBase<FunctionPtrType>(mgc)
    {
    };

    virtual ~GetEventValueCodeGenBase() {};

    llvm::Value *codeGen();

    typedef FunctionPtrType FunctionPtr;

    /**
     * default ret type is double, derived classes
     * must override if usign non-default func sic
     */
    llvm::Type *getRetType()
    {
        return llvm::Type::getDoubleTy(this->context);
    }

    /**
     * create a return type, a zero value should return the default type
     */
    llvm::Value *createRet(llvm::Value* value)
    {
        return value ? value :
                llvm::ConstantFP::get(this->context, llvm::APFloat(123.456));
    }

};

template <typename Derived, typename FunctionPtrType>
llvm::Value *GetEventValueCodeGenBase<Derived, FunctionPtrType>::codeGen()
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

    llvm::Type *retType = static_cast<Derived*>(this)->getRetType();

    llvm::BasicBlock *entry = this->codeGenHeader(Derived::FunctionName, retType,
            argTypes, argNames, args);

    const libsbml::ListOfEvents *events = this->model->getListOfEvents();

    ModelDataLoadSymbolResolver resolver(args[0], this->modelGenContext);

    ASTNodeCodeGen astCodeGen(this->builder, resolver, this->modelGenContext, args[0]);

    // default, return NaN
    llvm::BasicBlock *def = llvm::BasicBlock::Create(this->context, "default", this->function);
    this->builder.SetInsertPoint(def);

    llvm::Value *defRet = static_cast<Derived*>(this)->createRet(0);
    this->builder.CreateRet(defRet);

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
        resolver.flushCache();

        const libsbml::Event *event = events->get(i);

        const libsbml::ASTNode *math = static_cast<Derived*>(this)->getMath(event);

        // the requested value
        llvm::Value *value = astCodeGen.codeGen(math);

        // convert type to return type
        value = static_cast<Derived*>(this)->createRet(value);

        this->builder.CreateRet(value);
        s->addCase(llvm::ConstantInt::get(llvm::Type::getInt32Ty(this->context), i), block);
    }

    return this->verifyFunction();
}

typedef unsigned char (*GetEventTriggerCodeGen_FunctionPtr)(LLVMModelData*, size_t);

class GetEventTriggerCodeGen: public
    GetEventValueCodeGenBase<GetEventTriggerCodeGen,
    GetEventTriggerCodeGen_FunctionPtr>
{
public:
    GetEventTriggerCodeGen(const ModelGeneratorContext &mgc);
    ~GetEventTriggerCodeGen() {};

    const libsbml::ASTNode *getMath(const libsbml::Event *);

    static const char* FunctionName;
    static const char* IndexArgName;

    llvm::Type *getRetType();

    llvm::Value *createRet(llvm::Value*);
};

class GetEventPriorityCodeGen: public
    GetEventValueCodeGenBase<GetEventPriorityCodeGen>
{
public:
    GetEventPriorityCodeGen(const ModelGeneratorContext &mgc);
    ~GetEventPriorityCodeGen();

    const libsbml::ASTNode *getMath(const libsbml::Event *);

    static const char* FunctionName;
    static const char* IndexArgName;

private:
    libsbml::ASTNode *node;
};

class GetEventDelayCodeGen: public
    GetEventValueCodeGenBase<GetEventDelayCodeGen>
{
public:
    GetEventDelayCodeGen(const ModelGeneratorContext &mgc);
    ~GetEventDelayCodeGen();

    const libsbml::ASTNode *getMath(const libsbml::Event *);

    llvm::Value *createRet(llvm::Value* value)
    {
		// Return the value for the default label
		if (!value)
            return llvm::ConstantFP::get(this->context, llvm::APFloat(123.456));
		// If the delay evaluates to a double then just return it
		if (value->getType() == llvm::Type::getDoubleTy(context))
			return value;
		// Otherwise it's a boolean (i.e. an i1), so convert it to a double
		return this->builder.CreateCast(llvm::Instruction::CastOps::UIToFP, value, llvm::Type::getDoubleTy(context));
    }

    static const char* FunctionName;
    static const char* IndexArgName;
private:
    libsbml::ASTNode *node;
};


} /* namespace rr */




#endif /* RRLLVMGETVALUECODEGENBASE_H_ */
