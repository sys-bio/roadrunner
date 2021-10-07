/*
 * GetEventValuesCodeGen.cpp
 *
 *  Created on: Aug 10, 2013
 *      Author: andy
 */
#pragma hdrstop
#include "GetEventValuesCodeGen.h"
#include "LLVMException.h"
#include "ModelDataSymbolResolver.h"
#include "rrLogger.h"

#include <Poco/Logger.h>
#include <vector>

using namespace llvm;

using namespace libsbml;

namespace rrllvm
{

const char* GetEventTriggerCodeGen::FunctionName = "getEventTrigger";
const char* GetEventTriggerCodeGen::IndexArgName = "triggerIndx";

GetEventTriggerCodeGen::GetEventTriggerCodeGen(
        const ModelGeneratorContext &mgc) :
        GetEventValueCodeGenBase<GetEventTriggerCodeGen,
        GetEventTriggerCodeGen_FunctionPtr>(mgc)
{
}

llvm::Type *GetEventTriggerCodeGen::getRetType()
{
    return llvm::Type::getInt8Ty(context);
};

llvm::Value *GetEventTriggerCodeGen::createRet(llvm::Value *value)
{
    if (!value)
    {
        return llvm::ConstantInt::get(getRetType(), 0xff, false);
    }
    else
    {
        return builder.CreateIntCast(value, getRetType(), false);
    }
}

llvm::Value* GetEventTriggerCodeGen::getMath(
        const libsbml::Event* event, ASTNodeCodeGen& astCodeGen)
{
    const Trigger *trigger = event->getTrigger();
    const ASTNode* node =  trigger->getMath();
    return astCodeGen.codeGenBoolean(node);
}


const char* GetEventPriorityCodeGen::FunctionName = "getEventPriority";
const char* GetEventPriorityCodeGen::IndexArgName = "priorityIndx";

GetEventPriorityCodeGen::GetEventPriorityCodeGen(
        const ModelGeneratorContext &mgc) :
        GetEventValueCodeGenBase<GetEventPriorityCodeGen>(mgc),
        node(0)
{
}

GetEventPriorityCodeGen::~GetEventPriorityCodeGen()
{
    delete node;
}

llvm::Value* GetEventPriorityCodeGen::getMath(
    const libsbml::Event* event, ASTNodeCodeGen& astCodeGen)
{
    const ASTNode* ast = node;
    if (event->isSetPriority() && event->getPriority()->isSetMath())
    {
        ast = event->getPriority()->getMath();
    }
    else
    {
        if (!node)
        {
            node = new ASTNode(AST_REAL);
            node->setValue(0);
            ast = node;
        }
    }
    return astCodeGen.codeGenDouble(ast);
}

const char* GetEventDelayCodeGen::FunctionName = "getEventDelay";
const char* GetEventDelayCodeGen::IndexArgName = "delayIndx";

GetEventDelayCodeGen::GetEventDelayCodeGen(
        const ModelGeneratorContext &mgc) :
        GetEventValueCodeGenBase<GetEventDelayCodeGen>(mgc),
        node(0)
{
}

GetEventDelayCodeGen::~GetEventDelayCodeGen()
{
    delete node;
}

llvm::Value* GetEventDelayCodeGen::getMath(
    const libsbml::Event* event, ASTNodeCodeGen& astCodeGen)
{
    const ASTNode* ast = node;
    if (event->isSetDelay() && event->getDelay()->isSetMath())
    {
        ast = event->getDelay()->getMath();
    }
    else
    {
        if (!node)
        {
            node = new ASTNode(AST_REAL);
            node->setValue(0);
            ast = node;
        }
    }
    return astCodeGen.codeGenDouble(ast);
}




} /* namespace rr */




