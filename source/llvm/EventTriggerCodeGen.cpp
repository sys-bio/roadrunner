/*
 * EventTriggerCodeGen.cpp
 *
 *  Created on: Aug 11, 2013
 *      Author: andy
 */
#pragma hdrstop
#include "EventTriggerCodeGen.h"
#include "ModelDataIRBuilder.h"
#include "ModelDataSymbolResolver.h"
#include "ASTNodeCodeGen.h"

namespace rrllvm
{

using namespace llvm;
using namespace libsbml;

const char* EventTriggerCodeGen::FunctionName = "eventTrigger";

EventTriggerCodeGen::EventTriggerCodeGen(const ModelGeneratorContext& mgc) :
        EventCodeGenBase<EventTriggerCodeGen>(mgc)
{
}

EventTriggerCodeGen::~EventTriggerCodeGen()
{
}

bool EventTriggerCodeGen::eventCodeGen(llvm::Value *modelData,
        llvm::Value *data, const libsbml::Event* event)
{
    ModelDataIRBuilder mdBuilder(modelData, dataSymbols, builder);
    ModelDataLoadSymbolResolver mdLoadResolver(modelData, modelGenContext);
    ModelDataStoreSymbolResolver mdStoreResolver(modelData, model, modelSymbols,
            dataSymbols, builder, mdLoadResolver);
    ASTNodeCodeGen astCodeGen(builder, mdLoadResolver, modelGenContext, modelData);

    const ListOfEventAssignments *assignments =
            event->getListOfEventAssignments();

    for (uint id = 0; id < assignments->size(); ++id)
    {
        const EventAssignment *a = assignments->get(id);
        const ASTNode *math = a->getMath();
        Value *value = astCodeGen.codeGen(math);

        Value *loc = builder.CreateConstGEP1_32(data, id);
        builder.CreateStore(value, loc);
    }

    return true;
}

} /* namespace rr */


