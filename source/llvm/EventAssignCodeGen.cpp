/*
 * EventAssignCodeGen.cpp
 *
 *  Created on: Aug 11, 2013
 *      Author: andy
 */
#pragma hdrstop
#include "EventAssignCodeGen.h"
#include "ModelDataIRBuilder.h"
#include "ModelDataSymbolResolver.h"
#include "ASTNodeCodeGen.h"

namespace rrllvm
{

using namespace llvm;
using namespace libsbml;

const char* EventAssignCodeGen::FunctionName = "eventAssign";

EventAssignCodeGen::EventAssignCodeGen(const ModelGeneratorContext& mgc) :
        EventCodeGenBase<EventAssignCodeGen>(mgc)
{
}

EventAssignCodeGen::~EventAssignCodeGen()
{
}

bool EventAssignCodeGen::eventCodeGen(llvm::Value *modelData,
        llvm::Value *data, const libsbml::Event *event)
{
    ModelDataIRBuilder mdBuilder(modelData, dataSymbols, builder);
    ModelDataLoadSymbolResolver mdLoadResolver(modelData, modelGenContext);
    ModelDataStoreSymbolResolver mdStoreResolver(modelData, model, modelSymbols,
            dataSymbols, builder, mdLoadResolver);

    const ListOfEventAssignments *assignments = event->getListOfEventAssignments();

    for(uint id = 0; id < assignments->size(); ++id)
    {
        const EventAssignment *a = assignments->get(id);
        if (!a->isSetMath()) {
            continue;
        }
        Value *loc = builder.CreateConstGEP1_32(data, id);
        Value *value = builder.CreateLoad(loc, a->getVariable() + "_data");
        mdStoreResolver.storeSymbolValue(a->getVariable(), value);
    }

    return true;
}


} /* namespace rr */


