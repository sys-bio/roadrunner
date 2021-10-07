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

    //Go through the event assignments twice, to ensure that we change any compartment volumes *after* we change any species concentrations, so that the species amounts are assigned correctly.
    const libsbml::Model* model = event->getModel();
    for(uint id = 0; id < assignments->size(); ++id)
    {
        const EventAssignment *a = assignments->get(id);
        if (!a->isSetMath()) {
            continue;
        }
        if (model->getCompartment(a->getVariable()) == NULL)
        {
            Value* loc = builder.CreateConstGEP1_32(data, id);
            Value* value = builder.CreateLoad(loc, a->getVariable() + "_data");
            mdStoreResolver.storeSymbolValue(a->getVariable(), value);
        }
    }

    for (uint id = 0; id < assignments->size(); ++id)
    {
        const EventAssignment* a = assignments->get(id);
        if (!a->isSetMath()) {
            continue;
        }
        if (model->getCompartment(a->getVariable()) != NULL)
        {
            Value* loc = builder.CreateConstGEP1_32(data, id);
            Value* value = builder.CreateLoad(loc, a->getVariable() + "_data");
            mdStoreResolver.storeSymbolValue(a->getVariable(), value);
        }
    }

    return true;
}


} /* namespace rr */


