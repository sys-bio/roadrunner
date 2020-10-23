/*
 * EventTriggerCodeGen.h
 *
 *  Created on: Aug 11, 2013
 *      Author: andy
 */

#ifndef EVENTTRIGGERCODEGEN_H_
#define EVENTTRIGGERCODEGEN_H_

#include "EventCodeGenBase.h"

namespace rrllvm
{

typedef void (*EventTriggerCodeGen_FunctionPtr)(LLVMModelData*, size_t);

class EventTriggerCodeGen: public EventCodeGenBase<EventTriggerCodeGen>
{
public:
    EventTriggerCodeGen(const ModelGeneratorContext &mgc);
    ~EventTriggerCodeGen();

    bool eventCodeGen(llvm::Value *modelData, llvm::Value *data,
            const libsbml::Event* event);

    static const char* FunctionName;
};

} /* namespace rr */
#endif /* EVENTTRIGGERCODEGEN_H_ */
