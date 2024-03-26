/*
 * GetPiecewiseTriggerCodeGen.h
 *
 *  Created on: Aug 10, 2013
 *      Author: andy
 */

#ifndef RRLLVMGetPiecewiseTriggerCodeGen_H_
#define RRLLVMGetPiecewiseTriggerCodeGen_H_

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
    //Based on GetEventTriggerCodeGen (-LS)

    typedef unsigned char (*GetPiecewiseTriggerCodeGen_FunctionPtr)(LLVMModelData*, size_t);

    class GetPiecewiseTriggerCodeGen :
        public CodeGenBase<GetPiecewiseTriggerCodeGen_FunctionPtr>
    {
    public:
        GetPiecewiseTriggerCodeGen(const ModelGeneratorContext& mgc);
        virtual ~GetPiecewiseTriggerCodeGen() {};

        llvm::Value* codeGen();

        typedef GetPiecewiseTriggerCodeGen_FunctionPtr FunctionPtr;

        static const char* FunctionName;
        static const char* IndexArgName;

        llvm::Type* getRetType();

        llvm::Value* createRet(llvm::Value*);

    private:
        const std::vector<libsbml::ASTNode*>* piecewiseTriggers;
    };


} /* namespace rr */




#endif /* RRLLVMGETVALUECODEGENBASE_H_ */
