/*
 * GetPiecewiseTriggerCodeGen.h
 *
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

    /** @class GetPiecewiseTriggerCodeGen
    * Class for getting piecewise trigger values.
    */
    class GetPiecewiseTriggerCodeGen :
        public CodeGenBase<GetPiecewiseTriggerCodeGen_FunctionPtr>
    {
    public:
        GetPiecewiseTriggerCodeGen(const ModelGeneratorContext& mgc);
        virtual ~GetPiecewiseTriggerCodeGen() {};

        /**
         * The heart of the class, that creates code blocks for every piecewise
         * trigger in the model.
         */
        llvm::Value* codeGen();

        typedef GetPiecewiseTriggerCodeGen_FunctionPtr FunctionPtr;

        static const char* FunctionName;
        static const char* IndexArgName;

        /**
         * The ret type is a boolean, aka 'getInt8Ty'
         */
        llvm::Type* getRetType();

        /**
         * create a return type, a zero value should return the default type
         */
        llvm::Value* createRet(llvm::Value*);

    private:
        const std::vector<libsbml::ASTNode*>* piecewiseTriggers;
    };


} /* namespace rr */




#endif /* RRLLVMGETVALUECODEGENBASE_H_ */
