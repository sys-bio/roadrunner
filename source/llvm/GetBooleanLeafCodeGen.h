/*
 * GetBooleanLeafCodeGen.h
 *
 */

#ifndef RRLLVMGetBooleanLeafCodeGen_H_
#define RRLLVMGetBooleanLeafCodeGen_H_

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
    typedef unsigned char (*GetBooleanLeafCodeGen_FunctionPtr)(LLVMModelData*, size_t);

    /** @class GetBooleanLeafCodeGen
    * Class for getting the values of the relational leaves of event triggers.
    * These are extra roots for CVODE, so that a trigger that is true for only
    * part of a step is noticed.
    */
    class GetBooleanLeafCodeGen :
        public CodeGenBase<GetBooleanLeafCodeGen_FunctionPtr>
    {
    public:
        GetBooleanLeafCodeGen(const ModelGeneratorContext& mgc);
        virtual ~GetBooleanLeafCodeGen() {};

        /**
         * Creates a code block for every boolean leaf in the model.
         */
        llvm::Value* codeGen();

        typedef GetBooleanLeafCodeGen_FunctionPtr FunctionPtr;

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
        const std::vector<libsbml::ASTNode*>* booleanLeaves;
    };

} /* namespace rrllvm */

#endif /* RRLLVMGetBooleanLeafCodeGen_H_ */
