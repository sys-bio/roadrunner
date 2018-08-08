/*
 * CodeGenBase.h
 *
 *  Created on: Jul 4, 2013
 *      Author: andy
 */

#ifndef LLVMCodeGenBaseH
#define LLVMCodeGenBaseH

#include "ModelGeneratorContext.h"
#include "CodeGen.h"
#include "LLVMException.h"
#include "rrLogger.h"
#include <Poco/Logger.h>

using rr::Logger;
using rr::getLogger;

namespace rrllvm
{

typedef std::vector<std::string> StringVector;
typedef std::pair<std::string, int> StringIntPair;
typedef std::vector<StringIntPair> StringIntVector;

/**
 * a convenience class to pull the vars out of a context, and
 * store them as ivars. It can get tedious alwasy typing mgc.getThis
 * and mgc.getThat. Furthermore, its faster to access them as ivars
 * as it does not incur a func call each time.
 */
template <typename FunctionPtrType>
class CodeGenBase
{
public:
    llvm::Function* createFunction()
    {
        llvm::Function *func = (llvm::Function*)codeGen();

        if(functionPassManager)
        {
            functionPassManager->run(*func);
        }

		return func;
    }

    typedef FunctionPtrType FunctionPtr;

protected:
    CodeGenBase(const ModelGeneratorContext &mgc) :
            modelGenContext(mgc),
            model(mgc.getModel()),
            dataSymbols(mgc.getModelDataSymbols()),
            modelSymbols(mgc.getModelSymbols()),
            context(mgc.getContext()),
            module(mgc.getModule()),
            builder(mgc.getBuilder()),
            engine(mgc.getExecutionEngine()),
            options(mgc.getOptions()),
            function(0),
            functionPassManager(mgc.getFunctionPassManager())
    {
    };

    const ModelGeneratorContext &modelGenContext;

    /**
     * could potentially be null, everything else is guaranteed to be valid
     */
    const libsbml::Model *model;

    const LLVMModelDataSymbols &dataSymbols;
    const LLVMModelSymbols &modelSymbols;

    llvm::LLVMContext &context;
    llvm::Module *module;
    llvm::IRBuilder<> &builder;
    llvm::ExecutionEngine &engine;
    llvm::Function *function;

    /**
     * function pass manager. Null if no optimization.
     */
    llvm::legacy::FunctionPassManager *functionPassManager;

    /**
     * the options bit field that was passed into the top level load method.
     */
    const unsigned options;

    virtual llvm::Value *codeGen() = 0;

    template <size_t N>
    llvm::BasicBlock *codeGenHeader(const char* functionName,
            llvm::Type *retType,
            llvm::Type* (&argTypes)[N],
            const char* (&argNames)[N],
            llvm::Value* (&args)[N])
    {
        // make the set init value function

        llvm::FunctionType *funcType = llvm::FunctionType::get(retType,
                argTypes, false);
        function = llvm::Function::Create(funcType,
                llvm::Function::InternalLinkage,
                functionName, module);

        // Create a new basic block to start insertion into.
        llvm::BasicBlock *basicBlock = llvm::BasicBlock::Create(context,
                "entry", function);
        builder.SetInsertPoint(basicBlock);

        //assert(function->getArgumentList().size() == N);

        int i = 0;
        for (llvm::Function::arg_iterator ai = function->arg_begin();
                ai != function->arg_end(); ++ai, ++i)
        {
            llvm::Value *arg = ai;
            arg->setName(argNames[i]);
            args[i] = arg;
        }

        return basicBlock;
    }

    /**
     * the most common type of generated function takes a ModelData*,
     * and returns void.
     */
    llvm::BasicBlock *codeGenVoidModelDataHeader(const char* functionName,
            llvm::Value* &modelData)
    {
        llvm::Type *argTypes[] = {
            llvm::PointerType::get(
                ModelDataIRBuilder::getStructType(module), 0)
        };

        const char *argNames[] = { "modelData" };

        llvm::Value *args[] = { 0 };

        llvm::BasicBlock *basicBlock = codeGenHeader(functionName,
                llvm::Type::getVoidTy(context),
                    argTypes, argNames, args);

        modelData = args[0];
        return basicBlock;
    }

    llvm::Function *verifyFunction()
    {
        poco_information(getLogger(),
            string("function: ") + to_string(function));

        /// verifyFunction - Check a function for errors, printing messages on stderr.
        /// Return true if the function is corrupt.
//#if (LLVM_VERSION_MAJOR == 3) && (LLVM_VERSION_MINOR >= 5)
#if (LLVM_VERSION_MAJOR == 6)
        if (llvm::verifyFunction(*function))
#else
        if (llvm::verifyFunction(*function, llvm::AbortProcessAction))
#endif
        {
            poco_error(getLogger(),
                    "Corrupt Generated Function, "  + to_string(function));

            throw LLVMException("Generated function is corrupt, see stderr",
                    __FUNC__);
        }

        return function;
    }


    virtual ~CodeGenBase() {};


};

} /* namespace rr */
#endif /* LLVMCodeGenBaseH */
