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

#ifdef _MSC_VER
#pragma warning(disable: 4146)
#pragma warning(disable: 4141)
#pragma warning(disable: 4267)
#pragma warning(disable: 4624)
#endif

#include "llvm/IR/Mangler.h"

#ifdef _MSC_VER
#pragma warning(default: 4146)
#pragma warning(default: 4141)
#pragma warning(default: 4267)
#pragma warning(default: 4624)
#endif


using rr::Logger;
using rr::getLogger;

namespace rrllvm
{

    class Jit;

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
            //functionPassManager->run(*func);
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
            context(mgc.getJitNonOwning()->getModuleNonOwning()->getContext()),
            module(mgc.getJitNonOwning()->getModuleNonOwning()),
            builder(*mgc.getJitNonOwning()->getBuilderNonOwning()),
            options(mgc.getOptions()),
            function(0)
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
    llvm::Function *function;

    /**
     * function pass manager. Null if no optimization.
     */
    llvm::legacy::FunctionPassManager* functionPassManager;

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

        llvm::FunctionType *funcType = llvm::FunctionType::get(retType, argTypes, false);
        function = llvm::Function::Create(funcType,
                llvm::Function::ExternalLinkage,
                functionName, module);

        // Create a new basic block to start insertion into.
        llvm::BasicBlock *basicBlock = llvm::BasicBlock::Create(context,"entry", function);
        builder.SetInsertPoint(basicBlock);

        assert(function->arg_size() == N);

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
    llvm::BasicBlock *codeGenVoidModelDataHeader(const char* functionName,llvm::Value* &modelData)
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
        rrLogInfo << std::string("function: ") + to_string(function);

        /// verifyFunction - Check a function for errors, printing messages on stderr.
        /// Return true if the function is corrupt.
        if (llvm::verifyFunction(*function))
        {
            std::string err = "Corrupt Generated Function, "  + to_string(function);
            rrLogErr << err;

            throw LLVMException(err);
        }

        return function;
    }


    virtual ~CodeGenBase() {};


};

} /* namespace rr */
#endif /* LLVMCodeGenBaseH */
