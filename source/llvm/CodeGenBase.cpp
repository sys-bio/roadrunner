/*
 * rrLLVMCodeGenBase.cpp
 *
 *  Created on: Jul 18, 2013
 *      Author: andy
 */
#pragma hdrstop
#include "CodeGenBase.h"
#include "ASTNodeCodeGen.h"
#include "LLVMException.h"
#include "ModelGeneratorContext.h"
#include "SymbolForest.h"
#include "ASTNodeFactory.h"
#include "ModelDataIRBuilder.h"
#include "GetValueCodeGenBase.h"
#include <sbml/Model.h>

namespace rrllvm
{
using namespace libsbml;
using namespace llvm;
using namespace std;



class LLVMCodeGenTest
{

protected:

    LLVMCodeGenTest(const ModelGeneratorContext &mgc) :
            model(mgc.getModel()),
            dataSymbols(mgc.getModelDataSymbols()),
            modelSymbols(mgc.getModelSymbols()),
            context(mgc.getContext()),
            module(mgc.getModule()),
            builder(mgc.getBuilder()),
            engine(mgc.getExecutionEngine())
    {
    };

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
public:

    template <size_t N>
    static void test(const int (&Arr)[N], double out[N])
    {
        cout << N;

        for (int i = 0; i < N; i++) {
            out[i] = Arr[i];
        }
    }

    template <size_t N>
    void codeGenHeader(const char* functionName, llvm::Type *retType,
            llvm::Type (*argTypes)[N],
            const char (*argNames)[N],
            llvm::Value (*args)[N])
    {
        // make the set init value function

        FunctionType *funcType = FunctionType::get(retType, argTypes,
                false);
        Function *func = Function::Create(funcType, Function::InternalLinkage,
                functionName, module);

        // Create a new basic block to start insertion into.
        BasicBlock *basicBlock = BasicBlock::Create(context, "entry", func);
        builder.SetInsertPoint(basicBlock);

        assert(func->arg_size() == N);

        int i = 0;

        for (llvm::Function::arg_iterator ai = func->arg_begin();
                ai != func->arg_end(); ++ai, ++i)
        {
            llvm::Value *arg = ai;
            arg->setName(argNames[i]);
            args[i] = arg;
        }
    }


    virtual ~LLVMCodeGenTest() {};
};

void foo()
{
    //int vars[] = {1, 2, 3, 4};

    //double out[4];

    //LLVMCodeGenTest::test(vars, out);


}



}
