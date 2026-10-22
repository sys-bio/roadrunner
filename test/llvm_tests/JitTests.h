//
// Created by Ciaran on 25/10/2021.
//

#include "llvm/MCJit.h"
#include "gtest/gtest.h"
#include "TestModelFactory.h"
#include "rrLogger.h"
#include "rrRoadRunnerOptions.h"

using namespace rr;
using namespace rrllvm;

class JitTests : public ::testing::Test {
public:
    JitTests() = default;

    static llvm::Function *CreateFibFunction(llvm::Module *M) {
        llvm::LLVMContext &ctx = M->getContext();

        // Create the fib function and insert it into the model
        // This function is said to return an int and take an int
        // as parameter
        llvm::FunctionType *FibFTy = llvm::FunctionType::get(
                llvm::Type::getInt32Ty(ctx), {llvm::Type::getInt32Ty(ctx)}, false
        );

        llvm::Function *FibFn = llvm::Function::Create(FibFTy, llvm::Function::ExternalLinkage, "fib", M);

        //Add a basic memory block
        llvm::BasicBlock *BB = llvm::BasicBlock::Create(ctx, "EntryBlock", FibFn);

        // get pointers to the constants
        llvm::Value *One = llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 1);
        llvm::Value *Two = llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 2);

        // get ptr tointeger arg of the function
        llvm::Argument *ArgX = &*FibFn->arg_begin();
        ArgX->setName("AnArg");

        // the true block
        llvm::BasicBlock *RetBB = llvm::BasicBlock::Create(ctx, "return", FibFn);

        // and the exit block
        llvm::BasicBlock *RecurseB = llvm::BasicBlock::Create(ctx, "recurse", FibFn);

        // create the if arg < 2 goto exit
        llvm::Value *CondInst = new llvm::ICmpInst(BB->end(), llvm::ICmpInst::ICMP_SLE, ArgX, Two, "cond");
        llvm::BranchInst::Create(RetBB, RecurseB, CondInst, BB);
        llvm::ReturnInst::Create(ctx, One, RetBB);

        llvm::Value *sub = llvm::BinaryOperator::CreateSub(ArgX, One, "arg", RecurseB);
        llvm::Value *CallFibX1 = llvm::CallInst::Create(FibFn, sub, "fibx1", RecurseB);

        // create fib(x-2)
        sub = llvm::BinaryOperator::CreateSub(ArgX, Two, "arg", RecurseB);
        llvm::Value *CallFibX2 = llvm::CallInst::Create(FibFn, sub, "fibx2", RecurseB);

        // fib(x-1)+fib(x-2)
        llvm::Value *Sum = llvm::BinaryOperator::CreateAdd(CallFibX1, CallFibX2, "addresult", RecurseB);

        // Create the return instruction and add it to the basic block
        llvm::ReturnInst::Create(ctx, Sum, RecurseB);

        return FibFn;
    }
};

using fibonacciFnPtr = int (*)(int);
