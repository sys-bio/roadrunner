//
// Created by Ciaran on 15/11/2021.
//


#include "gtest/gtest.h"
#include "rrConfig.h"
#include "rrRoadRunnerOptions.h"
#include "JitFactory.h"
#include <memory>
#include "Jit.h"

using namespace rrllvm;
using namespace rr;

class JitFactoryTests : public ::testing::Test {
public:
    JitFactoryTests() = default;
};


TEST_F(JitFactoryTests, MCJitFromConfig){
    Config::setValue(Config::LLVM_BACKEND, Config::LLVM_BACKEND_VALUES::MCJIT);
    Jit* j = JitFactory::makeJitEngine();
    ASSERT_TRUE(typeid((*j)) == typeid(MCJit));
}

TEST_F(JitFactoryTests, LLJitFromConfig){
    Config::setValue(Config::LLVM_BACKEND, Config::LLVM_BACKEND_VALUES::LLJIT);
    Jit* j = JitFactory::makeJitEngine();
    ASSERT_TRUE(typeid((*j)) == typeid(LLJit));
}

TEST_F(JitFactoryTests, MCJitFromOpt){
    LoadSBMLOptions opt;
    opt.setLLVMBackend(LoadSBMLOptions::MCJIT);
    Jit* j = JitFactory::makeJitEngine(opt.modelGeneratorOpt);
    ASSERT_TRUE(typeid((*j)) == typeid(MCJit));
}

TEST_F(JitFactoryTests, LLJitFromOpt){
    LoadSBMLOptions opt;
    opt.setLLVMBackend(LoadSBMLOptions::LLJIT);
    Jit* j = JitFactory::makeJitEngine(opt.modelGeneratorOpt);
    ASSERT_TRUE(typeid((*j)) == typeid(LLJit));
}
