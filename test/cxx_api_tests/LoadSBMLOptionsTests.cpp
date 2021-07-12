//
// Created by Ciaran on 25/05/2021.
//

#include "gtest/gtest.h"
#include "TestModelFactory.h"
#include "rrRoadRunnerOptions.h" // for LoadSBMLOptions
#include "rrConfig.h"
#include <bitset>

using namespace rr;

/**
 * @brief tests for LoadSBMLOptions.
 * @developers These tests focus on the mechanism for turning options on and off
 * which (at least to me) was not obvious. Roadrunner LoadSBMLOptions uses a single
 * 16-bit binary string for storing options, 1 == True and 0 == false. Bitwise
 * operators are used for checking whether an option is on or off.
 * These tests do not focus on the semantic meaning of any individual option, which
 * should be tested in their own unit. (For instance "recompile" tests belong
 * inside the Model unit test).
 */
class LoadSBMLOptionsTests : public ::testing::Test {
public :

    LoadSBMLOptionsTests() {
        // Ensure everything is off before we begin tests
        Config::setValue(Config::LOADSBMLOPTIONS_CONSERVED_MOIETIES, false);
        Config::setValue(Config::LOADSBMLOPTIONS_RECOMPILE, false);
        Config::setValue(Config::LOADSBMLOPTIONS_READ_ONLY, false);
        Config::setValue(Config::LOADSBMLOPTIONS_MUTABLE_INITIAL_CONDITIONS, false);
        Config::setValue(Config::LOADSBMLOPTIONS_OPTIMIZE_GVN, false);
        Config::setValue(Config::LOADSBMLOPTIONS_OPTIMIZE_CFG_SIMPLIFICATION, false);
        Config::setValue(Config::LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_COMBINING, false);
        Config::setValue(Config::LOADSBMLOPTIONS_OPTIMIZE_DEAD_INST_ELIMINATION, false);
        Config::setValue(Config::LOADSBMLOPTIONS_OPTIMIZE_DEAD_CODE_ELIMINATION, false);
        Config::setValue(Config::LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_SIMPLIFIER, false);
        Config::setValue(Config::LOADSBMLOPTIONS_USE_MCJIT, false);
        Config::setValue(Config::LLVM_SYMBOL_CACHE, false);
    }

    void turnOptionOn(Config::Keys key, LoadSBMLOptions::ModelGeneratorOpt option) {
        Config::setValue(key, true);
        LoadSBMLOptions options;
        std::cout << "options.modelGeneratorOpt: " << std::bitset<16>(options.modelGeneratorOpt) << std::endl;
        std::cout << (options.modelGeneratorOpt & option) << std::endl;
        ASSERT_TRUE((bool(options.modelGeneratorOpt & option)));
    }

};

TEST_F(LoadSBMLOptionsTests, ConservedMoieties){
    turnOptionOn(Config::LOADSBMLOPTIONS_CONSERVED_MOIETIES, LoadSBMLOptions::CONSERVED_MOIETIES);
}

TEST_F(LoadSBMLOptionsTests, Recompile){
    turnOptionOn(Config::LOADSBMLOPTIONS_RECOMPILE, LoadSBMLOptions::RECOMPILE);
}

TEST_F(LoadSBMLOptionsTests, ReadOnly){
    turnOptionOn(Config::LOADSBMLOPTIONS_READ_ONLY, LoadSBMLOptions::READ_ONLY);
}

TEST_F(LoadSBMLOptionsTests, MutableInitialConditions){
    turnOptionOn(Config::LOADSBMLOPTIONS_MUTABLE_INITIAL_CONDITIONS, LoadSBMLOptions::MUTABLE_INITIAL_CONDITIONS);
}

TEST_F(LoadSBMLOptionsTests, OptimizeGVN){
    turnOptionOn(Config::LOADSBMLOPTIONS_OPTIMIZE_GVN, LoadSBMLOptions::OPTIMIZE_GVN);
}

TEST_F(LoadSBMLOptionsTests, OptimizeCFGSimpleflication){
    turnOptionOn(Config::LOADSBMLOPTIONS_OPTIMIZE_CFG_SIMPLIFICATION, LoadSBMLOptions::OPTIMIZE_CFG_SIMPLIFICATION);
}

TEST_F(LoadSBMLOptionsTests, OptimizeIntructionCombining){
    turnOptionOn(Config::LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_COMBINING, LoadSBMLOptions::OPTIMIZE_INSTRUCTION_COMBINING);
}

TEST_F(LoadSBMLOptionsTests, OptimizeDeadInstElimination){
    turnOptionOn(Config::LOADSBMLOPTIONS_OPTIMIZE_DEAD_INST_ELIMINATION, LoadSBMLOptions::OPTIMIZE_DEAD_INST_ELIMINATION);
}

TEST_F(LoadSBMLOptionsTests, OptimizeDeadCodeElimination){
    turnOptionOn(Config::LOADSBMLOPTIONS_OPTIMIZE_DEAD_CODE_ELIMINATION, LoadSBMLOptions::OPTIMIZE_DEAD_CODE_ELIMINATION);
}

TEST_F(LoadSBMLOptionsTests, OptimizeInstructionSimplifier){
    turnOptionOn(Config::LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_SIMPLIFIER, LoadSBMLOptions::OPTIMIZE_INSTRUCTION_SIMPLIFIER);
}

TEST_F(LoadSBMLOptionsTests, UseMCJit){
    turnOptionOn(Config::LOADSBMLOPTIONS_USE_MCJIT, LoadSBMLOptions::USE_MCJIT);
}

TEST_F(LoadSBMLOptionsTests, LLVMSymbolCache){
    turnOptionOn(Config::LLVM_SYMBOL_CACHE, LoadSBMLOptions::LLVM_SYMBOL_CACHE);
}

TEST_F(LoadSBMLOptionsTests, ConservedMoietyConversionGetterAndSetter){
    LoadSBMLOptions options;
    options.setConservedMoietyConversion(true);
    ASSERT_TRUE(options.getConservedMoietyConversion());
}







