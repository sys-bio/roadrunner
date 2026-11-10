//
// Created by Ciaran on 15/11/2021.
//

#ifndef ROADRUNNER_LLVMGENERATEDFUNCTIONTESTS_H
#define ROADRUNNER_LLVMGENERATEDFUNCTIONTESTS_H

#include "gtest/gtest.h"
#include "rrLogger.h"
#include "LLVMModelData.h"
#include "sbml/SBMLTypes.h"
#include "rrRoadRunnerOptions.h"
#include "JitFactory.h"

#include "EvalInitialConditionsCodeGen.h"
#include "GetValuesCodeGen.h"

using namespace rr;
using namespace rrllvm;
using namespace libsbml;

/**
 * These tests, and those which derive from it
 * are not yet a complete test suite. Instead they
 * serve to help me debug some issues and might as
 * well persist. Eventually it'd be nice to
 * expand on these tests (and see Derived classes)
 * to get comprehensive coverage of the llvm generated
 * code.
 */


/**
 * @brief Test suite for llvm part of the code
 */
class LLVMGeneratedFunctionTests : public ::testing::Test {
public:
    LLVMGeneratedFunctionTests() {
        //Logger::setLevel(Logger::LOG_INFORMATION);
    }

    void checkInitialEvaluationSpecies(int options, int speciesAmt, double compVol, double expectedConc);
};


#endif //ROADRUNNER_LLVMGENERATEDFUNCTIONTESTS_H
