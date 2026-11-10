//
// Created by Ciaran on 15/11/2021.
//

#include "gtest/gtest.h"
#include "rrLogger.h"
#include "LLVMModelData.h"
#include "SBMLInitialValueSymbolResolver.h"
#include "sbml/SBMLTypes.h"
#include "rrRoadRunnerOptions.h"
#include "JitFactory.h"
#include "TestModelFactory.h"

#include "LLVMGeneratedFunctionTests.h"
#include <EvalInitialConditionsCodeGen.h>
#include "GetValuesCodeGen.h"

using namespace rr;
using namespace rrllvm;
using namespace libsbml;

/**
 * @brief Test suite for llvm part of the code
 * beginning of a test suite for llvm generated Jitt'ed code.
 *
 */
class EvalInitialConditionsTests : public LLVMGeneratedFunctionTests {
public:
    EvalInitialConditionsTests() {
        //Logger::setLevel(Logger::LOG_INFORMATION);
    }
};

/**
 * Note on EvalInitialConditions (from comment where
 * ffunction is used, not docs):
    // initializes the model to the init values specified in the sbml, and
    // copies these to the initial initial conditions (not a typo),
    // sets the 'init(...)' values to the sbml specified init values.
 */

/**
 * Put 300K particles of S1 into Cell of size 12 (any units), what is the
 * concentration of S1?
 * 25000 * 12 = 300000
 * 300000 / 12 = 25000
 */
TEST_F(EvalInitialConditionsTests, CheckInitialSpeciesComputedCorrectlyMCJit) {
        LoadSBMLOptions opt;
    opt.setLLVMBackend(LoadSBMLOptions::MCJIT);
        checkInitialEvaluationSpecies(opt.modelGeneratorOpt, 300000, 12, 25000);
}

TEST_F(EvalInitialConditionsTests, CheckInitialSpeciesComputedCorrectlyLLJit) {
        LoadSBMLOptions opt;
    opt.setLLVMBackend(LoadSBMLOptions::LLJIT);
        checkInitialEvaluationSpecies(opt.modelGeneratorOpt, 300000, 12, 25000);
}

//
//TEST_F(EvalInitialConditionsTests, evalReactionRatesPtr) {
//
//}
//
//TEST_F(EvalInitialConditionsTests, getBoundarySpeciesAmountPtr) {
//
//}
//
//TEST_F(EvalInitialConditionsTests, getFloatingSpeciesAmountPtr) {
//
//}
//
//TEST_F(EvalInitialConditionsTests, getBoundarySpeciesConcentrationPtr) {
//
//}
//
//TEST_F(EvalInitialConditionsTests, getFloatingSpeciesConcentrationPtr) {
//
//}
//
//TEST_F(EvalInitialConditionsTests, getCompartmentVolumePtr) {
//
//}
//
//TEST_F(EvalInitialConditionsTests, getGlobalParameterPtr) {
//
//}
//
//TEST_F(EvalInitialConditionsTests, evalRateRuleRatesPtr) {
//
//}
//
//TEST_F(EvalInitialConditionsTests, getEventTriggerPtr) {
//
//}
//
//TEST_F(EvalInitialConditionsTests, getEventPriorityPtr) {
//
//}
//
//TEST_F(EvalInitialConditionsTests, getEventDelayPtr) {
//
//}
//
//TEST_F(EvalInitialConditionsTests, eventTriggerPtr) {
//
//}
//
//TEST_F(EvalInitialConditionsTests, eventAssignPtr) {
//
//}
//
//TEST_F(EvalInitialConditionsTests, evalVolatileStoichPtr) {
//
//}
//
//TEST_F(EvalInitialConditionsTests, evalConversionFactorPtr) {
//
//}
//
//TEST_F(EvalInitialConditionsTests, setBoundarySpeciesAmountPtr) {
//
//}
//
//TEST_F(EvalInitialConditionsTests, setFloatingSpeciesAmountPtr) {
//
//}
//
//TEST_F(EvalInitialConditionsTests, setBoundarySpeciesConcentrationPtr) {
//
//}
//
//TEST_F(EvalInitialConditionsTests, setFloatingSpeciesConcentrationPtr) {
//
//}
//
//TEST_F(EvalInitialConditionsTests, setCompartmentVolumePtr) {
//
//}
//
//TEST_F(EvalInitialConditionsTests, setGlobalParameterPtr) {
//
//}
//
//TEST_F(EvalInitialConditionsTests, getFloatingSpeciesInitConcentrationsPtr) {
//
//}
//
//TEST_F(EvalInitialConditionsTests, setFloatingSpeciesInitConcentrationsPtr) {
//
//}
//
//TEST_F(EvalInitialConditionsTests, getFloatingSpeciesInitAmountsPtr) {
//
//}
//
//TEST_F(EvalInitialConditionsTests, setFloatingSpeciesInitAmountsPtr) {
//
//}
//
//TEST_F(EvalInitialConditionsTests, getBoundarySpeciesInitConcentrationsPtr) {
//
//}
//
//TEST_F(EvalInitialConditionsTests, setBoundarySpeciesInitConcentrationsPtr) {
//
//}
//
//TEST_F(EvalInitialConditionsTests, getBoundarySpeciesInitAmountsPtr) {
//
//}
//
//TEST_F(EvalInitialConditionsTests, setBoundarySpeciesInitAmountsPtr) {
//
//}
//
//TEST_F(EvalInitialConditionsTests, getCompartmentInitVolumesPtr) {
//
//}
//
//TEST_F(EvalInitialConditionsTests, setCompartmentInitVolumesPtr) {
//
//}
//
//TEST_F(EvalInitialConditionsTests, getGlobalParameterInitValuePtr) {
//
//}
//
//TEST_F(EvalInitialConditionsTests, setGlobalParameterInitValuePtr) {
//
//}
//
//
