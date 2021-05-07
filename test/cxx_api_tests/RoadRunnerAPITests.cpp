//
// Created by Ciaran on 07/05/2021.
//

#include "gtest/gtest.h"

// todo move test model factor up one directory
#include "../sundials-tests/TestModelFactory.h"

#include "rrRoadRunner.h"
#include "rrConfig.h"
using namespace rr;

/**
 * This is a more of a stub test suite
 * than a full test suite at the moment.
 * It will eventually fully test every
 * aspect of the rr::RoadRunner API
 * For now, it only tests the things
 * that need a new test to fix a bug.
 */

class RoadRunnerAPITests : public ::testing::Test{

public:
    OpenLinearFlux openLinearFlux;
    RoadRunnerAPITests() = default;
};

TEST_F(RoadRunnerAPITests, DefaultJacobianMode){
    Setting x = Config::getValue(
            Config::ROADRUNNER_JACOBIAN_MODE
    );
    ASSERT_TRUE(x.get<int>() == Config::ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS);
}

TEST_F(RoadRunnerAPITests, SetJacobianModeToAmt){
    /**
     * Implicit type conversion has got us into a spot of trouble here.
     * The Config keys are being implicitly converted into an int,
     * rather than their original type, an unsigned int.
     */
    Config::setValue(Config::ROADRUNNER_JACOBIAN_MODE , Config::ROADRUNNER_JACOBIAN_MODE_AMOUNTS);
    Setting x = Config::getValue(
            Config::ROADRUNNER_JACOBIAN_MODE
    );
    ASSERT_TRUE(x.get<int>() == Config::ROADRUNNER_JACOBIAN_MODE_AMOUNTS);
}

TEST(not, finished){
    // complete the jacobian tests
    ASSERT_FALSE(true);
}

TEST_F(RoadRunnerAPITests, GetFullJacobian){
    RoadRunner rr(openLinearFlux.str());
    auto matrix = rr.getFullJacobian();
    std::cout << matrix << std::endl;
}
TEST_F(RoadRunnerAPITests, GetFullJacobianUsingConcMode){
    Config::setValue(Config::ROADRUNNER_JACOBIAN_MODE, Config::ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS);
    RoadRunner rr(openLinearFlux.str());
    auto matrix = rr.getFullJacobian();
    std::cout << matrix << std::endl;
}

TEST_F(RoadRunnerAPITests, GetFullJacobianUsingAmtMode){
    Config::setValue(Config::ROADRUNNER_JACOBIAN_MODE, Config::ROADRUNNER_JACOBIAN_MODE_AMOUNTS);
    RoadRunner rr(openLinearFlux.str());
    auto matrix = rr.getFullJacobian();
    std::cout << matrix << std::endl;
}

TEST_F(RoadRunnerAPITests, GetFullJacobianUsingAmtModeAsLong){
    /**
     * Some context for developers:
     * Python uses long for int values. So when
     * Config::setValue is used from Python, the Setting that
     * is implicitly created (second argument to Config::setValue) is a
     * std::int64_t, not an int! The Setting
     * is completely capable of handling an int64, but in other
     * places (getFullJacobian, for example) an int is expected,
     * generating an std::bad_variant_access error.
     * The solution is to use the Setting::getAs<int> command
     * instead of Setting::get<int>, which will do the
     * conversion, if one is needed.
     */
    Config::setValue(Config::ROADRUNNER_JACOBIAN_MODE, std::int64_t(Config::ROADRUNNER_JACOBIAN_MODE_AMOUNTS));
    RoadRunner rr(openLinearFlux.str());
    auto matrix = rr.getFullJacobian();
    std::cout << matrix << std::endl;
}
