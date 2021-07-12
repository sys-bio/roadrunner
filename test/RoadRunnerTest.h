//
// Created by Ciaran on 27/04/2021.
//

#ifndef ROADRUNNER_ROADRUNNERTEST_H
#define ROADRUNNER_ROADRUNNERTEST_H

#include "C/rrc_api.h"
#include "rrIniFile.h"
#include "rrLogger.h"
#include <filesystem>
#include <rr-libstruct/lsMatrix.h>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "RoadRunnerTestDirectory.h"

namespace fs = std::filesystem;
using fs::path;


/**
 * @brief Parent class for RoadRunner tests
 * @note this construct replaces main.cpp
 */
class RoadRunnerTest : public ::testing::Test {
public:

    RoadRunnerTest();

    ~RoadRunnerTest()  override = default;

    static void checkMatrixEqual(ls::DoubleMatrix expectedMatrix, ls::DoubleMatrix actualMatrix, double absError = 1e-3) ;

protected:
    /**
     * @brief check validity of the rrTestDir_ variable
     * @details A CMake configure_file operation is conducted on
     * RoadRunnerTestDirectory.h.in which puts it in the binary tree.
     * RoadRunnerTest reads this file and extracts the testdir variable
     * from it. If this method finds a known file under the testdir
     * folder, then the rrTestDir_ variable is validated. If this
     * test fails, it'll try to locate the same directory via the
     * testdir environment variable
     */
    void validateRoadRunnerTestDir();

    path rrTestDir_;
    path rrTestModelsDir_;
    path rrTestSbmlTestSuiteDir_;
};


#endif //ROADRUNNER_ROADRUNNERTEST_H
