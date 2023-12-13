//
// Created by Ciaran on 27/04/2021.
//

#include "RoadRunnerTest.h"

RoadRunnerTest::RoadRunnerTest()
    : rrTestDir_(getRoadRunnerTestDirectory())
    , rrPluginsBuildDir_(getRoadRunnerPluginBuildDirectory())
{
    RoadRunnerTest::validateRoadRunnerTestDir();
    rrTestModelsDir_ = rrTestDir_ / "models";
    if (!fs::exists(rrTestModelsDir_)) {
        std::cerr << "RoadRunner test models directory does not exist \""
                  << rrTestModelsDir_ << "\"";
    }
    rrTestSbmlTestSuiteDir_ = rrTestDir_ / "sbml-test-suite";
    if (!fs::exists(rrTestSbmlTestSuiteDir_)) {
        std::cerr << "RoadRunner sbml test suite directory does not exist \""
                  << rrTestSbmlTestSuiteDir_ << "\"";
    }

}

void RoadRunnerTest::validateRoadRunnerTestDir()
{
    // if we can locate <test dir>/RoadRunnerTest.h, then we know where
    // we are and can derive other directories from here.

    fs::path rrTestDotH = rrTestDir_ / fs::path("RoadRunnerTest.h");

    if (fs::exists(rrTestDotH)) {
        return;
    }
    // if not then we fall back on the old system of looking for the "testdir"
    // environment variable

    char *testdir = getenv("testdir");

    if (testdir) {
        rrTestDir_ = testdir;
    }
    if (rrTestDir_.string().empty()) {
        std::cerr << "Please set the 'testdir' environment variable before running tests.  "
                     "This should be the 'test/' directory of the roadrunner root directory."
                  << std::endl;
        return;
    }
    rrTestDotH = rrTestDir_ / "RoadRunnerTest.h";
    if (!fs::exists(rrTestDotH)) {
        std::cerr << "Could not locate a file know to exist in the roadrunner testdir (<roadrunnerRoot>/test). "
                     "Please check the accuracy of the testdir environment variable "
                     "(testdir = \"" << rrTestDir_ << "\"" << std::endl;
        return;
    }
}

void RoadRunnerTest::checkMatrixEqual(ls::DoubleMatrix expectedMatrix, ls::DoubleMatrix actualMatrix, double absError)
{
    if (expectedMatrix.numRows() != actualMatrix.numRows()) {
        std::cerr << "number of rows in expected Vs actual are not equal" << std::endl;
        ASSERT_EQ(expectedMatrix.numRows(), actualMatrix.numRows());
    }

    if (expectedMatrix.numCols() != actualMatrix.numCols()) {
        std::cerr << "number of cols in expected Vs actual are not equal" << std::endl;
        ASSERT_EQ(expectedMatrix.numCols(), actualMatrix.numCols());
    }

    for (int i = 0; i < expectedMatrix.numRows(); i++) {
        for (int j = 0; j < expectedMatrix.numCols(); j++) {
            //std::cout << "Comparing expected result: " << expectedMatrix[i][j]
            //          << " with actual result: " << actualMatrix[i][j] << std::endl;
            EXPECT_NEAR(expectedMatrix[i][j], actualMatrix[i][j], absError);
        }
    }
}


