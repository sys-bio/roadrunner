/**
 * Deprecated in favour of RoadRunnerTest
 */

#include "gtest/gtest.h"
#include "C/rrc_api.h"
#include "rrIniFile.h"
#include "rrLogger.h"
#include <filesystem>

using std::filesystem::path;
path     gRRTestDir= "";
path     gRROutputDir = "";
path     TestModelFileName = "";
rrc::RRHandle gRR = rrc::createRRInstance();;
rr::IniFile iniFile;

// question, is there a reason you wrote your own "main" rather than using gtest_main?
int main(int argc, char** argv)
{
    //Setup googletest
    ::testing::InitGoogleTest(&argc, argv);

    //Setup the global test directory:
    char* testdir = getenv("testdir");
    if (testdir) {
        gRRTestDir = testdir;
    }
    if (gRRTestDir.empty()) {
        std::cerr << "Please set the 'testdir' environment variable before running tests.  This should be the 'test/' directory of the roadrunner root directory." << std::endl;
        return 1;
    }

    //Setup the output directory (will be the relative directory "output/" by default)
    char* outdir = getenv("outdir");
    if (outdir) {
        gRROutputDir = outdir;
    }
    if (gRROutputDir.empty()) {
        gRROutputDir = "output";
    }

    //Set the logging level to 'notice':
    rr::Logger::enableConsoleLogging(rr::Logger::LOG_NOTICE);

    //Run the tests.
    int ret = RUN_ALL_TESTS();
    rrc::freeRRInstance(gRR);
    return ret;
}