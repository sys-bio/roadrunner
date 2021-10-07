#include "gtest/gtest.h"
#include "../test_util.h"
#include "C/rrc_api.h"
#include "rrLogger.h"
#include "rrUtils.h"
#include "rrIniFile.h"

#include "RoadRunnerTest.h"


#include <string>
#include <filesystem>
#include <rrRoadRunner.h>

using namespace std;
using namespace rrc;
using std::filesystem::path;


#include "CAPICoreTest.h"

class CAPIExceptions : public CAPICoreTest {
public:

    CAPIExceptions() = default;

};

TEST_F(CAPIExceptions, DATA_FILES) {
    path testDataFileName = rrTestModelsDir_ / "CAPICore";
    path testModelFileName;
    testDataFileName /= "TestModel_1.dat";

    ASSERT_TRUE(std::filesystem::exists(testDataFileName));
    ASSERT_TRUE(iniFile.Load(testDataFileName.string()));

    //clog << "Loaded test data from file: " << testDataFileName;
    if (iniFile.GetSection("SBML_FILE")) {
        rr::IniSection *sbml = iniFile.GetSection("SBML_FILE");
        rr::IniKey *fNameKey = sbml->GetKey("sbmlFile");
        if (fNameKey) {
            testModelFileName = fs::absolute(cAPICoreModelsDir / fNameKey->mValue);
            EXPECT_TRUE(std::filesystem::exists(testModelFileName));
        }
    } else {
        EXPECT_TRUE(false);
    }
}

TEST_F(CAPIExceptions, LOAD_SBML) {
    path TestModelFileName = cAPICoreModelsDir / "feedback.xml";
    ASSERT_TRUE(fs::exists(TestModelFileName));
    EXPECT_TRUE(loadSBMLFromFile(gRR, TestModelFileName.string().c_str()));
}

TEST_F(CAPIExceptions, SET_COMPUTE_AND_ASSIGN_CONSERVATION_LAWS) {
    path TestModelFileName = cAPICoreModelsDir / "Test_1.xml";
    loadSBMLFromFile(gRR, TestModelFileName.string().c_str());
    ASSERT_TRUE(gRR != NULL);
    bool res = setComputeAndAssignConservationLaws(gRR, true);
    EXPECT_TRUE(res);
    clog << "\nConservation laws: " << res << endl;

}


