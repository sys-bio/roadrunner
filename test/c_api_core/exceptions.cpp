#include "gtest/gtest.h"
#include "../test_util.h"
#include "C/rrc_api.h"
#include "rrLogger.h"
#include "rrUtils.h"
#include "rrIniFile.h"

#include <string>

//Add using clauses..
using namespace std;
using namespace rrc;

using rr::joinPath;
using rr::fileExists;

extern string gRRTestDir;
extern string gRROutputDir;
extern rr::IniFile iniFile;
extern RRHandle gRR;

//This tests is mimicking the Python tests
extern string TestModelFileName;

TEST(C_API_EXCEPTIONS, DATA_FILES)
{
    string testDataFileName = joinPath(gRRTestDir, "models/C_API_CORE/TestModel_1.dat");

    ASSERT_TRUE(fileExists(testDataFileName));
    ASSERT_TRUE(iniFile.Load(testDataFileName));

    //clog << "Loaded test data from file: " << testDataFileName;
    if (iniFile.GetSection("SBML_FILE"))
    {
        rr::IniSection* sbml = iniFile.GetSection("SBML_FILE");
        rr::IniKey* fNameKey = sbml->GetKey("sbmlFile");
        if (fNameKey)
        {
            TestModelFileName = joinPath(gRRTestDir + "models/C_API_CORE/", fNameKey->mValue);
            EXPECT_TRUE(fileExists(TestModelFileName));
        }
    }
    else
    {
        EXPECT_TRUE(false);
    }
}

TEST(C_API_EXCEPTIONS, LOAD_SBML)
{
    EXPECT_TRUE(loadSBMLFromFile(gRR, TestModelFileName.c_str()));
}

TEST(C_API_EXCEPTIONS, SET_COMPUTE_AND_ASSIGN_CONSERVATION_LAWS)
{
    ASSERT_TRUE(gRR != NULL);
    bool res = setComputeAndAssignConservationLaws(gRR, true);
    EXPECT_TRUE(res);
    clog << "\nConversation laws: " << res << endl;
}

TEST(C_API_EXCEPTIONS, GET_UNSCALED_ELASTICITY_COEFFICIENT)
{
    double test;
    bool val = getuEE(gRR, "J1", "S1", &test);

    // val = getuEE("J1", "S34", test);

     //val =  getuCC("J1", "S1", test);

}
