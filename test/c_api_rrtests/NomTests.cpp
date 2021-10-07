#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include "gtest/gtest.h"
#include "C/rrc_api.h"
#include "rrUtils.h"
#include "rrIniFile.h"
#include "rrException.h"
#include <filesystem>

#include "RoadRunnerTest.h"

using namespace std;
using namespace rr;
using namespace rrc;
using std::filesystem::path;

//extern path rrTestDir_;
//extern path gRROutputDir;
//extern RRHandle gRR;
//extern IniFile iniFile;

class NomTests : public RoadRunnerTest {
public:
    RRHandle gRR = createRRInstance();

    rr::IniFile iniFile;
    NomTests() = default;
    ~NomTests() {
        freeRRInstance(gRR);
    }

};

TEST_F(NomTests, NOM_TEST_DATA_FILES)
{
    string sec("NOM_TESTS");
    string key("InputFile");

    string NOMFileName = "NOM_Test.dat";

    NOMFileName = (rrTestDir_ / path("rrtest_files") / path(NOMFileName)).string();
    ASSERT_TRUE(std::filesystem::exists(NOMFileName));
    ASSERT_TRUE(iniFile.Load(NOMFileName));

    string TestModelFileName;
    //clog << "Loaded test data from file: " << NOMFileName;
    if (iniFile.GetSection(sec))
    {
        IniSection* sbml = iniFile.GetSection(sec);
        IniKey* fNameKey = sbml->GetKey(key);
        if (fNameKey)
        {
            TestModelFileName = (rrTestDir_ / path("rrtest_files") / fNameKey->mValue).string();
            EXPECT_TRUE(std::filesystem::exists(TestModelFileName));
        }
    }
    EXPECT_TRUE(loadSBMLFromFileE(gRR, TestModelFileName.c_str(), true));
}

TEST_F(NomTests, NOM_GET_NAME)
{
    string section("NOM_TESTS");
    string key("ModelName");

    if (iniFile.GetSection(section))
    {
        IniSection* isec = iniFile.GetSection(section);
        IniKey* ikey = isec->GetKey(key);
        if (ikey)
        {
            //======= Calling the function to be tested =============
            string name = getModelName(gRR);

            //======= Check that it worked out
            EXPECT_EQ(ikey->mValue, name);
        }
    }
}

TEST_F(NomTests, NOM_GET_NUMBER_OF_RULES)
{
    string section("NOM_TESTS");
    string key("NumberOfRules");
    if (iniFile.GetSection(section))
    {
        IniSection* iSec = iniFile.GetSection(section);
        IniKey* iKey = iSec->GetKey(key);
        if (iKey)
        {
            //======= Calling the function to be tested =============
            int nr = getNumberOfRules(gRR);

            //======= Check that it worked out
            EXPECT_EQ(iKey->AsInt(), nr);
        }
    }
}

TEST_F(NomTests, FREE_RR_INSTANCE)
{
    RRHandle r = createRRInstance();
    EXPECT_TRUE(freeRRInstance(r));
}
