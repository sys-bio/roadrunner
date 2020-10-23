#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include "gtest/gtest.h"
#include "C/rrc_api.h"
#include "rrUtils.h"
#include "rrIniFile.h"
#include "rrException.h"
#include "../test_util.h"

using namespace std;
using namespace rr;
using namespace rrc;

extern string gRRTestDir;
extern string gRROutputDir;
extern RRHandle gRR;

string LS_TestDataFileName = "models/C_API_CORE/LibStructTest.dat";
extern IniFile iniFile;
string LS_TestModelFileName;

TEST(LIBSTRUCT, testLibStructTestsDataFiles)
{
    string sec("LS_TESTS");
    string key("InputFile");

    string testDataFileName = joinPath(gRRTestDir, LS_TestDataFileName);
    ASSERT_TRUE(fileExists(testDataFileName));
    ASSERT_TRUE(iniFile.Load(testDataFileName));

    //clog << "Loaded test data from file: " << testDataFileName;
    if (iniFile.GetSection(sec))
    {
        IniSection* sbml = iniFile.GetSection(sec);
        IniKey* fNameKey = sbml->GetKey(key);
        if (fNameKey)
        {
            LS_TestModelFileName = joinPath(gRRTestDir + "models/C_API_CORE/", fNameKey->mValue);
            EXPECT_TRUE(fileExists(LS_TestModelFileName));
        }
    }
    EXPECT_TRUE(loadSBMLFromFileE(gRR, LS_TestModelFileName.c_str(), true));
}

TEST(LIBSTRUCT, getLinkMatrix)
{
    string section("getLinkMatrix");
    if (iniFile.GetSection(section))
    {
        IniSection* isec = iniFile.GetSection(section);
        ls::DoubleMatrix  ref = getDoubleMatrixFromString(isec->GetNonKeysAsString());

        //Testing call
        RRDoubleMatrix* actual = getLinkMatrix(gRR);

        //Check dimensions
        if (actual->RSize != ref.RSize() || actual->CSize != ref.CSize())
        {
            EXPECT_TRUE(false);
            return;
        }

        for (int row = 0; row < actual->RSize; row++)
        {
            for (int col = 0; col < actual->CSize; col++)
            {
                EXPECT_NEAR(ref(row, col), actual->Data[row * actual->CSize + col], 1e-6);
            }
        }
    }
}

TEST(LIBSTRUCT, getEigenValuesMatrix)
{
    string section("getEigenValuesMatrix");

    if (iniFile.GetSection(section))
    {
        IniSection* isec = iniFile.GetSection(section);
        ls::DoubleMatrix      ref = getDoubleMatrixFromString(isec->GetNonKeysAsString());
        RRDoubleMatrix* input = getLinkMatrix(gRR);

        //Testing call
        RRDoubleMatrix* actual = getEigenvaluesMatrix(input);
        /////////////////////////////////

        //Check dimensions
        if (actual->RSize != ref.RSize() || actual->CSize != ref.CSize())
        {
            EXPECT_TRUE(false);
            return;
        }

        for (int row = 0; row < actual->RSize; row++)
        {
            for (int col = 0; col < actual->CSize; col++)
            {
                EXPECT_NEAR(ref(row, col), actual->Data[row * actual->CSize + col], 1e-6);
            }
        }
    }
}

TEST(LIBSTRUCT, getEigenValuesVector)
{
    string section("getEigenvaluesVector");

    if (iniFile.GetSection(section))
    {
        IniSection* isec = iniFile.GetSection(section);
        vector< complex<double> >     ref = getComplexVectorFromString(isec->GetNonKeysAsString());
        RRDoubleMatrix* input = getLinkMatrix(gRR);
        //Testing call
        RRComplexVector* actual = getEigenvaluesVector(input);
        //////////////////////////////////////////////////////////////////

        //Check dimensions
        if (actual->Count != ref.size())
        {
            EXPECT_TRUE(false);
            return;
        }

        for (int row = 0; row < actual->Count; row++)
        {
            EXPECT_NEAR(real(ref[row]), actual->Data[row].re, 1e-6);
            EXPECT_NEAR(imag(ref[row]), actual->Data[row].imag, 1e-6);
        }
    }
}

TEST(LIBSTRUCT, FREE_RR_INSTANCE)
{
    EXPECT_TRUE(freeRRInstance(gRR));
    gRR = NULL;
}
