#include "gtest/gtest.h"
#include "rrLogger.h"
#include "rrRoadRunner.h"
#include "rrException.h"
#include "rrStringUtils.h"
#include "rrIniFile.h"
//#include "rrTestUtils.h"
#include "rrUtils.h"
#include "../test_util.h"

using namespace rr;
using namespace std;
using namespace ls;

extern string gRRTestDir;
extern string gRROutputDir;
extern string TestModelFileName;
extern IniFile iniFile;

//Global to this unit
RoadRunner *aRR = NULL;
string TestDataFileName     = "TestModel_1.dat";
string gTestDataFolder = joinPath(gRRTestDir, "tests");

//This test-suite tests various steady state functions, using the model TestModel_1.xml

//Test that model files and reference data for the tests in this suite are present
TEST(DISABLED_STEADY_STATE, DATA_FILES)
{
    gTestDataFolder = joinPath(gRRTestDir, "models/STEADY_STATE/");
    TestDataFileName = joinPath(gTestDataFolder, TestDataFileName);

    ASSERT_TRUE(fileExists(TestDataFileName));
    ASSERT_TRUE(iniFile.Load(TestDataFileName));
    //clog << "Loaded test data from file: " << TestDataFileName;
    if (iniFile.GetSection("SBML_FILES"))
    {
        IniSection* sbml = iniFile.GetSection("SBML_FILES");
        IniKey* fNameKey = sbml->GetKey("FNAME1");
        if (fNameKey)
        {
            TestModelFileName = joinPath(gTestDataFolder, fNameKey->mValue);
            ASSERT_TRUE(fileExists(TestModelFileName));
        }
    }
}

TEST(DISABLED_STEADY_STATE, LOAD_MODEL)
{
    aRR = new RoadRunner();
    EXPECT_TRUE(aRR != NULL);

    //Load the model
    aRR->setConservedMoietyAnalysis(true);
    aRR->load(TestModelFileName);
}

TEST(DISABLED_STEADY_STATE, COMPUTE_STEADY_STATE)
{
    //Compute Steady state
    if (aRR)
    {
        EXPECT_NEAR(0, aRR->steadyState(), 1e-6);
    }
    else
    {
        EXPECT_TRUE(false);
    }
}

TEST(DISABLED_STEADY_STATE, STEADY_STATE_CONCENTRATIONS)
{
    IniSection* aSection = iniFile.GetSection("STEADY_STATE_CONCENTRATIONS");
    //Read in the reference data, from the ini file
    if (!aSection || !aRR)
    {
        EXPECT_TRUE(false);
        return;
    }

    for (int i = 0; i < aSection->KeyCount(); i++)
    {
        IniKey* aKey = aSection->GetKey(i);
        double val = aRR->getValue(aKey->mKey);

        //Check concentrations
        EXPECT_NEAR(aKey->AsFloat(), val, 1e-6);
    }
}

//This test is using the function getValue("eigen_...")
TEST(DISABLED_STEADY_STATE, GET_EIGENVALUES_1)
{
    IniSection* aSection = iniFile.GetSection("EIGEN_VALUES");
    //Read in the reference data, from the ini file
    if (!aSection || !aRR)
    {
        EXPECT_TRUE(false);
        return;
    }

    vector<string> ids = aRR->getEigenValueIds();
    if (ids.size() != aSection->KeyCount())
    {
        EXPECT_TRUE(false);
        return;
    }

    for (int i = 0; i < aSection->KeyCount(); i++)
    {
        //Find correct eigenValue
        for (int j = 0; j < ids.size(); j++)
        {
            if (aSection->mKeys[i]->mKey == ids[j])
            {
                IniKey* aKey = aSection->GetKey(i);
                clog << "\n";
                clog << "Ref_EigenValue: " << aKey->mKey << ": " << aKey->mValue << endl;

                clog << "ID: " << ids[j] << "= " << aRR->getValue(ids[j]) << endl;

                EXPECT_NEAR(aKey->AsFloat(), aRR->getValue(ids[j]), 1e-6);
            }
        }
    }
}

TEST(DISABLED_STEADY_STATE, GET_EIGENVALUES_2)
{
    IniSection* aSection = iniFile.GetSection("EIGEN_VALUES");
    //Read in the reference data, from the ini file
    if (!aSection || !aRR)
    {
        EXPECT_TRUE(false);
        return;
    }

    std::vector<ls::Complex> eigenVals = aRR->getFullEigenValues();
    if (eigenVals.size() != aSection->KeyCount())
    {
        EXPECT_TRUE(false);
        return;
    }

    for (int i = 0; i < aSection->KeyCount(); i++)
    {
        IniKey* aKey = aSection->GetKey(i);
        clog << "\n";
        clog << "Ref_EigenValue: " << aKey->mKey << ": " << aKey->mValue << endl;

        double eigenVal = std::real(eigenVals[i]);

        clog << "EigenValue " << i << ": " << eigenVal << endl;
        EXPECT_NEAR(aKey->AsFloat(), eigenVal, 1e-6);
    }
}

TEST(DISABLED_STEADY_STATE, FULL_JACOBIAN)
{
    IniSection* aSection = iniFile.GetSection("FULL_JACOBIAN");
    if (!aSection)
    {
        EXPECT_TRUE(false);
        return;
    }

    DoubleMatrix fullJacobian = aRR->getFullJacobian();
    DoubleMatrix jRef = ParseMatrixFromText(aSection->GetNonKeysAsString());

    //Check dimensions
    if (fullJacobian.RSize() != jRef.RSize() || fullJacobian.CSize() != jRef.CSize())
    {
        EXPECT_TRUE(false);
        return;
    }

    clog << "Full Jacobian\n" << fullJacobian;
    expectArrayClose(jRef, fullJacobian, 1e-6);
}

TEST(DISABLED_STEADY_STATE, REDUCED_JACOBIAN)
{
    IniSection* aSection = iniFile.GetSection("REDUCED_REORDERED_JACOBIAN");
    if (!aSection)
    {
        EXPECT_TRUE(false);
        return;
    }

    DoubleMatrix fullJacobian = aRR->getReducedJacobian();
    DoubleMatrix jRef = ParseMatrixFromText(aSection->GetNonKeysAsString());

    //Check dimensions
    if (fullJacobian.RSize() != jRef.RSize() || fullJacobian.CSize() != jRef.CSize())
    {
        EXPECT_TRUE(false);
        return;
    }
    clog << "Reduced Jacobian\n" << fullJacobian;
    expectArrayClose(jRef, fullJacobian, 1e-6);
}

TEST(DISABLED_STEADY_STATE, FULL_REORDERED_JACOBIAN)
{
    IniSection* aSection = iniFile.GetSection("FULL_REORDERED_JACOBIAN");
    if (!aSection)
    {
        EXPECT_TRUE(false);
        return;
    }

    //Read in the reference data, from the ini file
    DoubleMatrix matrix = aRR->getFullReorderedJacobian();
    DoubleMatrix ref = ParseMatrixFromText(aSection->GetNonKeysAsString());

    cout << "Reference\n" << ref;
    cout << "matrix\n" << matrix;

    expectArrayClose(ref, matrix, 1e-6);
}

TEST(DISABLED_STEADY_STATE, REDUCED_REORDERED_JACOBIAN)
{
    IniSection* aSection = iniFile.GetSection("FULL_REORDERED_JACOBIAN");
    if (!aSection)
    {
        EXPECT_TRUE(false);
        return;
    }

    //Read in the reference data, from the ini file
    DoubleMatrix matrix = aRR->getReducedJacobian();
    DoubleMatrix ref = ParseMatrixFromText(aSection->GetNonKeysAsString());

    cout << "Reference\n" << ref;
    cout << "matrix\n" << matrix;

    expectArrayClose(ref, matrix, 1e-6);
}

TEST(DISABLED_STEADY_STATE, LINK_MATRIX)
{
    IniSection* aSection = iniFile.GetSection("LINK_MATRIX");
    if (!aSection)
    {
        EXPECT_TRUE(false);
        return;
    }

    //Read in the reference data, from the ini file
    DoubleMatrix matrix = aRR->getLinkMatrix();
    DoubleMatrix ref = ParseMatrixFromText(aSection->GetNonKeysAsString());

    expectArrayClose(ref, matrix, 1e-6);
}

TEST(DISABLED_STEADY_STATE, UNSCALED_ELASTICITY_MATRIX)
{
    IniSection* aSection = iniFile.GetSection("UNSCALED_ELASTICITY_MATRIX");
    if (!aSection)
    {
        EXPECT_TRUE(false);
        return;
    }

    //Read in the reference data, from the ini file
    DoubleMatrix matrix = aRR->getUnscaledElasticityMatrix();
    DoubleMatrix ref = ParseMatrixFromText(aSection->GetNonKeysAsString());

    expectArrayClose(ref, matrix, 1e-6);
}

TEST(DISABLED_STEADY_STATE, SCALED_ELASTICITY_MATRIX)
{
    IniSection* aSection = iniFile.GetSection("SCALED_ELASTICITY_MATRIX");
    if (!aSection)
    {
        EXPECT_TRUE(false);
        return;
    }

    //Read in the reference data, from the ini file
    DoubleMatrix matrix = aRR->getScaledElasticityMatrix();
    DoubleMatrix ref = ParseMatrixFromText(aSection->GetNonKeysAsString());

    expectArrayClose(ref, matrix, 1e-6);
}

TEST(DISABLED_STEADY_STATE, UNSCALED_CONCENTRATION_CONTROL_MATRIX)
{
    IniSection* aSection = iniFile.GetSection("UNSCALED_CONCENTRATION_CONTROL_MATRIX");
    if (!aSection)
    {
        EXPECT_TRUE(false);
        return;
    }

    //Read in the reference data, from the ini file
    DoubleMatrix matrix = aRR->getUnscaledConcentrationControlCoefficientMatrix();
    DoubleMatrix ref = ParseMatrixFromText(aSection->GetNonKeysAsString());

    expectArrayClose(ref, matrix, 1e-6);
}

TEST(DISABLED_STEADY_STATE, UNSCALED_FLUX_CONTROL_MATRIX)
{
    IniSection* aSection = iniFile.GetSection("UNSCALED_FLUX_CONTROL_MATRIX");
    if (!aSection)
    {
        EXPECT_TRUE(false);
        return;
    }

    //Read in the reference data, from the ini file
    DoubleMatrix matrix = aRR->getUnscaledFluxControlCoefficientMatrix();
    DoubleMatrix ref = ParseMatrixFromText(aSection->GetNonKeysAsString());

    expectArrayClose(ref, matrix, 1e-6);
}

TEST(DISABLED_STEADY_STATE, FREE_ROADRUNNER)
{
    delete aRR;
}
