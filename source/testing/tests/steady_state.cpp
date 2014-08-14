#include "unit_test/UnitTest++.h"
#include "rrLogger.h"
#include "rrRoadRunner.h"
#include "rrException.h"
#include "rrStringUtils.h"
#include "rrIniFile.h"
#include "rrTestUtils.h"
#include "rrUtils.h"

using namespace UnitTest;
using namespace rr;
using namespace std;

extern string             gTestDataFolder;
extern string             gCompiler;
extern string             gSupportCodeFolder;
extern string             gTempFolder;
extern string             gRRInstallFolder;


SUITE(SteadyState)
{
  //Global to this unit
  vector<RoadRunner*> aRRs;
  const char *cnames[] = {"TestModel_1.dat"};
  vector<string> TestDataFileNames(cnames, end(cnames));
  vector<string> TestModelFileNames;
  IniFile iniFile;

    //This test-suite tests various steady state functions, using the model TestModel_1.xml

    //Test that model files and reference data for the tests in this suite are present
    TEST(DATA_FILES)
    {
      for (size_t test=0; test<TestDataFileNames.size(); test++) {
        gTestDataFolder         = joinPath(gRRInstallFolder, "tests");
        TestModelFileNames.push_back(joinPath(gTestDataFolder, TestDataFileNames[test]));

        CHECK(fileExists(TestDataFileNames[test]));
        CHECK(iniFile.Load(TestDataFileNames[test]));
        clog<<"Loaded test data from file: "<< TestDataFileNames[test];
        if(iniFile.GetSection("SBML_FILES"))
        {
            IniSection* sbml = iniFile.GetSection("SBML_FILES");
            IniKey* fNameKey = sbml->GetKey("FNAME1");
            if(fNameKey)
            {
                TestModelFileNames[test]  = joinPath(gTestDataFolder, fNameKey->mValue);
                CHECK(fileExists(TestModelFileNames[test]));
            }
        }
      }
    }

    TEST(LOAD_MODEL)
    {
      for (size_t test=0; test<TestDataFileNames.size(); test++) {
        RoadRunner* aRR = new RoadRunner(gCompiler, gTempFolder, gSupportCodeFolder);
        CHECK(aRRs[test]!=NULL);

        //Load the model
        aRRs[test]->setConservedMoietyAnalysis(true);
        aRRs[test]->load(TestModelFileNames[test]);
        aRRs.push_back(aRRs[test]);
      }
    }

    TEST(COMPUTE_STEADY_STATE)
    {
      for (size_t test=0; test<aRRs.size(); test++) {
        //Compute Steady state
        if(aRRs[test])
        {
            CHECK_CLOSE(0, aRRs[test]->steadyState(), 1e-6);
        }
        else
        {
            CHECK(false);
        }
      }
    }

    TEST(STEADY_STATE_CONCENTRATIONS)
    {
      for (size_t test=0; test<aRRs.size(); test++) {
        IniSection* aSection = iniFile.GetSection("STEADY_STATE_CONCENTRATIONS");
        //Read in the reference data, from the ini file
        if(!aSection || !aRRs[test])
        {
            CHECK(false);
            return;
        }

        for(int i = 0 ; i < aSection->KeyCount(); i++)
        {
            IniKey *aKey = aSection->GetKey(i);
            double val = aRRs[test]->getValue(aKey->mKey);

            //Check concentrations
            CHECK_CLOSE(aKey->AsFloat(), val, 1e-6);
        }
      }
    }

    //This test is using the function getValue("eigen_...")
    TEST(GET_EIGENVALUES_1)
    {
      for (size_t test=0; test<aRRs.size(); test++) {
        IniSection* aSection = iniFile.GetSection("EIGEN_VALUES");
        //Read in the reference data, from the ini file
        if(!aSection || !aRRs[test])
        {
            CHECK(false);
            return;
        }

        vector<string> ids = aRRs[test]->getEigenvalueIds();
        if(ids.size() != aSection->KeyCount())
        {
            CHECK(false);
            return;
        }

        for(int i = 0 ; i < aSection->KeyCount(); i++)
        {
            //Find correct eigenValue
            for(int j = 0; j < ids.size(); j++)
            {
                if(aSection->mKeys[i]->mKey == ids[j])
                {
                    IniKey *aKey = aSection->GetKey(i);
                    clog<<"\n";
                    clog<<"Ref_EigenValue: "<<aKey->mKey<<": "<<aKey->mValue<<endl;

                    clog<<"ID: "<<ids[j]<<"= "<<aRRs[test]->getValue(ids[j])<<endl;

                    CHECK_CLOSE(aKey->AsFloat(), aRRs[test]->getValue(ids[j]), 1e-6);
                }
            }
        }
      }
    }

    TEST(GET_EIGENVALUES_2)
    {
      for (size_t test=0; test<aRRs.size(); test++) {
        IniSection* aSection = iniFile.GetSection("EIGEN_VALUES");
        //Read in the reference data, from the ini file
        if(!aSection || !aRRs[test])
        {
            CHECK(false);
            return;
        }

        vector<Complex> eigenVals = aRRs[test]->getEigenvaluesCpx();
        if(eigenVals.size() != aSection->KeyCount())
        {
            CHECK(false);
            return;
        }

        for(int i = 0 ; i < aSection->KeyCount(); i++)
        {
            IniKey *aKey = aSection->GetKey(i);
            clog<<"\n";
            clog<<"Ref_EigenValue: "<<aKey->mKey<<": "<<aKey->mValue<<endl;

            clog<<"EigenValue "<<i<<": "<<real(eigenVals[i])<<endl;
            CHECK_CLOSE(aKey->AsFloat(), real(eigenVals[i]), 1e-6);
        }
      }
    }

    TEST(FULL_JACOBIAN)
    {
      for (size_t test=0; test<aRRs.size(); test++) {
        IniSection* aSection = iniFile.GetSection("FULL_JACOBIAN");
           if(!aSection)
        {
            CHECK(false);
            return;
        }

        DoubleMatrix fullJacobian     = aRRs[test]->getFullJacobian();
        DoubleMatrix jRef             = ParseMatrixFromText(aSection->GetNonKeysAsString());

        //Check dimensions
        if(fullJacobian.RSize() != jRef.RSize() || fullJacobian.CSize() != jRef.CSize())
        {
            CHECK(false);
            return;
        }

        clog<<"Full Jacobian\n"<<fullJacobian;
        CHECK_ARRAY2D_CLOSE(jRef, fullJacobian, fullJacobian.RSize(),fullJacobian.CSize(), 1e-6);
      }
    }

    TEST(REDUCED_JACOBIAN)
    {
      for (size_t test=0; test<aRRs.size(); test++) {
        IniSection* aSection = iniFile.GetSection("REDUCED_REORDERED_JACOBIAN");
           if(!aSection)
        {
            CHECK(false);
            return;
        }

        DoubleMatrix fullJacobian     = aRRs[test]->getReducedJacobian();
        DoubleMatrix jRef             = ParseMatrixFromText(aSection->GetNonKeysAsString());

        //Check dimensions
        if(fullJacobian.RSize() != jRef.RSize() || fullJacobian.CSize() != jRef.CSize())
        {
            CHECK(false);
            return;
        }
        clog<<"Reduced Jacobian\n"<<fullJacobian;
        CHECK_ARRAY2D_CLOSE(jRef, fullJacobian, fullJacobian.RSize(),fullJacobian.CSize(), 1e-6);
      }
    }

    TEST(FULL_REORDERED_JACOBIAN)
    {
      for (size_t test=0; test<aRRs.size(); test++) {
        IniSection* aSection = iniFile.GetSection("FULL_REORDERED_JACOBIAN");
           if(!aSection)
        {
            CHECK(false);
            return;
        }

        //Read in the reference data, from the ini file
        DoubleMatrix matrix = aRRs[test]->getFullReorderedJacobian();
        DoubleMatrix ref = ParseMatrixFromText(aSection->GetNonKeysAsString());

        cout<<"Reference\n"<<ref;
        cout<<"matrix\n"<<matrix;

        //Check dimensions
        if(matrix.RSize() != ref.RSize() || matrix.CSize() != ref.CSize())
        {
            CHECK(false);
            return;
        }

        CHECK_ARRAY2D_CLOSE(ref, matrix, matrix.RSize(), matrix.CSize(), 1e-6);
      }
    }

    TEST(REDUCED_REORDERED_JACOBIAN)
    {
      for (size_t test=0; test<aRRs.size(); test++) {
        IniSection* aSection = iniFile.GetSection("FULL_REORDERED_JACOBIAN");
           if(!aSection)
        {
            CHECK(false);
            return;
        }

        //Read in the reference data, from the ini file
        DoubleMatrix matrix = aRRs[test]->getReducedJacobian();
        DoubleMatrix ref = ParseMatrixFromText(aSection->GetNonKeysAsString());

        cout<<"Reference\n"<<ref;
        cout<<"matrix\n"<<matrix;

        //Check dimensions
        if(matrix.RSize() != ref.RSize() || matrix.CSize() != ref.CSize())
        {
            CHECK(false);
            return;
        }

        CHECK_ARRAY2D_CLOSE(ref, matrix, matrix.RSize(), matrix.CSize(), 1e-6);
      }
    }

    TEST(LINK_MATRIX)
    {
      for (size_t test=0; test<aRRs.size(); test++) {
        IniSection* aSection = iniFile.GetSection("LINK_MATRIX");
           if(!aSection)
        {
            CHECK(false);
            return;
        }

        //Read in the reference data, from the ini file
        DoubleMatrix matrix     = aRRs[test]->getLinkMatrix();
        DoubleMatrix ref        = ParseMatrixFromText(aSection->GetNonKeysAsString());

        //Check dimensions
        if(matrix.RSize() != ref.RSize() || matrix.CSize() != ref.CSize())
        {
            CHECK(false);
            return;
        }

        CHECK_ARRAY2D_CLOSE(ref, matrix, matrix.RSize(), matrix.CSize(), 1e-6);
      }
    }

    TEST(UNSCALED_ELASTICITY_MATRIX)
    {
      for (size_t test=0; test<aRRs.size(); test++) {
        IniSection* aSection = iniFile.GetSection("UNSCALED_ELASTICITY_MATRIX");
           if(!aSection)
        {
            CHECK(false);
            return;
        }

        //Read in the reference data, from the ini file
        DoubleMatrix matrix     = aRRs[test]->getUnscaledElasticityMatrix();
        DoubleMatrix ref          = ParseMatrixFromText(aSection->GetNonKeysAsString());

        //Check dimensions
        if(matrix.RSize() != ref.RSize() || matrix.CSize() != ref.CSize())
        {
            CHECK(!"Wrong matrix dimensions" );
            return;
        }

        CHECK_ARRAY2D_CLOSE(ref, matrix, matrix.RSize(), matrix.CSize(), 1e-6);
      }
    }

    TEST(SCALED_ELASTICITY_MATRIX)
    {
      for (size_t test=0; test<aRRs.size(); test++) {
        IniSection* aSection = iniFile.GetSection("SCALED_ELASTICITY_MATRIX");
           if(!aSection)
        {
            CHECK(false);
            return;
        }

        //Read in the reference data, from the ini file
        DoubleMatrix matrix     = aRRs[test]->getScaledElasticityMatrix();
        DoubleMatrix ref          = ParseMatrixFromText(aSection->GetNonKeysAsString());

        //Check dimensions
        if(matrix.RSize() != ref.RSize() || matrix.CSize() != ref.CSize())
        {
            CHECK(!"Wrong matrix dimensions" );
            return;
        }

        CHECK_ARRAY2D_CLOSE(ref, matrix, matrix.RSize(), matrix.CSize(), 1e-6);
      }
    }

    TEST(UNSCALED_CONCENTRATION_CONTROL_MATRIX)
    {
      for (size_t test=0; test<aRRs.size(); test++) {
        IniSection* aSection = iniFile.GetSection("UNSCALED_CONCENTRATION_CONTROL_MATRIX");
           if(!aSection)
        {
            CHECK(false);
            return;
        }

        //Read in the reference data, from the ini file
        DoubleMatrix matrix     = aRRs[test]->getUnscaledConcentrationControlCoefficientMatrix();
        DoubleMatrix ref          = ParseMatrixFromText(aSection->GetNonKeysAsString());

        //Check dimensions
        if(matrix.RSize() != ref.RSize() || matrix.CSize() != ref.CSize())
        {
            CHECK(!"Wrong matrix dimensions" );
            return;
        }

        CHECK_ARRAY2D_CLOSE(ref, matrix, matrix.RSize(), matrix.CSize(), 1e-6);
      }
    }

    TEST(UNSCALED_FLUX_CONTROL_MATRIX)
    {
      for (size_t test=0; test<aRRs.size(); test++) {
        IniSection* aSection = iniFile.GetSection("UNSCALED_FLUX_CONTROL_MATRIX");
           if(!aSection)
        {
            CHECK(false);
            return;
        }

        //Read in the reference data, from the ini file
        DoubleMatrix matrix     = aRRs[test]->getUnscaledFluxControlCoefficientMatrix();
        DoubleMatrix ref          = ParseMatrixFromText(aSection->GetNonKeysAsString());

        //Check dimensions
        if(matrix.RSize() != ref.RSize() || matrix.CSize() != ref.CSize())
        {
            CHECK(!"Wrong matrix dimensions" );
            return;
        }

        CHECK_ARRAY2D_CLOSE(ref, matrix, matrix.RSize(), matrix.CSize(), 1e-6);
      }
    }

    TEST(FREE_ROADRUNNER)
    {
      for (size_t test=0; test<aRRs.size(); test++) {
        delete aRRs[test];
      }
    }
}

