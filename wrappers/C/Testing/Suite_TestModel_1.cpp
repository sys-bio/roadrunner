#include <string>
#include "rrLogger.h"
#include "rrUtils.h"
#include "rrIniFile.h"
#include "UnitTest++.h"
#include "rrc_api.h"
#include "src/TestUtils.h"

//using..
using namespace std;
using namespace UnitTest;
using namespace ls;
using namespace rrc;
using namespace rr;

using rr::joinPath;
using rr::fileExists;

extern string   gTempFolder;
extern string   gTestDataFolder;
extern string gCompiler;

//This tests is mimicking the Python tests
SUITE(TEST_MODELS)
{
  const char *cnames[] = {"Test_1.dat", "Test_2.dat", "Test_3.dat"};
  vector<string> TestDataFileNames(cnames, end(cnames));
  vector<string> TestModelFileNames;
  vector<RRHandle> gRRs;
  vector<IniFile*> iniFiles;

    TEST(DATA_FILES)
    {
      for(size_t test=0; test<TestDataFileNames.size(); test++) {
        IniFile* iniFile = new IniFile();
        gRRs.push_back(createRRInstanceEx(gTempFolder.c_str(), gCompiler.c_str()));
        TestModelFileNames.push_back(joinPath(gTestDataFolder, TestDataFileNames[test]));
        clog<<"Checking file: "<< TestModelFileNames[test] << endl;
        CHECK(fileExists(TestModelFileNames[test]));
        CHECK(iniFile->Load(TestModelFileNames[test]));

        clog<<"Loaded test data from file: "<< TestModelFileNames[test] << endl;
        if(iniFile->GetSection("SBML_FILES"))
        {
            IniSection* sbml = iniFile->GetSection("SBML_FILES");
            IniKey* fNameKey = sbml->GetKey("FNAME1");
            if(fNameKey)
            {
                TestModelFileNames[test]  = joinPath(gTestDataFolder, fNameKey->mValue);
                CHECK(fileExists(TestModelFileNames[test]));
            }
        }
        CHECK(loadSBMLFromFileE(gRRs[test], TestModelFileNames[test].c_str(), true));
        iniFiles.push_back(iniFile);
      }
    }

    TEST(SET_COMPUTE_AND_ASSIGN_CONSERVATION_LAWS)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        CHECK(gRRs[test]!=NULL);
        bool res = setComputeAndAssignConservationLaws(gRRs[test], true);
        CHECK(res);
        clog<<"\nConversation laws: "<<res<<endl;
      }
    }

    TEST(SET_STEADY_STATE_SELECTION_LIST)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        CHECK(gRRs[test]!=NULL);

        //Load reference data
        IniSection* aSection = iniFiles[test]->GetSection("Set Steady State Selection List");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        IniKey* aKey = aSection->GetKey("list");
        if(!aKey)
        {
            CHECK(false);
            return;
        }

        bool res = setSteadyStateSelectionList(gRRs[test], aKey->mValue.c_str());
        CHECK(res);
      }
    }

    TEST(GET_STEADY_STATE_SELECTION_LIST)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        IniSection* aSection = iniFiles[test]->GetSection("Get Steady State Selection List");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        IniKey* aKey = aSection->GetKey("list");
        if(!aKey)
        {
            CHECK(false);
            return;
        }

        RRStringArrayPtr list = getSteadyStateSelectionList(gRRs[test]);

        if(!list)
        {
            CHECK(false);
            return;
        }
        vector<string> selList = splitString(aKey->mValue," ,");
        CHECK(selList.size() == list->Count);
        for(int i = 0; i < selList.size(); i++)
        {
            CHECK(selList[i] == list->String[i]);
        }
        freeStringArray(list);
      }
    }

    TEST(COMPUTE_STEADY_STATE)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        double val;
        CHECK( steadyState(gRRs[test], &val));
        CHECK_CLOSE(0, val, 1e-4);
      }
    }

    TEST(SPECIES_CONCENTRATIONS)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        CHECK(gRRs[test]!=NULL);

        //Read in the reference data, from the ini file
        IniSection* aSection = iniFiles[test]->GetSection("Species Concentrations");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        for(int i = 0 ; i < aSection->KeyCount(); i++)
        {
            IniKey *aKey = aSection->GetKey(i);

            double val;

            //API Function getValue
            if(!getValue(gRRs[test], aKey->mKey.c_str(), &val))
            {
                CHECK(false);
            }

            //Check concentrations
            CHECK_CLOSE(aKey->AsFloat(), val, 1e-4);
            clog<<"\n";
            clog<<"Ref:\t"<<aKey->AsFloat()<<"\tActual:\t "<<val<<endl;
        }
      }
    }

    TEST(GET_SPECIES_INITIAL_CONCENTRATIONS)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        CHECK(gRRs[test]!=NULL);

        //Read in the reference data, from the ini file
        IniSection* aSection = iniFiles[test]->GetSection("Get Species Initial Concentrations");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        for(int i = 0 ; i < aSection->KeyCount(); i++)
        {
            IniKey *aKey = aSection->GetKey(i);
            double val;
            if(!getValue(gRRs[test], aKey->mKey.c_str(), &val))
            {
                CHECK(false);
            }

            //Check concentrations
            CHECK_CLOSE(aKey->AsFloat(), val, 1e-4);
            clog<<"\n";
            clog<<"Ref:\t"<<aKey->AsFloat()<<"\tActual:\t "<<val<<endl;
        }
      }
    }

    TEST(GET_SPECIES_INITIAL_CONCENTRATION_BY_INDEX)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        CHECK(gRRs[test]!=NULL);

        //Read in the reference data, from the ini file
        IniSection* aSection = iniFiles[test]->GetSection("Get Species Initial Concentrations By Index");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        RRStringArray *arr = getFloatingSpeciesIds(gRRs[test]);
        for(int i = 0 ; i < aSection->KeyCount(); i++)
        {
            IniKey *aKey = aSection->GetKey(i);
            double val;
            if(!getFloatingSpeciesInitialConcentrationByIndex(gRRs[test], i, &val))
            {
                CHECK(false);
            }

            //Check concentrations
            CHECK_CLOSE(aKey->AsFloat(), val, 1e-4);
            clog<<"\n";
            clog<<"Ref:\t"<<aKey->AsFloat()<<"\tActual:\t "<<val<<endl;
        }
      }
    }

    TEST(SET_SPECIES_INITIAL_CONCENTRATION_BY_INDEX)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        CHECK(gRRs[test]!=NULL);

        //Read in the reference data, from the ini file
        IniSection* aSection = iniFiles[test]->GetSection("Set Species Initial Concentrations By Index");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        RRStringArray *arr = getFloatingSpeciesIds(gRRs[test]);
        for(int i = 0 ; i < aSection->KeyCount(); i++)
        {
            IniKey *aKey = aSection->GetKey(i);

            //Set the value..
            setFloatingSpeciesInitialConcentrationByIndex(gRRs[test], i, aKey->AsFloat());

            double val;
            //Read it back
            if(!getFloatingSpeciesInitialConcentrationByIndex(gRRs[test], i, &val))
            {
                CHECK(false);
            }

            //Check concentrations
            CHECK_CLOSE(aKey->AsFloat(), val, 1e-4);
            clog<<"\n";
            clog<<"Ref:\t"<<aKey->AsFloat()<<"\tActual:\t "<<val<<endl;
        }

        double val;
        reset(gRRs[test]);
        steadyState(gRRs[test], &val);
      }
    }

    TEST(SET_SPECIES_INITIAL_CONCENTRATIONS)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        CHECK(gRRs[test]!=NULL);

        //Read in the reference data, from the ini file
        IniSection* aSection = iniFiles[test]->GetSection("Set Species Initial Concentrations");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        for(int i = 0 ; i < aSection->KeyCount(); i++)
        {
            IniKey *aKey = aSection->GetKey(i);

            //Set the value..
            setValue(gRRs[test], aKey->mKey.c_str(), aKey->AsFloat());

            double val;
            //Read it back
            if(!getValue(gRRs[test], aKey->mKey.c_str(), &val))
            {
                CHECK(false);
            }

            //Check concentrations
            CHECK_CLOSE(aKey->AsFloat(), val, 1e-4);
            clog<<"\n";
            clog<<"Ref:\t"<<aKey->AsFloat()<<"\tActual:\t "<<val<<endl;
        }

        //We need to set back the values to concentrations, after running steady state..
        double val;
        reset(gRRs[test]);
        steadyState(gRRs[test], &val);
      }
    }

    TEST(FLUXES)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        CHECK(gRRs[test]!=NULL);

        //Read in the reference data, from the ini file
        IniSection* aSection = iniFiles[test]->GetSection("Fluxes");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        for(int i = 0 ; i < aSection->KeyCount(); i++)
        {
            IniKey *aKey = aSection->GetKey(i);
            double val;

            //API Function getValue
            if(!getValue(gRRs[test], aKey->mKey.c_str(), &val))
            {
                CHECK(false);
            }

            //Check concentrations
            CHECK_CLOSE(aKey->AsFloat(), val, 1e-4);
            clog<<"\n";
            clog<<"Ref:\t"<<aKey->AsFloat()<<"\tActual:\t "<<val<<endl;
        }
      }
    }

    TEST(FULL_JACOBIAN)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        IniSection* aSection = iniFiles[test]->GetSection("Full Jacobian");
        if(!aSection)
        {
            continue;
        }

        RRDoubleMatrixPtr    jActual     = getFullJacobian(gRRs[test]);
        ls::DoubleMatrix     jRef        = getDoubleMatrixFromString(aSection->GetNonKeysAsString());

        //Check dimensions
        if(jActual->RSize != jRef.RSize() || jActual->CSize != jRef.CSize())
        {
            CHECK(false);
            return;
        }

        for(int row = 0; row < jActual->RSize; row++)
        {
            for(int col = 0; col < jActual->CSize; col++)
            {
              CHECK_CLOSE(jRef(row,col), jActual->Data[row*jActual->CSize + col], 1e-4);
            }
        }
        //Clean up...
        freeMatrix(jActual);
      }
    }

    TEST(INDIVIDUAL_EIGENVALUES)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        IniSection* aSection = iniFiles[test]->GetSection("Individual EigenValues");
        if(!aSection)
        {
            continue;
        }

        for(int i = 0 ; i < aSection->KeyCount(); i++)
        {
            IniKey *aKey = aSection->GetKey(i);
            clog<<"\n";
            clog<<"Ref_EigenValue: "<<aKey->mKey<<": "<<aKey->mValue<<endl;

            double val;

            string eigenValueLabel ="eigen(" + aKey->mKey + ")";
            if(!getValue(gRRs[test], eigenValueLabel.c_str(), &val))
            {
                CHECK(false);
            }

            clog<<"EigenValue "<<i<<": "<<val<<endl;
            CHECK_CLOSE(aKey->AsFloat(), val, 1e-4);
        }
      }
    }

    TEST(GET_EIGENVALUE_MATRIX)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        CHECK(gRRs[test]!=NULL);

        setComputeAndAssignConservationLaws(gRRs[test], true);

        IniSection* aSection = iniFiles[test]->GetSection("Eigenvalue Matrix");

        //Read in the reference data, from the ini file
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        ls::DoubleMatrix     ref = getDoubleMatrixFromString(aSection->GetNonKeysAsString());

        RRDoubleMatrixPtr matrix = getEigenvalues(gRRs[test]);
        if(!matrix)
        {
            CHECK(false);
            return;
        }

        if(!matrix || matrix->RSize != ref.RSize())
        {
            CHECK(false);
            return;
        }

        clog<<"Reference Matrix:\n";
        clog<<ref<<endl;

        clog<<"Calculated Matrix:";
        clog<<matrixToString(matrix);

        for(int i = 0 ; i < ref.RSize(); i++)
        {
            double val;
            if(!getMatrixElement(matrix, i , 0, &val))
            {
                CHECK(false);
            }
            CHECK_CLOSE(ref(i,0), val, 1e-4);

        }
        freeMatrix(matrix);
      }
    }


    TEST(STOICHIOMETRY_MATRIX)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        CHECK(gRRs[test]!=NULL);

        //Read in the reference data, from the ini file
        IniSection* aSection = iniFiles[test]->GetSection("Stoichiometry Matrix");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        ls::DoubleMatrix     ref         = getDoubleMatrixFromString(aSection->GetNonKeysAsString());

        RRDoubleMatrixPtr matrix = getStoichiometryMatrix(gRRs[test]);
        if(!matrix)
        {
            CHECK(false);
            return;
        }

        if(!matrix || matrix->RSize != ref.RSize())
        {
            CHECK(false);
            return;
        }

        clog<<"Reference Matrix:\n";
        clog<<ref<<endl;

        clog<<"Calculated Matrix:";
        clog<<matrixToString(matrix);

        for(int i = 0 ; i < ref.RSize(); i++)
        {
            double val;
            if(!getMatrixElement(matrix, i , 0, &val))
            {
                CHECK(false);
            }
            CHECK_CLOSE(ref(i,0), val, 1e-4);
        }
        freeMatrix(matrix);
      }
    }

    TEST(LINK_MATRIX)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        clog<<"\n==== LINK_MATRIX ====\n\n";
        CHECK(gRRs[test]!=NULL);

        //Read in the reference data, from the ini file
        IniSection* aSection = iniFiles[test]->GetSection("Link Matrix");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

           ls::DoubleMatrix     ref         = getDoubleMatrixFromString(aSection->GetNonKeysAsString());

        RRDoubleMatrixPtr matrix = getLinkMatrix(gRRs[test]);
        if(!matrix)
        {
            CHECK(false);
            return;
        }

        if(!matrix || matrix->RSize != ref.RSize())
        {
            CHECK(false);
            return;
        }

        clog<<"Reference Matrix:\n";
        clog<<ref<<endl;

        clog<<"Calculated Matrix:";
        clog<<matrixToString(matrix);

        for(int i = 0 ; i < ref.RSize(); i++)
        {
            double val;
            if(!getMatrixElement(matrix, i , 0, &val))
            {
                CHECK(false);
            }
            CHECK_CLOSE(ref(i,0), val, 1e-4);
        }
        freeMatrix(matrix);
      }
    }

    TEST(UNSCALED_ELASTICITY_MATRIX)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        clog<<"\n==== UNSCALED_ELASTICITY_MATRIX ====\n\n";
        CHECK(gRRs[test]!=NULL);

        //Read in the reference data, from the ini file
        IniSection* aSection = iniFiles[test]->GetSection("Unscaled Elasticity Matrix");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

           ls::DoubleMatrix     ref         = getDoubleMatrixFromString(aSection->GetNonKeysAsString());

        RRDoubleMatrixPtr matrix = getUnscaledElasticityMatrix(gRRs[test]);
        if(!matrix)
        {
            CHECK(false);
            return;
        }

        if(!matrix || matrix->RSize != ref.RSize())
        {
            CHECK(false);
            return;
        }

        clog<<"Reference Matrix:\n";
        clog<<ref<<endl;

        clog<<"Calculated Matrix:";
        clog<<matrixToString(matrix);

        for(int i = 0 ; i < ref.RSize(); i++)
        {
            double val;
            if(!getMatrixElement(matrix, i , 0, &val))
            {
                CHECK(false);
            }
            CHECK_CLOSE(ref(i,0), val, 1e-4);
        }
        freeMatrix(matrix);
      }
    }

    TEST(SCALED_ELASTICITY_MATRIX)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        clog<<"\n==== SCALED_ELASTICITY_MATRIX ====\n\n";
        CHECK(gRRs[test]!=NULL);

        //Read in the reference data, from the ini file
        IniSection* aSection = iniFiles[test]->GetSection("Scaled Elasticity Matrix");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        ls::DoubleMatrix     ref         = getDoubleMatrixFromString(aSection->GetNonKeysAsString());

        double testnum;
        steadyState(gRRs[test], &testnum);
        RRDoubleMatrixPtr matrix = getScaledElasticityMatrix(gRRs[test]);
        if(!matrix)
        {
            CHECK(false);
            return;
        }

        clog<<"Reference Matrix:\n";
        clog<<ref<<endl;

        clog<<"Calculated Matrix:";
        clog<<matrixToString(matrix);

        if(!matrix || matrix->RSize != ref.RSize())
        {
            CHECK(false);
            return;
        }

        for(int i = 0 ; i < ref.RSize(); i++)
        {
            double val;
            if(!getMatrixElement(matrix, i , 0, &val))
            {
                CHECK(false);
            }
            CHECK_CLOSE(ref(i,0), val, 1e-4);

        }
        freeMatrix(matrix);
      }
    }

    TEST(UNSCALED_CONCENTRATION_CONTROL_MATRIX)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        clog<<"\n==== UNSCALED_CONCENTRATION_CONTROL_MATRIX ====\n\n";

        CHECK(gRRs[test]!=NULL);

        //Read in the reference data, from the ini file
        IniSection* aSection = iniFiles[test]->GetSection("UnScaled Concentration Control Matrix");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        ls::DoubleMatrix     ref         = getDoubleMatrixFromString(aSection->GetNonKeysAsString());

        RRDoubleMatrixPtr matrix = getUnscaledConcentrationControlCoefficientMatrix(gRRs[test]);
        if(!matrix)
        {
            CHECK(false);
            return;
        }

        clog<<"Reference Matrix:\n";
        clog<<ref<<endl;

        clog<<"Calculated Matrix:";
        clog<<matrixToString(matrix);

        if(!matrix || matrix->RSize != ref.RSize())
        {
            CHECK(false);
            return;
        }

        for(int i = 0 ; i < ref.RSize(); i++)
        {
            double val;
            if(!getMatrixElement(matrix, i , 0, &val))
            {
                CHECK(false);
            }
            CHECK_CLOSE(ref(i,0), val, 1e-4);

        }
        freeMatrix(matrix);
      }
    }

    TEST(SCALED_CONCENTRATION_CONTROL_MATRIX)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        clog<<"\n==== SCALED_CONCENTRATION_CONTROL_MATRIX ====\n\n";
        CHECK(gRRs[test]!=NULL);

        //Read in the reference data, from the ini file
        IniSection* aSection = iniFiles[test]->GetSection("Scaled Concentration Control Matrix");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        ls::DoubleMatrix     ref         = getDoubleMatrixFromString(aSection->GetNonKeysAsString());

        RRDoubleMatrixPtr matrix = getScaledConcentrationControlCoefficientMatrix(gRRs[test]);
        if(!matrix)
        {
            CHECK(false);
            return;
        }

          clog<<"Reference Matrix:\n";
        clog<<ref<<endl;

          clog<<"Calculated Matrix:";
        clog<<matrixToString(matrix);

        if(!matrix || matrix->RSize != ref.RSize())
        {
            CHECK(false);
            return;
        }

        for(int i = 0 ; i < ref.RSize(); i++)
        {
            double val;
            if(!getMatrixElement(matrix, i , 0, &val))
            {
                CHECK(false);
            }
            CHECK_CLOSE(ref(i,0), val, 1e-4);

        }
        freeMatrix(matrix);
      }
    }

    TEST(UNSCALED_FLUX_CONTROL_MATRIX)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        clog<<"\n==== UNSCALED_FLUX_CONTROL_MATRIX ====\n\n";
        CHECK(gRRs[test]!=NULL);

        //Read in the reference data, from the ini file
        IniSection* aSection = iniFiles[test]->GetSection("Unscaled Flux Control Matrix");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        ls::DoubleMatrix     ref         = getDoubleMatrixFromString(aSection->GetNonKeysAsString());

        RRDoubleMatrixPtr matrix = getUnscaledFluxControlCoefficientMatrix(gRRs[test]);
        if(!matrix)
        {
            CHECK(false);
            return;
        }

          clog<<"Reference Matrix:\n";
        clog<<ref<<endl;

          clog<<"Calculated Matrix:";
        clog<<matrixToString(matrix);

        if(!matrix || matrix->RSize != ref.RSize())
        {
            CHECK(false);
            return;
        }

        for(int i = 0 ; i < ref.RSize(); i++)
        {
            double val;
            if(!getMatrixElement(matrix, i , 0, &val))
            {
                CHECK(false);
            }
            CHECK_CLOSE(ref(i,0), val, 1e-4);

        }
        freeMatrix(matrix);
      }
    }

    TEST(SCALED_FLUX_CONTROL_MATRIX)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        clog<<"\n==== SCALED_FLUX_CONTROL_MATRIX ====\n\n";
        CHECK(gRRs[test]!=NULL);

        //Read in the reference data, from the ini file
        IniSection* aSection = iniFiles[test]->GetSection("Scaled Flux Control Matrix");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        ls::DoubleMatrix     ref         = getDoubleMatrixFromString(aSection->GetNonKeysAsString());

        RRDoubleMatrixPtr matrix = getScaledFluxControlCoefficientMatrix(gRRs[test]);
        if(!matrix)
        {
            CHECK(false);
            return;
        }

          clog<<"Reference Matrix:\n";
        clog<<ref<<endl;

          clog<<"Calculated Matrix:";
        clog<<matrixToString(matrix);

        if(!matrix || matrix->RSize != ref.RSize())
        {
            CHECK(false);
            return;
        }

        for(int i = 0 ; i < ref.RSize(); i++)
        {
            double val;
            if(!getMatrixElement(matrix, i , 0, &val))
            {
                CHECK(false);
            }
            CHECK_CLOSE(ref(i,0), val, 1e-4);

        }
        freeMatrix(matrix);
      }
    }

    TEST(FLOATING_SPECIES_IDS)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        IniSection* aSection = iniFiles[test]->GetSection("Floating Species Ids");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        IniKey* aKey = aSection->GetKey("list");
        if(!aKey)
        {
            CHECK(false);
            return;
        }

        RRStringArrayPtr list = getFloatingSpeciesIds(gRRs[test]);

        if(!list)
        {
            CHECK(false);
            return;
        }
        vector<string> selList = splitString(aKey->mValue," ,");
        CHECK(selList.size() == list->Count);
        for(int i = 0; i < selList.size(); i++)
        {
            CHECK(selList[i] == list->String[i]);
        }
        freeStringArray(list);
      }
    }

    TEST(BOUNDARY_SPECIES_IDS)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        IniSection* aSection = iniFiles[test]->GetSection("Boundary Species Ids");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        IniKey* aKey = aSection->GetKey("list");
        if(!aKey)
        {
            CHECK(false);
            return;
        }

        RRStringArrayPtr list = getBoundarySpeciesIds(gRRs[test]);

        if(!list)
        {
            CHECK(false);
            return;
        }
        vector<string> selList = splitString(aKey->mValue," ,");
        CHECK(selList.size() == list->Count);
        for(int i = 0; i < selList.size(); i++)
        {
            CHECK(selList[i] == list->String[i]);
        }
        freeStringArray(list);
      }
    }

    TEST(GLOBAL_PARAMETER_IDS)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        IniSection* aSection = iniFiles[test]->GetSection("Global Parameter Ids");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        IniKey* aKey = aSection->GetKey("list");
        if(!aKey)
        {
            CHECK(false);
            return;
        }

        RRStringArrayPtr list = getGlobalParameterIds(gRRs[test]);

        if(!list)
        {
            CHECK(false);
            return;
        }
        vector<string> selList = splitString(aKey->mValue," ,");
        CHECK(selList.size() == list->Count);
        for(int i = 0; i < selList.size(); i++)
        {
            CHECK(selList[i] == list->String[i]);
        }
        freeStringArray(list);
      }
    }

    TEST(COMPARTMENT_IDS)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        IniSection* aSection = iniFiles[test]->GetSection("Compartment Ids");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        IniKey* aKey = aSection->GetKey("list");
        if(!aKey)
        {
            CHECK(false);
            return;
        }

        RRStringArrayPtr list = getCompartmentIds(gRRs[test]);

        if(!list)
        {
            CHECK(false);
            return;
        }
        vector<string> selList = splitString(aKey->mValue," ,");
        CHECK(selList.size() == list->Count);
        for(int i = 0; i < selList.size(); i++)
        {
            CHECK(selList[i] == list->String[i]);
        }
        freeStringArray(list);
      }
    }

    TEST(REACTION_IDS)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        IniSection* aSection = iniFiles[test]->GetSection("Reaction Ids");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        IniKey* aKey = aSection->GetKey("list");
        if(!aKey)
        {
            CHECK(false);
            return;
        }

        RRStringArrayPtr list = getReactionIds(gRRs[test]);

        if(!list)
        {
            CHECK(false);
            return;
        }
        vector<string> selList = splitString(aKey->mValue," ,");
        CHECK(selList.size() == list->Count);
        for(int i = 0; i < selList.size(); i++)
        {
            CHECK(selList[i] == list->String[i]);
        }
        freeStringArray(list);
      }
    }

    TEST(SPECIES_INITIAL_CONDITION_IDS)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        IniSection* aSection = iniFiles[test]->GetSection("Species Initial Condition Ids");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        IniKey* aKey = aSection->GetKey("list");
        if(!aKey)
        {
            CHECK(false);
            return;
        }

        RRStringArrayPtr list = getFloatingSpeciesInitialConditionIds(gRRs[test]);

        if(!list)
        {
            CHECK(false);
            return;
        }
        vector<string> selList = splitString(aKey->mValue," ,");
        CHECK(selList.size() == list->Count);
        for(int i = 0; i < selList.size(); i++)
        {
            CHECK(selList[i] == list->String[i]);
        }
        freeStringArray(list);
      }
    }

    TEST(GET_EIGENVALUE_IDS)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        IniSection* aSection = iniFiles[test]->GetSection("Get Eigenvalue Ids");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        IniKey* aKey = aSection->GetKey("list");
        if(!aKey)
        {
            CHECK(false);
            return;
        }

        RRStringArrayPtr list = getEigenvalueIds(gRRs[test]);

        if(!list)
        {
            CHECK(false);
            return;
        }
        vector<string> selList = splitString(aKey->mValue," ,");
        CHECK(selList.size() == list->Count);
        for(int i = 0; i < selList.size(); i++)
        {
            CHECK(selList[i] == list->String[i]);
        }
        freeStringArray(list);
      }
    }

    TEST(GET_RATES_OF_CHANGE_IDS)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        IniSection* aSection = iniFiles[test]->GetSection("Get Rates Of Change Ids");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        IniKey* aKey = aSection->GetKey("list");
        if(!aKey)
        {
            CHECK(false);
            return;
        }

        RRStringArrayPtr list = getRatesOfChangeIds(gRRs[test]);

        if(!list)
        {
            CHECK(false);
            return;
        }
        vector<string> selList = splitString(aKey->mValue," ,");
        CHECK(selList.size() == list->Count);
        for(int i = 0; i < selList.size(); i++)
        {
            CHECK(selList[i] == list->String[i]);
        }
        freeStringArray(list);
      }
    }

    TEST(SET_STEADY_STATE_SELECTION_LIST_2)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        CHECK(gRRs[test]!=NULL);

        //Load reference data
        IniSection* aSection = iniFiles[test]->GetSection("Set Steady State Selection List 2");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        IniKey* aKey = aSection->GetKey("list");
        if(!aKey)
        {
            CHECK(false);
            return;
        }

        bool res = setSteadyStateSelectionList(gRRs[test], aKey->mValue.c_str());
        CHECK(res);
      }
    }

    TEST(GET_STEADY_STATE_SELECTION_LIST_2)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        IniSection* aSection = iniFiles[test]->GetSection("Get Steady State Selection List 2");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        IniKey* aKey = aSection->GetKey("list");
        if(!aKey)
        {
            CHECK(false);
            return;
        }

        RRStringArrayPtr list = getSteadyStateSelectionList(gRRs[test]);

        if(!list)
        {
            CHECK(false);
            return;
        }
        vector<string> selList = splitString(aKey->mValue," ,");
        CHECK(selList.size() == list->Count);
        for(int i = 0; i < selList.size(); i++)
        {
            CHECK(selList[i] == list->String[i]);
        }
        freeStringArray(list);
      }
    }

    TEST(SET_TIME_COURSE_SELECTION_LIST)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        CHECK(gRRs[test]!=NULL);

        //Load reference data
        IniSection* aSection = iniFiles[test]->GetSection("Set Time Course Selection List");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        IniKey* aKey = aSection->GetKey("list");
        if(!aKey)
        {
            CHECK(false);
            return;
        }

        bool res = setTimeCourseSelectionList(gRRs[test], aKey->mValue.c_str());
        CHECK(res);
      }
    }

    TEST(GET_TIME_COURSE_SELECTION_LIST)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        IniSection* aSection = iniFiles[test]->GetSection("Get Time Course Selection List");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        IniKey* aKey = aSection->GetKey("list");
        if(!aKey)
        {
            CHECK(false);
            return;
        }

        RRStringArrayPtr list = getTimeCourseSelectionList(gRRs[test]);

        if(!list)
        {
            CHECK(false);
            return;
        }
        vector<string> selList = splitString(aKey->mValue," ,");
        CHECK(selList.size() == list->Count);
        for(int i = 0; i < selList.size(); i++)
        {
            CHECK(selList[i] == list->String[i]);
        }
        freeStringArray(list);
      }
    }

    TEST(COMPUTE_STEADY_STATE_VALUES)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        IniSection* aSection = iniFiles[test]->GetSection("Compute Steady State Values");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        RRVector* values = computeSteadyStateValues(gRRs[test]);

        if(!values || values->Count != aSection->KeyCount())
        {
            freeVector(values);
            CHECK(false);
            return;
        }

        for(int i = 0 ; i < aSection->KeyCount(); i++)
        {
            IniKey *aKey = aSection->GetKey(i);

            //Check concentrations
            CHECK_CLOSE(aKey->AsFloat(), values->Data[i], 1e-4);
            clog<<"\n";
            clog<<"Ref:\t"<<aKey->AsFloat()<<"\tActual:\t "<<values->Data[i]<<endl;
        }
        freeVector(values);
      }
    }

    TEST(FLOATING_SPECIES_CONCENTRATIONS)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        CHECK(gRRs[test]!=NULL);

        //Read in the reference data, from the ini file
        IniSection* aSection = iniFiles[test]->GetSection("Floating Species Concentrations");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        RRVector* values = getFloatingSpeciesConcentrations(gRRs[test]);

        if(!values || values->Count != aSection->KeyCount())
        {
            CHECK(false);
            freeVector(values);
            return;
        }

        for(int i = 0 ; i < aSection->KeyCount(); i++)
        {
            IniKey *aKey = aSection->GetKey(i);

            //Check concentrations
            CHECK_CLOSE(aKey->AsFloat(), values->Data[i], 1e-4);
            clog<<"\n";
            clog<<"Ref:\t"<<aKey->AsFloat()<<"\tActual:\t "<<values->Data[i]<<endl;
        }
        freeVector(values);
      }
    }

    TEST(BOUNDARY_SPECIES_CONCENTRATIONS)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        CHECK(gRRs[test]!=NULL);

        //Read in the reference data, from the ini file
        IniSection* aSection = iniFiles[test]->GetSection("Boundary Species Concentrations");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        RRVector* values = getBoundarySpeciesConcentrations(gRRs[test]);

        if(!values || values->Count != aSection->KeyCount())
        {
            CHECK(false);
            freeVector(values);
            return;
        }

        for(int i = 0 ; i < aSection->KeyCount(); i++)
        {
            IniKey *aKey = aSection->GetKey(i);

            //Check concentrations
            CHECK_CLOSE(aKey->AsFloat(), values->Data[i], 1e-4);
            clog<<"\n";
            clog<<"Ref:\t"<<aKey->AsFloat()<<"\tActual:\t "<<values->Data[i]<<endl;
        }
        freeVector(values);
      }
    }

    TEST(GET_GLOBABL_PARAMETER_VALUES)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        CHECK(gRRs[test]!=NULL);

        //Read in the reference data, from the ini file
        IniSection* aSection = iniFiles[test]->GetSection("Get Global Parameter Values");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        IniKey* aKey = aSection->GetKey("list");
        if(!aKey)
        {
            CHECK(false);
            return;
        }

        RRVector* values = getGlobalParameterValues(gRRs[test]);

        vector<string> refList = splitString(aKey->mValue," ,");

        if(!values || values->Count != refList.size())
        {
            CHECK(false);
            freeVector(values);
            return;
        }

        for(int i = 0 ; i < refList.size(); i++)
        {

            //Check concentrations
            CHECK_CLOSE(toDouble(refList[i]), values->Data[i], 1e-4);
            clog<<"\n";
            clog<<"Ref:\t"<<toDouble(refList[i])<<"\tActual:\t "<<values->Data[i]<<endl;
        }
        freeVector(values);
      }
    }

    TEST(GET_INITIAL_FLOATING_SPECIES_CONCENTRATIONS)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        CHECK(gRRs[test]!=NULL);

        //Read in the reference data, from the ini file
        IniSection* aSection = iniFiles[test]->GetSection("Get Initial Floating Species Concs");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        IniKey* aKey = aSection->GetKey("list");
        if(!aKey)
        {
            CHECK(false);
            return;
        }

        RRVector* values = getFloatingSpeciesInitialConcentrations(gRRs[test]);

        vector<string> refList = splitString(aKey->mValue," ,");

        if(!values || values->Count != refList.size())
        {
            CHECK(false);
            freeVector(values);
            return;
        }

        for(int i = 0 ; i < refList.size(); i++)
        {

            //Check concentrations
            CHECK_CLOSE(toDouble(refList[i]), values->Data[i], 1e-4);
            clog<<"\n";
            clog<<"Ref:\t"<<toDouble(refList[i])<<"\tActual:\t "<<values->Data[i]<<endl;
        }
        freeVector(values);
      }
    }

    TEST(GET_REACTION_RATES)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        CHECK(gRRs[test]!=NULL);

        //Read in the reference data, from the ini file
        IniSection* aSection = iniFiles[test]->GetSection("Get Reaction Rates");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        IniKey* aKey = aSection->GetKey("list");
        if(!aKey)
        {
            CHECK(false);
            return;
        }

        RRVector* values = getReactionRates(gRRs[test]);

        vector<string> refList = splitString(aKey->mValue," ,");

        if(!values || values->Count != refList.size())
        {
            CHECK(false);
            freeVector(values);
            return;
        }

        for(int i = 0 ; i < refList.size(); i++)
        {

            //Check concentrations
            CHECK_CLOSE(toDouble(refList[i]), values->Data[i], 1e-4);
            clog<<"\n";
            clog<<"Ref:\t"<<toDouble(refList[i])<<"\tActual:\t "<<values->Data[i]<<endl;
        }
        freeVector(values);
      }
    }

    TEST(GET_REACTION_RATE_BY_INDEX)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        CHECK(gRRs[test]!=NULL);

        //Read in the reference data, from the ini file
        IniSection* aSection = iniFiles[test]->GetSection("Get Reaction Rates");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        IniKey* aKey = aSection->GetKey("list");
        if(!aKey)
        {
            CHECK(false);
            return;
        }

        vector<string> refList = splitString(aKey->mValue," ,");

        if(refList.size() != getNumberOfReactions(gRRs[test]))
        {
            CHECK(false);
            return;
        }

        for(int i = 0 ; i < refList.size(); i++)
        {
            double value;
              if(!getReactionRate(gRRs[test], i, &value))
              {
                CHECK(false);
              }

            //Check concentrations
            CHECK_CLOSE(toDouble(refList[i]), value, 1e-4);
            clog<<"\n";
            clog<<"Ref:\t"<<toDouble(refList[i])<<"\tActual:\t "<<value<<endl;
        }
      }
    }

    TEST(NUMBER_OF_DEPENDENT_SPECIES)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        IniSection* aSection = iniFiles[test]->GetSection("Number of Dependent Species");
        if(!aSection)
        {
            continue;
        }

        if(aSection->KeyCount() < 1)
        {
            CHECK(false);
          }

          IniKey *aKey = aSection->GetKey(0);
          int val = getNumberOfDependentSpecies(gRRs[test]);

          CHECK(aKey->AsInt() ==  val);
      }
    }

    TEST(NUMBER_OF_INDEPENDENT_SPECIES)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        IniSection* aSection = iniFiles[test]->GetSection("Number of Independent Species");
        if(!aSection)
        {
            continue;
        }

        if(aSection->KeyCount() < 1)
        {
            CHECK(false);
          }

          IniKey *aKey = aSection->GetKey(0);
          int val = getNumberOfIndependentSpecies(gRRs[test]);

          CHECK(aKey->AsInt() ==  val);
      }
    }

    TEST(NUMBER_OF_RULES)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        IniSection* aSection = iniFiles[test]->GetSection("Number of Rules");
        if(!aSection)
        {
            continue;
        }

        if(aSection->KeyCount() < 1)
        {
            CHECK(false);
          }

          IniKey *aKey = aSection->GetKey(0);
          int val = getNumberOfRules(gRRs[test]);

          CHECK(aKey->AsInt() ==  val);
      }
    }

    TEST(GET_RATES_OF_CHANGE)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        CHECK(gRRs[test]!=NULL);

        //Read in the reference data, from the ini file
        IniSection* aSection = iniFiles[test]->GetSection("Get Rates Of Change");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        IniKey* aKey = aSection->GetKey("list");
        if(!aKey)
        {
            CHECK(false);
            return;
        }

        RRVector* values = getRatesOfChange(gRRs[test]);

        vector<string> refList = splitString(aKey->mValue," ,");

        if(!values || values->Count != refList.size())
        {
            CHECK(false);
            freeVector(values);
            return;
        }

        for(int i = 0 ; i < refList.size(); i++)
        {
            //Check concentrations
            CHECK_CLOSE(toDouble(refList[i]), values->Data[i], 1e-4);
            clog<<"\n";
            clog<<"Ref:\t"<<toDouble(refList[i])<<"\tActual:\t "<<values->Data[i]<<endl;
        }
        freeVector(values);
      }
    }

    TEST(GET_REACTION_RATES_EX)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        CHECK(gRRs[test]!=NULL);

        //Read in the reference data, from the ini file
        IniSection* aSection = iniFiles[test]->GetSection("Get Reaction Rates Ex");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        IniKey* conc = aSection->GetKey("list1");
        if(!conc)
        {
            CHECK(false);
            return;
        }

        IniKey* refs = aSection->GetKey("list2");
        if(!refs)
        {
            CHECK(false);
            return;
        }

          vector<string> refList = splitString(refs->mValue," ");

          vector<string> concList = splitString(conc->mValue," ");
          RRVector* aVector = createVector(concList.size());

          for(int i = 0; i < concList.size(); i++)
          {
            aVector->Data[i] = toDouble(concList[i]);
          }

        RRVector* values = getReactionRatesEx(gRRs[test], aVector);


        if(!values || values->Count != refList.size())
        {
            CHECK(false);
            freeVector(values);
            freeVector(aVector);
            return;
        }

        for(int i = 0 ; i < refList.size(); i++)
        {
            //Check concentrations
            CHECK_CLOSE(toDouble(refList[i]), values->Data[i], 1e-4);
            clog<<"\n";
            clog<<"Ref:\t"<<toDouble(refList[i])<<"\tActual:\t "<<values->Data[i]<<endl;
        }
        freeVector(values);
        freeVector(aVector);
      }
    }

    TEST(GET_RATES_OF_CHANGE_EX)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        CHECK(gRRs[test]!=NULL);

        //Read in the reference data, from the ini file
        IniSection* aSection = iniFiles[test]->GetSection("Get Rates Of Change Ex");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        IniKey* conc = aSection->GetKey("list1");
        if(!conc)
        {
            CHECK(false);
            return;
        }

        IniKey* refs = aSection->GetKey("list2");
        if(!refs)
        {
            CHECK(false);
            return;
        }

          vector<string> refList = splitString(refs->mValue," ");

          vector<string> concList = splitString(conc->mValue," ");
          RRVector* aVector = createVector(concList.size());

          for(int i = 0; i < concList.size(); i++)
          {
            aVector->Data[i] = toDouble(concList[i]);
          }

          RRVector* values = getRatesOfChangeEx(gRRs[test], aVector);


        if(!values || values->Count != refList.size())
        {
            CHECK(false);
            freeVector(values);
            freeVector(aVector);
            return;
        }

        for(int i = 0 ; i < refList.size(); i++)
        {
            //Check concentrations
            CHECK_CLOSE(toDouble(refList[i]), values->Data[i], 1e-4);
            clog<<"\n";
            clog<<"Ref:\t"<<toDouble(refList[i])<<"\tActual:\t "<<values->Data[i]<<endl;
        }
        freeVector(values);
        freeVector(aVector);
      }
    }

    TEST(GET_RATE_OF_CHANGE_BY_INDEX)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        CHECK(gRRs[test]!=NULL);

        //Read in the reference data, from the ini file
        IniSection* aSection = iniFiles[test]->GetSection("Get Rates Of Change Ex");
        if(!aSection || !gRRs[test])
        {
            continue;
        }

        IniKey* conc = aSection->GetKey("list1");
        if(!conc)
        {
            CHECK(false);
            return;
        }

        IniKey* refs = aSection->GetKey("list2");
        if(!refs)
        {
            CHECK(false);
            return;
        }

          vector<string> refList = splitString(refs->mValue," ");

          vector<string> concList = splitString(conc->mValue," ");
          RRVector* aVector = createVector(concList.size());

          for(int i = 0; i < concList.size(); i++)
          {
            aVector->Data[i] = toDouble(concList[i]);
          }

        for(int i = 0 ; i < refList.size(); i++)
        {
            double value;
              if(!getRateOfChange(gRRs[test], i, &value))
              {
                CHECK(false);
              }

            //Check concentrations
            CHECK_CLOSE(toDouble(refList[i]), value, 1e-4);
            clog<<"\n";
            clog<<"Ref:\t"<<toDouble(refList[i])<<"\tActual:\t "<<value<<endl;
        }
        freeVector(aVector);
      }
    }

    TEST(FREE_RR_INSTANCE)
    {
      for(size_t test=0; test<gRRs.size(); test++) {
        CHECK(freeRRInstance(gRRs[test]));
        gRRs[test] = NULL;
        delete iniFiles[test];
        iniFiles[test] = NULL;
      }
    }
}
