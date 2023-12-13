#include <string>
#include <algorithm>
#include <sys/types.h>

#ifndef WIN32
#include <dirent.h>
#endif

#include "Poco/Path.h"
#include "Poco/Glob.h"
#include "gtest/gtest.h"
#include "rrConfig.h"
#include "rrIniFile.h"
#include "rrLogger.h"
#include "rrRoadRunner.h"
#include "rrUtils.h"
#include "C/rrc_api.h"
#include "C/rrc_cpp_support.h"
#include "../test_util.h"
#include <filesystem>
#include "RoadRunnerTest.h"

//using..
using namespace std;
using namespace ls;
using namespace rrc;
using namespace rr;
using std::filesystem::path;


class CApiRRTests : public RoadRunnerTest {
public:

    path rrTestFileDir = rrTestDir_ / "rrtest_files";

    RRHandle gRR = createRRInstance();

    IniFile iniFile;

    CApiRRTests() = default;
    ~CApiRRTests() {
        freeRRInstance(gRR);
    }

    void checkRRTest(const std::string &fname) {

        path rrTestFileName = (rrTestFileDir / (fname + ".rrtest"));
        check_LoadData(rrTestFileName);
        std::cout << "Checking file " << rrTestFileName << endl;
        std::cout << " which has " << iniFile.GetNumberOfSections() << " sections " << std::endl;
        ASSERT_TRUE(gRR != NULL);
        if (check_Unimplemented())
        {
            std::cout << "Skipping this test." << endl;
            return;
        }
        for (size_t sec = 0; sec < iniFile.GetNumberOfSections(); sec++) {
            IniSection *aSection = iniFile.GetSection(sec);
            aSection->mIsUsed = true;
            std::string sectionName = toLower(aSection->mName);

            //std::cout << "running section name : " << sectionName << std::endl;

            if (sectionName == "sbml") {
                continue; //Already managed this section in DATA_FILES
            } else if (sectionName == "conservation laws") {
                check_SET_COMPUTE_AND_ASSIGN_CONSERVATION_LAWS(aSection);
                continue;
            } else if (sectionName == "set steady state selection list") {
                check_SET_STEADY_STATE_SELECTION_LIST(aSection);
                continue;
            } else if (sectionName == "get steady state selection list") {
                check_GET_STEADY_STATE_SELECTION_LIST(aSection);
                continue;
            } else if (sectionName == "species concentrations") {
                check_SPECIES_CONCENTRATIONS(aSection);
                continue;
            } else if (sectionName == "get species initial concentrations") {
                check_GET_SPECIES_INITIAL_CONCENTRATIONS(aSection);
                continue;
            } else if (sectionName == "get species initial concentrations by index") {
                check_GET_SPECIES_INITIAL_CONCENTRATION_BY_INDEX(aSection);
                continue;
            } else if (sectionName == "get initial floating species concs") {
                check_GET_INITIAL_FLOATING_SPECIES_CONCENTRATIONS(aSection);
                continue;
            } else if (sectionName == "set species initial concentrations by index") {
                check_SET_SPECIES_INITIAL_CONCENTRATION_BY_INDEX(aSection);
                continue;
            } else if (sectionName == "set species initial concentrations") {
                check_SET_SPECIES_INITIAL_CONCENTRATIONS(aSection);
                continue;
            } else if (sectionName == "steady state fluxes") {
                check_STEADY_STATE_FLUXES(aSection);
                continue;
            } else if (sectionName == "full jacobian") {
                check_FULL_JACOBIAN(aSection);
                continue;
            } else if (sectionName == "reduced jacobian") {
                check_REDUCED_JACOBIAN(aSection);
                continue;
            } else if (sectionName == "amount jacobian") {
                check_AMOUNT_JACOBIAN(aSection);
                continue;
            } else if (sectionName == "individual eigenvalues") {
                check_INDIVIDUAL_EIGENVALUES(aSection);
                continue;
            } else if (sectionName == "individual amount eigenvalues") {
                check_INDIVIDUAL_AMOUNT_EIGENVALUES(aSection);
                continue;
            } else if (sectionName == "eigenvalue matrix") {
                check_GET_EIGENVALUE_MATRIX(aSection);
                continue;
            } else if (sectionName == "eigenvalue amount matrix") {
                check_GET_EIGENVALUE_AMOUNT_MATRIX(aSection);
                continue;
            } else if (sectionName == "reduced eigenvalue matrix") {
                check_GET_REDUCED_EIGENVALUE_MATRIX(aSection);
                continue;
            } else if (sectionName == "stoichiometry matrix") {
                check_STOICHIOMETRY_MATRIX(aSection);
                continue;
            } else if (sectionName == "reduced stoichiometry matrix") {
                check_REDUCED_STOICHIOMETRY_MATRIX(aSection);
                continue;
            } else if (sectionName == "link matrix") {
                check_LINK_MATRIX(aSection);
                continue;
            } else if (sectionName == "unscaled elasticity matrix") {
                check_UNSCALED_ELASTICITY_MATRIX(aSection);
                continue;
            } else if (sectionName == "unscaled elasticity amount matrix") {
                check_UNSCALED_ELASTICITY_AMOUNT_MATRIX(aSection);
                continue;
            } else if (sectionName == "scaled elasticity matrix") {
                check_SCALED_ELASTICITY_MATRIX(aSection);
                continue;
            } else if (sectionName == "scaled elasticity amount matrix") {
                check_SCALED_ELASTICITY_AMOUNT_MATRIX(aSection);
                continue;
            } else if (sectionName == "unscaled concentration control matrix") {
                check_UNSCALED_CONCENTRATION_CONTROL_MATRIX(aSection);
                continue;
            } else if (sectionName == "scaled concentration control matrix") {
                check_SCALED_CONCENTRATION_CONTROL_MATRIX(aSection);
                continue;
            } else if (sectionName == "unscaled flux control matrix") {
                check_UNSCALED_FLUX_CONTROL_MATRIX(aSection);
                continue;
            } else if (sectionName == "scaled flux control matrix") {
                check_SCALED_FLUX_CONTROL_MATRIX(aSection);
                continue;
            } else if (sectionName == "get control coefficient") {
                check_GET_CONTROL_COEFFICIENT(aSection);
                continue;
            } else if (sectionName == "test reset") {
                check_CHECK_RESET(aSection);
                continue;
            } else if (sectionName == "test resetall") {
                check_CHECK_RESETALL(aSection);
                continue;
            } else if (sectionName == "test resettoorigin") {
                check_CHECK_RESETTOORIGIN(aSection);
                continue;
            } else if (sectionName == "check rk4 output") {
                check_CHECK_RK4_OUTPUT(aSection);
                continue;
            } else if (sectionName == "check rk45 output") {
                check_CHECK_RK45_OUTPUT(aSection);
                continue;
            } else if (sectionName == "test setvalues") {
                check_CHECK_SETVALUES(aSection);
                continue;
            } else if (sectionName == "floating species ids") {
                check_FLOATING_SPECIES_IDS(aSection);
                continue;
            } else if (sectionName == "boundary species ids") {
                check_BOUNDARY_SPECIES_IDS(aSection);
                continue;
            } else if (sectionName == "global parameter ids") {
                check_GLOBAL_PARAMETER_IDS(aSection);
                continue;
            } else if (sectionName == "compartment ids") {
                check_COMPARTMENT_IDS(aSection);
                continue;
            } else if (sectionName == "reaction ids") {
                check_REACTION_IDS(aSection);
                continue;
            } else if (sectionName == "species initial concentration ids") {
                check_SPECIES_INITIAL_CONDITION_IDS(aSection);
                continue;
            } else if (sectionName == "get eigenvalue ids") {
                check_GET_EIGENVALUE_IDS(aSection);
                continue;
            } else if (sectionName == "get rates of change ids") {
                check_GET_RATES_OF_CHANGE_IDS(aSection);
                continue;
            } else if (sectionName == "set steady state selection list 2") {
                check_SET_STEADY_STATE_SELECTION_LIST_2(aSection);
                continue;
            } else if (sectionName == "get steady state selection list 2") {
                check_GET_STEADY_STATE_SELECTION_LIST_2(aSection);
                continue;
            } else if (sectionName == "set time course selection list") {
                check_SET_TIME_COURSE_SELECTION_LIST(aSection);
                continue;
            } else if (sectionName == "get time course selection list") {
                check_GET_TIME_COURSE_SELECTION_LIST(aSection);
                continue;
            } else if (sectionName == "compute steady state values") {
                check_COMPUTE_STEADY_STATE_VALUES(aSection);
                continue;
            } else if (sectionName == "floating species concentrations") {
                check_FLOATING_SPECIES_CONCENTRATIONS(aSection);
                continue;
            } else if (sectionName == "boundary species concentrations") {
                check_BOUNDARY_SPECIES_CONCENTRATIONS(aSection);
                continue;
            } else if (sectionName == "get global parameter values") {
                check_GET_GLOBAL_PARAMETER_VALUES(aSection);
                continue;
            } else if (sectionName == "get reaction rates") {
                check_GET_REACTION_RATES(aSection);
                continue;
            } else if (sectionName == "get reaction rates by index") {
                check_GET_REACTION_RATE_BY_INDEX(aSection);
                continue;
            } else if (sectionName == "number of dependent species") {
                check_NUMBER_OF_DEPENDENT_SPECIES(aSection);
                continue;
            } else if (sectionName == "number of independent species") {
                check_NUMBER_OF_INDEPENDENT_SPECIES(aSection);
                continue;
            } else if (sectionName == "number of rate rules") {
                check_NUMBER_OF_RATE_RULES(aSection);
                continue;
            } else if (sectionName == "get rates of change") {
                check_GET_RATES_OF_CHANGE(aSection);
                continue;
            } else if (sectionName == "get reaction rates ex") {
                check_GET_REACTION_RATES_EX(aSection);
                continue;
            } else if (sectionName == "get rates of change ex") {
                check_GET_RATES_OF_CHANGE_EX(aSection);
                continue;
            } else if (sectionName == "get rates of change by index") {
                check_GET_RATES_OF_CHANGE_BY_INDEX(aSection);
                continue;
            } else if (sectionName == "amount/concentration jacobians") {
                check_AMOUNT_CONCENTRATION_JACOBIANS(aSection);
                continue;
            } else if (sectionName == "check default time step") {
                check_CHECK_DEFAULT_TIME_STEP(aSection);
                continue;
            } else if (sectionName == "check simulate points vs steps") {
                check_CHECK_SIMULATE_POINTS_VS_STEPS(aSection);
                continue;
            } else if (sectionName == "check monotonic timepoints") {
                check_CHECK_MONOTONIC_TIMEPOINTS(aSection);
                continue;
            } else if (sectionName == "check event pre and postfire timepoints") {
                check_CHECK_EVENT_PRE_AND_POSTFIRE_TIMEPOINTS(aSection);
                continue;
            } else if (sectionName == "test resetconservedtotal") {
                check_CHECK_RESETCONSERVEDTOTAL(aSection);
                continue;
            } else {
                clog << "Unused section:\t" << aSection->mName << endl;
                //EXPECT_TRUE(false);
                continue;
            }
        }
    }


    /// Function to get around inability of C to switch on strings, courtesy
    ///  of Nick at https://stackoverflow.com/questions/650162/why-the-switch-statement-cannot-be-applied-on-strings
    constexpr unsigned int
    hashc(const char *s, int off = 0) {
        return !s[off] ? 5381 : (hashc(s, off + 1) * 33) ^ s[off];
    }

    string Trim(const string &input) {
        string ret = trim(input, '\n');
        if (!ret.empty() && ret[0] == '"' && ret[ret.size() - 1] == '"') {
            ret = ret.replace(0, 1, "");
            ret = ret.replace(ret.size() - 1, 1, "");
        }
        return ret;
    }

    void compareJacobians(RRHandle gRR) {
        RoadRunner *rri = castToRoadRunner(gRR);
        ls::DoubleMatrix jFull = rri->getFullJacobian();
        ls::DoubleMatrix jReduced = rri->getReducedJacobian();

        //Check dimensions
        ASSERT_TRUE(jFull.RSize() == jReduced.RSize());
        ASSERT_TRUE(jFull.CSize() == jReduced.CSize());

        for (int row = 0; row < jFull.RSize(); row++) {
            for (int col = 0; col < jFull.CSize(); col++) {
                EXPECT_NEAR(jFull(row, col), jReduced(row, col), 1e-6);
            }
        }
    }

    void compareMatrices(const ls::DoubleMatrix &ref, const ls::DoubleMatrix &calc) {
        //clog << "Reference Matrix:" << endl;
        //clog << ref << endl;

        //clog << "Calculated Matrix:" << endl;
        //clog << calc << endl;

        ASSERT_TRUE(calc.RSize() == ref.RSize());
        ASSERT_TRUE(calc.CSize() == ref.CSize());

        for (int i = 0; i < ref.RSize(); i++) {
            for (int j = 0; j < ref.CSize(); j++) {
                if (abs(ref(i, j) - calc(i, j)) > abs((ref(i, j) + 1e-7) * 1e-4))
                    //clog <<  "check close failed zzxx\n";
                    EXPECT_NEAR(ref(i, j), calc(i, j), abs((ref(i, j) + 1e-7) * 1e-4));
            }
        }
    }


    void compareMatrices(const ls::DoubleMatrix &ref, const std::vector<ls::Complex> calc) {
        //clog<<"Reference Matrix:" << endl;
        //clog<<ref<<endl;

        //clog<<"Calculated Matrix:" << endl;
        //for(int i = 0; i < calc.size(); ++i)
        //{
        //    clog << calc[i] << endl;
        //}

        ASSERT_TRUE(ref.CSize() == 2);
        ASSERT_TRUE(calc.size() == ref.RSize());

        for (int i = 0; i < ref.RSize(); i++) {
            EXPECT_NEAR(ref(i, 0), std::real(calc[i]), abs((ref(i, 0) + 1e-7) * 1e-4));
            EXPECT_NEAR(ref(i, 1), std::imag(calc[i]), abs((ref(i, 1) + 1e-7) * 1e-4));
        }
    }

    void compareMatrices(const ls::DoubleMatrix &ref, RRDoubleMatrixPtr calc) {
        ASSERT_TRUE(calc);

        //clog<<"Reference Matrix:\n";
        //clog<<ref<<endl;

        //clog<<"Calculated Matrix:";
        //clog<<matrixToString(calc);

        ASSERT_TRUE(calc->RSize == ref.RSize());
        ASSERT_TRUE(calc->CSize == ref.CSize());

        for (int i = 0; i < ref.RSize(); i++) {
            for (int j = 0; j < ref.CSize(); j++) {
                double val;
                if (!getMatrixElement(calc, i, j, &val)) {
                    EXPECT_TRUE(false);
                }
                EXPECT_NEAR(ref(i, j), val, abs((val + 1e-7) * 1e-4));
            }
        }
    }

    void trySteadyState(RRHandle &gRR) {
        double val;
        for (int i = 0; i < 10; i++) {
            EXPECT_TRUE(steadyState(gRR, &val));
            if (val < 1e-5) break;
        }
        EXPECT_NEAR(0, val, 1e-5);
    }

    int findCloseTimepoint(const DoubleMatrix *m, double t, double tol) {
        for (int i = 0; i < m->RSize(); ++i) {
            if (abs((*m)(i, 0) - t) <= tol)
                return i;
        }
        return -1;
    }

    std::set<path> getTestFiles(const std::string &dir) {
        std::set<path> result;
        for (auto p : fs::directory_iterator(rrTestFileDir)) {
            if (p.path().extension() == ".rrtest") {
                result.insert(p.path());
            }
        }
        return result;
    }

//These tests are intended to duplicate the Python tests
    void check_LoadData(path rrTestFileName) {
        // need to re-assign it, Load does not clear old data.;
        iniFile.Clear();

        EXPECT_TRUE(std::filesystem::exists(rrTestFileName));
        EXPECT_TRUE(iniFile.Load(rrTestFileName.string()));

        IniSection *sbmlsec = iniFile.GetSection("SBML");
        if (!sbmlsec) {
            EXPECT_TRUE(false);
            rrLog(Logger::LOG_FATAL) << "No 'SBML' section found in " << rrTestFileName;
            return;
        }
        sbmlsec->mIsUsed = true;
        string sbml = sbmlsec->GetNonKeysAsString();
        if (sbml.find('<') == string::npos) {
            sbml = (rrTestDir_ / path("rrtest_files") / sbml).string();
            EXPECT_TRUE(std::filesystem::exists(sbml));
        }
        if (!loadSBMLEx(gRR, sbml.c_str(), true)) {
            EXPECT_TRUE(false);
            rrLog(Logger::LOG_FATAL) << "Unable to load SBML:" << endl << sbml;
        }
    }

    void check_SET_COMPUTE_AND_ASSIGN_CONSERVATION_LAWS(IniSection *aSection) {
        bool res = setComputeAndAssignConservationLaws(gRR, true);
        EXPECT_TRUE(res);
        //clog << "\nConservation laws: " << res << endl;
    }

    void check_SET_STEADY_STATE_SELECTION_LIST(IniSection *aSection) {
        clog << "==== SET_STEADY_STATE_SELECTION_LIST ====" << endl;
        aSection->mIsUsed = true;

        setComputeAndAssignConservationLaws(gRR, true);

        string keys = Trim(aSection->GetNonKeysAsString());

        bool res = setSteadyStateSelectionList(gRR, keys.c_str());
        EXPECT_TRUE(res);

        //Actually calculate the steady state:
        trySteadyState(gRR);
    }

    void check_GET_STEADY_STATE_SELECTION_LIST(IniSection *aSection) {
        clog << "==== GET_STEADY_STATE_SELECTION_LIST ====" << endl;
        aSection->mIsUsed = true;

        string keys = Trim(aSection->GetNonKeysAsString());

        RRStringArrayPtr list = getSteadyStateSelectionList(gRR);

        if (!list) {
            EXPECT_TRUE(false);
            return;
        }
        vector<string> selList = splitString(keys, " ,");
        if (selList.size() != list->Count) {
            EXPECT_TRUE(false);
            freeStringArray(list);
            return;
        }
        for (int i = 0; i < selList.size(); i++) {
            EXPECT_EQ(selList[i], list->String[i]);;
        }
        freeStringArray(list);
    }

    void check_SPECIES_CONCENTRATIONS(IniSection *aSection) {
        clog << "==== SPECIES_CONCENTRATIONS ====" << endl;
        aSection->mIsUsed = true;

        string valstr = Trim(aSection->GetNonKeysAsString());
        vector<string> vals = splitString(valstr, " ");
        for (int i = 0; i < vals.size(); i++) {
            double val;

            //API Function getValue
            if (!getValue(gRR, vals[i].c_str(), &val)) {
                EXPECT_TRUE(false);
            }

            //Check concentrations
            EXPECT_NEAR(toDouble(vals[i]), val, 1e-6);
            //clog << "\n";
            //clog << "Ref:\t" << vals[i] << "\tActual:\t " << val << endl;
        }
    }

    void check_GET_SPECIES_INITIAL_CONCENTRATIONS(IniSection *aSection) {
        clog << "==== GET_SPECIES_INITIAL_CONCENTRATIONS ====" << endl;
        aSection->mIsUsed = true;

        for (int i = 0; i < aSection->KeyCount(); i++) {
            IniKey *aKey = aSection->GetKey(i);
            double val;
            if (!getValue(gRR, aKey->mKey.c_str(), &val)) {
                EXPECT_TRUE(false);
            }

            //Check concentrations
            EXPECT_NEAR(aKey->AsFloat(), val, 1e-6);
            //clog << "\n";
            //clog << "Ref:\t" << aKey->AsFloat() << "\tActual:\t " << val << endl;
        }
    }

    void check_GET_SPECIES_INITIAL_CONCENTRATION_BY_INDEX(IniSection *aSection) {
        clog << "==== GET_SPECIES_INITIAL_CONCENTRATION_BY_INDEX ====" << endl;
        aSection->mIsUsed = true;

        string valstr = Trim(aSection->GetNonKeysAsString());
        vector<string> vals = splitString(valstr, " ");
        for (int i = 0; i < static_cast<int>(vals.size()); i++) {
            IniKey *aKey = aSection->GetKey(i);
            double val;
            ASSERT_TRUE(getFloatingSpeciesInitialConcentrationByIndex(gRR, i, &val));

            //Check concentrations
            EXPECT_NEAR(toDouble(vals[i]), val, 1e-6);
            //clog << "\n";
            //clog << "Ref:\t" << toDouble(vals[i]) << "\tActual:\t " << val << endl;
        }
    }

    void check_GET_INITIAL_FLOATING_SPECIES_CONCENTRATIONS(IniSection *aSection) {
        clog << "==== GET_INITIAL_FLOATING_SPECIES_CONCENTRATIONS ====" << endl;
        aSection->mIsUsed = true;

        string keys = Trim(aSection->GetNonKeysAsString());

        RRVector *values = getFloatingSpeciesInitialConcentrations(gRR);

        vector<string> refList = splitString(keys, " ,");

        if (!values || values->Count != refList.size()) {
            EXPECT_TRUE(false);
            freeVector(values);
            return;
        }

        for (int i = 0; i < refList.size(); i++) {

            //Check concentrations
            EXPECT_NEAR(toDouble(refList[i]), values->Data[i], 1e-6);
            //clog <<"\n";
            //clog <<"Ref:\t" << toDouble(refList[i]) << "\tActual:\t " << values->Data[i] << endl;
        }
        freeVector(values);
    }

    void check_SET_SPECIES_INITIAL_CONCENTRATION_BY_INDEX(IniSection *aSection) {
        clog << "==== SET_SPECIES_INITIAL_CONCENTRATION_BY_INDEX ====" << endl;
        aSection->mIsUsed = true;

        string valstr = Trim(aSection->GetNonKeysAsString());
        vector<string> vals = splitString(valstr, " ");
        for (int i = 0; i < static_cast<int>(vals.size()); i++) {
            //Set the value..
            double newval = toDouble(vals[i]);
            setFloatingSpeciesInitialConcentrationByIndex(gRR, i, newval);

            double val;
            //Read it back
            if (!getFloatingSpeciesInitialConcentrationByIndex(gRR, i, &val)) {
                EXPECT_TRUE(false);
            }

            //Check concentrations
            EXPECT_NEAR(newval, val, 1e-6);
            //clog <<"\n";
            //clog <<"Ref:\t" << newval << "\tActual:\t " << val << endl;
        }
    }

    void check_SET_SPECIES_INITIAL_CONCENTRATIONS(IniSection *aSection) {
        clog << "==== SET_SPECIES_INITIAL_CONCENTRATIONS ====" << endl;
        aSection->mIsUsed = true;

        for (int i = 0; i < aSection->KeyCount(); i++) {
            IniKey *aKey = aSection->GetKey(i);

            //Set the value..
            setValue(gRR, aKey->mKey.c_str(), aKey->AsFloat());

            double val;
            //Read it back
            if (!getValue(gRR, aKey->mKey.c_str(), &val)) {
                EXPECT_TRUE(false);
            }

            //Check concentrations
            EXPECT_NEAR(aKey->AsFloat(), val, 1e-6);
            //clog <<"\n";
            //clog <<"Ref:\t" << aKey->AsFloat() << "\tActual:\t " << val << endl;
        }

        //We need to set back the values to concentrations, after running steady state..
        reset(gRR);
        trySteadyState(gRR);
    }

    void check_STEADY_STATE_FLUXES(IniSection *aSection) {
        clog << "==== STEADY STATE FLUXES ====" << endl;
        aSection->mIsUsed = true;

        reset(gRR);
        trySteadyState(gRR);

        for (int i = 0; i < aSection->KeyCount(); i++) {
            IniKey *aKey = aSection->GetKey(i);
            double val;

            //API Function getValue
            if (!getValue(gRR, aKey->mKey.c_str(), &val)) {
                EXPECT_TRUE(false);
            }

            //Check concentrations
            EXPECT_NEAR(aKey->AsFloat(), val, max(1e-9, abs(val * 1e-5)));
            //clog <<"\n";
            //clog <<"Ref:\t" << aKey->AsFloat() << "\tActual:\t " << val << endl;
        }
    }

    void check_FULL_JACOBIAN(IniSection *aSection) {
        clog << "==== FULL JACOBIAN ====" << endl;
        aSection->mIsUsed = true;

        Config::setValue(Config::ROADRUNNER_JACOBIAN_MODE,
                         (unsigned) Config::ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS);
        RoadRunner *rri = castToRoadRunner(gRR);


        ls::DoubleMatrix jActual = rri->getFullJacobian();
        ls::DoubleMatrix jRef = getDoubleMatrixFromString(aSection->GetNonKeysAsString());

        compareMatrices(jRef, jActual);
    }

    void check_REDUCED_JACOBIAN(IniSection *aSection) {
        clog << "==== REDUCED JACOBIAN ====" << endl;
        aSection->mIsUsed = true;

        Config::setValue(Config::ROADRUNNER_JACOBIAN_MODE,
                         (unsigned) Config::ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS);
        RoadRunner *rri = castToRoadRunner(gRR);

        ls::DoubleMatrix jActual = rri->getReducedJacobian();
        ls::DoubleMatrix jRef = getDoubleMatrixFromString(aSection->GetNonKeysAsString());

        compareMatrices(jRef, jActual);
    }

    void check_AMOUNT_JACOBIAN(IniSection *aSection) {
        clog << "==== AMOUNT JACOBIAN ====" << endl;
        aSection->mIsUsed = true;

        Config::setValue(Config::ROADRUNNER_JACOBIAN_MODE, (unsigned) Config::ROADRUNNER_JACOBIAN_MODE_AMOUNTS);
        RoadRunner *rri = castToRoadRunner(gRR);

        ls::DoubleMatrix jActual = rri->getFullJacobian();
        ls::DoubleMatrix jRef = getDoubleMatrixFromString(aSection->GetNonKeysAsString());

        compareMatrices(jRef, jActual);
    }

    void check_INDIVIDUAL_EIGENVALUES(IniSection *aSection) {
        clog << "==== INDIVIDUAL_EIGENVALUES ====" << endl;
        aSection->mIsUsed = true;

        RoadRunner *rri = castToRoadRunner(gRR);

        Config::setValue(Config::ROADRUNNER_JACOBIAN_MODE,
                         (unsigned) Config::ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS);
        for (int i = 0; i < aSection->KeyCount(); i++) {
            IniKey *aKey = aSection->GetKey(i);
            //clog <<"\n";
            //clog <<"Ref_EigenValue: " << aKey->mKey << ": " << aKey->mValue << endl;

            string eigenValueLabel = "eigenReal(" + aKey->mKey + ")";
            double val = rri->getValue(eigenValueLabel.c_str());

            //clog <<"EigenValue " << i << ": " << val << endl;
            EXPECT_NEAR(aKey->AsFloat(), val, abs(val * 1e-4));
        }
    }

    void check_INDIVIDUAL_AMOUNT_EIGENVALUES(IniSection *aSection) {
        clog << "==== INDIVIDUAL AMOUNT EIGENVALUES ====" << endl;
        aSection->mIsUsed = true;

        Config::setValue(Config::ROADRUNNER_JACOBIAN_MODE, (unsigned) Config::ROADRUNNER_JACOBIAN_MODE_AMOUNTS);
        RoadRunner *rri = castToRoadRunner(gRR);
        for (int i = 0; i < aSection->KeyCount(); i++) {
            IniKey *aKey = aSection->GetKey(i);
            //clog <<"\n";
            //clog <<"Ref_EigenValue: " << aKey->mKey << ": " << aKey->mValue << endl;

            string eigenValueLabel = "eigenReal(" + aKey->mKey + ")";
            double val = rri->getValue(eigenValueLabel.c_str());

            //clog <<"EigenValue " << i << ": " << val << endl;
            EXPECT_NEAR(aKey->AsFloat(), val, 1e-5);
        }
    }

    void check_GET_EIGENVALUE_MATRIX(IniSection *aSection) {
        clog << "==== GET_EIGENVALUE_MATRIX ====" << endl;
        aSection->mIsUsed = true;

        Config::setValue(Config::ROADRUNNER_JACOBIAN_MODE,
                         (unsigned) Config::ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS);
        RoadRunner *rri = castToRoadRunner(gRR);
        ls::DoubleMatrix ref = getDoubleMatrixFromString(aSection->GetNonKeysAsString());
        std::vector<ls::Complex> eigen = rri->getFullEigenValues();
        compareMatrices(ref, eigen);
    }


    void check_GET_EIGENVALUE_AMOUNT_MATRIX(IniSection *aSection) {
        clog << "==== GET_EIGENVALUE AMOUNT MATRIX ====" << endl;
        aSection->mIsUsed = true;

        Config::setValue(Config::ROADRUNNER_JACOBIAN_MODE, (unsigned) Config::ROADRUNNER_JACOBIAN_MODE_AMOUNTS);
        RoadRunner *rri = castToRoadRunner(gRR);
        ls::DoubleMatrix ref = getDoubleMatrixFromString(aSection->GetNonKeysAsString());
        std::vector<ls::Complex> eigen = rri->getFullEigenValues();

        compareMatrices(ref, eigen);
    }


    void check_GET_REDUCED_EIGENVALUE_MATRIX(IniSection *aSection) {
        clog << "==== GET_REDUCED_EIGENVALUE_MATRIX ====" << endl;
        aSection->mIsUsed = true;

        Config::setValue(Config::ROADRUNNER_JACOBIAN_MODE,
                         (unsigned) Config::ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS);
        RoadRunner *rri = castToRoadRunner(gRR);
        ls::DoubleMatrix ref = getDoubleMatrixFromString(aSection->GetNonKeysAsString());
        std::vector<ls::Complex> eigen = rri->getReducedEigenValues();

        compareMatrices(ref, eigen);
    }


    void check_STOICHIOMETRY_MATRIX(IniSection *aSection) {
        clog << "==== STOICHIOMETRY_MATRIX ====" << endl;
        aSection->mIsUsed = true;

        ls::DoubleMatrix ref = getDoubleMatrixFromString(aSection->GetNonKeysAsString());
        RoadRunner *rri = castToRoadRunner(gRR);
        ls::DoubleMatrix matrix = rri->getFullStoichiometryMatrix();
        compareMatrices(ref, matrix);
    }

    void check_REDUCED_STOICHIOMETRY_MATRIX(IniSection *aSection) {
        clog << "==== REDUCED STOICHIOMETRY_MATRIX ====" << endl;
        aSection->mIsUsed = true;

        ls::DoubleMatrix ref = getDoubleMatrixFromString(aSection->GetNonKeysAsString());
        RoadRunner *rri = castToRoadRunner(gRR);
        ls::DoubleMatrix matrix = rri->getReducedStoichiometryMatrix();
        compareMatrices(ref, matrix);
    }

    void check_LINK_MATRIX(IniSection *aSection) {
        clog << "==== LINK_MATRIX ====" << endl;
        aSection->mIsUsed = true;

        ls::DoubleMatrix ref = getDoubleMatrixFromString(aSection->GetNonKeysAsString());

        RRDoubleMatrixPtr matrix = getLinkMatrix(gRR);
        compareMatrices(ref, matrix);
        freeMatrix(matrix);
    }

    void check_UNSCALED_ELASTICITY_MATRIX(IniSection *aSection) {
        clog << "==== UNSCALED_ELASTICITY_MATRIX ====" << endl;
        aSection->mIsUsed = true;

        Config::setValue(Config::ROADRUNNER_JACOBIAN_MODE,
                         (unsigned) Config::ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS);
        RoadRunner *rri = castToRoadRunner(gRR);
        //trySteadyState(gRR);
        ls::DoubleMatrix ref = getDoubleMatrixFromString(aSection->GetNonKeysAsString());
        ls::DoubleMatrix matrix = rri->getUnscaledElasticityMatrix();

        compareMatrices(ref, matrix);
    }

    void check_UNSCALED_ELASTICITY_AMOUNT_MATRIX(IniSection *aSection) {
        clog << "==== UNSCALED_ELASTICITY_AMOUNT_MATRIX ====" << endl;
        aSection->mIsUsed = true;

        Config::setValue(Config::ROADRUNNER_JACOBIAN_MODE, (unsigned) Config::ROADRUNNER_JACOBIAN_MODE_AMOUNTS);
        RoadRunner *rri = castToRoadRunner(gRR);
        //trySteadyState(gRR);
        ls::DoubleMatrix ref = getDoubleMatrixFromString(aSection->GetNonKeysAsString());
        ls::DoubleMatrix matrix = rri->getUnscaledElasticityMatrix();

        compareMatrices(ref, matrix);
    }

    void check_SCALED_ELASTICITY_MATRIX(IniSection *aSection) {
        clog << "==== SCALED_ELASTICITY_MATRIX ====" << endl;
        aSection->mIsUsed = true;

        Config::setValue(Config::ROADRUNNER_JACOBIAN_MODE,
                         (unsigned) Config::ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS);
        RoadRunner *rri = castToRoadRunner(gRR);
        //trySteadyState(gRR);
        ls::DoubleMatrix ref = getDoubleMatrixFromString(aSection->GetNonKeysAsString());
        ls::DoubleMatrix matrix = rri->getScaledElasticityMatrix();

        compareMatrices(ref, matrix);
    }

    void check_SCALED_ELASTICITY_AMOUNT_MATRIX(IniSection *aSection) {
        clog << "==== SCALED_ELASTICITY_AMOUNT_MATRIX ====" << endl;
        aSection->mIsUsed = true;

        Config::setValue(Config::ROADRUNNER_JACOBIAN_MODE, (unsigned) Config::ROADRUNNER_JACOBIAN_MODE_AMOUNTS);
        RoadRunner *rri = castToRoadRunner(gRR);
        //trySteadyState(gRR);
        ls::DoubleMatrix ref = getDoubleMatrixFromString(aSection->GetNonKeysAsString());
        ls::DoubleMatrix matrix = rri->getScaledElasticityMatrix();

        compareMatrices(ref, matrix);
    }

    void check_UNSCALED_CONCENTRATION_CONTROL_MATRIX(IniSection *aSection) {
        clog << "==== UNSCALED_CONCENTRATION_CONTROL_MATRIX ====" << endl;
        aSection->mIsUsed = true;

        Config::setValue(Config::ROADRUNNER_JACOBIAN_MODE,
                         (unsigned) Config::ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS);

        ls::DoubleMatrix ref = getDoubleMatrixFromString(aSection->GetNonKeysAsString());

        reset(gRR);
        RRDoubleMatrixPtr matrix = getUnscaledConcentrationControlCoefficientMatrix(gRR);
        compareMatrices(ref, matrix);
        freeMatrix(matrix);
    }

    void check_SCALED_CONCENTRATION_CONTROL_MATRIX(IniSection *aSection) {
        clog << "==== SCALED_CONCENTRATION_CONTROL_MATRIX ====" << endl;
        aSection->mIsUsed = true;

        Config::setValue(Config::ROADRUNNER_JACOBIAN_MODE,
                         (unsigned) Config::ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS);

        ls::DoubleMatrix ref = getDoubleMatrixFromString(aSection->GetNonKeysAsString());

        reset(gRR);
        RRDoubleMatrixPtr matrix = getScaledConcentrationControlCoefficientMatrix(gRR);
        compareMatrices(ref, matrix);
        freeMatrix(matrix);
    }

    void check_UNSCALED_FLUX_CONTROL_MATRIX(IniSection *aSection) {
        clog << "==== UNSCALED_FLUX_CONTROL_MATRIX ====" << endl;
        aSection->mIsUsed = true;

        ls::DoubleMatrix ref = getDoubleMatrixFromString(aSection->GetNonKeysAsString());

        reset(gRR);
        RRDoubleMatrixPtr matrix = getUnscaledFluxControlCoefficientMatrix(gRR);
        compareMatrices(ref, matrix);
        freeMatrix(matrix);
    }

    void check_SCALED_FLUX_CONTROL_MATRIX(IniSection *aSection) {
        clog << "==== SCALED_FLUX_CONTROL_MATRIX ====" << endl;
        aSection->mIsUsed = true;

        ls::DoubleMatrix ref = getDoubleMatrixFromString(aSection->GetNonKeysAsString());

        reset(gRR);

        RRDoubleMatrixPtr matrix = getScaledFluxControlCoefficientMatrix(gRR);
        compareMatrices(ref, matrix);
        freeMatrix(matrix);
    }

    void check_GET_CONTROL_COEFFICIENT(IniSection *aSection) {
        clog << "==== GET_CONTROL_COEFFICIENT ====" << endl;
        aSection->mIsUsed = true;

        string keys = Trim(aSection->GetNonKeysAsString());
        vector<string> refList = splitString(keys, " ,");

        const char *variable = refList[0].c_str();
        const char *parameter = refList[1].c_str();
        double value = toDouble(refList[2]);

        getCC(gRR, variable, parameter, &value);

        EXPECT_NEAR(toDouble(refList[2]), value, 1e-6);
        //clog <<"\n";
        //clog <<"Ref:\t" << toDouble(refList[2]) << "\tActual:\t " << value << endl;
    }

    void check_CHECK_RESET(IniSection *aSection) {
        clog << "==== CHECK_RESET ====" << endl;
        aSection->mIsUsed = true;

        string keys = Trim(aSection->GetNonKeysAsString());
        vector<string> refList = splitString(keys, " ,");

        const char *f = refList[0].c_str();
        double f_value = toDouble(refList[1]);
        const char *k = refList[2].c_str();
        double k_value = toDouble(refList[3]);
        const char *d = refList[4].c_str();
        double d_value = toDouble(refList[5]);

        double f_value_r;
        double k_value_r;
        double d_value_r;

        setValue(gRR, f, f_value);
        setValue(gRR, k, k_value);
        setValue(gRR, d, d_value);

        reset(gRR);

        getValue(gRR, f, &f_value_r);
        getValue(gRR, k, &k_value_r);
        getValue(gRR, d, &d_value_r);

        EXPECT_TRUE(f_value != f_value_r);
        EXPECT_EQ(k_value, k_value_r);
        EXPECT_EQ(d_value, d_value_r);
    }

    void check_CHECK_RESETALL(IniSection *aSection) {
        clog << "==== CHECK_RESET_ALL ====" << endl;
        aSection->mIsUsed = true;

        string keys = Trim(aSection->GetNonKeysAsString());
        vector<string> refList = splitString(keys, " ,");

        const char *k = refList[0].c_str();
        double k_value = toDouble(refList[1]);
        const char *d = refList[2].c_str();
        double d_value = toDouble(refList[3]);

        double k_value_r;
        double d_value_r;

        setValue(gRR, k, k_value);
        setValue(gRR, d, d_value);

        resetAll(gRR);

        getValue(gRR, k, &k_value_r);
        getValue(gRR, d, &d_value_r);

        EXPECT_NE(k_value, k_value_r);
        EXPECT_NE(d_value, d_value_r);

        setValue(gRR, ("init(" + refList[0] + ")").c_str(), k_value);
        setValue(gRR, ("init(" + refList[2] + ")").c_str(), d_value);

        resetAll(gRR);

        getValue(gRR, k, &k_value_r);
        getValue(gRR, d, &d_value_r);

        EXPECT_EQ(k_value, k_value_r);
        EXPECT_EQ(d_value, d_value_r);
    }

    void check_CHECK_RESETTOORIGIN(IniSection *aSection) {
        clog << "==== CHECK_RESET_TO_ORIGIN ====" << endl;
        aSection->mIsUsed = true;

        string keys = Trim(aSection->GetNonKeysAsString());
        vector<string> refList = splitString(keys, " ,");

        const char *d = refList[0].c_str();
        string initstr = "init(" + refList[0] + ")";
        const char* d_init = initstr.c_str();
        double d_value = toDouble(refList[1]);

        double d_value_r;

        setValue(gRR, d, d_value);
        setValue(gRR, d_init, d_value);

        resetToOrigin(gRR);

        getValue(gRR, d, &d_value_r);

        EXPECT_NE(d_value, d_value_r);
    }

    void check_CHECK_RK4_OUTPUT(IniSection *aSection) {
        clog << "==== CHECK_RK4_OUTPUT ====" << endl;
        clog << "Skipping this test: setting rk4 and rk45 integrators is currently broken in this testing environment." << endl;
        return;
        aSection->mIsUsed = true;

        RoadRunner *rri = castToRoadRunner(gRR);
        SimulateOptions opt;
        opt.start = 0;
        opt.duration = 10;

        // cvode
        //clog <<endl << "  simulate with " << opt.start << ", " << opt.duration << ", " << opt.steps << "\n";
        const DoubleMatrix *cvode = rri->simulate(&opt);

        // rk4
        rri->setIntegrator("rk4");
        //clog <<endl << "  simulate with " << opt.start << ", " << opt.duration << ", " << opt.steps << "\n";
        const DoubleMatrix *rk4 = rri->simulate(&opt);

        for (int k = 0; k < cvode->CSize(); k++) {
            EXPECT_NEAR((*cvode)(cvode->RSize() - 1, k), (*rk4)(rk4->RSize() - 1, k), 1e-6);
        }
    }

    void check_CHECK_RK45_OUTPUT(IniSection *aSection) {
        clog << "==== CHECK_RK45_OUTPUT ====" << endl;
        clog << "Skipping this test: setting rk4 and rk45 integrators is currently broken in this testing environment." << endl;
        return;
        aSection->mIsUsed = true;

        RoadRunner *rri = castToRoadRunner(gRR);
        SimulateOptions opt;
        opt.start = 0;
        opt.duration = 10;

        // cvode
        //clog <<endl << "  simulate with " << opt.start << ", " << opt.duration << ", " << opt.steps << "\n";
        const DoubleMatrix *cvode = rri->simulate(&opt);

        // rk4
        rri->setIntegrator("rk45");
        //clog <<endl << "  simulate with " << opt.start << ", " << opt.duration << ", " << opt.steps << "\n";
        const DoubleMatrix *rk45 = rri->simulate(&opt);

        for (int k = 0; k < cvode->CSize(); k++) {
            EXPECT_NEAR((*cvode)(cvode->RSize() - 1, k), (*rk45)(rk45->RSize() - 1, k), 1e-6);
        }
    }

// Placeholder for testing setValues which are not implemented in C++ yet
    void check_CHECK_SETVALUES(IniSection *aSection) {
        clog << "==== CHECK_SETVALUES ====" << endl;
        aSection->mIsUsed = true;

        EXPECT_TRUE(true);
    }

    void check_FLOATING_SPECIES_IDS(IniSection *aSection) {
        clog << "==== FLOATING_SPECIES_IDS ====" << endl;
        aSection->mIsUsed = true;

        string keys = Trim(aSection->GetNonKeysAsString());

        RRStringArrayPtr list = getFloatingSpeciesIds(gRR);

        vector<string> selList = splitString(keys, " ,");

        if (!list && selList.size()) {
            EXPECT_TRUE(false);
            return;
        }
        ASSERT_TRUE((selList.size() == 0 && !list) || (selList.size() == list->Count));
        if (list) {
            for (int i = 0; i < selList.size(); i++) {
                EXPECT_EQ(selList[i], list->String[i]);;
            }
            freeStringArray(list);
        }
    }

    void check_BOUNDARY_SPECIES_IDS(IniSection *aSection) {
        clog << "==== BOUNDARY_SPECIES_IDS ====" << endl;
        aSection->mIsUsed = true;

        string keys = Trim(aSection->GetNonKeysAsString());

        RRStringArrayPtr list = getBoundarySpeciesIds(gRR);

        vector<string> selList = splitString(keys, " ,");

        if (!list && selList.size()) {
            EXPECT_TRUE(false);
            return;
        }
        EXPECT_TRUE((selList.size() == 0 && !list) || (selList.size() == list->Count));
        if (list) {
            for (int i = 0; i < selList.size(); i++) {
                EXPECT_EQ(selList[i], list->String[i]);;
            }
            freeStringArray(list);
        }
    }

    void check_GLOBAL_PARAMETER_IDS(IniSection *aSection) {
        clog << "==== GLOBAL_PARAMETER_IDS ====" << endl;
        aSection->mIsUsed = true;

        string keys = Trim(aSection->GetNonKeysAsString());

        RRStringArrayPtr list = getGlobalParameterIds(gRR);

        vector<string> selList = splitString(keys, " ,");

        if (!list && selList.size()) {
            EXPECT_TRUE(false);
            return;
        }
        EXPECT_TRUE((selList.size() == 0 && !list) || (selList.size() == list->Count));
        if (list) {
            for (int i = 0; i < selList.size(); i++) {
                EXPECT_EQ(selList[i], list->String[i]);;
            }
            freeStringArray(list);
        }
    }

    void check_COMPARTMENT_IDS(IniSection *aSection) {
        clog << "==== COMPARTMENT_IDS ====" << endl;
        aSection->mIsUsed = true;

        string keys = Trim(aSection->GetNonKeysAsString());

        RRStringArrayPtr list = getCompartmentIds(gRR);

        if (!list) {
            EXPECT_TRUE(false);
            return;
        }
        vector<string> selList = splitString(keys, " ,");
        EXPECT_TRUE(selList.size() == list->Count);
        for (int i = 0; i < selList.size(); i++) {
            EXPECT_EQ(selList[i], list->String[i]);;
        }
        freeStringArray(list);
    }

    void check_REACTION_IDS(IniSection *aSection) {
        clog << "==== REACTION_IDS ====" << endl;
        aSection->mIsUsed = true;

        string keys = Trim(aSection->GetNonKeysAsString());

        RRStringArrayPtr list = getReactionIds(gRR);

        vector<string> selList = splitString(keys, " ,");

        ASSERT_TRUE((selList.size() == 0 && !list) || (selList.size() == list->Count));
        if (list) {
            for (int i = 0; i < selList.size(); i++) {
                EXPECT_EQ(selList[i], list->String[i]);
            }
            freeStringArray(list);
        }
    }

    void check_SPECIES_INITIAL_CONDITION_IDS(IniSection *aSection) {
        clog << "==== SPECIES_INITIAL_CONDITION_IDS ====" << endl;
        aSection->mIsUsed = true;

        string keys = Trim(aSection->GetNonKeysAsString());

        RRStringArrayPtr list = getFloatingSpeciesInitialConditionIds(gRR);

        ASSERT_TRUE(list != NULL);
        vector<string> selList = splitString(keys, " ,");
        ASSERT_TRUE(selList.size() == list->Count);
        for (int i = 0; i < selList.size(); i++) {
            EXPECT_EQ(selList[i], list->String[i]);
        }
        freeStringArray(list);
    }

    void check_GET_EIGENVALUE_IDS(IniSection *aSection) {
        clog << "==== GET_EIGENVALUE_IDS ====" << endl;
        aSection->mIsUsed = true;

        string keys = Trim(aSection->GetNonKeysAsString());

        RRStringArrayPtr list = getEigenvalueIds(gRR);

        ASSERT_TRUE(list);
        vector<string> selList = splitString(keys, " ,");
        ASSERT_TRUE(selList.size() == list->Count);
        for (int i = 0; i < selList.size(); i++) {
            EXPECT_EQ(selList[i], list->String[i]);;
        }
        freeStringArray(list);
    }

    void check_GET_RATES_OF_CHANGE_IDS(IniSection *aSection) {
        clog << "==== GET_RATES_OF_CHANGE_IDS ====" << endl;
        aSection->mIsUsed = true;

        string keys = Trim(aSection->GetNonKeysAsString());

        RRStringArrayPtr list = getRatesOfChangeIds(gRR);

        ASSERT_TRUE(list);
        vector<string> selList = splitString(keys, " ,");
        EXPECT_TRUE(selList.size() == list->Count);
        for (int i = 0; i < selList.size(); i++) {
            EXPECT_EQ(selList[i], list->String[i]);;
        }
        freeStringArray(list);
    }

    void check_SET_STEADY_STATE_SELECTION_LIST_2(IniSection *aSection) {
        clog << "==== SET_STEADY_STATE_SELECTION_LIST_2 ====" << endl;
        aSection->mIsUsed = true;

        string keys = Trim(aSection->GetNonKeysAsString());

        bool res = setSteadyStateSelectionList(gRR, keys.c_str());
        EXPECT_TRUE(res);

        //Actually calculate the steady state:
        trySteadyState(gRR);
    }

    void check_GET_STEADY_STATE_SELECTION_LIST_2(IniSection *aSection) {
        clog << "==== GET_STEADY_STATE_SELECTION_LIST_2 ====" << endl;
        aSection->mIsUsed = true;

        string keys = Trim(aSection->GetNonKeysAsString());

        RRStringArrayPtr list = getSteadyStateSelectionList(gRR);

        ASSERT_TRUE(list);
        vector<string> selList = splitString(keys, " ,");
        EXPECT_TRUE(selList.size() == list->Count);
        for (int i = 0; i < selList.size(); i++) {
            EXPECT_EQ(selList[i], list->String[i]);;
        }
        freeStringArray(list);
    }

    void check_SET_TIME_COURSE_SELECTION_LIST(IniSection *aSection) {
        clog << "==== SET_TIME_COURSE_SELECTION_LIST ====" << endl;
        aSection->mIsUsed = true;

        string keys = Trim(aSection->GetNonKeysAsString());

        bool res = setTimeCourseSelectionList(gRR, keys.c_str());
        EXPECT_TRUE(res);
    }

    void check_GET_TIME_COURSE_SELECTION_LIST(IniSection *aSection) {
        clog << "==== GET_TIME_COURSE_SELECTION_LIST ====" << endl;
        aSection->mIsUsed = true;

        string keys = Trim(aSection->GetNonKeysAsString());

        RRStringArrayPtr list = getTimeCourseSelectionList(gRR);

        ASSERT_TRUE(list);
        vector<string> selList = splitString(keys, " ,");
        EXPECT_TRUE(selList.size() == list->Count);
        for (int i = 0; i < selList.size(); i++) {
            EXPECT_EQ(selList[i], list->String[i]);;
        }
        freeStringArray(list);
    }

    void check_COMPUTE_STEADY_STATE_VALUES(IniSection *aSection) {
        clog << "==== COMPUTE_STEADY_STATE_VALUES ====" << endl;
        aSection->mIsUsed = true;

        RRVector *values = computeSteadyStateValues(gRR);

        string valstr = Trim(aSection->GetNonKeysAsString());
        vector<string> vals = splitString(valstr, " ");
        for (int i = 0; i < vals.size(); i++) {
            //Check concentrations
            EXPECT_NEAR(toDouble(vals[i]), values->Data[i], 1e-6);
            //clog <<"\n";
            //clog <<"Ref:\t" << vals[i] << "\tActual:\t " << values->Data[i] << endl;
        }
        freeVector(values);
    }

    void check_FLOATING_SPECIES_CONCENTRATIONS(IniSection *aSection) {
        clog << "==== FLOATING_SPECIES_CONCENTRATIONS ====" << endl;
        aSection->mIsUsed = true;

        RRVector *values = getFloatingSpeciesConcentrations(gRR);

        string valstr = Trim(aSection->GetNonKeysAsString());
        vector<string> vals = splitString(valstr, " ");
        for (int i = 0; i < vals.size(); i++) {
            //Check concentrations
            EXPECT_NEAR(toDouble(vals[i]), values->Data[i], 1e-6);
            //clog <<"\n";
            //clog <<"Ref:\t" << vals[i] << "\tActual:\t " << values->Data[i] << endl;
        }
        freeVector(values);
    }

    void check_BOUNDARY_SPECIES_CONCENTRATIONS(IniSection *aSection) {
        clog << "==== BOUNDARY_SPECIES_CONCENTRATIONS ====" << endl;
        aSection->mIsUsed = true;

        RRVector *values = getBoundarySpeciesConcentrations(gRR);

        string valstr = Trim(aSection->GetNonKeysAsString());
        vector<string> vals = splitString(valstr, " ");
        for (int i = 0; i < vals.size(); i++) {
            //Check concentrations
            EXPECT_NEAR(toDouble(vals[i]), values->Data[i], 1e-6);
            //clog <<"\n";
            //clog <<"Ref:\t" << vals[i] << "\tActual:\t " << values->Data[i] << endl;
        }
        freeVector(values);
    }

    void check_GET_GLOBAL_PARAMETER_VALUES(IniSection *aSection) {
        clog << "==== GET_GLOBAL_PARAMETER_VALUES ====" << endl;
        aSection->mIsUsed = true;

        string keys = Trim(aSection->GetNonKeysAsString());

        RRVector *values = getGlobalParameterValues(gRR);

        vector<string> refList = splitString(keys, " ,");

        if (!values || values->Count != refList.size()) {
            EXPECT_TRUE(false);
            freeVector(values);
            return;
        }

        for (int i = 0; i < refList.size(); i++) {

            //Check concentrations
            EXPECT_NEAR(toDouble(refList[i]), values->Data[i], 1e-6);
            //clog <<"\n";
            //clog <<"Ref:\t" << toDouble(refList[i]) << "\tActual:\t " << values->Data[i] << endl;
        }
        freeVector(values);
    }

    void check_GET_REACTION_RATES(IniSection *aSection) {
        clog << "==== GET_REACTION_RATES ====" << endl;
        aSection->mIsUsed = true;

        string keys = Trim(aSection->GetNonKeysAsString());

        RRVector *values = getReactionRates(gRR);

        vector<string> refList = splitString(keys, " ,");

        if (!values || values->Count != refList.size()) {
            EXPECT_TRUE(false && " ");
            freeVector(values);
            return;
        }

        for (int i = 0; i < refList.size(); i++) {

            //Check concentrations
            EXPECT_NEAR(toDouble(refList[i]), values->Data[i], max(1e-9, abs(values->Data[i] * 1e-4)));
            //clog <<"\n";
            //clog <<"Ref:\t" << toDouble(refList[i]) << "\tActual:\t " << values->Data[i] << endl;
        }
        freeVector(values);
    }

    void check_GET_REACTION_RATE_BY_INDEX(IniSection *aSection) {
        clog << "==== GET_REACTION_RATE_BY_INDEX ====" << endl;
        aSection->mIsUsed = true;

        string keys = Trim(aSection->GetNonKeysAsString());

        vector<string> refList = splitString(keys, " ,");

        if (refList.size() != getNumberOfReactions(gRR)) {
            EXPECT_TRUE(false);
            return;
        }

        for (int i = 0; i < refList.size(); i++) {
            double value;
            if (!getReactionRate(gRR, i, &value)) {
                EXPECT_TRUE(false);
            }

            //Check concentrations
            EXPECT_NEAR(toDouble(refList[i]), value, 1e-6);
            //clog <<"\n";
            //clog <<"Ref:\t" << toDouble(refList[i]) << "\tActual:\t " << value << endl;
        }
    }

    void check_NUMBER_OF_DEPENDENT_SPECIES(IniSection *aSection) {
        clog << "==== NUMBER_OF_DEPENDENT_SPECIES ====" << endl;
        aSection->mIsUsed = true;

        string key = Trim(aSection->GetNonKeysAsString());
        int val = getNumberOfDependentSpecies(gRR);

        EXPECT_TRUE(toInt(key) == val);
    }

    void check_NUMBER_OF_INDEPENDENT_SPECIES(IniSection *aSection) {
        clog << "==== NUMBER_OF_INDEPENDENT_SPECIES ====" << endl;
        aSection->mIsUsed = true;

        string key = Trim(aSection->GetNonKeysAsString());
        int val = getNumberOfIndependentSpecies(gRR);

        EXPECT_TRUE(toInt(key) == val);
    }

    void check_NUMBER_OF_RATE_RULES(IniSection *aSection) {
        clog << "==== NUMBER_OF_RATE_RULES ====" << endl;
        aSection->mIsUsed = true;

        string key = Trim(aSection->GetNonKeysAsString());
        int val = getNumberOfRules(gRR);

        EXPECT_TRUE(toInt(key) == val);
    }

    void check_GET_RATES_OF_CHANGE(IniSection *aSection) {
        clog << "==== GET_RATES_OF_CHANGE ====" << endl;
        aSection->mIsUsed = true;

        string keys = Trim(aSection->GetNonKeysAsString());

        RRVector *values = getRatesOfChange(gRR);

        vector<string> refList = splitString(keys, " ,");

        if (!values || values->Count != refList.size()) {
            EXPECT_TRUE(false);
            freeVector(values);
            return;
        }

        for (int i = 0; i < refList.size(); i++) {
            //Check concentrations
            EXPECT_NEAR(toDouble(refList[i]), values->Data[i], 1e-6);
            //clog <<"\n";
            //clog <<"Ref:\t" << toDouble(refList[i]) << "\tActual:\t " << values->Data[i] << endl;
        }
        freeVector(values);
    }

    void check_GET_REACTION_RATES_EX(IniSection *aSection) {
        clog << "==== GET_REACTION_RATES_EX ====" << endl;
        aSection->mIsUsed = true;

        IniKey *conc = aSection->GetKey("concentrations");
        if (!conc) {
            EXPECT_TRUE(false);
            return;
        }

        IniKey *refs = aSection->GetKey("rates");
        if (!refs) {
            EXPECT_TRUE(false);
            return;
        }

        vector<string> refList = splitString(refs->mValue, " ");

        vector<string> concList = splitString(conc->mValue, " ");
        RRVector *aVector = createVector(static_cast<int>(concList.size()));

        for (int i = 0; i < concList.size(); i++) {
            aVector->Data[i] = toDouble(concList[i]);
        }

        RRVector *values = getReactionRatesEx(gRR, aVector);


        if (!values || values->Count != refList.size()) {
            EXPECT_TRUE(false);
            freeVector(values);
            freeVector(aVector);
            return;
        }

        for (int i = 0; i < refList.size(); i++) {
            //Check concentrations
            EXPECT_NEAR(toDouble(refList[i]), values->Data[i], 1e-6);
            //clog <<"\n";
            //clog <<"Ref:\t" << toDouble(refList[i]) << "\tActual:\t " << values->Data[i] << endl;
        }
        freeVector(values);
        freeVector(aVector);
    }

    void check_GET_RATES_OF_CHANGE_EX(IniSection *aSection) {
        clog << "==== GET_RATES_OF_CHANGE_EX ====" << endl;
        aSection->mIsUsed = true;

        IniKey *conc = aSection->GetKey("concentrations");
        if (!conc) {
            EXPECT_TRUE(false);
            return;
        }

        IniKey *refs = aSection->GetKey("rates");
        if (!refs) {
            EXPECT_TRUE(false);
            return;
        }

        vector<string> refList = splitString(refs->mValue, " ");

        vector<string> concList = splitString(conc->mValue, " ");
        RRVector *aVector = createVector(static_cast<int>(concList.size()));

        for (int i = 0; i < concList.size(); i++) {
            aVector->Data[i] = toDouble(concList[i]);
        }

        RRVector *values = getRatesOfChangeEx(gRR, aVector);


        if (!values || values->Count != refList.size()) {
            EXPECT_TRUE(false);
            freeVector(values);
            freeVector(aVector);
            return;
        }

        for (int i = 0; i < refList.size(); i++) {
            //Check concentrations
            EXPECT_NEAR(toDouble(refList[i]), values->Data[i], 1e-6);
            //clog <<"\n";
            //clog <<"Ref:\t" << toDouble(refList[i]) << "\tActual:\t " << values->Data[i] << endl;
        }
        freeVector(values);
        freeVector(aVector);
    }

    void check_GET_RATES_OF_CHANGE_BY_INDEX(IniSection *aSection) {
        clog << "==== GET_RATES_OF_CHANGE_BY_INDEX ====" << endl;
        aSection->mIsUsed = true;

        string keys = Trim(aSection->GetNonKeysAsString());

        vector<string> refList = splitString(keys, " ,");

        if (refList.size() != getNumberOfFloatingSpecies(gRR)) {
            EXPECT_TRUE(false);
            return;
        }

        for (int i = 0; i < refList.size(); i++) {
            double value;
            if (!getRateOfChange(gRR, i, &value)) {
                EXPECT_TRUE(false);
            }

            //Check concentrations
            EXPECT_NEAR(toDouble(refList[i]), value, 1e-6);
            //clog <<"\n";
            //clog <<"Ref:\t" << toDouble(refList[i]) << "\tActual:\t " << value << endl;
        }
    }

    void check_AMOUNT_CONCENTRATION_JACOBIANS(IniSection *aSection) {
        clog << "==== AMOUNT_CONCENTRATION_JACOBIANS ====" << endl;
        aSection->mIsUsed = true;
        rr::Setting saved = Config::getValue(Config::ROADRUNNER_JACOBIAN_MODE);
        Config::setValue(Config::ROADRUNNER_JACOBIAN_MODE, (unsigned) Config::ROADRUNNER_JACOBIAN_MODE_AMOUNTS);
        compareJacobians(gRR);
        Config::setValue(Config::ROADRUNNER_JACOBIAN_MODE,
                         (unsigned) Config::ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS);
        compareJacobians(gRR);
        Config::setValue(Config::ROADRUNNER_JACOBIAN_MODE, saved);
    }

    void check_CHECK_DEFAULT_TIME_STEP(IniSection *aSection) {
        clog << "==== CHECK_DEFAULT_TIME_STEP ====" << endl;
        aSection->mIsUsed = true;

        string keys = Trim(aSection->GetNonKeysAsString());
        vector<string> refList = splitString(keys, " ,");

        int NumPts = toInt(refList[0]);

        RoadRunner *rri = castToRoadRunner(gRR);

        SimulateOptions opt;
        const DoubleMatrix *result = rri->simulate(&opt);

        if (result->RSize() != NumPts) {
            EXPECT_TRUE(false);
            //clog <<"Default time step does not match" << endl;
        }
    }

    void check_CHECK_SIMULATE_POINTS_VS_STEPS(IniSection *aSection) {
        clog << "==== CHECK SIMULATE POINTS VS STEPS ====" << endl;
        // Python-only test
        aSection->mIsUsed = true;
    }

    void check_CHECK_MONOTONIC_TIMEPOINTS(IniSection *aSection) {
        clog << "==== CHECK MONOTONIC TIMEPOINTS ====" << endl;

        aSection->mIsUsed = true;
        string keys = Trim(aSection->GetNonKeysAsString());
        vector<string> refList = splitString(keys, " ,");

        RoadRunner *rri = castToRoadRunner(gRR);
        SimulateOptions opt;
        opt.start = toDouble(refList.at(0));
        opt.duration = toDouble(refList.at(1));

        // For variable step
        rri->getIntegrator()->setValue("variable_step_size", true);
        const DoubleMatrix *result = rri->simulate(&opt);

        for (int k = 1; k < result->RSize(); ++k) {
            EXPECT_TRUE(((*result)(k - 1, 0) < (*result)(k, 0)));
        }

        // For fixed step
        opt.steps = toInt(refList.at(2));
        rri->getIntegrator()->setValue("variable_step_size", false);
        //clog <<endl << "  simulate with " << opt.start << ", " << opt.duration << ", " << opt.steps << "\n";
        result = rri->simulate(&opt);

        for (int k = 1; k < result->RSize(); ++k) {
            EXPECT_TRUE(((*result)(k - 1, 0) < (*result)(k, 0)));
        }
    }

    void check_CHECK_EVENT_PRE_AND_POSTFIRE_TIMEPOINTS(IniSection *aSection) {
        clog << "==== CHECK EVENT PRE AND POSTFIRE TIMEPOINTS ====" << endl;

        aSection->mIsUsed = true;
        string keys = Trim(aSection->GetNonKeysAsString());
        vector<string> refList = splitString(keys, " ,");

        RoadRunner *rri = castToRoadRunner(gRR);
        SimulateOptions opt;
        opt.start = 0;
        opt.duration = 10;

        // For variable step
        rri->getIntegrator()->setValue("variable_step_size", true);
        const DoubleMatrix *result = rri->simulate(&opt);

        double max_tol = 0.0001;

        for (vector<string>::const_iterator z = refList.begin(); z != refList.end(); ++z) {
            double t = toDouble(*z);
            int i = findCloseTimepoint(result, t, max_tol);
            if (i < 0 || i + 1 >= result->RSize()) {
                EXPECT_TRUE(false);
            } else {
                EXPECT_TRUE(abs((*result)(i + 1, 0) - (*result)(i, 0)) < max_tol);
            }
        }
    }

    void check_CHECK_RESETCONSERVEDTOTAL(IniSection *aSection) {
        clog << "==== CHECK_RESET_CONSERVED_TOTAL ====" << endl;
        aSection->mIsUsed = true;

        string keys = Trim(aSection->GetNonKeysAsString());
        vector<string> refList = splitString(keys, " ,");

        const char *ct = refList[0].c_str();

        double ct_val;

        RRVector *values = computeSteadyStateValues(gRR);

        setValue(gRR, ct, 1000.);

        reset(gRR);

        getValue(gRR, ct, &ct_val);

        EXPECT_NE(ct_val, 1000.);

        resetAll(gRR);

        freeVector(values);
    }

    void check_CHECK_UNUSED_TESTS() {
        for (int i = 0; i < iniFile.GetNumberOfSections(); i++) {
            if (!iniFile.GetSection(i)->mIsUsed && iniFile.GetSection(i)->mName != "Conservation Laws") {
            }
        }
    }

    void check_FREE_RR_INSTANCE() {
        EXPECT_TRUE(freeRRInstance(gRR));
        gRR = NULL;
    }

    bool check_Unimplemented() {
        if (iniFile.GetSection("Add Parameter") != NULL) {
            clog <<"Testing editing models is currently not supported outside of Python." << endl;
            return true;
        }

        if (iniFile.GetSection("Add Rule") != NULL) {
            clog <<"Testing editing models is currently not supported outside of Python." << endl;
            return true;
        }

        if (iniFile.GetSection("Add Species") != NULL) {
            clog <<"Testing editing models is currently not supported outside of Python." << endl;
            return true;
        }

        if (iniFile.GetSection("Add Reaction") != NULL) {
            clog <<"Testing editing models is currently not supported outside of Python." << endl;
            return true;
        }

        if (iniFile.GetSection("Add Event Assignment") != NULL) {
            clog <<"Testing editing models is currently not supported outside of Python." << endl;
            return true;
        }

        if (iniFile.GetSection("Remove Compartment") != NULL) {
            clog <<"Testing editing models is currently not supported outside of Python." << endl;
            return true;
        }

        if (iniFile.GetSection("Remove Parameter") != NULL) {
            clog <<"Testing editing models is currently not supported outside of Python." << endl;
            return true;
        }

        if (iniFile.GetSection("Remove Rule") != NULL) {
            clog <<"Testing editing models is currently not supported outside of Python." << endl;
            return true;
        }

        if (iniFile.GetSection("Remove Species") != NULL) {
            clog <<"Testing editing models is currently not supported outside of Python." << endl;
            return true;
        }

        if (iniFile.GetSection("Set Kinetic Law") != NULL) {
            clog <<"Testing editing models is currently not supported outside of Python." << endl;
            return true;
        }

        if (iniFile.GetSection("Set Vector Amount Absolute Tolerance") != NULL) {
            clog <<"Testing setting vector tolerances is currently not supported outside of Python." << endl;
            return true;
        }

        if (iniFile.GetSection("Set Scalar Concentration Absolute Tolerance") != NULL) {
            clog <<"Testing setting vector tolerances is currently not supported outside of Python." << endl;
            return true;
        }

        if (iniFile.GetSection("Set Vector Concentration Absolute Tolerance") != NULL) {
            clog <<"Testing setting vector tolerances is currently not supported outside of Python." << endl;
            return true;
        }

        if (iniFile.GetSection("Gillespie Seed") != NULL) {
            clog <<"Testing setting the Gillespie seed is currently not supported outside of Python." << endl;
            return true;
        }

        if (iniFile.GetSection("Gillespie Value") != NULL) {
            clog <<"Testing setting the Gillespie value is currently not supported outside of Python." << endl;
            return true;
        }

        if (iniFile.GetSection("Gillespie Fixed Time Interval") != NULL) {
            clog <<"Testing setting the Gillespie fixed time interval is currently not supported outside of Python." << endl;
            return true;
        }

        if (iniFile.GetSection("Gillespie Fixed Time Interval") != NULL) {
            clog << "Testing setting the Gillespie fixed time interval is currently not supported outside of Python." << endl;
            return true;
        }

        //Since this won't break any other tests, it's fine to run the rest of the rrtest file, even if we don't implement this section yet.
        //if (iniFile.GetSection("Get Variable End Time") != NULL) {
        //    clog <<"Testing setting vector tolerances is currently not supported outside of Python." << endl;
        //    return true;
        //}
        return false;
    }

};


TEST_F(CApiRRTests, TestAbsolute_Tolerance1) {
    checkRRTest("Absolute_Tolerance1");
}

TEST_F(CApiRRTests, TestAdd_Event_Assignment1) {
    checkRRTest("Add_Event_Assignment1");
}

TEST_F(CApiRRTests, TestAdd_Reactions1) {
    checkRRTest("Add_Reactions1");
}

TEST_F(CApiRRTests, TestAdd_Reactions2) {
    checkRRTest("Add_Reactions2");
}

TEST_F(CApiRRTests, TestAdd_Species1) {
    checkRRTest("Add_Species1");
}

TEST_F(CApiRRTests, TestBimolecular_end) {
    checkRRTest("Bimolecular_end");
}

TEST_F(CApiRRTests, TestComp) {
    checkRRTest("Comp");
}

TEST_F(CApiRRTests, TestConserved_Cycle) {
    checkRRTest("Conserved_Cycle");
}

TEST_F(CApiRRTests, TestConserved_Cycle_and_Branch) {
    checkRRTest("Conserved_Cycle_and_Branch");
}

TEST_F(CApiRRTests, TestCycle) {
    checkRRTest("Cycle");
}

TEST_F(CApiRRTests, TestCycle_across_branches) {
    checkRRTest("Cycle_across_branches");
}

TEST_F(CApiRRTests, TestCycle_to_Input_and_Branch) {
    checkRRTest("Cycle_to_Input_and_Branch");
}

TEST_F(CApiRRTests, TestEvent_timings_276) {
    checkRRTest("Event_timings_276");
}

TEST_F(CApiRRTests, TestFour_Steps) {
    checkRRTest("Four_Steps");
}

TEST_F(CApiRRTests, TestFutile_Cycle) {
    checkRRTest("Futile_Cycle");
}

TEST_F(CApiRRTests, TestGillespie) {
    checkRRTest("Gillespie");
}

TEST_F(CApiRRTests, TestMultiBranch2) {
    checkRRTest("MultiBranch2");
}

TEST_F(CApiRRTests, TestMultibranch1) {
    checkRRTest("Multibranch1");
}

TEST_F(CApiRRTests, TestRatesOfChange) {
    checkRRTest("RatesOfChange");
}

TEST_F(CApiRRTests, TestRemove_Compartment1) {
    checkRRTest("Remove_Compartment1");
}

TEST_F(CApiRRTests, TestRemove_Parameter1) {
    checkRRTest("Remove_Parameter1");
}

TEST_F(CApiRRTests, TestRemove_Parameter2) {
    checkRRTest("Remove_Parameter2");
}

TEST_F(CApiRRTests, TestRemove_Rules1) {
    checkRRTest("Remove_Rules1");
}

TEST_F(CApiRRTests, TestRemove_Species1) {
    checkRRTest("Remove_Species1");
}

TEST_F(CApiRRTests, TestSet_Kinetic_Law1) {
    checkRRTest("Set_Kinetic_Law1");
}

TEST_F(CApiRRTests, TestSimpleUniUni) {
    checkRRTest("SimpleUniUni");
}

TEST_F(CApiRRTests, TestSimple_Branch) {
    checkRRTest("Simple_Branch");
}

TEST_F(CApiRRTests, TestTest_1) {
    checkRRTest("Test_1");
}

TEST_F(CApiRRTests, TestTest_2) {
    checkRRTest("Test_2");
}

TEST_F(CApiRRTests, TestTest_3) {
    checkRRTest("Test_3");
}

TEST_F(CApiRRTests, TestThree_Steps) {
    checkRRTest("Three_Steps");
}

TEST_F(CApiRRTests, TestTwo_Cycles) {
    checkRRTest("Two_Cycles");
}

TEST_F(CApiRRTests, TestTwo_Steps) {
    checkRRTest("Two_Steps");
}

TEST_F(CApiRRTests, Testjacobian_1) {
    checkRRTest("jacobian_1");
}

TEST_F(CApiRRTests, Testoscli) {
    checkRRTest("oscli");
}

TEST_F(CApiRRTests, Testreversible_Jacobian) {
    checkRRTest("reversible_Jacobian");
}

TEST_F(CApiRRTests, Testvariable_time_step) {
    checkRRTest("variable_time_step");
}

