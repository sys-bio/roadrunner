#include "rrLogger.h"
#include "rrRoadRunner.h"
#include "rrException.h"
#include "rrStringUtils.h"
#include "gtest/gtest.h"
#include "LLVMExecutableModel.h"
#include "rrConfig.h"
#include <filesystem>
#include "RoadRunnerTest.h"

using std::filesystem::path;

using namespace rr;

class ModelAnalysisTests : public RoadRunnerTest {
public:
    path modelAnalysisModelsDir ;

    ModelAnalysisTests() {
        modelAnalysisModelsDir =  rrTestModelsDir_ / "ModelAnalysis";
    }
};


TEST_F(ModelAnalysisTests, SimulateWithSameTimes) {
    RoadRunner rr((modelAnalysisModelsDir / "BIOMD0000000035_url.xml").string());
    std::vector<double> times;
    times.push_back(0);
    times.push_back(1);
    times.push_back(1);
    times.push_back(10);
    SimulateOptions opt;
    opt.times = times;
    const ls::DoubleMatrix* result = rr.simulate(&opt);
    EXPECT_EQ(result->numRows(), 4);
    EXPECT_EQ(result->Element(0, 0), 0);
    EXPECT_EQ(result->Element(1, 0), 1);
    EXPECT_EQ(result->Element(2, 0), 1);
    EXPECT_EQ(result->Element(3, 0), 10);
    for (unsigned int col = 0; col < result->numCols(); col++) {
        EXPECT_EQ(result->Element(1, col), result->Element(2, col));
    }
}

//TEST_F(ModelAnalysisTests, SimulateWithSameStartTimes) {
//    RoadRunner rr((modelAnalysisModelsDir / "BIOMD0000000035_url.xml").string());
//    std::vector<double> times;
//    times.push_back(0);
//    times.push_back(0);
//    times.push_back(1);
//    times.push_back(10);
//    SimulateOptions opt;
//    opt.times = times;
//    const ls::DoubleMatrix* result = rr.simulate(&opt);
//    EXPECT_EQ(result->numRows(), 4);
//    EXPECT_EQ(result->Element(0, 0), 0);
//    EXPECT_EQ(result->Element(1, 0), 0);
//    EXPECT_EQ(result->Element(2, 0), 1);
//    EXPECT_EQ(result->Element(3, 0), 10);
//    for (unsigned int col = 0; col < result->numCols(); col++) {
//        EXPECT_EQ(result->Element(0, col), result->Element(1, col));
//    }
//}

TEST_F(ModelAnalysisTests, SimulateWithSameEndTimes) {
    RoadRunner rr((modelAnalysisModelsDir / "BIOMD0000000035_url.xml").string());
    std::vector<double> times;
    times.push_back(0);
    times.push_back(1);
    times.push_back(10);
    times.push_back(10);
    SimulateOptions opt;
    opt.times = times;
    const ls::DoubleMatrix* result = rr.simulate(&opt);
    EXPECT_EQ(result->numRows(), 4);
    EXPECT_EQ(result->Element(0, 0), 0);
    EXPECT_EQ(result->Element(1, 0), 1);
    EXPECT_EQ(result->Element(2, 0), 10);
    EXPECT_EQ(result->Element(3, 0), 10);
    for (unsigned int col = 0; col < result->numCols(); col++) {
        EXPECT_EQ(result->Element(2, col), result->Element(3, col));
    }
}

TEST_F(ModelAnalysisTests, issue1031) {
    //Config::setValue(Config::LLVM_BACKEND, Config::LLVM_BACKEND_VALUES::LLJIT);
    rr::RoadRunner rr((modelAnalysisModelsDir / "zero_rate_at_steady_state.xml").string());

    rr.steadyState();
    ls::DoubleMatrix sfcc = rr.getScaledFluxControlCoefficientMatrix();

    int AKnum = 14;
    ASSERT_STREQ("AK", rr.getReactionIds()[AKnum].c_str());

    for (unsigned int rxn = 0; rxn < sfcc.CSize(); rxn++)
    {
        EXPECT_EQ(sfcc.Element(AKnum, rxn), 0.0);
    }

}


TEST_F(ModelAnalysisTests, issue1020_full) {
    //Config::setValue(Config::LLVM_BACKEND, Config::LLVM_BACKEND_VALUES::LLJIT);
    rr::RoadRunner rr((modelAnalysisModelsDir / "Jarrah2014.xml").string());

    ls::DoubleMatrix ues = rr.getUnscaledElasticityMatrix();
    ls::DoubleMatrix jacobian = rr.getFullJacobian();

    rr = ((modelAnalysisModelsDir / "Jarrah2014_timevar.xml").string());
    ls::DoubleMatrix ues_tv = rr.getUnscaledElasticityMatrix();
    ls::DoubleMatrix jacobian_tv = rr.getFullJacobian();

    ASSERT_EQ(ues.CSize(), ues_tv.CSize());
    ASSERT_EQ(ues.RSize(), ues_tv.RSize());
    for (unsigned int c = 0; c < ues.CSize(); c++)
    {
        for (unsigned int r = 0; r < ues.RSize(); r++)
        {
            EXPECT_NEAR(ues.Element(r, c), ues_tv.Element(r, c), 0.0001);
        }
    }

    ASSERT_EQ(jacobian.CSize(), jacobian_tv.CSize());
    ASSERT_EQ(jacobian.RSize(), jacobian_tv.RSize());
    for (unsigned int c = 0; c < jacobian.CSize(); c++)
    {
        for (unsigned int r = 0; r < jacobian.RSize(); r++)
        {
            EXPECT_NEAR(jacobian.Element(r, c), jacobian_tv.Element(r, c), 0.0001);
        }
    }
}


TEST_F(ModelAnalysisTests, issue1020_reduced) {
    //Config::setValue(Config::LLVM_BACKEND, Config::LLVM_BACKEND_VALUES::LLJIT);
    rr::RoadRunner rr((modelAnalysisModelsDir / "Jarrah2014.xml").string());

    ls::DoubleMatrix jacobian = rr.getReducedJacobian();

    rr = ((modelAnalysisModelsDir / "Jarrah2014_timevar.xml").string());
    ls::DoubleMatrix jacobian_tv = rr.getReducedJacobian();

    ASSERT_EQ(jacobian.CSize(), jacobian_tv.CSize());
    ASSERT_EQ(jacobian.RSize(), jacobian_tv.RSize());
    for (unsigned int c = 0; c < jacobian.CSize(); c++)
    {
        for (unsigned int r = 0; r < jacobian.RSize(); r++)
        {
            EXPECT_NEAR(jacobian.Element(r, c), jacobian_tv.Element(r, c), 0.0001);
        }
    }
}


TEST_F(ModelAnalysisTests, issue976a) {
    //The only thing this test needs to do is not crash--copying MCJit models was having problems.
    //Config::setValue(Config::LLVM_BACKEND, Config::LLVM_BACKEND_VALUES::LLJIT);
    rr::RoadRunner roadrunner((modelAnalysisModelsDir / "BIOMD0000000021.xml").string());
    double t_start = 0;
    double delta_time = 0.5;

    //create copy for checkpoint
    rr::RoadRunner roadrunnerB;

    // save checkpoint
    roadrunnerB = roadrunner;
    std::stringstream* saved = roadrunnerB.saveStateS();
    roadrunner.oneStep(t_start, delta_time);
    //std::cout << "P1: " << roadrunner.getValue("[P1]") << std::endl;

    //go back to checkpoint
    roadrunner.loadStateS(saved);
}


TEST_F(ModelAnalysisTests, issue976b) {
    //The only thing this test needs to do is not crash--copying MCJit models was having problems.
    rr::RoadRunner roadrunner((modelAnalysisModelsDir / "BIOMD0000000021.xml").string());
    double t_start = 0;
    double delta_time = 0.5;

    // save State for checkpoint
    std::stringstream* roadrunner_state = roadrunner.saveStateS();

    roadrunner.oneStep(t_start, delta_time);
    //std::cout << "P1: " << roadrunner.getValue("[P1]") << std::endl;

    //go back to checkpoint if FEM did not converged
    roadrunner.loadStateS(roadrunner_state);

    //save State again
    std::stringstream* roadrunner_state2 = roadrunner.saveStateS();

    //rerun with smaller timestep
    roadrunner.oneStep(t_start, (delta_time / 2));

    //std::cout << "P1: " << roadrunner.getValue("[P1]") << std::endl;

    //go back to checkpoint if FEM did not converged
    roadrunner.loadStateS(roadrunner_state2);

    //rerun with smaller timestep
    roadrunner.oneStep(t_start, (delta_time / 4));

    //std::cout << "P1: " << roadrunner.getValue("[P1]") << std::endl;
}


TEST_F(ModelAnalysisTests, checkGetRatesOfChangeIds) {
    //If a model has non-species with rates of change, 'getRatesOfChange' works fine,
    // but the labels for those rates were wrong. This tests that they were fixed.
    RoadRunner rr((modelAnalysisModelsDir / "ratesOfChange.xml").string());
    vector<double> out = rr.getRatesOfChange();
    vector<string> outids = rr.getRateOfChangeIds();
    ASSERT_EQ(out.size(), outids.size());
    EXPECT_NEAR(out[0], 0.7, 0.0001);
    EXPECT_NEAR(out[1], 3.0, 0.0001);
    EXPECT_NEAR(out[2], 0.5, 0.0001);
    EXPECT_NEAR(out[3], -.3, 0.0001);
    EXPECT_NEAR(out[4], 0.3, 0.0001);
    EXPECT_NEAR(out[5], 0.0, 0.0001);
    EXPECT_STREQ(outids[0].c_str(), "S3'");
    EXPECT_STREQ(outids[1].c_str(), "C'");
    EXPECT_STREQ(outids[2].c_str(), "k1'");
    EXPECT_STREQ(outids[3].c_str(), "S1'");
    EXPECT_STREQ(outids[4].c_str(), "S2'");
    EXPECT_STREQ(outids[5].c_str(), "S5'");

    ls::DoubleMatrix out2m = rr.getRatesOfChangeNamedArray();
    vector<string> cols = out2m.getColNames();
    EXPECT_EQ(cols, outids);
}


TEST_F(ModelAnalysisTests, checkGetRatesOfChangeIdsNonBoundarySpeciesRate) {
    //A species can be 'boundary=false' and still have a rate rule if it
    // never shows up in a reaction.  This makes it 'floating', which makes
    // things more awkward for situations like this.
    RoadRunner rr((modelAnalysisModelsDir / "ratesOfChange.xml").string());
    rr.setBoundary("S3", false);
    vector<double> out = rr.getRatesOfChange();
    vector<string> outids = rr.getRateOfChangeIds();
    ASSERT_EQ(out.size(), outids.size());
    EXPECT_NEAR(out[0], 0.7, 0.0001);
    EXPECT_NEAR(out[1], 3.0, 0.0001);
    EXPECT_NEAR(out[2], 0.5, 0.0001);
    EXPECT_NEAR(out[3], -.3, 0.0001);
    EXPECT_NEAR(out[4], 0.3, 0.0001);
    EXPECT_NEAR(out[5], 0.0, 0.0001);
    EXPECT_STREQ(outids[0].c_str(), "S3'");
    EXPECT_STREQ(outids[1].c_str(), "C'");
    EXPECT_STREQ(outids[2].c_str(), "k1'");
    EXPECT_STREQ(outids[3].c_str(), "S1'");
    EXPECT_STREQ(outids[4].c_str(), "S2'");
    EXPECT_STREQ(outids[5].c_str(), "S5'");

    ls::DoubleMatrix out2m = rr.getRatesOfChangeNamedArray();
    vector<string> cols = out2m.getColNames();
    EXPECT_EQ(cols, outids);
}


TEST_F(ModelAnalysisTests, checkGetRatesOfChangeIdsConservedMoiety) {
    //As above, but in a model with conserved moieties
    RoadRunner rr((modelAnalysisModelsDir / "conserved_cycle.xml").string());
    rr.setConservedMoietyAnalysis(true);
    vector<double> out = rr.getRatesOfChange();
    vector<string> outids = rr.getRateOfChangeIds();
    ASSERT_EQ(out.size(), outids.size());
    EXPECT_NEAR(out[0], -0.10748794268115947, 0.000001);
    EXPECT_STREQ(outids[0].c_str(), "S1'");
    ls::DoubleMatrix out2m = rr.getRatesOfChangeNamedArray();
    vector<string> cols = out2m.getColNames();
    EXPECT_EQ(cols, outids);

    rr.setConservedMoietyAnalysis(false);
    out = rr.getRatesOfChange();
    outids = rr.getRateOfChangeIds();
    ASSERT_EQ(out.size(), outids.size());
    EXPECT_NEAR(out[0], -0.10748794268115947, 0.000001);
    EXPECT_NEAR(out[1], 0.10748794268115947, 0.000001);
    EXPECT_STREQ(outids[0].c_str(), "S1'");
    EXPECT_STREQ(outids[1].c_str(), "S2'");
    out2m = rr.getRatesOfChangeNamedArray();
    cols = out2m.getColNames();
    EXPECT_EQ(cols, outids);
}


TEST_F(ModelAnalysisTests, checkGetFullStoichimetryMatrixWarningMsg) {
    //If a model has a lot of reactions but only a few species, it would sometimes
    // get too small of a scratch space to use in lapack.  The only way to tell
    // is that lapack would complain to the screen, so this test can only
    // tell you something's wrong if you look at it and see the lapack error.
    RoadRunner rr((modelAnalysisModelsDir / "ansatz.xml").string());
    ls::DoubleMatrix out = rr.getFullStoichiometryMatrix();
}


TEST_F(ModelAnalysisTests, checkUCCForSet) {
    RoadRunner rr((modelAnalysisModelsDir / "BIOMD0000000021.xml").string());
    //Behind the scenes, we get and then set various values to calculate ucc, and 
    // have to ensure that we don't worry about setting something that's already set.
    double val = rr.getValue("ucc(P0_to_P1, V_mT)");
    EXPECT_NEAR(val, 7.84724, 0.001);
}

TEST_F(ModelAnalysisTests, getConcentrationRateSimple) {
    RoadRunner rr((modelAnalysisModelsDir / "threestep.xml").string());
    double S1_conc_rate = rr.getValue("[S1]'");
    double S1_rate = rr.getValue("S1'");
    EXPECT_NEAR(S1_conc_rate, -1.3, 0.001);
    EXPECT_NEAR(S1_rate, -1.3, 0.001);
}

TEST_F(ModelAnalysisTests, getConcentrationRateComplex) {
    RoadRunner rr((modelAnalysisModelsDir / "threestep_varC.xml").string());
    double S1_rate = rr.getValue("S1'");
    double S1_conc = rr.getValue("[S1]");
    double comp = rr.getValue("C");
    double comp_rate = rr.getValue("C'");
    double S1_conc_rate = rr.getValue("[S1]'");
    EXPECT_NEAR(S1_rate, -1.3, 0.001);
    EXPECT_NEAR(S1_conc, 10, 0.001);
    EXPECT_NEAR(comp, 2, 0.001);
    EXPECT_NEAR(comp_rate, 2, 0.001);
    EXPECT_NEAR(S1_conc_rate, (S1_rate - (S1_conc * comp_rate))/comp, 0.001);
}

TEST_F(ModelAnalysisTests, getConcentrationRateBoundary) {
    RoadRunner rr((modelAnalysisModelsDir / "threestep_varC.xml").string());
    rr.setBoundary("S1", true, false);
    rr.setHasOnlySubstanceUnits("S1", true, false);
    rr.addRateRule("S1", "-1.3");
    double S1_rate = rr.getValue("S1'");
    double S1_conc = rr.getValue("[S1]");
    double comp = rr.getValue("C");
    double comp_rate = rr.getValue("C'");
    double S1_conc_rate = rr.getValue("[S1]'");
    EXPECT_NEAR(S1_rate, -1.3, 0.001);
    EXPECT_NEAR(S1_conc, 10, 0.001);
    EXPECT_NEAR(comp, 2, 0.001);
    EXPECT_NEAR(comp_rate, 2, 0.001);
    EXPECT_NEAR(S1_conc_rate, (S1_rate - (S1_conc * comp_rate)) / comp, 0.001);
}

TEST_F(ModelAnalysisTests, getConcentrationRateFail) {
    RoadRunner rr((modelAnalysisModelsDir / "threestep.xml").string());
    rr.addAssignmentRule("C", "3");
    EXPECT_THROW(
        double S1_conc_rate = rr.getValue("[S1]'"), 
        std::invalid_argument);
}

TEST_F(ModelAnalysisTests, SameJacobians1) {
    RoadRunner rr((modelAnalysisModelsDir / "apap_liver_core_9.xml").string());

    rr.setValue("Vext", 0.1 * 0.2);
    rr.setValue("Vli", 0.1 * (0.3 + (1 - 0.3 - 0.2)));

    SimulateOptions opt;
    opt.start = 0;
    opt.duration = 0.1;
    opt.steps = 1;
    rr.simulate(&opt);

    double origVext = rr.getValue("Vext");
    double origapap = rr.getValue("apap");

    ls::DoubleMatrix jf = rr.getFullJacobian();
    ls::DoubleMatrix jr = rr.getReducedJacobian();

    ASSERT_EQ(jf.CSize(), jr.CSize());
    ASSERT_EQ(jf.RSize(), jr.RSize());

    for (unsigned int c = 0; c < jf.CSize(); c++)
    {
        for (unsigned int r = 0; r < jf.RSize(); r++)
        {
            EXPECT_NEAR(jf.Element(r, c), jr.Element(r, c), 0.01);
        }
    }

    EXPECT_NEAR(origVext, rr.getValue("Vext"), 0.0001);
    EXPECT_NEAR(origapap, rr.getValue("apap"), 0.0001);
}


TEST_F(ModelAnalysisTests, SameJacobians2) {
    RoadRunner rr((modelAnalysisModelsDir / "apap_liver_core_simplified.xml").string());

    rr.setValue("Vext", 0.1 * 0.2);
    rr.setValue("Vli", 0.1 * (0.3 + (1 - 0.3 - 0.2)));

    SimulateOptions opt;
    opt.start = 0;
    opt.duration = 0.1;
    opt.steps = 1;
    rr.simulate(&opt);

    double origVext = rr.getValue("Vext");
    double origapap = rr.getValue("apap");

    ls::DoubleMatrix jf = rr.getFullJacobian();
    ls::DoubleMatrix jr = rr.getReducedJacobian();

    ASSERT_EQ(jf.CSize(), jr.CSize());
    ASSERT_EQ(jf.RSize(), jr.RSize());

    for (unsigned int c = 0; c < jf.CSize(); c++)
    {
        for (unsigned int r = 0; r < jf.RSize(); r++)
        {
            EXPECT_NEAR(jf.Element(r, c), jr.Element(r, c), 0.01);
        }
    }

    EXPECT_NEAR(origVext, rr.getValue("Vext"), 0.0001);
    EXPECT_NEAR(origapap, rr.getValue("apap"), 0.0001);
}


TEST_F(ModelAnalysisTests, SameJacobians3) {
    RoadRunner rr((modelAnalysisModelsDir / "apap_liver_core_volchange.xml").string());

    SimulateOptions opt;
    opt.start = 0;
    opt.duration = 1;
    opt.steps = 10;
    rr.simulate(&opt);

    ls::DoubleMatrix jf = rr.getFullJacobian();
    ls::DoubleMatrix jr = rr.getReducedJacobian();

    double origVext = rr.getValue("Vext");
    double origapap = rr.getValue("apap");

    ASSERT_EQ(jf.CSize(), jr.CSize());
    ASSERT_EQ(jf.RSize(), jr.RSize());

    for (unsigned int c = 0; c < jf.CSize(); c++)
    {
        for (unsigned int r = 0; r < jf.RSize(); r++)
        {
            EXPECT_NEAR(jf.Element(r, c), jr.Element(r, c), 0.01);
        }
    }

    EXPECT_NEAR(origVext, rr.getValue("Vext"), 0.0001);
    EXPECT_NEAR(origapap, rr.getValue("apap"), 0.0001);
}


TEST_F(ModelAnalysisTests, SimulateCVODEFromNegativeStartGeneral) {
    //Event:  at S1 > 500: S1 = 300;
    RoadRunner rr((modelAnalysisModelsDir / "negstart_event.xml").string());
    SimulateOptions opt;
    opt.start = -2;
    opt.duration = 10;
    opt.steps = 120;
    const ls::DoubleMatrix* result = rr.simulate(&opt);
    ASSERT_EQ(result->CSize(), 2);
    ASSERT_EQ(result->RSize(), 121);
    //Spot checks for when the events fire.
    EXPECT_NEAR(result->Element(2, 1), 308.3, 1);
    EXPECT_NEAR(result->Element(23, 1), 300.8, 1);
    EXPECT_NEAR(result->Element(44, 1), 493.3, 1);
    EXPECT_NEAR(result->Element(45, 1), 302.5, 1);
    EXPECT_NEAR(result->Element(67, 1), 304.2, 1);
    EXPECT_NEAR(result->Element(89, 1), 305.8, 1);

}

TEST_F(ModelAnalysisTests, SimulateCVODEFromNegativeStart_Time) {
    //Event:  at time > -1.1: S1 = 0;
    RoadRunner rr((modelAnalysisModelsDir / "negstart_event_time.xml").string());
    SimulateOptions opt;
    opt.start = -2;
    opt.duration = 10;
    opt.steps = 120;
    const ls::DoubleMatrix* result = rr.simulate(&opt);
    ASSERT_EQ(result->CSize(), 2);
    ASSERT_EQ(result->RSize(), 121);
    //Spot checks for when the events fire.
    EXPECT_NEAR(result->Element(2, 1), 508.3, 1);
    EXPECT_NEAR(result->Element(11, 1), 1.8, 1);
    EXPECT_NEAR(result->Element(24, 1), 121, 1);
    EXPECT_NEAR(result->Element(120, 1), 1001, 1);
}

TEST_F(ModelAnalysisTests, SimulateCVODEFromNegativeStart_TimeDelay) {
    //Event:  at 0.5 after time > -1.1: S1 = 0;
    RoadRunner rr((modelAnalysisModelsDir / "negstart_event_time_delay.xml").string());
    SimulateOptions opt;
    opt.start = -2;
    opt.duration = 10;
    opt.steps = 120;
    const ls::DoubleMatrix* result = rr.simulate(&opt);
    ASSERT_EQ(result->CSize(), 2);
    ASSERT_EQ(result->RSize(), 121);
    //Spot checks for when the events fire.
    EXPECT_NEAR(result->Element(2, 1), 508.3, 1);
    EXPECT_NEAR(result->Element(17, 1), 1.8, 1);
    EXPECT_NEAR(result->Element(24, 1), 66, 1);
    EXPECT_NEAR(result->Element(120, 1), 946, 1);
}

TEST_F(ModelAnalysisTests, SimulateCVODEFromNegativeStart_T0fire) {
    //Event:  at S1 > 500, t0=false: S1 = 300;
    RoadRunner rr((modelAnalysisModelsDir / "negstart_event_t0fire.xml").string());
    SimulateOptions opt;
    opt.start = -2;
    opt.duration = 10;
    opt.steps = 120;
    const ls::DoubleMatrix* result = rr.simulate(&opt);
    ASSERT_EQ(result->CSize(), 2);
    ASSERT_EQ(result->RSize(), 121);
    //Spot checks for when the events fire.
    EXPECT_EQ(result->Element(0, 1), 300);
    EXPECT_NEAR(result->Element(21, 1), 492.5, 1);
    EXPECT_NEAR(result->Element(22, 1), 301.7, 1);
    EXPECT_NEAR(result->Element(44, 1), 303.3, 1);
    EXPECT_NEAR(result->Element(120, 1), 400, 1);
}

TEST_F(ModelAnalysisTests, SimulateCVODEFromNegativeStart_Combo) {
    //Events:  E0: at 0.3 after time > -1: S1 = 0;
    //         E1: at time > 1: S1 = 100;
    //         E2: at 0.5 after S1 > 500: S1 = 300;
    RoadRunner rr((modelAnalysisModelsDir / "negstart_event_combo.xml").string());
    SimulateOptions opt;
    opt.start = -2;
    opt.duration = 10;
    opt.steps = 120;
    const ls::DoubleMatrix* result = rr.simulate(&opt);
    ASSERT_EQ(result->CSize(), 2);
    ASSERT_EQ(result->RSize(), 121);
    //Spot checks for when the events fire.
    EXPECT_NEAR(result->Element(2, 1), 508.3, 1);
    EXPECT_NEAR(result->Element(8, 1), 308.3, 1);
    EXPECT_NEAR(result->Element(16, 1), 3.7, 1);
    EXPECT_NEAR(result->Element(36, 1), 187.0, 1);
    EXPECT_NEAR(result->Element(37, 1), 109.2, 1);
    EXPECT_NEAR(result->Element(80, 1), 503.3, 1);
    EXPECT_NEAR(result->Element(86, 1), 303.3, 1);
    EXPECT_NEAR(result->Element(120, 1), 360, 1);
}

//T0 event firing when start != 0 is generally broken, not only for negative starts.  Disabling for now; will file separate issue.
TEST_F(ModelAnalysisTests, DISABLED_SimulateCVODEFromNegativeStart_T0fireDelay) {
    //Event:  at 0.5 after S1 > 500, t0=false: S1 = 300;
    RoadRunner rr((modelAnalysisModelsDir / "negstart_event_t0fire_delay.xml").string());
    SimulateOptions opt;
    opt.start = -2;
    opt.duration = 10;
    opt.steps = 120;
    const ls::DoubleMatrix* result = rr.simulate(&opt);
    ASSERT_EQ(result->CSize(), 2);
    ASSERT_EQ(result->RSize(), 121);
    //Spot checks for when the events fire.
    EXPECT_NEAR(result->Element(0, 1), 550, 1);
    EXPECT_NEAR(result->Element(21, 1), 492.5, 1);
    EXPECT_NEAR(result->Element(22, 1), 301.7, 1);
    EXPECT_NEAR(result->Element(44, 1), 303.3, 1);
    EXPECT_NEAR(result->Element(120, 1), 400, 1);
    std::cout << *result;
}

TEST_F(ModelAnalysisTests, SimulateGillespieFromNegativeStart_General) {
    //Event:  at S1 > 500: S1 = 300;
    RoadRunner rr((modelAnalysisModelsDir / "negstart_event.xml").string());
    rr.setIntegrator("gillespie");
    rr.getIntegrator()->setValue("variable_step_size", false);
    rr.getIntegrator()->setValue("seed", -1);
    //std::cout << rr.getSeed("gillespie") << std::endl;
    SimulateOptions opt;
    opt.start = -2;
    opt.duration = 10;
    opt.steps = 120;
    const ls::DoubleMatrix* result = rr.simulate(&opt);

    for (int i = 0; i <= opt.steps; i++)
    {
        EXPECT_LE(result->Element(i, 1), 500);
        EXPECT_GE(result->Element(i, 1), 300);
    }
}

TEST_F(ModelAnalysisTests, SimulateGillespieFromNegativeStart_T0fire) {
    //Event:  at S1 > 500, t0=false: S1 = 300;
    RoadRunner rr((modelAnalysisModelsDir / "negstart_event_t0fire.xml").string());
    rr.setIntegrator("gillespie");
    rr.getIntegrator()->setValue("variable_step_size", false);
    rr.getIntegrator()->setValue("seed", -1);
    SimulateOptions opt;
    opt.start = -2;
    opt.duration = 10;
    opt.steps = 120;
    const ls::DoubleMatrix* result = rr.simulate(&opt);

    EXPECT_EQ(result->Element(0, 1), 300);
    for (int i = 1; i <= opt.steps; i++)
    {
        EXPECT_LE(result->Element(i, 1), 500);
        EXPECT_GE(result->Element(i, 1), 300);
    }
}

TEST_F(ModelAnalysisTests, NonZeroStarts_CVODE) {
    //Event:  at S1 > 500, t0=false: S1 = 300;
    RoadRunner rr((modelAnalysisModelsDir / "BIOMD0000000035_url.xml").string());
    rr.setIntegrator("cvode");
    //rr.getIntegrator()->setValue("variable_step_size", false);
    SimulateOptions opt;
    opt.start = 0;
    opt.duration = 10;
    opt.steps = 50;
    ls::DoubleMatrix s0result(*(rr.simulate(&opt)));
    rr.reset(int(SelectionRecord::SelectionType::ALL));
    opt.start = -2;
    ls::DoubleMatrix sneg2result(*(rr.simulate(&opt)));
    rr.reset(int(SelectionRecord::SelectionType::ALL));
    opt.start = 2;
    ls::DoubleMatrix s2result(*(rr.simulate(&opt)));


    for (int i = 0; i <= opt.steps; i++)
    {
        EXPECT_NEAR(s0result.Element(i, 0), sneg2result.Element(i, 0) + 2, 0.01);
        EXPECT_NEAR(s0result.Element(i, 0), s2result.Element(i, 0) - 2, 0.01);

        for (int j = 1; j < s0result.CSize(); j++)
        {
            EXPECT_NEAR(s0result.Element(i, j), sneg2result.Element(i, j), 0.01);
            EXPECT_NEAR(s0result.Element(i, j), s2result.Element(i, j), 0.01);
        }
    }
}

TEST_F(ModelAnalysisTests, NonZeroStarts_RK4) {
    //Event:  at S1 > 500, t0=false: S1 = 300;
    RoadRunner rr((modelAnalysisModelsDir / "threestep.xml").string());
    rr.setIntegrator("rk4");
    //rr.getIntegrator()->setValue("variable_step_size", false);
    SimulateOptions opt;
    opt.start = 0;
    opt.duration = 10;
    opt.steps = 50;
    ls::DoubleMatrix s0result(*(rr.simulate(&opt)));
    rr.reset(int(SelectionRecord::SelectionType::ALL));
    opt.start = -2;
    ls::DoubleMatrix sneg2result(*(rr.simulate(&opt)));
    rr.reset(int(SelectionRecord::SelectionType::ALL));
    opt.start = 2;
    ls::DoubleMatrix s2result(*(rr.simulate(&opt)));

    for (int i = 0; i <= opt.steps; i++)
    {
        EXPECT_NEAR(s0result.Element(i, 0), sneg2result.Element(i, 0) + 2, 0.01);
        EXPECT_NEAR(s0result.Element(i, 0), s2result.Element(i, 0) - 2, 0.01);

        for (int j = 1; j < s0result.CSize(); j++)
        {
            EXPECT_NEAR(s0result.Element(i, j), sneg2result.Element(i, j), 0.01);
            EXPECT_NEAR(s0result.Element(i, j), s2result.Element(i, j), 0.01);
        }
    }
}


TEST_F(ModelAnalysisTests, NonZeroStarts_RK45) {
    //Event:  at S1 > 500, t0=false: S1 = 300;
    RoadRunner rr((modelAnalysisModelsDir / "threestep.xml").string());
    rr.setIntegrator("rk45");
    rr.getIntegrator()->setValue("variable_step_size", false);
    SimulateOptions opt;
    opt.start = 0;
    opt.duration = 10;
    opt.steps = 50;
    ls::DoubleMatrix s0result(*(rr.simulate(&opt)));
    rr.reset(int(SelectionRecord::SelectionType::ALL));
    opt.start = -2;
    ls::DoubleMatrix sneg2result(*(rr.simulate(&opt)));
    rr.reset(int(SelectionRecord::SelectionType::ALL));
    opt.start = 2;
    ls::DoubleMatrix s2result(*(rr.simulate(&opt)));

    for (int i = 0; i <= opt.steps; i++)
    {
        EXPECT_NEAR(s0result.Element(i, 0), sneg2result.Element(i, 0) + 2, 0.01);
        EXPECT_NEAR(s0result.Element(i, 0), s2result.Element(i, 0) - 2, 0.01);

        for (int j = 1; j < s0result.CSize(); j++)
        {
            EXPECT_NEAR(s0result.Element(i, j), sneg2result.Element(i, j), 0.01);
            EXPECT_NEAR(s0result.Element(i, j), s2result.Element(i, j), 0.01);
        }
    }
}

TEST_F(ModelAnalysisTests, NonZeroStarts_Gillespie) {
    RoadRunner rr((modelAnalysisModelsDir / "threestep.xml").string());
    rr.setIntegrator("gillespie");
    rr.getIntegrator()->setValue("variable_step_size", false);
    rr.getIntegrator()->setValue("seed", 1001);
    SimulateOptions opt;
    opt.start = 0;
    opt.duration = 10;
    opt.steps = 50;
    ls::DoubleMatrix s0result(*(rr.simulate(&opt)));
    rr.reset(int(SelectionRecord::SelectionType::ALL));
    rr.getIntegrator()->setValue("seed", 1001);
    opt.start = -2;
    ls::DoubleMatrix sneg2result(*(rr.simulate(&opt)));
    rr.reset(int(SelectionRecord::SelectionType::ALL));
    rr.getIntegrator()->setValue("seed", 1001);
    opt.start = 2;
    ls::DoubleMatrix s2result(*(rr.simulate(&opt)));

    for (int i = 0; i <= opt.steps; i++)
    {
        EXPECT_NEAR(s0result.Element(i, 0), sneg2result.Element(i, 0) + 2, 0.01);
        EXPECT_NEAR(s0result.Element(i, 0), s2result.Element(i, 0) - 2, 0.01);

        for (int j = 1; j < s0result.CSize(); j++)
        {
            EXPECT_NEAR(s0result.Element(i, j), sneg2result.Element(i, j), 0.01);
            EXPECT_NEAR(s0result.Element(i, j), s2result.Element(i, j), 0.01);
        }
    }
}


TEST_F(ModelAnalysisTests, GetRateOfConservedSpecies) {
    RoadRunner rr((modelAnalysisModelsDir / "conserved_cycle.xml").string());
    rr.setConservedMoietyAnalysis(true);

    EXPECT_THROW(
        {
            try
            {
                rr.getValue("S2'");
            }
            catch (const std::invalid_argument& e)
            {
                // Test that it has the correct message.
                EXPECT_STREQ(e.what(), "No rate available for floating species S2: if conserved moieties are enabled, this species may be defined by an implied assignment rule instead, and its rate cannot be determined.");
                throw e;
            }
        }, std::invalid_argument);

    std::vector<std::string> selections;
    selections.push_back("time");
    selections.push_back("S2'");
    rr.setSelections(selections);
    EXPECT_THROW(
        {
            try
            {
                rr.simulate();
            }
            catch (const std::invalid_argument& e)
            {
                // Test that it has the correct message.
                EXPECT_STREQ(e.what(), "No rate available for floating species S2: if conserved moieties are enabled, this species may be defined by an implied assignment rule instead, and its rate cannot be determined.");
                throw e;
            }
        }, std::invalid_argument);
}


TEST_F(ModelAnalysisTests, ResetBoundarySpeciesRate) {
    RoadRunner rr((modelAnalysisModelsDir / "boundary_change_rate.xml").string());
    double S1 = rr.getValue("init(S1)");
    EXPECT_EQ(S1, 3);

    rr.simulate();
    S1 = rr.getValue("init(S1)");
    EXPECT_EQ(S1, 3);
    S1 = rr.getValue("S1");
    EXPECT_EQ(S1, 23);

    rr.reset(SelectionRecord::BOUNDARY);
    S1 = rr.getValue("S1");
    EXPECT_EQ(S1, 3);

    //Since it's being changed by a rate rule, it should also be reset by default:
    rr.simulate();
    rr.reset();
    S1 = rr.getValue("S1");
    EXPECT_EQ(S1, 3);

    //Now check to ensure we don't reset this when we reset something else instead:
    rr.simulate();
    rr.reset(SelectionRecord::TIME);
    S1 = rr.getValue("S1");
    EXPECT_EQ(S1, 23);

    //Now check what happens when we change the initial value:
    rr.setValue("init(S1)", 10);
    rr.simulate();
    rr.reset();
    S1 = rr.getValue("S1");
    EXPECT_EQ(S1, 10);
}


TEST_F(ModelAnalysisTests, ResetBoundarySpeciesEvent) {
    RoadRunner rr((modelAnalysisModelsDir / "boundary_change_event.xml").string());
    double S1 = rr.getValue("init(S1)");
    EXPECT_EQ(S1, 3);

    rr.simulate();
    S1 = rr.getValue("init(S1)");
    EXPECT_EQ(S1, 3);
    S1 = rr.getValue("S1");
    EXPECT_EQ(S1, 4);

    rr.reset(SelectionRecord::BOUNDARY);
    S1 = rr.getValue("S1");
    EXPECT_EQ(S1, 3);

    //Since it's being changed by an event, it should not be reset by default:
    rr.simulate();
    rr.reset();
    S1 = rr.getValue("S1");
    EXPECT_EQ(S1, 4);

    //Now check to ensure we don't reset this when we reset something else instead:
    rr.reset(int(SelectionRecord::ALL));
    rr.simulate();
    rr.reset(SelectionRecord::TIME);
    S1 = rr.getValue("S1");
    EXPECT_EQ(S1, 4);

    //Now check what happens when we change the initial value:
    rr.setValue("init(S1)", 10);
    rr.simulate();
    rr.reset(SelectionRecord::BOUNDARY);
    S1 = rr.getValue("S1");
    EXPECT_EQ(S1, 10);
}


TEST_F(ModelAnalysisTests, ResetFloatingSpeciesRate) {
    RoadRunner rr((modelAnalysisModelsDir / "floating_change_rate.xml").string());
    double S1 = rr.getValue("init(S1)");
    EXPECT_EQ(S1, 3);

    rr.simulate();
    S1 = rr.getValue("init(S1)");
    EXPECT_EQ(S1, 3);
    S1 = rr.getValue("S1");
    EXPECT_EQ(S1, 23);

    rr.reset(SelectionRecord::FLOATING);
    S1 = rr.getValue("S1");
    EXPECT_EQ(S1, 3);

    //Since it's being changed by a rate rule, it should definitely be reset by default:
    rr.simulate();
    rr.reset();
    S1 = rr.getValue("S1");
    EXPECT_EQ(S1, 3);

    //Now check to ensure we don't reset this when we reset something else instead:
    rr.simulate();
    rr.reset(SelectionRecord::TIME);
    S1 = rr.getValue("S1");
    EXPECT_EQ(S1, 23);

    //Now check what happens when we change the initial value:
    rr.setValue("init(S1)", 10);
    rr.simulate();
    rr.reset();
    S1 = rr.getValue("S1");
    EXPECT_EQ(S1, 10);
}


TEST_F(ModelAnalysisTests, ResetFloatingSpeciesEvent) {
    RoadRunner rr((modelAnalysisModelsDir / "floating_change_event.xml").string());
    double S1 = rr.getValue("init(S1)");
    EXPECT_EQ(S1, 3);

    rr.simulate();
    S1 = rr.getValue("init(S1)");
    EXPECT_EQ(S1, 3);
    S1 = rr.getValue("S1");
    EXPECT_EQ(S1, 4);

    rr.reset(SelectionRecord::FLOATING);
    S1 = rr.getValue("S1");
    EXPECT_EQ(S1, 3);

    //Since it's a floating species, even though it's only being changed by an event, it should still be reset by default:
    rr.simulate();
    rr.reset();
    S1 = rr.getValue("S1");
    EXPECT_EQ(S1, 3);

    //Now check to ensure we don't reset this when we reset something else instead:
    rr.simulate();
    rr.reset(SelectionRecord::TIME);
    S1 = rr.getValue("S1");
    EXPECT_EQ(S1, 4);

    //Now check what happens when we change the initial value:
    rr.setValue("init(S1)", 10);
    rr.simulate();
    rr.reset(SelectionRecord::FLOATING);
    S1 = rr.getValue("S1");
    EXPECT_EQ(S1, 10);
}


TEST_F(ModelAnalysisTests, ResetCompartmentFromRate) {
    RoadRunner rr((modelAnalysisModelsDir / "compartment_change_rate.xml").string());
    double comp = rr.getValue("init(C1)");
    EXPECT_EQ(comp, 5);

    rr.simulate();
    comp = rr.getValue("init(C1)");
    EXPECT_EQ(comp, 5);
    comp = rr.getValue("C1");
    EXPECT_EQ(comp, 15);

    rr.reset(SelectionRecord::COMPARTMENT);
    comp = rr.getValue("C1");
    EXPECT_EQ(comp, 5);

    //Since it's being changed by a rate rule, it should also be reset by default:
    rr.simulate();
    rr.reset();
    comp = rr.getValue("C1");
    EXPECT_EQ(comp, 5);

    //Now check to ensure we don't reset this when we reset something else instead:
    rr.simulate();
    rr.reset(SelectionRecord::TIME);
    comp = rr.getValue("C1");
    EXPECT_EQ(comp, 15);

    //Now check what happens when we change the initial value:
    rr.setValue("init(C1)", 10);
    rr.simulate();
    rr.reset(SelectionRecord::COMPARTMENT);
    comp = rr.getValue("C1");
    EXPECT_EQ(comp, 10);

}


TEST_F(ModelAnalysisTests, ResetCompartmentFromEvent) {
    RoadRunner rr((modelAnalysisModelsDir / "compartment_change_event.xml").string());
    double comp = rr.getValue("init(C1)");
    EXPECT_EQ(comp, 5);

    rr.simulate();
    comp = rr.getValue("init(C1)");
    EXPECT_EQ(comp, 5);
    comp = rr.getValue("C1");
    EXPECT_EQ(comp, 2);

    rr.reset(SelectionRecord::COMPARTMENT);
    comp = rr.getValue("C1");
    EXPECT_EQ(comp, 5);

    //Since it's being changed by an event, it should not be reset by default:
    rr.simulate();
    rr.reset();
    comp = rr.getValue("C1");
    EXPECT_EQ(comp, 2);

    //Now check to ensure we don't reset this when we reset something else instead:
    rr.reset(int(SelectionRecord::ALL));
    rr.simulate();
    rr.reset(SelectionRecord::TIME);
    comp = rr.getValue("C1");
    EXPECT_EQ(comp, 2);

    //Now check what happens when we change the initial value:
    rr.setValue("init(C1)", 10);
    rr.simulate();
    rr.reset(SelectionRecord::COMPARTMENT);
    comp = rr.getValue("C1");
    EXPECT_EQ(comp, 10);

}


TEST_F(ModelAnalysisTests, ResetParamFromRate) {
    RoadRunner rr((modelAnalysisModelsDir / "parameter_change_rate.xml").string());
    double P1 = rr.getValue("init(P1)");
    EXPECT_EQ(P1, 3);

    rr.simulate();
    P1 = rr.getValue("init(P1)");
    EXPECT_EQ(P1, 3);
    P1 = rr.getValue("P1");
    EXPECT_EQ(P1, 23);

    rr.reset(SelectionRecord::GLOBAL_PARAMETER);
    P1 = rr.getValue("P1");
    EXPECT_EQ(P1, 3);

    //Since it's being changed by a rate rule, it should also be reset by default:
    rr.simulate();
    rr.reset();
    P1 = rr.getValue("P1");
    EXPECT_EQ(P1, 3);

    //Now check to ensure we don't reset this when we reset something else instead:
    rr.simulate();
    rr.reset(SelectionRecord::TIME);
    P1 = rr.getValue("P1");
    EXPECT_EQ(P1, 23);

    //Now check what happens when we change the initial value:
    rr.setValue("init(P1)", 10);
    rr.simulate();
    rr.reset(SelectionRecord::GLOBAL_PARAMETER);
    P1 = rr.getValue("P1");
    EXPECT_EQ(P1, 10);

}


TEST_F(ModelAnalysisTests, ResetParamFromEvent) {
    RoadRunner rr((modelAnalysisModelsDir / "parameter_change_event.xml").string());
    double P1 = rr.getValue("init(P1)");
    EXPECT_EQ(P1, 3);

    rr.simulate();
    P1 = rr.getValue("init(P1)");
    EXPECT_EQ(P1, 3);
    P1 = rr.getValue("P1");
    EXPECT_EQ(P1, 4);

    rr.reset(SelectionRecord::GLOBAL_PARAMETER);
    P1 = rr.getValue("P1");
    EXPECT_EQ(P1, 3);

    //Since it's being changed by an event, it should not be reset by default:
    rr.simulate();
    rr.reset();
    P1 = rr.getValue("P1");
    EXPECT_EQ(P1, 4);

    //Now check to ensure we don't reset this when we reset something else instead:
    rr.reset(int(SelectionRecord::ALL));
    rr.simulate();
    rr.reset(SelectionRecord::TIME);
    P1 = rr.getValue("P1");
    EXPECT_EQ(P1, 4);

    //Now check what happens when we change the initial value:
    rr.setValue("init(P1)", 10);
    rr.simulate();
    rr.reset(SelectionRecord::GLOBAL_PARAMETER);
    P1 = rr.getValue("P1");
    EXPECT_EQ(P1, 10);

}


TEST_F(ModelAnalysisTests, ResetManyCompartments) {
    //Here's the model in Antimony:
    //C1 = 1;
    //E0: at time > 1: C1 = 10;
    //C2: = C4/2
    //C3 = 3
    //C3' = 0.5
    //C4 = 3 + C1
    //C5 = 4 + C1
    //C5' = 1.5
    //C6 = 6
    //E1: at time > 1.1: C6 = 60


    RoadRunner rr((modelAnalysisModelsDir / "many_compartments.xml").string());
    double C1 = rr.getValue("init(C1)");
    EXPECT_EQ(C1, 1);
    double C2 = rr.getValue("init(C2)");
    EXPECT_EQ(C2, 2);
    double C3 = rr.getValue("init(C3)");
    EXPECT_EQ(C3, 3);
    double C4 = rr.getValue("init(C4)");
    EXPECT_EQ(C4, 4);
    double C5 = rr.getValue("init(C5)");
    EXPECT_EQ(C5, 5);
    double C6 = rr.getValue("init(C6)");
    EXPECT_EQ(C6, 6);

    //Simulate and reset all compartments:
    rr.simulate();
    rr.reset(SelectionRecord::COMPARTMENT);
    C1 = rr.getValue("C1");
    EXPECT_EQ(C1, 1);
    C2 = rr.getValue("C2");
    EXPECT_EQ(C2, 2);
    C3 = rr.getValue("C3");
    EXPECT_EQ(C3, 3);
    C4 = rr.getValue("C4");
    EXPECT_EQ(C4, 4);
    C5 = rr.getValue("C5");
    EXPECT_EQ(C5, 5);
    C6 = rr.getValue("C6");
    EXPECT_EQ(C6, 6);

    //Simulate and only reset compartments changed by a rate rule:
    rr.simulate();
    rr.reset();
    C1 = rr.getValue("C1");
    EXPECT_EQ(C1, 10);
    C2 = rr.getValue("C2");
    EXPECT_EQ(C2, 2);
    C3 = rr.getValue("C3");
    EXPECT_EQ(C3, 3);
    C4 = rr.getValue("C4");
    EXPECT_EQ(C4, 4);
    C5 = rr.getValue("C5");
    EXPECT_EQ(C5, 14);
    C6 = rr.getValue("C6");
    EXPECT_EQ(C6, 60);

    //Simulate and only reset compartments changed by a rate rule:
    rr.simulate();
    rr.reset(SelectionRecord::RATE);
    C1 = rr.getValue("C1");
    EXPECT_EQ(C1, 10);
    C2 = rr.getValue("C2");
    EXPECT_EQ(C2, 2);
    C3 = rr.getValue("C3");
    EXPECT_EQ(C3, 3);
    C4 = rr.getValue("C4");
    EXPECT_EQ(C4, 4);
    C5 = rr.getValue("C5");
    EXPECT_EQ(C5, 14);
    C6 = rr.getValue("C6");
    EXPECT_EQ(C6, 60);

    //Now check what happens when we change the initial value:
    rr.setValue("init(C1)", 15);
    rr.simulate();
    rr.reset(SelectionRecord::COMPARTMENT);
    C1 = rr.getValue("C1");
    EXPECT_EQ(C1, 15);
    C2 = rr.getValue("C2");
    EXPECT_EQ(C2, 9);
    C3 = rr.getValue("C3");
    EXPECT_EQ(C3, 3);
    C4 = rr.getValue("C4");
    EXPECT_EQ(C4, 18);
    C5 = rr.getValue("C5");
    EXPECT_EQ(C5, 19);
    C6 = rr.getValue("C6");
    EXPECT_EQ(C6, 6);

}


TEST_F(ModelAnalysisTests, ResetManyFloatingSpecies) {
    RoadRunner rr((modelAnalysisModelsDir / "many_floating_species.xml").string());
    double C1 = rr.getValue("init(C1)");
    EXPECT_EQ(C1, 1);
    double C2 = rr.getValue("init(C2)");
    EXPECT_EQ(C2, 2);
    double C3 = rr.getValue("init(C3)");
    EXPECT_EQ(C3, 3);
    double C4 = rr.getValue("init(C4)");
    EXPECT_EQ(C4, 4);
    double C5 = rr.getValue("init(C5)");
    EXPECT_EQ(C5, 5);
    double C6 = rr.getValue("init(C6)");
    EXPECT_EQ(C6, 6);

    //Simulate and reset all floating species:
    rr.simulate();
    rr.reset(SelectionRecord::FLOATING);
    C1 = rr.getValue("C1");
    EXPECT_EQ(C1, 1);
    C2 = rr.getValue("C2");
    EXPECT_EQ(C2, 2);
    C3 = rr.getValue("C3");
    EXPECT_EQ(C3, 3);
    C4 = rr.getValue("C4");
    EXPECT_EQ(C4, 4);
    C5 = rr.getValue("C5");
    EXPECT_EQ(C5, 5);
    C6 = rr.getValue("C6");
    EXPECT_EQ(C6, 6);

    //Simulate and reset the default, which includes all floating species:
    rr.simulate();
    rr.reset();
    C1 = rr.getValue("C1");
    EXPECT_EQ(C1, 1);
    C2 = rr.getValue("C2");
    EXPECT_EQ(C2, 2);
    C3 = rr.getValue("C3");
    EXPECT_EQ(C3, 3);
    C4 = rr.getValue("C4");
    EXPECT_EQ(C4, 4);
    C5 = rr.getValue("C5");
    EXPECT_EQ(C5, 5);
    C6 = rr.getValue("C6");
    EXPECT_EQ(C6, 6);

    //Simulate and only reset floating species changed by a rate rule:
    rr.simulate();
    rr.reset(SelectionRecord::RATE);
    C1 = rr.getValue("C1");
    EXPECT_EQ(C1, 10);
    C2 = rr.getValue("C2");
    EXPECT_EQ(C2, 2);
    C3 = rr.getValue("C3");
    EXPECT_EQ(C3, 3);
    C4 = rr.getValue("C4");
    EXPECT_EQ(C4, 4);
    C5 = rr.getValue("C5");
    //Bug:  see https://github.com/sys-bio/roadrunner/issues/819
    //EXPECT_EQ(C5, 14);
    C6 = rr.getValue("C6");
    EXPECT_EQ(C6, 60);

    //Now check what happens when we change the initial value:
    rr.setValue("init(C1)", 15);
    rr.simulate();
    rr.reset(SelectionRecord::FLOATING);
    C1 = rr.getValue("C1");
    EXPECT_EQ(C1, 15);
    C2 = rr.getValue("C2");
    EXPECT_EQ(C2, 9);
    C3 = rr.getValue("C3");
    EXPECT_EQ(C3, 3);
    C4 = rr.getValue("C4");
    EXPECT_EQ(C4, 18);
    C5 = rr.getValue("C5");
    EXPECT_EQ(C5, 19);
    C6 = rr.getValue("C6");
    EXPECT_EQ(C6, 6);
}


TEST_F(ModelAnalysisTests, ResetManyBoundarySpecies) {
    RoadRunner rr((modelAnalysisModelsDir / "many_boundary_species.xml").string());
    double C1 = rr.getValue("init(C1)");
    EXPECT_EQ(C1, 1);
    double C2 = rr.getValue("init(C2)");
    EXPECT_EQ(C2, 2);
    double C3 = rr.getValue("init(C3)");
    EXPECT_EQ(C3, 3);
    double C4 = rr.getValue("init(C4)");
    EXPECT_EQ(C4, 4);
    double C5 = rr.getValue("init(C5)");
    EXPECT_EQ(C5, 5);
    double C6 = rr.getValue("init(C6)");
    EXPECT_EQ(C6, 6);

    //Simulate and reset all boundary species:
    rr.simulate();
    rr.reset(SelectionRecord::BOUNDARY);
    C1 = rr.getValue("C1");
    EXPECT_EQ(C1, 1);
    C2 = rr.getValue("C2");
    EXPECT_EQ(C2, 2);
    C3 = rr.getValue("C3");
    EXPECT_EQ(C3, 3);
    C4 = rr.getValue("C4");
    EXPECT_EQ(C4, 4);
    C5 = rr.getValue("C5");
    EXPECT_EQ(C5, 5);
    C6 = rr.getValue("C6");
    EXPECT_EQ(C6, 6);

    //Simulate and reset the default, which includes only boundary species changed by a rate rule:
    rr.simulate();
    rr.reset();
    C1 = rr.getValue("C1");
    EXPECT_EQ(C1, 10);
    C2 = rr.getValue("C2");
    EXPECT_EQ(C2, 2);
    C3 = rr.getValue("C3");
    EXPECT_EQ(C3, 3);
    C4 = rr.getValue("C4");
    EXPECT_EQ(C4, 4);
    //Bug:  see https://github.com/sys-bio/roadrunner/issues/819
    C5 = rr.getValue("C5");
    //EXPECT_EQ(C5, 14);
    C6 = rr.getValue("C6");
    EXPECT_EQ(C6, 60);

    //Simulate and only reset boundary species changed by a rate rule:
    rr.simulate();
    rr.reset(SelectionRecord::RATE);
    C1 = rr.getValue("C1");
    EXPECT_EQ(C1, 10);
    C2 = rr.getValue("C2");
    EXPECT_EQ(C2, 2);
    C3 = rr.getValue("C3");
    EXPECT_EQ(C3, 3);
    C4 = rr.getValue("C4");
    EXPECT_EQ(C4, 4);
    C5 = rr.getValue("C5");
    //EXPECT_EQ(C5, 14);
    C6 = rr.getValue("C6");
    EXPECT_EQ(C6, 60);

    //Now check what happens when we change the initial value:
    rr.setValue("init(C1)", 15);
    rr.simulate();
    rr.reset(SelectionRecord::BOUNDARY);
    C1 = rr.getValue("C1");
    EXPECT_EQ(C1, 15);
    C2 = rr.getValue("C2");
    EXPECT_EQ(C2, 9);
    C3 = rr.getValue("C3");
    EXPECT_EQ(C3, 3);
    C4 = rr.getValue("C4");
    EXPECT_EQ(C4, 18);
    C5 = rr.getValue("C5");
    EXPECT_EQ(C5, 19);
    C6 = rr.getValue("C6");
    EXPECT_EQ(C6, 6);

}


TEST_F(ModelAnalysisTests, ResetManyParameters) {
    RoadRunner rr((modelAnalysisModelsDir / "many_parameters.xml").string());
    double C1 = rr.getValue("init(C1)");
    EXPECT_EQ(C1, 1);
    double C2 = rr.getValue("init(C2)");
    EXPECT_EQ(C2, 2);
    double C3 = rr.getValue("init(C3)");
    EXPECT_EQ(C3, 3);
    double C4 = rr.getValue("init(C4)");
    EXPECT_EQ(C4, 4);
    double C5 = rr.getValue("init(C5)");
    EXPECT_EQ(C5, 5);
    double C6 = rr.getValue("init(C6)");
    EXPECT_EQ(C6, 6);

    //Simulate and reset all parameters:
    rr.simulate();
    rr.reset(SelectionRecord::GLOBAL_PARAMETER);
    C1 = rr.getValue("C1");
    EXPECT_EQ(C1, 1);
    C2 = rr.getValue("C2");
    EXPECT_EQ(C2, 2);
    C3 = rr.getValue("C3");
    EXPECT_EQ(C3, 3);
    C4 = rr.getValue("C4");
    EXPECT_EQ(C4, 4);
    C5 = rr.getValue("C5");
    EXPECT_EQ(C5, 5);
    C6 = rr.getValue("C6");
    EXPECT_EQ(C6, 6);

    //Simulate and reset the default, which includes only those parameters changed by a rate rule:
    rr.simulate();
    rr.reset();
    C1 = rr.getValue("C1");
    EXPECT_EQ(C1, 10);
    C2 = rr.getValue("C2");
    EXPECT_EQ(C2, 2);
    C3 = rr.getValue("C3");
    EXPECT_EQ(C3, 3);
    C4 = rr.getValue("C4");
    EXPECT_EQ(C4, 4);
    C5 = rr.getValue("C5");
    EXPECT_EQ(C5, 14);
    C6 = rr.getValue("C6");
    EXPECT_EQ(C6, 60);

    //Simulate and only reset parameters changed by a rate rule:
    rr.simulate();
    rr.reset(SelectionRecord::RATE);
    C1 = rr.getValue("C1");
    EXPECT_EQ(C1, 10);
    C2 = rr.getValue("C2");
    EXPECT_EQ(C2, 2);
    C3 = rr.getValue("C3");
    EXPECT_EQ(C3, 3);
    C4 = rr.getValue("C4");
    EXPECT_EQ(C4, 4);
    C5 = rr.getValue("C5");
    EXPECT_EQ(C5, 14);
    C6 = rr.getValue("C6");
    EXPECT_EQ(C6, 60);

    //Now check what happens when we change the initial value:
    rr.setValue("init(C1)", 15);
    rr.simulate();
    rr.reset(SelectionRecord::GLOBAL_PARAMETER);
    C1 = rr.getValue("C1");
    EXPECT_EQ(C1, 15);
    C2 = rr.getValue("C2");
    EXPECT_EQ(C2, 9);
    C3 = rr.getValue("C3");
    EXPECT_EQ(C3, 3);
    C4 = rr.getValue("C4");
    EXPECT_EQ(C4, 18);
    C5 = rr.getValue("C5");
    EXPECT_EQ(C5, 19);
    C6 = rr.getValue("C6");
    EXPECT_EQ(C6, 6);

}


TEST_F(ModelAnalysisTests, ResetConservedCycles) {
    RoadRunner rr((modelAnalysisModelsDir / "conserved_cycle.xml").string());
    rr.setConservedMoietyAnalysis(true);
    rr.steadyState();
    rr.setValue("_CSUM0", 1000);
    rr.reset();
    double val = rr.getValue("_CSUM0");
    EXPECT_NEAR(val, 8.09, 0.0001);
}


TEST_F(ModelAnalysisTests, AnalysisFunctionsWithEvent) {
    RoadRunner *rr = new RoadRunner((modelAnalysisModelsDir / "event.xml").string());
    ls::DoubleMatrix jacobian = rr->getFullJacobian();
    EXPECT_EQ(jacobian.CSize(), 2);

    std::vector<ls::Complex> eigens = rr->getFullEigenValues();
    EXPECT_EQ(eigens.size(), 2);

    ls::DoubleMatrix elasticities = rr->getScaledElasticityMatrix();
    EXPECT_EQ(elasticities.CSize(), 2);

    //ls::DoubleMatrix  fluxccm= rr->getScaledFluxControlCoefficientMatrix();
    //CHECK(fluxccm.CSize() == 2);
    //std::cout << "   PASS" << std::endl << std::endl;
    delete rr;
}


TEST_F(ModelAnalysisTests, EigenvalueNamedArrays) {
    RoadRunner* rr = new RoadRunner((modelAnalysisModelsDir / "BIOMD0000000035_url.xml").string());

    ls::DoubleMatrix eigens = rr->getFullEigenValuesNamedArray();
    EXPECT_EQ(eigens.size(), 18);
    EXPECT_EQ(eigens.RSize(), 9);
    EXPECT_EQ(eigens.CSize(), 2);
    for (unsigned int r = 0; r < 9; r++) {
        EXPECT_GT(eigens(r, 0), -105);
        EXPECT_GT(eigens(r, 1), -15);
    }

    delete rr;
}


TEST_F(ModelAnalysisTests, ReducedEigenvalueNamedArrays) {
    RoadRunner rr((modelAnalysisModelsDir / "conserved_cycle.xml").string());

    rr.setConservedMoietyAnalysis(true);
    ls::DoubleMatrix eigens = rr.getReducedEigenValuesNamedArray();
    EXPECT_EQ(eigens.size(), 2);
    ASSERT_EQ(eigens.RSize(), 1);
    ASSERT_EQ(eigens.CSize(), 2);
    vector<string> colnames = eigens.getColNames();
    EXPECT_STREQ(colnames[0].c_str(), "real");
    EXPECT_STREQ(colnames[1].c_str(), "imaginary");
    vector<string> rownames = eigens.getRowNames();
    EXPECT_STREQ(rownames[0].c_str(), "S1");
}


TEST_F(ModelAnalysisTests, GetEventIDs) {
    RoadRunner *rr = new RoadRunner((modelAnalysisModelsDir / "event.xml").string());

    list<string> eventids;
    rrllvm::LLVMExecutableModel *mod = static_cast<rrllvm::LLVMExecutableModel *>(rr->getModel());
    mod->getEventIds(eventids);
    EXPECT_EQ(eventids.size(), 1);
    EXPECT_EQ("_E0", *eventids.begin());
    delete rr;
}


TEST_F(ModelAnalysisTests, SetGetHasOnlySubstanceUnits) {
    RoadRunner rr((modelAnalysisModelsDir / "many_floating_species.xml").string());
    EXPECT_TRUE(rr.getHasOnlySubstanceUnits("C1"));
    EXPECT_FALSE(rr.getHasOnlySubstanceUnits("C2"));
    rr.setHasOnlySubstanceUnits("C1", false);
    rr.setHasOnlySubstanceUnits("C2", true);
    EXPECT_FALSE(rr.getHasOnlySubstanceUnits("C1"));
    EXPECT_TRUE(rr.getHasOnlySubstanceUnits("C2"));
}


TEST_F(ModelAnalysisTests, SimulateGillespieMaxRows) {
    RoadRunner *rr = new RoadRunner((modelAnalysisModelsDir / "BIOMD0000000035_url.xml").string());
    rr->setIntegrator("gillespie");
    rr->getIntegrator()->setValue("max_output_rows", 100);
    SimulateOptions opts = rr->getSimulateOptions();

    const ls::DoubleMatrix *results = rr->simulate(&opts);
    EXPECT_EQ(results->RSize(), 100);

    rr->getIntegrator()->setValue("max_output_rows", 1000);
    results = rr->simulate(&opts);
    EXPECT_EQ(results->RSize(), 1000);

    delete rr;
}

TEST_F(ModelAnalysisTests, SimulateAndCheckTime) {
    RoadRunner* rr = new RoadRunner((modelAnalysisModelsDir / "BIOMD0000000035_url.xml").string());
    rr->simulate(0, 5, 10);
    EXPECT_NEAR(rr->getValue("time"), 5.0, 0.00001);

    rr->reset();
    rr->setIntegrator("gillespie");
    rr->simulate(0, 5, 10);
    EXPECT_NEAR(rr->getValue("time"), 5.0, 0.001);

    delete rr;
}

TEST_F(ModelAnalysisTests, SimulateGillespieZeroDuration) {
    RoadRunner *rr = new RoadRunner((modelAnalysisModelsDir / "BIOMD0000000035_url.xml").string());
    rr->setIntegrator("gillespie");
    SimulateOptions opts = rr->getSimulateOptions();
    opts.duration = 0;
    opts.steps = 100;

    const ls::DoubleMatrix *results = rr->simulate(&opts);
    EXPECT_EQ(results->RSize(), 100);

    opts.steps = 1000;
    results = rr->simulate(&opts);
    EXPECT_EQ(results->RSize(), 1000);

    delete rr;
}

TEST_F(ModelAnalysisTests, SimulateGillespieDuration) {
    RoadRunner *rr = new RoadRunner((modelAnalysisModelsDir / "BIOMD0000000035_url.xml").string());
    rr->setIntegrator("gillespie");
    SimulateOptions opts = rr->getSimulateOptions();
    opts.duration = 0.5;

    const ls::DoubleMatrix *results = rr->simulate(&opts);
    EXPECT_NEAR(results->Element(results->numRows() - 1, 0), 0.5, 0.0001);

    opts.start = 0;
    opts.duration = 0.7;
    results = rr->simulate(&opts);
    EXPECT_NEAR(results->Element(results->numRows() - 1, 0), 0.7, 0.0001);

    delete rr;
}

TEST_F(ModelAnalysisTests, SimulateAccordingToDocs) {
    RoadRunner rr((modelAnalysisModelsDir / "BIOMD0000000035_url.xml").string());
    SimulateOptions opt;
    opt.start = 0;
    opt.duration = 10;
    opt.steps = 1000;
    const ls::DoubleMatrix* result = rr.simulate(&opt);

}

TEST_F(ModelAnalysisTests, SimulateWithTimes) {
    RoadRunner rr((modelAnalysisModelsDir / "BIOMD0000000035_url.xml").string());
    std::vector<double> times;
    times.push_back(0);
    times.push_back(1);
    times.push_back(5);
    times.push_back(10);
    SimulateOptions opt;
    opt.times = times;
    const ls::DoubleMatrix* result = rr.simulate(&opt);
    EXPECT_EQ(result->numRows(), 4);
    EXPECT_EQ(result->Element(0, 0), 0);
    EXPECT_EQ(result->Element(1, 0), 1);
    EXPECT_EQ(result->Element(2, 0), 5);
    EXPECT_EQ(result->Element(3, 0), 10);
}

TEST_F(ModelAnalysisTests, SimulateWithTimesFunction) {
    RoadRunner rr((modelAnalysisModelsDir / "BIOMD0000000035_url.xml").string());
    std::vector<double> times;
    times.push_back(0);
    times.push_back(1);
    times.push_back(5);
    times.push_back(10);
    const ls::DoubleMatrix* result = rr.simulate(times);
    EXPECT_EQ(result->numRows(), 4);
    EXPECT_EQ(result->Element(0, 0), 0);
    EXPECT_EQ(result->Element(1, 0), 1);
    EXPECT_EQ(result->Element(2, 0), 5);
    EXPECT_EQ(result->Element(3, 0), 10);
}

TEST_F(ModelAnalysisTests, ResetAfterControlCalc) {
    RoadRunner rr((modelAnalysisModelsDir / "conserved_cycle.xml").string());
    double pre = rr.getValue("S1");
    double val = rr.getValue("ucc(J0, Vm)");
    double post = rr.getValue("S1");

    EXPECT_EQ(pre, post);
}

TEST_F(ModelAnalysisTests, Stoichiometry_Reactant_Or_Product) {
    RoadRunner rr((modelAnalysisModelsDir / "get_set_stoichiometry.xml").string());

    // get the initial stoichiometry value with parameter and stoich(Species,Reaction)
    EXPECT_EQ(rr.getValue("m"), 2);
    EXPECT_EQ(rr.getValue("stoich(S2,_J0)"), 2);

    // set and get with parameter
    rr.setValue("m", 3);
    EXPECT_EQ(rr.getValue("m"), 3);

    // set and get with stoich(Species,Reaction)
    rr.setValue("stoich(S2,_J0)", 4);
    EXPECT_EQ(rr.getValue("stoich(S2,_J0)"), 4);

    // set with parameter and get with stoich(Species,Reaction)
    rr.setValue("m", 5);
    EXPECT_EQ(rr.getValue("stoich(S2,_J0)"), 5);

    // set with stoich(Species,Reaction) and get with parameter
    rr.setValue("stoich(S2,_J0)", 6);
    EXPECT_EQ(rr.getValue("m"), 6);
}

TEST_F(ModelAnalysisTests, Stoichiometry_Reactant_Or_Product_With_Switching_Conserved_Moiety_ON_And_Off) {
    RoadRunner rr((modelAnalysisModelsDir / "get_set_stoichiometry.xml").string());

    rr.setConservedMoietyAnalysis(true);

    // get the initial stoichiometry value with parameter and stoich(Species,Reaction)
    EXPECT_THROW(rr.getValue("m"), rrllvm::LLVMException);
    EXPECT_THROW(rr.getValue("stoich(S2,_J0)"), rrllvm::LLVMException);

    // set the stoichiometry value with parameter and stoich(Species,Reaction)
    EXPECT_THROW(rr.setValue("m", 3), rrllvm::LLVMException);
    EXPECT_THROW(rr.setValue("stoich(S2,_J0)", 3), rrllvm::LLVMException);

    rr.setConservedMoietyAnalysis(false);

    // get the initial stoichiometry value with parameter and stoich(Species,Reaction)
    EXPECT_EQ(rr.getValue("m"), 2);
    EXPECT_EQ(rr.getValue("stoich(S2,_J0)"), 2);

    // set and get with parameter
    rr.setValue("m", 3);
    EXPECT_EQ(rr.getValue("m"), 3);

    // set and get with stoich(Species,Reaction)
    rr.setValue("stoich(S2,_J0)", 4);
    EXPECT_EQ(rr.getValue("stoich(S2,_J0)"), 4);

    // set with parameter and get with stoich(Species,Reaction)
    rr.setValue("m", 5);
    EXPECT_EQ(rr.getValue("stoich(S2,_J0)"), 5);

    // set with stoich(Species,Reaction) and get with parameter
    rr.setValue("stoich(S2,_J0)", 6);
    EXPECT_EQ(rr.getValue("m"), 6);

    rr.setConservedMoietyAnalysis(true);

    // get the initial stoichiometry value with parameter and stoich(Species,Reaction)
    EXPECT_THROW(rr.getValue("m"), rrllvm::LLVMException);
    EXPECT_THROW(rr.getValue("stoich(S2,_J0)"), rrllvm::LLVMException);

    // set the stoichiometry value with parameter and stoich(Species,Reaction)
    EXPECT_THROW(rr.setValue("m", 3), rrllvm::LLVMException);
    EXPECT_THROW(rr.setValue("stoich(S2,_J0)", 3), rrllvm::LLVMException);
}

TEST_F(ModelAnalysisTests, Stoichiometry_MultiReactantProduct) {
    RoadRunner rr((modelAnalysisModelsDir / "get_set_stoichiometry.xml").string());

    // get the initial stoichiometry value with parameter and stoich(Species,Reaction)
    EXPECT_THROW(rr.getValue("n"), rrllvm::LLVMException);
    EXPECT_THROW(rr.getValue("stoich(S1,_J0)"), rrllvm::LLVMException);

    // set the initial stoichiometry value with parameter and stoich(Species,Reaction)
    EXPECT_THROW(rr.setValue("n", 3), rrllvm::LLVMException);
    EXPECT_THROW(rr.setValue("stoich(S1,_J0)", 3), rrllvm::LLVMException);
}

TEST_F(ModelAnalysisTests, SetSeed_Get_Seed_Value_Check) {
    RoadRunner rr((modelAnalysisModelsDir / "simple_distrib_model.xml").string());
    EXPECT_NE(rr.getSeed(), -1);
    rr.setSeed(42);
    EXPECT_EQ(rr.getSeed(), 42);
    rr.setSeed(-1);
    EXPECT_NE(rr.getSeed(), -1);
    rr.setIntegrator("gillespie");
    EXPECT_NE(rr.getSeed("gillespie"), -1);
    EXPECT_EQ(rr.getSeed("gillespie"), rr.getIntegrator()->getValue("seed").getAs<int64_t>());
    rr.setSeed(42);
    EXPECT_EQ(rr.getSeed("gillespie"), 42);
    EXPECT_EQ(rr.getSeed("gillespie"), rr.getIntegrator()->getValue("seed").getAs<int64_t>());
    rr.resetSeed();
    EXPECT_EQ(Config::getValue(Config::RANDOM_SEED).getAs<int64_t>(), -1);
    EXPECT_NE(rr.getSeed(), -1);
    EXPECT_NE(rr.getSeed("gillespie"), -1);
    EXPECT_EQ(rr.getSeed("gillespie"), rr.getIntegrator()->getValue("seed").getAs<int64_t>());
}

TEST_F(ModelAnalysisTests, SetSeed_Get_Seed_Value_With_Reset_Model) {
    RoadRunner rr1((modelAnalysisModelsDir / "simple_distrib_model.xml").string());
    RoadRunner rr2((modelAnalysisModelsDir / "simple_distrib_model.xml").string());
    rr1.setSeed(42);
    rr2.setSeed(42);
    rr1.simulate();
    rr2.simulate();
    EXPECT_NEAR(rr1.getValue("a"), rr2.getValue("a"), 0.0000001);
}

TEST_F(ModelAnalysisTests, SetSeed_Get_Seed_Value_Uniform_Without_Reset_Model) {
    RoadRunner rr1((modelAnalysisModelsDir / "uniform_distrib_model.xml").string());
    RoadRunner rr2((modelAnalysisModelsDir / "uniform_distrib_model.xml").string());
    rr1.simulate(0, 5, 50);
    rr2.simulate(0, 5, 50);
    rr1.setSeed(42, false);
    rr2.setSeed(42, false);
    rr1.simulate(5, 10, 50);
    rr2.simulate(5, 10, 50);
    EXPECT_NEAR(rr1.getValue("x"), rr2.getValue("x"), 0.0000001);
}

TEST_F(ModelAnalysisTests, Set_Gillespie_Random_Seed) {
    RoadRunner rr1((modelAnalysisModelsDir / "gillespie_random_seed.xml").string());
    RoadRunner rr2((modelAnalysisModelsDir / "gillespie_random_seed.xml").string());
    rr1.setIntegrator("gillespie");
    rr1.setSeed(-1, false);
    rr1.simulate(0, 10, 2);
    rr2.setIntegrator("gillespie");
    rr2.setSeed(-1, false);
    rr2.simulate(0, 10, 2);
    EXPECT_NE(rr1.getValue("S2"), rr2.getValue("S2"));
}

