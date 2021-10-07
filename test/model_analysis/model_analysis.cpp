#include "rrLogger.h"
#include "rrRoadRunner.h"
#include "rrException.h"
#include "rrStringUtils.h"
#include "gtest/gtest.h"
#include "LLVMExecutableModel.h"
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
    //EXPECT_NEAR(result->Element(0, 1), 550, 1);
    //EXPECT_NEAR(result->Element(21, 1), 492.5, 1);
    //EXPECT_NEAR(result->Element(22, 1), 301.7, 1);
    //EXPECT_NEAR(result->Element(44, 1), 303.3, 1);
    //EXPECT_NEAR(result->Element(120, 1), 400, 1);
    std::cout << *result;
}

TEST_F(ModelAnalysisTests, SimulateGillespieFromNegativeStart_General) {
    //Event:  at S1 > 500: S1 = 300;
    RoadRunner rr((modelAnalysisModelsDir / "negstart_event.xml").string());
    rr.setIntegrator("gillespie");
    rr.getIntegrator()->setValue("variable_step_size", false);
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

TEST_F(ModelAnalysisTests, SimulateGillespieFromNegativeStart_Time) {
    //Event:  at time > -1.1: S1 = 0;
    RoadRunner rr((modelAnalysisModelsDir / "negstart_event_time.xml").string());
    rr.setIntegrator("gillespie");
    rr.getIntegrator()->setValue("variable_step_size", false);
    SimulateOptions opt;
    opt.start = -2;
    opt.duration = 10;
    opt.steps = 120;
    const ls::DoubleMatrix* result = rr.simulate(&opt);

    EXPECT_LE(result->Element(11, 1), 20);
    EXPECT_NEAR(result->Element(24, 1), 100, 50);
}

TEST_F(ModelAnalysisTests, SimulateGillespieFromNegativeStart_Time_Delay) {
    //Event:  at 0.5 after time > -1.1: S1 = 0;
    RoadRunner rr((modelAnalysisModelsDir / "negstart_event_time_delay.xml").string());
    rr.setIntegrator("gillespie");
    rr.getIntegrator()->setValue("variable_step_size", false);
    SimulateOptions opt;
    opt.start = -2;
    opt.duration = 10;
    opt.steps = 120;
    const ls::DoubleMatrix* result = rr.simulate(&opt);

    EXPECT_GE(result->Element(2, 1), 500);
    EXPECT_LE(result->Element(17, 1), 20);
    EXPECT_NEAR(result->Element(24, 1), 66, 30);
    EXPECT_NEAR(result->Element(120, 1), 946, 50);
}

TEST_F(ModelAnalysisTests, SimulateGillespieFromNegativeStart_T0fire) {
    //Event:  at S1 > 500, t0=false: S1 = 300;
    RoadRunner rr((modelAnalysisModelsDir / "negstart_event_t0fire.xml").string());
    rr.setIntegrator("gillespie");
    rr.getIntegrator()->setValue("variable_step_size", false);
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

TEST_F(ModelAnalysisTests, SimulateGillespieFromNegativeStart_Combo) {
    //Events:  E0: at 0.3 after time > -1: S1 = 0;
    //         E1: at time > 1: S1 = 100;
    //         E2: at 0.5 after S1 > 500: S1 = 300;
    RoadRunner rr((modelAnalysisModelsDir / "negstart_event_combo.xml").string());
    rr.setIntegrator("gillespie");
    rr.getIntegrator()->setValue("variable_step_size", false);
    SimulateOptions opt;
    opt.start = -2;
    opt.duration = 10;
    opt.steps = 120;
    const ls::DoubleMatrix* result = rr.simulate(&opt);

    EXPECT_NEAR(result->Element(2, 1), 508, 10);
    EXPECT_NEAR(result->Element(8, 1), 308, 10);
    EXPECT_LE(result->Element(16, 1), 15);
    EXPECT_NEAR(result->Element(36, 1), 187, 20);
    EXPECT_NEAR(result->Element(37, 1), 110, 10);
    EXPECT_NEAR(result->Element(80, 1), 503, 30);
    EXPECT_NEAR(result->Element(86, 1), 304, 10);
    EXPECT_NEAR(result->Element(120, 1), 360, 40);
    for (int i = 1; i <= opt.steps; i++)
    {
        EXPECT_LE(result->Element(i, 1), 580);
        EXPECT_GE(result->Element(i, 1), 0);
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
    rr.reset(SelectionRecord::SelectionType::ALL);
    opt.start = -2;
    ls::DoubleMatrix sneg2result(*(rr.simulate(&opt)));
    rr.reset(SelectionRecord::SelectionType::ALL);
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
    rr.reset(SelectionRecord::SelectionType::ALL);
    opt.start = -2;
    ls::DoubleMatrix sneg2result(*(rr.simulate(&opt)));
    rr.reset(SelectionRecord::SelectionType::ALL);
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
    rr.reset(SelectionRecord::SelectionType::ALL);
    opt.start = -2;
    ls::DoubleMatrix sneg2result(*(rr.simulate(&opt)));
    rr.reset(SelectionRecord::SelectionType::ALL);
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


//I don't think setting the seed is working for this one.
TEST_F(ModelAnalysisTests, DISABLED_NonZeroStarts_Gillespie) {
    RoadRunner rr((modelAnalysisModelsDir / "threestep.xml").string());
    rr.setIntegrator("gillespie");
    rr.getIntegrator()->setValue("variable_step_size", false);
    rr.getIntegrator()->setValue("seed", 1001);
    SimulateOptions opt;
    opt.start = 0;
    opt.duration = 10;
    opt.steps = 50;
    ls::DoubleMatrix s0result(*(rr.simulate(&opt)));
    rr.reset(SelectionRecord::SelectionType::ALL);
    opt.start = -2;
    ls::DoubleMatrix sneg2result(*(rr.simulate(&opt)));
    rr.reset(SelectionRecord::SelectionType::ALL);
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
    rr.reset(SelectionRecord::ALL);
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
    rr.reset(SelectionRecord::ALL);
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
    rr.reset(SelectionRecord::ALL);
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

