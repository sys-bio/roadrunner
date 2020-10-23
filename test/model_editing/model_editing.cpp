#include "gtest/gtest.h"
#include "rrRoadRunner.h"
#include "rrException.h"
#include "rrUtils.h"
#include "rrTestSuiteModelSimulation.h"
#include "sbml/SBMLTypes.h"
#include "sbml/SBMLReader.h"
#include "../test_util.h"

using namespace testing;
using namespace rr;
using namespace std;


extern string gRRTestDir;
extern string gRROutputDir;

bool RunModelEditingTest(void(*modification)(RoadRunner*), std::string version = "l2v4");
bool RunTestModelFromScratch(void(*generate)(RoadRunner*), std::string version = "l2v4");
bool RunTestWithEdit(const string& version, int caseNumber, void(*edit)(RoadRunner*, libsbml::SBMLDocument*), std::string editName);
void readdAllReactions(RoadRunner* rri, libsbml::SBMLDocument* doc);
void readdAllSpecies(RoadRunner* rri, libsbml::SBMLDocument* doc);
void readdAllEvents(RoadRunner* rri, libsbml::SBMLDocument* doc);
void removeAndReaddAllSpecies(RoadRunner* rri, libsbml::SBMLDocument* doc);
void removeAndReaddAllReactions(RoadRunner* rri, libsbml::SBMLDocument* doc);
void removeAndReaddAllCompartments(RoadRunner* rri, libsbml::SBMLDocument* doc);


TEST(MODEL_EDITING_TEST_SUITE, CLEAR_MODEL_1)
{
    RoadRunner rri;
    rri.addCompartment("compartment", 3.14159);
    rri.addSpecies("S1", "compartment", 1.0, false, false);
    ASSERT_TRUE(rri.isModelLoaded());
    rri.clearModel();
    ASSERT_TRUE(!rri.isModelLoaded());
    rri.addCompartment("compartment", 3.14159);
    rri.addSpecies("S2", "compartment", 2.0, false, false);
    ASSERT_TRUE(rri.getNumberOfFloatingSpecies() == 1);
}
TEST(MODEL_EDITING_TEST_SUITE, ALLOW_EVENT_ASSIGNMENT_AND_RATE_LAW_1)
{
    try
    {
        RoadRunner rri;
        rri.addParameter("k1", 0.0);
        rri.addRateRule("k1", "1");
        rri.addEvent("e1", true, "k1 > 1");
        rri.addEventAssignment("e1", "k1", "1");
    }
    catch (std::exception& ex)
    {
        string error = ex.what();
        cerr << "ALLOW_EVENT_ASSIGNMENT_AND_RATE_LAW_1 Exception: " << error << endl;
        ASSERT_TRUE(false);
    }
}
TEST(MODEL_EDITING_TEST_SUITE, ALLOW_EVENT_ASSIGNMENT_AND_RATE_LAW_2)
{
    try
    {
        RoadRunner rri;
        rri.addParameter("k1", 0.0);
        rri.addEvent("e1", true, "k1 > 1");
        rri.addEventAssignment("e1", "k1", "1");
        rri.addRateRule("k1", "1");
    }
    catch (std::exception& ex)
    {
        string error = ex.what();
        cerr << "ALLOW_EVENT_ASSIGNMENT_AND_RATE_LAW_2 Exception: " << error << endl;
        ASSERT_TRUE(false);
    }
}
TEST(MODEL_EDITING_TEST_SUITE, ALLOW_EVENT_ASSIGNMENT_AND_INITIAL_ASSIGNMENT_1)
{
    try
    {
        RoadRunner rri;
        rri.addParameter("k1", 0.0);
        rri.addRateRule("k1", "1");
        rri.addInitialAssignment("k1", "1");
    }
    catch (std::exception& ex)
    {
        string error = ex.what();
        cerr << "ALLOW_EVENT_ASSIGNMENT_AND_RATE_LAW_1 Exception: " << error << endl;
        ASSERT_TRUE(false);
    }
}
TEST(MODEL_EDITING_TEST_SUITE, ALLOW_EVENT_ASSIGNMENT_AND_INITIAL_ASSIGNMENT_2)
{
    try
    {
        RoadRunner rri;
        rri.addParameter("k1", 0.0);
        rri.addInitialAssignment("k1", "1");
        rri.addRateRule("k1", "1");
    }
    catch (std::exception& ex)
    {
        string error = ex.what();
        cerr << "ALLOW_EVENT_ASSIGNMENT_AND_RATE_LAW_2 Exception: " << error << endl;
        ASSERT_TRUE(false);
    }
}
TEST(MODEL_EDITING_TEST_SUITE, SET_CONSERVED_MOIETY_ANALYSIS)
{
    RoadRunner rri(gRRTestDir + "sbml-test-suite/semantic/00001/00001-sbml-l2v4.xml");
    rri.setGlobalParameterByIndex(0, 100);
    rri.setConservedMoietyAnalysis(true);

    ASSERT_TRUE(rri.getGlobalParameterByIndex(0) == 100);
}

TEST(MODEL_EDITING_TEST_SUITE, REMOVE_ASSIGNMENT_RULE_2)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri) {
        rri->removeRules("k1", true);
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, REMOVE_ASSIGNMENT_RULE_3)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri) {
        rri->removeRules("k1", false);
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, REMOVE_RATE_RULE_1)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri) {
        rri->removeRules("k1", false);
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, REMOVE_RATE_RULE_2)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri) {
        rri->removeRules("k1", true);
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, REMOVE_RATE_RULE_3)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri) {
        rri->simulate();
        rri->removeRules("k1", false);
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, REMOVE_RATE_RULE_4)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri) {
        rri->simulate();
        rri->removeRules("k1", true);
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, REMOVE_INITIAL_ASSIGNMENT_RULE_1)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri) {
        rri->removeInitialAssignment("k1");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, REMOVE_INITIAL_ASSIGNMENT_RULE_2)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri) {
        rri->simulate();
        rri->removeInitialAssignment("k1");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, ADD_REACTION_1)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri) {
        rri->addReaction("reaction2", { "S2" }, { "S1" }, "k1*S2", true);
        }));
}
TEST(MODEL_EDITING_TEST_SUITE, REMOVE_REACTION_1)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri) {
        rri->removeReaction("reaction2");
        }));
}
TEST(MODEL_EDITING_TEST_SUITE, ADD_REACTION_2)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri) {
        rri->addReaction("reaction2", { "2S1", "S2" }, { "2S2" }, "compartment * k1 * S1 + compartment * k1 * S2", true);
        }));
}
TEST(MODEL_EDITING_TEST_SUITE, REMOVE_REACTION_2)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri) {
        rri->removeReaction("reaction2");
        }));
}
TEST(MODEL_EDITING_TEST_SUITE, REMOVE_REACTION_3)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri) {
        rri->removeSpecies("S2");
        }));
}
TEST(MODEL_EDITING_TEST_SUITE, ADD_SPECIES_1)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->addSpecies("S3", "compartment", 0.0015, true, false);
        }));
}
TEST(MODEL_EDITING_TEST_SUITE, ADD_REACTION_3)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->addSpecies("S3", "compartment", 0.015, false, false);
            rri->addReaction("reaction3", { "S2" }, { "S3" }, "k2*S2");
        }));
}
TEST(MODEL_EDITING_TEST_SUITE, ADD_REACTION_4)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->addSpecies("S3", "compartment", 0.015, false);
            rri->addReaction("reaction3", { "S2" }, { "S3" }, "k2*S2");
        }));
}
TEST(MODEL_EDITING_TEST_SUITE, ADD_REACTION_5)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->addSpecies("S3", "compartment", 0.15, false, false);
            rri->addReaction("reaction3", { "S3" }, { "S1" }, "k2*S3");
        }));
}
TEST(MODEL_EDITING_TEST_SUITE, ADD_REACTION_6)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->addReaction("reaction3", { "S3" }, { "S4" }, "C * k3 * S3 * time");
        }));
}
TEST(MODEL_EDITING_TEST_SUITE, REMOVE_SPECIES_1)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->removeSpecies("S2", false);
            rri->addSpecies("S3", "compartment", 0.00030, false, false);
            rri->addReaction("reaction1", { "S1" }, { "S3" }, "k1*S1", false);
            rri->addReaction("reaction2", { "S3" }, { "S1" }, "k2*S3", true);
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, REMOVE_SPECIES_2)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->removeSpecies("S4");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, ADD_EVENT_1)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->addEvent("event1", true, "S1 > 0.00015", false);
            rri->addEventAssignment("event1", "S1", "1", true);
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, EVENT_PRIORITY_1)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->addEvent("_E0", true, "time >= 0.99", false);
            rri->addPriority("_E0", "1", false);
            rri->addEventAssignment("_E0", "S1", "4", false);
            rri->addEventAssignment("_E0", "S2", "5", false);
            rri->addEventAssignment("_E0", "S3", "6", false);

            rri->addEvent("_E1", true, "time >= 0.99", false);
            rri->addPriority("_E1", "0", false);
            rri->addEventAssignment("_E1", "S1", "1", false);
            rri->addEventAssignment("_E1", "S2", "2", false);
            rri->addEventAssignment("_E1", "S3", "3", true);
        }, "l3v1"));
}

TEST(MODEL_EDITING_TEST_SUITE, EVENT_DELAY_1)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->addDelay("event1", "1");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, REMOVE_EVENT_1)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->removeEvent("event1");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, ADD_PARAMETER_1)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->addParameter("k1", 0.75, false);
            rri->addReaction("reaction1", { "S1", "S2" }, { "S3" }, "k1*S1*S2");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, ADD_COMPARTMENT_1)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->addCompartment("compartment", 1, false);
            rri->addReaction("reaction1", { "S1", "S2" }, { "2S2" }, "compartment * k1 * S1 * S2", true);
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, ADD_COMPARTMENT_2)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->addCompartment("compartment", 1, false);
            rri->addReaction("reaction1", { "S1" }, { "S3" }, "compartment * k1 * S1");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, ADD_ASSIGNMENT_RULE_1)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->setBoundary("S1", true);
            rri->addAssignmentRule("S1", "7");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, ADD_ASSIGNMENT_RULE_2)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->setBoundary("S1", true);
            rri->addAssignmentRule("S1", "7");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, ADD_ASSIGNMENT_RULE_3)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->setBoundary("S3", true);
            rri->addAssignmentRule("S3", "k1 * S2");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, ADD_RATE_RULE_1)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->setBoundary("S1", true);
            rri->addRateRule("S1", "7");
        }));
}


TEST(MODEL_EDITING_TEST_SUITE, ADD_RATE_RULE_2)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->setBoundary("S1", true);
            rri->setBoundary("S2", true);
            rri->addRateRule("S1", "-1 * k1 * S1", false);
            rri->addRateRule("S2", "k1 * S1", true);
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, ADD_RATE_RULE_3)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->setBoundary("S1", true);
            rri->setBoundary("S2", true);
            rri->addRateRule("S1", "-1 * k1 * S1");
            rri->addRateRule("S2", "k1 * S1");
        }));
}


TEST(MODEL_EDITING_TEST_SUITE, SET_KINETIC_LAW_1)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->setKineticLaw("reaction2", "compartment * k2 * S3");
            ASSERT_TRUE(rri->getKineticLaw("reaction2") == "compartment * k2 * S3");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, SET_KINETIC_LAW_2)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->setKineticLaw("reaction1", "compartment * k1 * S1 * S2");
            rri->setKineticLaw("reaction2", "compartment * k2 * S3 * S4");
            ASSERT_TRUE(rri->getKineticLaw("reaction1") == "compartment * k1 * S1 * S2");
            ASSERT_TRUE(rri->getKineticLaw("reaction2") == "compartment * k2 * S3 * S4");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, SET_KINETIC_LAW_3)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->setKineticLaw("reaction1", "compartment * k1 * S1 * S2", false);
            rri->setKineticLaw("reaction2", "compartment * k2 * S3 * S4", true);
            ASSERT_TRUE(rri->getKineticLaw("reaction1") == "compartment * k1 * S1 * S2");
            ASSERT_TRUE(rri->getKineticLaw("reaction2") == "compartment * k2 * S3 * S4");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, TRANSFORM_1)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->removeSpecies("S1", true);
            rri->removeSpecies("S2", true);
            rri->addSpecies("S1", "compartment", 0.001);
            rri->addSpecies("S2", "compartment", 0.001);
            rri->addSpecies("S3", "compartment", 0.002);
            rri->addSpecies("S4", "compartment", 0.001);
            rri->removeParameter("k1", true);
            rri->addParameter("k1", 750, true);
            rri->addParameter("k2", 250, true);
            rri->addReaction("reaction1", { "S1", "S2" }, { "S3","S4" }, "compartment * k1 * S1 * S2", true);
            rri->addReaction("reaction2", { "S3", "S4" }, { "S1", "S2" }, "compartment * k2 * S3 * S4", true);
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, PAUSE_1)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->simulate();
            rri->addReaction("reaction3", { "S3" }, { "S2" }, "compartment * k2 * S3 * S4");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, PAUSE_2)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->simulate();
            rri->removeReaction("reaction2");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, PAUSE_3)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->simulate();
            rri->addSpecies("S5", "compartment", 0.001, true);
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, PAUSE_4)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->simulate();
            rri->removeSpecies("S4");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, PAUSE_5)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->simulate();
            rri->setKineticLaw("reaction3", "sin(S2)");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, PAUSE_6)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->simulate();
            rri->addEvent("event1", true, "S2 > 0.0004", false);
            rri->addEventAssignment("event1", "S1", "0.1", true);
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, PAUSE_7)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->simulate();
            rri->removeEvent("event1");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, PAUSE_8)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->simulate();
            rri->addRateRule("k1", "7");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, PAUSE_9)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->simulate();
            rri->removeRules("k1");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, PAUSE_10)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->simulate();
            rri->addDelay("event1", "0.2");
        }));
}


TEST(MODEL_EDITING_TEST_SUITE, ADD_EVENT_ASSIGNMENT_1)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->addEventAssignment("event2", "S3", "1");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, REMOVE_RULES_2)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->simulate();
            rri->removeRules("S4");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, ADD_EVENT_ASSIGNMENT_2)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->simulate();
            rri->addEventAssignment("event1", "S3", "1");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, ADD_RATE_RULE_PARAMETER_1)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->addRateRule("k1", "1000000");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, ADD_RATE_RULE_PARAMETER_2)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->addParameter("k1", 1000000);
            rri->addRateRule("k1", "1000000");
            rri->addReaction("reaction1", { "S1", "S2" }, { "S3","S4" }, "compartment * k1 * S1 * S2");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, ADD_TRIGGER_1)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->addTrigger("event1", "S1 < 0.75");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, ADD_TRIGGER_2)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->addTrigger("event1", "S1 < 0.75");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, REMOVE_COMPARTMENT_1)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->removeCompartment("compartment");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, REMOVE_COMPARTMENT_2)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->removeCompartment("compartment1");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, REMOVE_EVENT_ASSIGNMENT_1)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->removeEventAssignments("event1", "S2");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, REMOVE_EVENT_ASSIGNMENT_2)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->removeEventAssignments("event1", "S1");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, REMOVE_PARAMETER_1)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->removeParameter("k4");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, ADD_DELAY_2)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->addDelay("event1", "1", false);
            rri->addDelay("event2", "0.5");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, ADD_PRIORITY_2)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->addEvent("Rinc", true, "(time - reset) >= 0.01", false);
            rri->addEventAssignment("Rinc", "reset", "time", false);
            rri->addEventAssignment("Rinc", "R", "R + 0.01", false);
            rri->addPriority("Rinc", "-1", false);

            rri->addEvent("Qinc2", true, "(time - reset2) >= 0.01", false);
            rri->addEventAssignment("Qinc2", "reset2", "time", false);
            rri->addEventAssignment("Qinc2", "Q2", "Q2 + 0.01", false);
            rri->addPriority("Qinc2", "-1", false);

            rri->addEvent("Rinc2", true, "(time - reset2) >= 0.01", false);
            rri->addEventAssignment("Rinc2", "reset2", "time", false);
            rri->addEventAssignment("Rinc2", "R2", "R2 + 0.01", false);
            rri->addPriority("Rinc2", "1");

            rri->setSelections({ "R", "Q", "reset", "R2", "Q2", "reset2" });
        }, "l3v1"));
}

TEST(MODEL_EDITING_TEST_SUITE, REMOVE_PARAM_RECURSE_1)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->removeParameter("k2");
        }));
}


TEST(MODEL_EDITING_TEST_SUITE, REMOVE_PARAM_RECURSE_2)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->removeParameter("k4");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, FROM_SCRATCH_1)
{
    ASSERT_TRUE(RunTestModelFromScratch([](RoadRunner* rri)
        {
            rri->addCompartment("compartment", 1);
            rri->addSpecies("S1", "compartment", 0.00015, true);
            rri->addSpecies("S2", "compartment", 0);
            rri->addParameter("k1", 1);
            rri->addReaction("reaction1", { "S1" }, { "S2" }, "compartment * k1 * S1");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, FROM_SCRATCH_2)
{
    ASSERT_TRUE(RunTestModelFromScratch([](RoadRunner* rri)
        {
            rri->addCompartment("compartment", 1);
            rri->addSpecies("S1", "compartment", 1, true);
            rri->addSpecies("S2", "compartment", 0, true);
            rri->addParameter("k1", 1);
            rri->addReaction("reaction1", { "S1" }, { "S2" }, "compartment * k1 * S1");
            rri->addEvent("event1", true, "S1 < 0.1");
            rri->addDelay("event1", "1");
            rri->addEventAssignment("event1", "S1", "1");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, FROM_SCRATCH_3)
{
    ASSERT_TRUE(RunTestModelFromScratch([](RoadRunner* rri)
        {
            rri->addCompartment("compartment", 1);
            rri->addSpecies("S1", "compartment", 0, true, true);
            rri->addRateRule("S1", "7");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, FROM_SCRATCH_4)
{
    ASSERT_TRUE(RunTestModelFromScratch([](RoadRunner* rri)
        {
            rri->addCompartment("compartment", 1);
            rri->addSpecies("S1", "compartment", 7, true, true);
            rri->addAssignmentRule("S1", "7");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, FROM_SCRATCH_5)
{
    ASSERT_TRUE(RunTestModelFromScratch([](RoadRunner* rri)
        {
            rri->addCompartment("compartment", 1);
            rri->addSpecies("S1", "compartment", 1, true);
            rri->addSpecies("S2", "compartment", 1.5e-15, true);
            rri->addSpecies("S3", "compartment", 1, true, true);
            rri->addParameter("k1", 0.75);
            rri->addParameter("k2", 50);
            rri->addAssignmentRule("S3", "k1*S2");
            rri->addReaction("reaction1", { "S1" }, { "S2" }, "compartment * k2 * S1");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, FROM_SCRATCH_1_L2V5)
{
    RoadRunner rri(2, 5);
    rri.addCompartment("compartment", 1);
    rri.addSpecies("S1", "compartment", 0.00015, true);
    rri.addSpecies("S2", "compartment", 0, true);
    rri.addParameter("k1", 1);
    rri.addReaction("reaction1", { "S1" }, { "S2" }, "compartment * k1 * S1");
    ASSERT_TRUE(testValidateSBML(rri.getCurrentSBML()));
}

TEST(MODEL_EDITING_TEST_SUITE, FROM_SCRATCH_1_L3V2)
{
    RoadRunner rri;
    rri.addCompartment("compartment", 1);
    rri.addSpecies("S1", "compartment", 0.00015);
    rri.addSpecies("S2", "compartment", 0);
    rri.addParameter("k1", 1);
    rri.addReaction("reaction1", { "S1" }, { "S2" }, "compartment * k1 * S1");
    ASSERT_TRUE(testValidateSBML(rri.getCurrentSBML()));
}

TEST(MODEL_EDITING_TEST_SUITE, FROM_SCRATCH_2_L3V2)
{
    RoadRunner rri;
    rri.addCompartment("compartment", 1);
    rri.addSpecies("S1", "compartment", 0.00015);
    rri.addSpecies("S2", "compartment", 0);
    rri.addSpecies("S3", "compartment", 0);
    rri.addParameter("k1", 1);
    rri.addReaction("reaction1", { "S1" }, { "S2" }, "compartment * k1 * S1 * S3");
    ASSERT_TRUE(testValidateSBML(rri.getCurrentSBML()));
}

TEST(MODEL_EDITING_TEST_SUITE, FROM_SCRATCH_3_L3V2)
{
    RoadRunner rri;
    rri.addCompartment("compartment", 1);
    rri.addSpecies("S1", "compartment", 0.00015);
    rri.addSpecies("S2", "compartment", 0);
    rri.addSpecies("S3", "compartment", 0);
    rri.addParameter("k1", 1);
    rri.addReaction("reaction1", { "S1" }, { "S2" }, "S3");
    ASSERT_TRUE(testValidateSBML(rri.getCurrentSBML()));
}

TEST(MODEL_EDITING_TEST_SUITE, FROM_SCRATCH_2_VALID_L3V1)
{
    RoadRunner rri(3, 1);
    rri.addCompartment("compartment", 1);
    rri.addSpecies("S1", "compartment", 0.00015);
    rri.addSpecies("S2", "compartment", 0);
    rri.addSpecies("S3", "compartment", 0);
    rri.addParameter("k1", 1);
    rri.addReaction("reaction1", { "S1" }, { "S2" }, "compartment * k1 * S1 * S3");
    ASSERT_TRUE(testValidateSBML(rri.getCurrentSBML()));
}

TEST(MODEL_EDITING_TEST_SUITE, FROM_SCRATCH_3_VALID_L3V1)
{
    RoadRunner rri(3, 1);
    rri.addCompartment("compartment", 1);
    rri.addSpecies("S1", "compartment", 0.00015);
    rri.addSpecies("S2", "compartment", 0);
    rri.addSpecies("S3", "compartment", 0);
    rri.addParameter("k1", 1);
    rri.addReaction("reaction1", { "S1" }, { "S2" }, "S3");
    ASSERT_TRUE(testValidateSBML(rri.getCurrentSBML()));
}


TEST(MODEL_EDITING_TEST_SUITE, FROM_SCRATCH_1_L3V1)
{
    ASSERT_TRUE(RunTestModelFromScratch([](RoadRunner* rri)
        {
            rri->addCompartment("compartment", 1);
            rri->addSpecies("S1", "compartment", 0.00015);
            rri->addSpecies("S2", "compartment", 0);
            rri->addParameter("k1", 1);
            rri->addReaction("reaction1", { "S1" }, { "S2" }, "compartment * k1 * S1");
        }, "l3v1"));
}

TEST(MODEL_EDITING_TEST_SUITE, FROM_SCRATCH_2_L3V1)
{
    ASSERT_TRUE(RunTestModelFromScratch([](RoadRunner* rri)
        {
            rri->addCompartment("compartment", 1);
            rri->addSpecies("S1", "compartment", 1);
            rri->addSpecies("S2", "compartment", 0);
            rri->addParameter("k1", 1);
            rri->addReaction("reaction1", { "S1" }, { "S2" }, "compartment * k1 * S1");
            rri->addEvent("event1", true, "S1 < 0.1");
            rri->addDelay("event1", "1");
            rri->addEventAssignment("event1", "S1", "1");
        }, "l3v1"));
}

TEST(MODEL_EDITING_TEST_SUITE, FROM_SCRATCH_3_L3V1)
{
    ASSERT_TRUE(RunTestModelFromScratch([](RoadRunner* rri)
        {
            rri->addCompartment("compartment", 1);
            rri->addSpecies("S1", "compartment", 0, false, true);
            rri->addRateRule("S1", "7");
        }, "l3v1"));
}

TEST(MODEL_EDITING_TEST_SUITE, FROM_SCRATCH_4_L3V1)
{
    ASSERT_TRUE(RunTestModelFromScratch([](RoadRunner* rri)
        {
            rri->addCompartment("compartment", 1);
            rri->addSpecies("S1", "compartment", 7, false, true);
            rri->addAssignmentRule("S1", "7");
        }, "l3v1"));
}

TEST(MODEL_EDITING_TEST_SUITE, FROM_SCRATCH_5_L3V1)
{
    ASSERT_TRUE(RunTestModelFromScratch([](RoadRunner* rri)
        {
            rri->addCompartment("compartment", 1);
            rri->addSpecies("S1", "compartment", 1);
            rri->addSpecies("S2", "compartment", 1.5e-15);
            rri->addSpecies("S3", "compartment", 1, false, true);
            rri->addParameter("k1", 0.75);
            rri->addParameter("k2", 50);
            rri->addAssignmentRule("S3", "k1*S2");
            rri->addReaction("reaction1", { "S1" }, { "S2" }, "compartment * k2 * S1");
        }, "l3v1"));
}

TEST(MODEL_EDITING_TEST_SUITE, FROM_SCRATCH_6)
{
    ASSERT_TRUE(RunTestModelFromScratch([](RoadRunner* rri)
        {
            rri->addParameter("Q", 0);
            rri->addParameter("R", 0);
            rri->addParameter("reset", 0);
            rri->addParameter("Q2", 0);
            rri->addParameter("R2", 0);
            rri->addParameter("reset2", 0);

            rri->addEvent("Qinc", true, "(time - reset) >= 0.01", false);
            rri->addEventAssignment("Qinc", "reset", "time", false);
            rri->addEventAssignment("Qinc", "Q", "Q + 0.01", false);
            rri->addPriority("Qinc", "1");

            rri->addEvent("Rinc", true, "(time - reset) >= 0.01", false);
            rri->addEventAssignment("Rinc", "reset", "time", false);
            rri->addEventAssignment("Rinc", "R", "R + 0.01", false);
            rri->addPriority("Rinc", "-1", false);

            rri->addEvent("Qinc2", true, "(time - reset2) >= 0.01", false);
            rri->addEventAssignment("Qinc2", "reset2", "time", false);
            rri->addEventAssignment("Qinc2", "Q2", "Q2 + 0.01", false);
            rri->addPriority("Qinc2", "-1", false);

            rri->addEvent("Rinc2", true, "(time - reset2) >= 0.01", false);
            rri->addEventAssignment("Rinc2", "reset2", "time", false);
            rri->addEventAssignment("Rinc2", "R2", "R2 + 0.01", false);
            rri->addPriority("Rinc2", "1");
        }, "l3v1"));
}

TEST(MODEL_EDITING_TEST_SUITE, SET_PERSISTENT_1)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->setPersistent("event1", true, false);
            rri->setPersistent("event2", true);
        }, "l3v1"));
}

TEST(MODEL_EDITING_TEST_SUITE, SET_PERSISTENT_2)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->setPersistent("event1", false);
        }, "l3v1"));
}

TEST(MODEL_EDITING_TEST_SUITE, SET_CONSTANT_1)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->setConstant("k1", false, false);
            rri->addRateRule("k1", "0.5");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, SET_CONSTANT_2)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->setConstant("S1", false, false);
            rri->setBoundary("S1", true);
            rri->addRateRule("S1", "7");
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, SET_HAS_ONLY_SUBSTANCE_UNITS_1)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->setHasOnlySubstanceUnits("S1", true, false);
            rri->setHasOnlySubstanceUnits("S2", true);
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, SET_HAS_ONLY_SUBSTANCE_UNITS_2)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->setHasOnlySubstanceUnits("S1", true, false);
            rri->setHasOnlySubstanceUnits("S2", true, false);
            rri->setHasOnlySubstanceUnits("S3", true, false);
            rri->setHasOnlySubstanceUnits("S4", true);
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, SET_HAS_ONLY_SUBSTANCE_UNITS_3)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->setHasOnlySubstanceUnits("S1", false, false);
            rri->setHasOnlySubstanceUnits("S2", false);
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, SET_INITIAL_CONCENTRATION_1)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->addSpecies("S1", "C", 0.0);
            rri->setInitConcentration("S1", 0.0004, false);
            rri->addSpecies("S2", "C", 0.0);
            rri->setInitConcentration("S2", 0.00048, false);
            rri->addSpecies("S3", "C", 0.0);
            rri->setInitConcentration("S3", 0.0008, false);
            rri->addSpecies("S4", "C", 0.0);
            rri->setInitConcentration("S4", 0.0004);

            rri->addReaction("reaction1", { "S1", "S2" }, { "S3", "S4" }, "C * k1 * S1 * S2", false);
            rri->addReaction("reaction2", { "S3", "S4" }, { "S1", "S2" }, "C * k2 * S3 * S4");

            rri->addEvent("event1", true, "S4 > S2", false);
            rri->addEventAssignment("event1", "S1", "1/5000", false);
            rri->addEventAssignment("event1", "S4", "1/5000", true);
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, SET_INITIAL_CONCENTRATION_2)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->setInitConcentration("S1", 0.0004, false);
            rri->setInitConcentration("S2", 0.00048, false);
            rri->setInitConcentration("S3", 0.0008, false);
            rri->setInitConcentration("S4", 0.0004);
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, SET_INITIAL_AMOUNT_1)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->setInitAmount("S1", 0.00015, false);
            rri->setInitAmount("S2", 0);
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, SET_INITIAL_AMOUNT_2)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->addSpecies("S1", "compartment");
            rri->setInitAmount("S1", 0.00015, false);
            rri->addSpecies("S2", "compartment");
            rri->setInitAmount("S2", 0);

            rri->addReaction("reaction1", { "S1" }, { "S2" }, "compartment * k1 * S1");
            rri->reset();
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, SET_REVERSIBLE_1)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->setReversible("reaction2", true);
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, SET_REVERSIBLE_2)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->setReversible("reaction1", false);
        }));
}

TEST(MODEL_EDITING_TEST_SUITE, SET_TRIGGER_INITIAL_VALUE_1)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->setTriggerInitialValue("event1", false);
        }, "l3v1"));
}

TEST(MODEL_EDITING_TEST_SUITE, SET_TRIGGER_INITIAL_VALUE_2)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->setTriggerInitialValue("event1", true);
        }, "l3v1"));
}

TEST(MODEL_EDITING_TEST_SUITE, SET_TRIGGER_INITIAL_VALUE_3)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            rri->setTriggerInitialValue("event1", true);
        }, "l3v1"));
}

TEST(MODEL_EDITING_TEST_SUITE, RETAIN_ABSOLUTE_TOLERANCES_1)
{
    RoadRunner* rri = new RoadRunner(gRRTestDir + "sbml-test-suite/semantic/00001/00001-sbml-l2v4.xml");
    rri->getIntegrator()->setIndividualTolerance("S1", 5.0);
    rri->getIntegrator()->setIndividualTolerance("S2", 3.0);
    rri->removeSpecies("S2");
    EXPECT_TRUE(rri->getIntegrator()->getConcentrationTolerance().size() == 1) << "RETAIN_ABSOLUTE_TOLERANCES_1 failed, rri->getIntegrator()->getConcentrationTolerance().size() != 1" << endl;

    EXPECT_TRUE(rri->getIntegrator()->getConcentrationTolerance()[0] == 5.0) << "RETAIN_ABSOLUTE_TOLERANCES_1 failed, rri->getIntegrator()->getConcentrationTolerance()[0] == 5.0" << endl;
    delete rri;
}

TEST(MODEL_EDITING_TEST_SUITE, RETAIN_ABSOLUTE_TOLERANCES_2)
{
    RoadRunner* rri = new RoadRunner(gRRTestDir + "sbml-test-suite/semantic/00001/00001-sbml-l2v4.xml");
    rri->getIntegrator()->setIndividualTolerance("S1", 5.0);
    rri->getIntegrator()->setIndividualTolerance("S2", 3.0);
    rri->removeSpecies("S1");
    EXPECT_TRUE(rri->getIntegrator()->getConcentrationTolerance().size() == 1) << "RETAIN_ABSOLUTE_TOLERANCES_2 failed, rri->getIntegrator()->getConcentrationTolerance().size() != 1" << endl;

    EXPECT_TRUE(rri->getIntegrator()->getConcentrationTolerance()[0] == 3.0) << "RETAIN_ABSOLUTE_TOLERANCES_2 failed, rri->getIntegrator()->getConcentrationTolerance()[0] == 3.0" << endl;
    delete rri;
}

TEST(MODEL_EDITING_TEST_SUITE, READD_SPECIES)
{
    //Remove and readd all the species from some SBML models that have no rate rules
    for (int i : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22
        , 23, 24, 25, 26, 27, 28, 29, 33, 34, 35})
    {
        EXPECT_TRUE(RunTestWithEdit("l2v4", i, removeAndReaddAllSpecies, "removeAndReaddAllSpecies")) << "Remove and re-add all species for SBML Test " << i << " failed";
    }
}
TEST(MODEL_EDITING_TEST_SUITE, READD_REACTION)
{
    for (int i = 1; i <= 38; i++)
    {
        EXPECT_TRUE(RunTestWithEdit("l2v4", i, removeAndReaddAllReactions, "removeAndReaddAllReactions")) << "SBML Test " << i << " failed";
    }
}

TEST(MODEL_EDITING_TEST_SUITE, READD_COMPARTMENTS)
{
    for (int i : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22
        , 23, 24, 25, 26, 27, 28, 29, 33, 34, 35})
    {
        EXPECT_TRUE(RunTestWithEdit("l2v4", i, removeAndReaddAllCompartments, "removeAndReaddAllCompartments")) << "SBML Test " << i << " failed";
    }
}

TEST(MODEL_EDITING_TEST_SUITE, ONE_ASSIGNMENT_RULE)
{
    ASSERT_TRUE(RunModelEditingTest([](RoadRunner* rri)
        {
            // Do nothing. This is here for a convenient test of issue #610, where
            // reaction rates should be updated due to assignmentRules but are not
            // due to optimization.
        }));
}



bool RunModelEditingTest(void(*modification)(RoadRunner*), std::string version)
{
	bool result(false);
	RoadRunner rr;
    UnitTest::GetInstance()->current_test_info()->test_case_name();

	string testName(UnitTest::GetInstance()->current_test_info()->name());
	string suiteName(UnitTest::GetInstance()->current_test_info()->test_suite_name());

	libsbml::SBMLDocument* doc = nullptr;

	try
	{
		rr.getIntegrator()->setValue("stiff", false);

		if (!createFolder(gRROutputDir))
		{
			string msg("Failed creating output folder for data output: " + gRROutputDir);
			throw(rr::Exception(msg));
		}
		//Create subfolder for data output
		string dataOutputFolder = joinPath(gRROutputDir, testName);

		if (!createFolder(dataOutputFolder))
		{
			string msg("Failed creating output folder for data output: " + dataOutputFolder);
			throw(rr::Exception(msg));
		}

		TestSuiteModelSimulation simulation(dataOutputFolder);

		simulation.UseEngine(&rr);

		//Read SBML models.....
		simulation.SetCaseNumber(0);

		string modelFilePath = getModelDir(gRRTestDir, suiteName, testName);
		string modelFileName = testName + "-sbml-" + version + ".xml";
		string settingsFileName = testName + "-settings.txt";

		//The following will load and compile and simulate the sbml model in the file
		simulation.SetModelFilePath(modelFilePath);
		simulation.SetModelFileName(modelFileName);
		simulation.ReCompileIfDllExists(true);
		simulation.CopyFilesToOutputFolder();
		rr.setConservedMoietyAnalysis(false);

		libsbml::SBMLReader reader;
		std::string fullPath = modelFilePath + "/" + modelFileName;
		doc = reader.readSBML(fullPath);

		if (!simulation.LoadSBMLFromFile())
		{
			throw(Exception("Failed loading sbml from file"));
		}


		//Check first if file exists first
		if (!fileExists(fullPath))
		{
			Log(Logger::LOG_ERROR) << "sbml file " << fullPath << " not found";
			throw(Exception("No such SBML file: " + fullPath));
		}


		LoadSBMLOptions opt;

		// don't generate cache for models
		opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::RECOMPILE;

		opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::MUTABLE_INITIAL_CONDITIONS;

		opt.modelGeneratorOpt = opt.modelGeneratorOpt & ~LoadSBMLOptions::READ_ONLY;

		opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::OPTIMIZE_CFG_SIMPLIFICATION;

		opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::OPTIMIZE_GVN;


		rr.load(fullPath, &opt);

		//Then read settings file if it exists..
		if (!simulation.LoadSettings(joinPath(modelFilePath, settingsFileName)))
		{
			throw(Exception("Failed loading simulation settings"));
		}
		modification(&rr);
		//Then Simulate model
		if (!simulation.Simulate())
		{
			throw(Exception("Failed running simulation"));
		}

		//Write result
		if (!simulation.SaveResult())
		{
			//Failed to save data
			throw(Exception("Failed saving result"));
		}

		if (!simulation.LoadReferenceData(modelFilePath + "/" + testName + "-results.csv"))
		{
			throw(Exception("Failed Loading reference data"));
		}

		simulation.CreateErrorData();
		result = simulation.Pass();
		result = simulation.SaveAllData() && result;
		result = simulation.SaveModelAsXML(dataOutputFolder) && result;
		result = testValidateSBML(rr.getCurrentSBML()) && result;
	}
	catch (std::exception& ex)
	{
		string error = ex.what();
		cerr << "Case " << testName << ": Exception: " << error << endl;
		if (doc) {
			delete doc;
		}
		return false;
	}

	delete doc;

	return result;
}

bool RunTestModelFromScratch(void(*generate)(RoadRunner*), std::string version)
{
	bool result(false);
	int level = version.at(1) - '0';
	int versionNum = version.at(3) - '0';
	RoadRunner rr(level, versionNum);


    string testName(UnitTest::GetInstance()->current_test_info()->name());
    string suiteName(UnitTest::GetInstance()->current_test_info()->test_suite_name());

	try
	{
		rr.getIntegrator()->setValue("stiff", false);

		if (!createFolder(gRROutputDir))
		{
			string msg("Failed creating output folder for data output: " + gRROutputDir);
			throw(rr::Exception(msg));
		}
		//Create subfolder for data output
		string dataOutputFolder = joinPath(gRROutputDir, testName);

		if (!createFolder(dataOutputFolder))
		{
			string msg("Failed creating output folder for data output: " + dataOutputFolder);
			throw(rr::Exception(msg));
		}

		TestSuiteModelSimulation simulation(dataOutputFolder);

		simulation.UseEngine(&rr);

		//Read SBML models.....
        simulation.SetCaseNumber(0);

        string modelFilePath = getModelDir(gRRTestDir, suiteName, testName);
        string modelFileName = testName + "-sbml-" + version + ".xml";
        string settingsFileName = testName + "-settings.txt";

		//The following will load and compile and simulate the sbml model in the file
		simulation.SetModelFilePath(modelFilePath);
		simulation.SetModelFileName(modelFileName);
		simulation.ReCompileIfDllExists(true);
		simulation.CopyFilesToOutputFolder();
		rr.setConservedMoietyAnalysis(false);

		libsbml::SBMLReader reader;
		std::string fullPath = modelFilePath + "/" + modelFileName;

		//Check first if file exists first
		if (!fileExists(fullPath))
		{
			Log(Logger::LOG_ERROR) << "sbml file " << fullPath << " not found";
			throw(Exception("No such SBML file: " + fullPath));
		}


		LoadSBMLOptions opt;

		// don't generate cache for models
		opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::RECOMPILE;

		opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::MUTABLE_INITIAL_CONDITIONS;

		opt.modelGeneratorOpt = opt.modelGeneratorOpt & ~LoadSBMLOptions::READ_ONLY;

		opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::OPTIMIZE_CFG_SIMPLIFICATION;

		opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::OPTIMIZE_GVN;



		//Then read settings file if it exists..
		if (!simulation.LoadSettings(joinPath(modelFilePath, settingsFileName)))
		{
			throw(Exception("Failed loading simulation settings"));
		}
		generate(&rr);
		//Then Simulate model
		if (!simulation.Simulate())
		{
			throw(Exception("Failed running simulation"));
		}

		//Write result
		if (!simulation.SaveResult())
		{
			//Failed to save data
			throw(Exception("Failed saving result"));
		}

		if (!simulation.LoadReferenceData(modelFilePath + "/" + testName + "-results.csv"))
		{
			throw(Exception("Failed Loading reference data"));
		}

		simulation.CreateErrorData();
		result = simulation.Pass();
		result = simulation.SaveAllData() && result;
		result = simulation.SaveModelAsXML(dataOutputFolder) && result;
		result = testValidateSBML(rr.getCurrentSBML()) && result;
	}
	catch (std::exception& ex)
	{
		string error = ex.what();
		cerr << "Case " << testName << ": Exception: " << error << endl;
		return false;
	}

	return result;
}

bool RunTestWithEdit(const string& version, int caseNumber, void(*edit)(RoadRunner*, libsbml::SBMLDocument*), std::string editName)
{
    //Log(Logger::LOG_NOTICE) << "Running Test: " << editName << ", " << caseNumber << endl;
    bool result(false);
	RoadRunner rr;

	//Setup environment
	//setTempFolder(gRR, gTempFolder.c_str());
	libsbml::SBMLDocument* doc;

	try
	{
        string dataOutputFolder = joinPath(gRROutputDir, editName);
        string dummy;
		string logFileName;
		string settingsFileName;

		rr.getIntegrator()->setValue("stiff", false);

		//Create a log file name
		createTestSuiteFileNameParts(caseNumber, ".log", dummy, logFileName, settingsFileName, dummy);
		if (!createFolder(dataOutputFolder))
		{
			string msg("Failed creating output folder for data output: " + dataOutputFolder);
			throw(rr::Exception(msg));
		}
		//Create subfolder for data output
		dataOutputFolder = joinPath(dataOutputFolder, getTestSuiteSubFolderName(caseNumber));

		if (!createFolder(dataOutputFolder))
		{
			string msg("Failed creating output folder for data output: " + dataOutputFolder);
			throw(rr::Exception(msg));
		}

		TestSuiteModelSimulation simulation(dataOutputFolder);

		simulation.UseEngine(&rr);

		//Read SBML models.....
		string modelFilePath(gRRTestDir + "sbml-test-suite/semantic/");
		string modelFileName;

		simulation.SetCaseNumber(caseNumber);
		createTestSuiteFileNameParts(caseNumber, "-sbml-" + version + ".xml", modelFilePath, modelFileName, settingsFileName, dummy);

		//The following will load and compile and simulate the sbml model in the file
		simulation.SetModelFilePath(modelFilePath);
		simulation.SetModelFileName(modelFileName);
		simulation.ReCompileIfDllExists(true);
		simulation.CopyFilesToOutputFolder();
		//setTempFolder(gRR, simulation.GetDataOutputFolder().c_str());
		rr.setConservedMoietyAnalysis(false);

		libsbml::SBMLReader reader;
		std::string fullPath = modelFilePath + "/" + modelFileName;
		doc = reader.readSBML(fullPath);

		if (!simulation.LoadSBMLFromFile())
		{
			throw(Exception("Failed loading sbml from file"));
		}


		//Check first if file exists first
		if (!fileExists(fullPath))
		{
			Log(Logger::LOG_ERROR) << "sbml file " << fullPath << " not found";
			throw(Exception("No such SBML file: " + fullPath));
		}

		LoadSBMLOptions opt;

		// don't generate cache for models
		opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::RECOMPILE;

		opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::MUTABLE_INITIAL_CONDITIONS;

		opt.modelGeneratorOpt = opt.modelGeneratorOpt & ~LoadSBMLOptions::READ_ONLY;

		opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::OPTIMIZE_CFG_SIMPLIFICATION;

		opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::OPTIMIZE_GVN;


		rr.load(fullPath, &opt);

		//Then read settings file if it exists..
		string settingsOveride("");
		if (!simulation.LoadSettings(settingsOveride))
		{
			throw(Exception("Failed loading simulation settings"));
		}
		//Perform the model editing action
		edit(&rr, doc);
		//Then Simulate model
		if (!simulation.Simulate())
		{
			throw(Exception("Failed running simulation"));
		}

		//Write result
		if (!simulation.SaveResult())
		{
			//Failed to save data
			throw(Exception("Failed saving result"));
		}

		if (!simulation.LoadReferenceData())
		{
			throw(Exception("Failed Loading reference data"));
		}

		simulation.CreateErrorData();
		result = simulation.Pass();
		result = simulation.SaveAllData() && result;
		result = simulation.SaveModelAsXML(dataOutputFolder) && result;
		result = testValidateSBML(rr.getCurrentSBML()) && result;
	}
	catch (std::exception& ex)
	{
		string error = ex.what();
		cerr << "Case " << caseNumber << ": Exception: " << error << endl;
		delete doc;
		return false;
	}

	delete doc;

	return result;
}

void readdAllReactions(RoadRunner* rri, libsbml::SBMLDocument* doc)
{
	libsbml::ListOfReactions* reactionsToAdd = doc->getModel()->getListOfReactions();
	std::vector<std::string> currReactionIds = rri->getReactionIds();
	for (int i = 0; i < reactionsToAdd->size(); i++)
	{
		libsbml::Reaction* next = reactionsToAdd->get(i);
		if (std::find(currReactionIds.begin(), currReactionIds.end(), next->getId()) ==
			currReactionIds.end()) {
			char* nextSBML = next->toSBML();
			rri->addReaction(nextSBML);
			free(nextSBML);
		}
	}
}

void readdAllSpecies(RoadRunner* rri, libsbml::SBMLDocument* doc)
{
	libsbml::ListOfSpecies* speciesToAdd = doc->getModel()->getListOfSpecies();
	std::vector<std::string> currSpeciesIds = rri->getBoundarySpeciesIds();

	for (std::string s : rri->getFloatingSpeciesIds())
		currSpeciesIds.push_back(s);

	libsbml::Species* next;
	for (int i = 0; i < speciesToAdd->size(); i++)
	{
		next = speciesToAdd->get(i);
		if (std::find(currSpeciesIds.begin(), currSpeciesIds.end(), next->getId()) == currSpeciesIds.end())
		{
			rri->addSpecies(next->getId(), next->getCompartment(), next->getInitialAmount(), next->getHasOnlySubstanceUnits(), next->getBoundaryCondition(), next->getSubstanceUnits());
		}
	}
}

void readdAllEvents(RoadRunner* rri, libsbml::SBMLDocument* doc)
{
	libsbml::ListOfEvents* eventsToAdd = doc->getModel()->getListOfEvents();

	libsbml::Event* next;
	for (int i = 0; i < eventsToAdd->size(); i++)
	{
		next = eventsToAdd->get(i);
		char* L3_str = libsbml::SBML_formulaToL3String(next->getTrigger()->getMath());
		rri->addEvent(next->getId(), next->getUseValuesFromTriggerTime(), L3_str);
		free(L3_str);
		if (next->isSetDelay())
		{
			L3_str = libsbml::SBML_formulaToL3String(next->getDelay()->getMath());
			rri->addDelay(next->getId(), L3_str);
			free(L3_str);
		}
		libsbml::ListOfEventAssignments* nextAssignments = next->getListOfEventAssignments();
		for (int j = 0; j < nextAssignments->size(); j++)
		{
			libsbml::EventAssignment* nextEA = nextAssignments->get(j);
			L3_str = libsbml::SBML_formulaToL3String(nextEA->getMath());
			rri->addEventAssignment(next->getId(), nextEA->getVariable(), L3_str);
			free(L3_str);
		}
	}
}

/*
* Pre: the model must not contain any rules reliant on species values
*/
void removeAndReaddAllSpecies(RoadRunner* rri, libsbml::SBMLDocument* doc)
{
	//Remove all species
	std::vector<std::string> floatingSpeciesIds = rri->getFloatingSpeciesIds();
	for (std::string sid : floatingSpeciesIds)
	{
		rri->removeSpecies(sid);
	}

	std::vector<std::string> boundarySpeciesIds = rri->getBoundarySpeciesIds();
	for (std::string sid : boundarySpeciesIds)
	{
		rri->removeSpecies(sid);
	}

	//Readd all species
	//libsbml::ListOfSpecies *speciesToAdd = doc->getModel()->getListOfSpecies();
	//if (speciesToAdd->size() > 0)
	//{
	//	libsbml::Species *next;
	//	for (int i = 0; i < speciesToAdd->size(); i++)
	//	{
	//		next = speciesToAdd->get(i);
	//		rri->addSpecies(next->getId(), next->getCompartment(), next->getInitialConcentration(), "concentration", false);
	//	}
	//}
	readdAllSpecies(rri, doc);

	readdAllReactions(rri, doc);

	readdAllEvents(rri, doc);
}

void removeAndReaddAllReactions(RoadRunner* rri, libsbml::SBMLDocument* doc)
{
	std::vector<std::string> reactionIds = rri->getReactionIds();
	for (std::string rid : reactionIds)
	{
		rri->removeReaction(rid);
	}
	readdAllReactions(rri, doc);
}

void removeAndReaddAllCompartments(RoadRunner* rri, libsbml::SBMLDocument* doc)
{
	std::vector<std::string> compartmentIds = rri->getCompartmentIds();
	for (std::string cid : compartmentIds)
	{
		rri->removeCompartment(cid, false);
	}

	libsbml::ListOfCompartments* compartmentsToAdd = doc->getModel()->getListOfCompartments();
	for (int i = 0; i < compartmentsToAdd->size(); i++)
	{
		libsbml::Compartment* next = compartmentsToAdd->get(i);
		rri->addCompartment(next->getId(), next->getVolume(), true);
	}
	readdAllSpecies(rri, doc);
	readdAllReactions(rri, doc);
	readdAllEvents(rri, doc);
}
