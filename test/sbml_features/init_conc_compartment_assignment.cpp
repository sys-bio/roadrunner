// Regression tests for issue #1319:
// A species' initialConcentration was ignored (the species came out 0, or
// inf/NaN depending on the rule) when its compartment size was governed by a
// time-dependent assignment rule. The species amount was being converted from
// concentration using a compartment volume evaluated against the pre-simulation
// sentinel time (-inf) rather than the initial time (t = 0).
#include "gtest/gtest.h"
#include "rrRoadRunner.h"
#include "rrExecutableModel.h"
#include <cmath>
#include <string>
#include "RoadRunnerTest.h"

using namespace testing;
using namespace rr;
using std::string;

class InitConcCompartmentAssignment : public RoadRunnerTest {
public:
    InitConcCompartmentAssignment() = default;

    // Build the minimal model from the issue, with the compartment V driven by
    // the supplied assignment-rule expression. X has initialConcentration 0.05.
    static string comp_init_model(const string &compartmentRule) {
        return
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
            "<sbml xmlns=\"http://www.sbml.org/sbml/level3/version1/core\" level=\"3\" version=\"1\">\n"
            "  <model id=\"init_bug\">\n"
            "    <listOfCompartments>\n"
            "      <compartment id=\"V\" spatialDimensions=\"3\" size=\"1\" constant=\"false\"/>\n"
            "    </listOfCompartments>\n"
            "    <listOfSpecies>\n"
            "      <species id=\"X\" compartment=\"V\" initialConcentration=\"0.05\"\n"
            "               hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
            "    </listOfSpecies>\n"
            "    <listOfRules>\n"
            "      <assignmentRule variable=\"V\">\n"
            "        <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n"
            + compartmentRule +
            "        </math>\n"
            "      </assignmentRule>\n"
            "    </listOfRules>\n"
            "  </model>\n"
            "</sbml>\n";
    }
};

// V := 1 + time. At t = 0, V = 1, so X should start at its stated
// initialConcentration of 0.05 (amount 0.05). Before the fix this read 0.
TEST_F(InitConcCompartmentAssignment, OnePlusTime) {
    RoadRunner rr(comp_init_model(
        "          <apply><plus/>\n"
        "            <cn type=\"integer\">1</cn>\n"
        "            <csymbol encoding=\"text\" definitionURL=\"http://www.sbml.org/sbml/symbols/time\">time</csymbol>\n"
        "          </apply>\n"));
    EXPECT_DOUBLE_EQ(rr.getValue("V"), 1.0);
    EXPECT_DOUBLE_EQ(rr.getValue("[X]"), 0.05);
    EXPECT_DOUBLE_EQ(rr.getValue("X"), 0.05);   // amount = conc * V(0)
}

// V := 2 / (1 + time). At t = 0, V = 2. [X] must still be 0.05
// (amount 0.10). Before the fix this read -inf.
TEST_F(InitConcCompartmentAssignment, TwoOverOnePlusTime) {
    RoadRunner rr(comp_init_model(
        "          <apply><divide/>\n"
        "            <cn type=\"integer\">2</cn>\n"
        "            <apply><plus/>\n"
        "              <cn type=\"integer\">1</cn>\n"
        "              <csymbol encoding=\"text\" definitionURL=\"http://www.sbml.org/sbml/symbols/time\">time</csymbol>\n"
        "            </apply>\n"
        "          </apply>\n"));
    EXPECT_DOUBLE_EQ(rr.getValue("V"), 2.0);
    EXPECT_DOUBLE_EQ(rr.getValue("[X]"), 0.05);
    EXPECT_DOUBLE_EQ(rr.getValue("X"), 0.10);
}

// V := 2 + 0*time. At t = 0, V = 2. [X] must still be 0.05.
// Before the fix this read NaN (0 * (-inf) = NaN).
TEST_F(InitConcCompartmentAssignment, TwoPlusZeroTime) {
    RoadRunner rr(comp_init_model(
        "          <apply><plus/>\n"
        "            <cn type=\"integer\">2</cn>\n"
        "            <apply><times/>\n"
        "              <cn type=\"integer\">0</cn>\n"
        "              <csymbol encoding=\"text\" definitionURL=\"http://www.sbml.org/sbml/symbols/time\">time</csymbol>\n"
        "            </apply>\n"
        "          </apply>\n"));
    EXPECT_DOUBLE_EQ(rr.getValue("V"), 2.0);
    EXPECT_DOUBLE_EQ(rr.getValue("[X]"), 0.05);
    EXPECT_DOUBLE_EQ(rr.getValue("X"), 0.10);
}

// The initial concentration must survive an explicit reset as well, since reset
// re-runs the initial-condition evaluation.
TEST_F(InitConcCompartmentAssignment, SurvivesReset) {
    RoadRunner rr(comp_init_model(
        "          <apply><plus/>\n"
        "            <cn type=\"integer\">1</cn>\n"
        "            <csymbol encoding=\"text\" definitionURL=\"http://www.sbml.org/sbml/symbols/time\">time</csymbol>\n"
        "          </apply>\n"));
    rr.getModel()->setTime(10.0);
    EXPECT_DOUBLE_EQ(rr.getValue("V"), 11.0);
    EXPECT_DOUBLE_EQ(rr.getValue("[X]"), 0.05 / 11.0);
    //NOTE:  even though we're not resetting time, time gets reset anyway.  If we ever fix this, the following will need to be changed.
    rr.reset(SelectionRecord::INITIAL_FLOATING_AMOUNT);
    EXPECT_DOUBLE_EQ(rr.getValue("time"), 0.0);
    EXPECT_DOUBLE_EQ(rr.getValue("V"), 1.0);
    EXPECT_DOUBLE_EQ(rr.getValue("[X]"), 0.05);
    EXPECT_DOUBLE_EQ(rr.getValue("X"), 0.05);
}
