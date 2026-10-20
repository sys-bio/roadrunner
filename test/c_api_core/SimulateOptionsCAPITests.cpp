#include "gtest/gtest.h"
#include "C/rrc_api.h"
#include "C/rrc_types.h"
#include "RoadRunnerTest.h"
#include <filesystem>

using namespace rrc;
using std::filesystem::path;

// https://github.com/sys-bio/roadrunner/issues/1339
// setTimeStart/setTimeEnd/setNumPoints/setTimes mutate a persistent
// SimulateOptions field-by-field across repeated simulate() calls, so
// simulateEx/simulateTimes/gillespieEx and friends need to clear whichever
// of 'times' or the steps-triplet a previous call left behind. These tests
// exercise that directly through the C API's handle-based functions.
class SimulateOptionsCAPITests : public RoadRunnerTest {
public:
    path modelFile = rrTestModelsDir_ / "ModelAnalysis" / "no_steady_state.xml";
    path gillespieModelFile = rrTestModelsDir_ / "ModelAnalysis" / "gillespie_random_seed.xml";
};

TEST_F(SimulateOptionsCAPITests, simulateExThenSimulateTimes) {
    RRHandle rr = createRRInstance();
    ASSERT_TRUE(loadSBMLFromFileE(rr, modelFile.string().c_str(), true));

    // Steps-based simulation: start=0, duration=9, steps=3 (4 points).
    RRCDataPtr result = simulateEx(rr, 0, 9, 4);
    ASSERT_NE(result, nullptr);
    freeRRCData(result);
    ASSERT_TRUE(reset(rr));

    // Switch to a times-based simulation via simulateTimes.
    double times[] = { 0, 1, 2, 9 };
    result = simulateTimes(rr, times, 4);
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result->RSize, 4);
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(result->Data[i * result->CSize + 0], times[i]);
    }
    freeRRCData(result);
    freeRRInstance(rr);
}

TEST_F(SimulateOptionsCAPITests, simulateTimesThenSimulateEx) {
    RRHandle rr = createRRInstance();
    ASSERT_TRUE(loadSBMLFromFileE(rr, modelFile.string().c_str(), true));

    double times[] = { 0, 1, 5, 10, 20 };
    RRCDataPtr result = simulateTimes(rr, times, 5);
    ASSERT_NE(result, nullptr);
    freeRRCData(result);
    ASSERT_TRUE(reset(rr));

    // Switch back to a steps-based simulation via simulateEx.
    result = simulateEx(rr, 0, 6, 4);
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result->RSize, 4);
    EXPECT_EQ(result->Data[0 * result->CSize + 0], 0);
    EXPECT_EQ(result->Data[3 * result->CSize + 0], 6);
    freeRRCData(result);
    freeRRInstance(rr);
}

TEST_F(SimulateOptionsCAPITests, largerSimulateTimesAfterSimulateTimes) {
    RRHandle rr = createRRInstance();
    ASSERT_TRUE(loadSBMLFromFileE(rr, modelFile.string().c_str(), true));

    double times1[] = { 0, 5, 10 };
    RRCDataPtr result = simulateTimes(rr, times1, 3);
    ASSERT_NE(result, nullptr);
    freeRRCData(result);
    ASSERT_TRUE(reset(rr));

    double times2[] = { 0, 1, 2, 3, 10 };
    result = simulateTimes(rr, times2, 5);
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result->RSize, 5);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(result->Data[i * result->CSize + 0], times2[i]);
    }
    freeRRCData(result);
    freeRRInstance(rr);
}

TEST_F(SimulateOptionsCAPITests, smallerSimulateTimesAfterSimulateTimes) {
    RRHandle rr = createRRInstance();
    ASSERT_TRUE(loadSBMLFromFileE(rr, modelFile.string().c_str(), true));

    double times1[] = { 0, 1, 2, 3, 10 };
    RRCDataPtr result = simulateTimes(rr, times1, 5);
    ASSERT_NE(result, nullptr);
    freeRRCData(result);
    ASSERT_TRUE(reset(rr));

    double times2[] = { 0, 5, 10 };
    result = simulateTimes(rr, times2, 3);
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result->RSize, 3);
    for (int i = 0; i < 3; i++) {
        EXPECT_EQ(result->Data[i * result->CSize + 0], times2[i]);
    }
    freeRRCData(result);
    freeRRInstance(rr);
}

// Same bug, exercised via the raw setters instead of the bundling *Ex
// functions -- setTimeStart/setTimeEnd/setNumPoints/setTimes must each
// clear the complementary state themselves, since simulateEx/simulateTimes
// are just call-sequences of these.
TEST_F(SimulateOptionsCAPITests, rawSettersTimesAfterSteps) {
    RRHandle rr = createRRInstance();
    ASSERT_TRUE(loadSBMLFromFileE(rr, modelFile.string().c_str(), true));

    ASSERT_TRUE(setTimeStart(rr, 0));
    ASSERT_TRUE(setTimeEnd(rr, 9));
    ASSERT_TRUE(setNumPoints(rr, 4));
    RRCDataPtr result = simulate(rr);
    ASSERT_NE(result, nullptr);
    freeRRCData(result);
    ASSERT_TRUE(reset(rr));

    double times[] = { 0, 1, 2, 9 };
    ASSERT_TRUE(setTimes(rr, times, 4));
    result = simulate(rr);
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result->RSize, 4);
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(result->Data[i * result->CSize + 0], times[i]);
    }
    freeRRCData(result);
    freeRRInstance(rr);
}

TEST_F(SimulateOptionsCAPITests, rawSettersStepsAfterTimes) {
    RRHandle rr = createRRInstance();
    ASSERT_TRUE(loadSBMLFromFileE(rr, modelFile.string().c_str(), true));

    double times[] = { 0, 1, 5, 10, 20 };
    ASSERT_TRUE(setTimes(rr, times, 5));
    RRCDataPtr result = simulate(rr);
    ASSERT_NE(result, nullptr);
    freeRRCData(result);
    ASSERT_TRUE(reset(rr));

    ASSERT_TRUE(setTimeStart(rr, 0));
    ASSERT_TRUE(setTimeEnd(rr, 6));
    ASSERT_TRUE(setNumPoints(rr, 4));
    result = simulate(rr);
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result->RSize, 4);
    EXPECT_EQ(result->Data[0 * result->CSize + 0], 0);
    EXPECT_EQ(result->Data[3 * result->CSize + 0], 6);
    freeRRCData(result);
    freeRRInstance(rr);
}

// gillespieOnGridEx has the identical bundling pattern (setTimeStart/
// setTimeEnd/setNumPoints, then a bare gillespieOnGrid call) and needs the
// identical fix.
TEST_F(SimulateOptionsCAPITests, gillespieOnGridExStepsAfterSimulateTimes) {
    RRHandle rr = createRRInstance();
    ASSERT_TRUE(loadSBMLFromFileE(rr, gillespieModelFile.string().c_str(), true));

    double times[] = { 0, 1, 5, 10, 20 };
    RRCDataPtr result = simulateTimes(rr, times, 5);
    ASSERT_NE(result, nullptr);
    freeRRCData(result);
    ASSERT_TRUE(reset(rr));

    result = gillespieOnGridEx(rr, 0, 6, 4);
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result->RSize, 4);
    EXPECT_EQ(result->Data[0 * result->CSize + 0], 0);
    EXPECT_EQ(result->Data[3 * result->CSize + 0], 6);
    freeRRCData(result);
    freeRRInstance(rr);
}
