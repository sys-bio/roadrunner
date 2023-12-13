
#include "gtest/gtest.h"
#include "rrSelectionRecord.h"
#include "rrRoadRunner.h"
#include "TestModelFactory.h"
#include "RoadRunnerTest.h"
#include "llvm/LLVMException.h"

using namespace rr;
using namespace testing;

/**
 * This is a difficult set of tests to retroactively write.
 *
 */

class SelectionRecordTests: public RoadRunnerTest {
public:
    SelectionRecordTests() {
        //std::cout << TestModelFactory("SimpleFlux")->str() << std::endl;
    };
};


TEST_F(SelectionRecordTests, TIME) {
    TestModel* testModel = TestModelFactory("SimpleFlux");
    RoadRunner rr(testModel->str());
    SelectionRecord record = rr.createSelection("time");
    EXPECT_EQ(record.selectionType, SelectionRecord::TIME);
    EXPECT_EQ(record.index, -1);
    EXPECT_EQ(record.p1, "time");
    EXPECT_EQ(record.p2, "");
    delete testModel;
}

TEST_F(SelectionRecordTests, REACTION_RATE) {
    TestModel* testModel = TestModelFactory("SimpleFlux");
    RoadRunner rr(testModel->str());
    SelectionRecord record = rr.createSelection("_J1");
    EXPECT_EQ(record.selectionType, SelectionRecord::REACTION_RATE);
    EXPECT_EQ(record.index, 1);
    EXPECT_EQ(record.p1, "_J1");
    EXPECT_EQ(record.p2, "");
    delete testModel;
}

TEST_F(SelectionRecordTests, BOUNDARY_CONCENTRATION) {
    TestModel* testModel = TestModelFactory("SimpleFlux");
    RoadRunner rr(testModel->str());
    rr.setBoundary("S1", true, true);
    SelectionRecord record = rr.createSelection("[S1]");
    EXPECT_EQ(record.selectionType, SelectionRecord::BOUNDARY_CONCENTRATION);
    EXPECT_STREQ(record.to_string().c_str(), "[S1]");
    EXPECT_EQ(record.index, 0);
    EXPECT_EQ(record.p1, "S1");
    EXPECT_EQ(record.p2, "");
    delete testModel;
}

TEST_F(SelectionRecordTests, BOUNDARY_AMOUNT) {
    TestModel* testModel = TestModelFactory("SimpleFlux");
    RoadRunner rr(testModel->str());
    rr.setBoundary("S1", true, true);
    SelectionRecord record = rr.createSelection("S1");
    EXPECT_EQ(record.selectionType, SelectionRecord::BOUNDARY_AMOUNT);
    EXPECT_STREQ(record.to_string().c_str(), "S1");
    EXPECT_EQ(record.index, 0);
    EXPECT_EQ(record.p1, "S1");
    EXPECT_EQ(record.p2, "");
    delete testModel;
}

TEST_F(SelectionRecordTests, UNKNOWN) {
    TestModel* testModel = TestModelFactory("SimpleFlux");
    RoadRunner rr(testModel->str());
    EXPECT_THROW(
        SelectionRecord record = rr.createSelection("random");,
        Exception
    );
    delete testModel;
}

TEST_F(SelectionRecordTests, FLOATING_AMOUNT){
    TestModel* testModel = TestModelFactory("SimpleFlux");
    RoadRunner rr(testModel->str());
    SelectionRecord record = rr.createSelection("S2");
    EXPECT_EQ(record.selectionType, SelectionRecord::FLOATING_AMOUNT);
    EXPECT_STREQ(record.to_string().c_str(), "S2");
    EXPECT_EQ(record.index, 1);
    EXPECT_EQ(record.p1, "S2");
    EXPECT_EQ(record.p2, "");
    delete testModel;
}

TEST_F(SelectionRecordTests, FLOATING_CONCENTRATION) {
    TestModel* testModel = TestModelFactory("SimpleFlux");
    RoadRunner rr(testModel->str());
    SelectionRecord record = rr.createSelection("[S2]");
    EXPECT_EQ(record.selectionType, SelectionRecord::FLOATING_CONCENTRATION);
    EXPECT_STREQ(record.to_string().c_str(), "[S2]");
    EXPECT_EQ(record.index, 1);
    EXPECT_EQ(record.p1, "S2");
    EXPECT_EQ(record.p2, "");
    delete testModel;
}

TEST_F(SelectionRecordTests, INITIAL_FLOATING_AMOUNT){
    TestModel* testModel = TestModelFactory("SimpleFlux");
    RoadRunner rr(testModel->str());
    SelectionRecord record = rr.createSelection("init(S2)");
    EXPECT_EQ(record.selectionType, SelectionRecord::INITIAL_FLOATING_AMOUNT);
    EXPECT_STREQ(record.to_string().c_str(), "init(S2)");
    EXPECT_EQ(record.index, 1);
    EXPECT_EQ(record.p1, "S2");
    EXPECT_EQ(record.p2, "");
    delete testModel;
}

TEST_F(SelectionRecordTests, INITIAL_FLOATING_CONCENTRATION) {
    TestModel* testModel = TestModelFactory("SimpleFlux");
    RoadRunner rr(testModel->str());
    SelectionRecord record = rr.createSelection("init([S2])");
    EXPECT_EQ(record.selectionType, SelectionRecord::INITIAL_FLOATING_CONCENTRATION);
    EXPECT_STREQ(record.to_string().c_str(), "init([S2])");
    EXPECT_EQ(record.index, 1);
    EXPECT_EQ(record.p1, "S2");
    EXPECT_EQ(record.p2, "");
    delete testModel;
}

TEST_F(SelectionRecordTests, INITIAL_BOUNDARY_AMOUNT) {
    TestModel* testModel = TestModelFactory("SimpleFlux");
    RoadRunner rr(testModel->str());
    rr.setBoundary("S1", true, true);
    SelectionRecord record = rr.createSelection("init(S1)");
    EXPECT_EQ(record.selectionType, SelectionRecord::INITIAL_BOUNDARY_AMOUNT);
    EXPECT_STREQ(record.to_string().c_str(), "init(S1)");
    EXPECT_EQ(record.index, 0);
    EXPECT_EQ(record.p1, "S1");
    EXPECT_EQ(record.p2, "");
    delete testModel;
}

TEST_F(SelectionRecordTests, INITIAL_BOUNDARY_CONCENTRATION) {
    TestModel* testModel = TestModelFactory("SimpleFlux");
    RoadRunner rr(testModel->str());
    rr.setBoundary("S1", true, true);
    SelectionRecord record = rr.createSelection("init([S1])");
    EXPECT_EQ(record.selectionType, SelectionRecord::INITIAL_BOUNDARY_CONCENTRATION);
    EXPECT_STREQ(record.to_string().c_str(), "init([S1])");
    EXPECT_EQ(record.index, 0);
    EXPECT_EQ(record.p1, "S1");
    EXPECT_EQ(record.p2, "");
    delete testModel;
}

TEST_F(SelectionRecordTests, EIGENVALUE_COMPLEX){
    TestModel* testModel = TestModelFactory("SimpleFlux");
    RoadRunner rr(testModel->str());
    SelectionRecord record = rr.createSelection("eigen(S1)");
    EXPECT_EQ(record.selectionType, SelectionRecord::EIGENVALUE_COMPLEX);
    EXPECT_STREQ(record.to_string().c_str(), "eigen(S1)");
    EXPECT_EQ(record.index, 0);
    EXPECT_EQ(record.p1, "S1");
    EXPECT_EQ(record.p2, "");
    delete testModel;
}

TEST_F(SelectionRecordTests, EIGENVALUE_REAL) {
    TestModel* testModel = TestModelFactory("SimpleFlux");
    RoadRunner rr(testModel->str());
    SelectionRecord record = rr.createSelection("eigenReal(S1)");
    EXPECT_EQ(record.selectionType, SelectionRecord::EIGENVALUE_REAL);
    EXPECT_STREQ(record.to_string().c_str(), "eigenReal(S1)");
    EXPECT_EQ(record.index, 0);
    EXPECT_EQ(record.p1, "S1");
    EXPECT_EQ(record.p2, "");
    delete testModel;
}

TEST_F(SelectionRecordTests, EIGENVALUE_IMAG) {
    TestModel* testModel = TestModelFactory("SimpleFlux");
    RoadRunner rr(testModel->str());
    SelectionRecord record = rr.createSelection("eigenImag(S1)");
    EXPECT_EQ(record.selectionType, SelectionRecord::EIGENVALUE_IMAG);
    EXPECT_STREQ(record.to_string().c_str(), "eigenImag(S1)");
    EXPECT_EQ(record.index, 0);
    EXPECT_EQ(record.p1, "S1");
    EXPECT_EQ(record.p2, "");
    delete testModel;
}

TEST_F(SelectionRecordTests, STOICHIOMETRY){
    TestModel* testModel = TestModelFactory("SimpleFlux");
    RoadRunner rr(testModel->str());
    SelectionRecord record = rr.createSelection("stoich(S1, _J1)");
    EXPECT_EQ(record.selectionType, SelectionRecord::STOICHIOMETRY);
    EXPECT_STREQ(record.to_string().c_str(), "stoich(S1, _J1)");
    EXPECT_EQ(record.index, -1);
    EXPECT_EQ(record.p1, "S1");
    EXPECT_EQ(record.p2, "_J1");
    delete testModel;
}

TEST_F(SelectionRecordTests, CONSERVED_MOIETY){
    RoadRunner rr((rrTestModelsDir_ / "ModelAnalysis" / "conserved_cycle.xml").string());
    rr.setConservedMoietyAnalysis(true);
    SelectionRecord record = rr.createSelection("_CSUM0");
    //Conserved moieties come through as global parameters, and not conserved moieties.
    EXPECT_EQ(record.selectionType, SelectionRecord::GLOBAL_PARAMETER);
    EXPECT_STREQ(record.to_string().c_str(), "_CSUM0");
    EXPECT_EQ(record.index, 1);
    EXPECT_EQ(record.p1, "_CSUM0");
    EXPECT_EQ(record.p2, "");
}

TEST_F(SelectionRecordTests, EVENT){
    RoadRunner rr((rrTestModelsDir_ / "ModelAnalysis" / "event.xml").string());
    EXPECT_THROW(
        SelectionRecord record = rr.createSelection("_E0"),
        Exception
    );
}

TEST_F(SelectionRecordTests, COMPARTMENT) {
    TestModel* testModel = TestModelFactory("SimpleFlux");
    RoadRunner rr(testModel->str());
    SelectionRecord record = rr.createSelection("default_compartment");
    EXPECT_EQ(record.selectionType, SelectionRecord::COMPARTMENT);
    EXPECT_STREQ(record.to_string().c_str(), "default_compartment");
    EXPECT_EQ(record.index, 0);
    EXPECT_EQ(record.p1, "default_compartment");
    EXPECT_EQ(record.p2, "");
    delete testModel;
}

TEST_F(SelectionRecordTests, GLOBAL_PARAMETER) {
    TestModel* testModel = TestModelFactory("SimpleFlux");
    RoadRunner rr(testModel->str());
    SelectionRecord record = rr.createSelection("kf");
    EXPECT_EQ(record.selectionType, SelectionRecord::GLOBAL_PARAMETER);
    EXPECT_STREQ(record.to_string().c_str(), "kf");
    EXPECT_EQ(record.index, 0);
    EXPECT_EQ(record.p1, "kf");
    EXPECT_EQ(record.p2, "");
    delete testModel;
}

TEST_F(SelectionRecordTests, FLOATING_AMOUNT_RATE){
    TestModel* testModel = TestModelFactory("SimpleFlux");
    RoadRunner rr(testModel->str());
    SelectionRecord record = rr.createSelection("S1'");
    EXPECT_EQ(record.selectionType, SelectionRecord::FLOATING_AMOUNT_RATE);
    EXPECT_STREQ(record.to_string().c_str(), "S1'");
    EXPECT_EQ(record.index, 0);
    EXPECT_EQ(record.p1, "S1");
    EXPECT_EQ(record.p2, "");
    delete testModel;
}

TEST_F(SelectionRecordTests, FLOATING_CONCENTRATION_RATE){
    TestModel* testModel = TestModelFactory("SimpleFlux");
    RoadRunner rr(testModel->str());
    SelectionRecord record = rr.createSelection("[S1]'");
    EXPECT_EQ(record.selectionType, SelectionRecord::FLOATING_CONCENTRATION_RATE);
    EXPECT_STREQ(record.to_string().c_str(), "[S1]'");
    EXPECT_EQ(record.index, 0);
    EXPECT_EQ(record.p1, "S1");
    EXPECT_EQ(record.p2, "");
    delete testModel;
}

TEST_F(SelectionRecordTests, GLOBAL_PARAMETER_RATE){
    TestModel* testModel = TestModelFactory("SimpleFlux");
    RoadRunner rr(testModel->str());
    SelectionRecord record = rr.createSelection("kf'");
    EXPECT_EQ(record.selectionType, SelectionRecord::GLOBAL_PARAMETER_RATE);
    EXPECT_STREQ(record.to_string().c_str(), "kf'");
    EXPECT_EQ(record.index, 0);
    EXPECT_EQ(record.p1, "kf");
    EXPECT_EQ(record.p2, "");
    delete testModel;
}

TEST_F(SelectionRecordTests, COMPARTMENT_RATE) {
    TestModel* testModel = TestModelFactory("SimpleFlux");
    RoadRunner rr(testModel->str());
    SelectionRecord record = rr.createSelection("default_compartment'");
    EXPECT_EQ(record.selectionType, SelectionRecord::COMPARTMENT_RATE);
    EXPECT_STREQ(record.to_string().c_str(), "default_compartment'");
    EXPECT_EQ(record.index, 0);
    EXPECT_EQ(record.p1, "default_compartment");
    EXPECT_EQ(record.p2, "");
    delete testModel;
}

TEST_F(SelectionRecordTests, INITIAL_COMPARTMENT){
    TestModel* testModel = TestModelFactory("SimpleFlux");
    RoadRunner rr(testModel->str());
    SelectionRecord record = rr.createSelection("init(default_compartment)");
    EXPECT_EQ(record.selectionType, SelectionRecord::INITIAL_COMPARTMENT);
    EXPECT_STREQ(record.to_string().c_str(), "init(default_compartment)");
    EXPECT_EQ(record.index, 0);
    EXPECT_EQ(record.p1, "default_compartment");
    EXPECT_EQ(record.p2, "");
    delete testModel;
}

TEST_F(SelectionRecordTests, INITIAL_GLOBAL_PARAMETER){
    TestModel* testModel = TestModelFactory("SimpleFlux");
    RoadRunner rr(testModel->str());
    SelectionRecord record = rr.createSelection("init(kf)");
    EXPECT_EQ(record.selectionType, SelectionRecord::INITIAL_GLOBAL_PARAMETER);
    EXPECT_STREQ(record.to_string().c_str(), "init(kf)");
    EXPECT_EQ(record.index, 0);
    EXPECT_EQ(record.p1, "kf");
    EXPECT_EQ(record.p2, "");
    delete testModel;
}

TEST_F(SelectionRecordTests, UNSCALED_ELASTICITY) {
    TestModel* testModel = TestModelFactory("SimpleFlux");
    RoadRunner rr(testModel->str());
    SelectionRecord record = rr.createSelection("uec(_J1, S2)");
    EXPECT_STREQ(record.to_string().c_str(), "uec(_J1, S2)");
    EXPECT_EQ(record.selectionType, SelectionRecord::UNSCALED_ELASTICITY);
    EXPECT_EQ(record.index, -1);
    EXPECT_EQ(record.p1, "_J1");
    EXPECT_EQ(record.p2, "S2");
    delete testModel;
}

TEST_F(SelectionRecordTests, ELASTICITY) {
    TestModel* testModel = TestModelFactory("SimpleFlux");
    RoadRunner rr(testModel->str());
    SelectionRecord record = rr.createSelection("ec(_J1, S2)");
    EXPECT_STREQ(record.to_string().c_str(), "ec(_J1, S2)");
    EXPECT_EQ(record.selectionType, SelectionRecord::ELASTICITY);
    EXPECT_EQ(record.index, -1);
    EXPECT_EQ(record.p1, "_J1");
    EXPECT_EQ(record.p2, "S2");
    delete testModel;
}

TEST_F(SelectionRecordTests, UNSCALED_CONTROL) {
    TestModel* testModel = TestModelFactory("SimpleFlux");
    RoadRunner rr(testModel->str());
    SelectionRecord record = rr.createSelection("ucc(_J1, kf)");
    EXPECT_STREQ(record.to_string().c_str(), "ucc(_J1, kf)");
    EXPECT_EQ(record.selectionType, SelectionRecord::UNSCALED_CONTROL);
    EXPECT_EQ(record.index, -1);
    EXPECT_EQ(record.p1, "_J1");
    EXPECT_EQ(record.p2, "kf");
    delete testModel;
}

TEST_F(SelectionRecordTests, CONTROL) {
    TestModel* testModel = TestModelFactory("SimpleFlux");
    RoadRunner rr(testModel->str());
    SelectionRecord record = rr.createSelection("cc(_J1, kf)");
    EXPECT_STREQ(record.to_string().c_str(), "cc(_J1, kf)");
    EXPECT_EQ(record.selectionType, SelectionRecord::CONTROL);
    EXPECT_EQ(record.index, -1);
    EXPECT_EQ(record.p1, "_J1");
    EXPECT_EQ(record.p2, "kf");
    delete testModel;
}

TEST_F(SelectionRecordTests, AllIDs) {
    RoadRunner rr((rrTestModelsDir_ / "ModelAnalysis" / "conserved_cycle.xml").string());
    //Add a few boundary species:
    rr.addSpeciesAmount("S3", "default_compartment", 2.2, false, true);
    rr.addSpeciesAmount("S4", "default_compartment", 2.2, false, true);
    rr.addSpeciesAmount("S5", "default_compartment", 2.2, false, true);
    rr.addAssignmentRule("S3", "S2+1");
    rr.addRateRule("S4", "S1 - S2");
    rr.addRateRule("default_compartment", "0.1");

    //Set to test _CSUM values.
    rr.setConservedMoietyAnalysis(true);

    list<string> ids;
    vector<string> stoichids;
    rr.getIds(SelectionRecord::ALL, ids);
    for (list<string>::iterator id = ids.begin(); id != ids.end(); id++)
    {
        SelectionRecord record = rr.createSelection(*id);
        if ((*id).find("cc(") < 2) {
            EXPECT_THROW(rr.getValue(record), CoreException);
            std::cout << *id << ": [uncalculatable; no steady state]" << std::endl;
        }
        else if ((*id).find("stoich(") < 2) {
            EXPECT_THROW(rr.getValue(record), rrllvm::LLVMException);
            std::cout << *id << ": [uncalculatable when conserved moieties are on]" << std::endl;
            stoichids.push_back(*id);
        }
        else {
            double val = rr.getValue(record);
            std::cout << *id << ": " << val << std::endl;
        }
    }
    //Set to test stoich values.
    rr.setConservedMoietyAnalysis(false);
    for (size_t sid = 0; sid < stoichids.size(); sid++) {
        SelectionRecord record = rr.createSelection(stoichids[sid]);
        double val = rr.getValue(record);
        std::cout << stoichids[sid] << ": " << val << std::endl;
    }
}

TEST_F(SelectionRecordTests, MODIFY_STEADY_STATE_SELECTIONS_LIST) {
    RoadRunner rr((rrTestModelsDir_ / "ModelAnalysis" / "long_selection_record.xml").string());
    std::string users_favorite_steady_state_selection_record = "x0";
    std::vector<std::string> steadyStateSelectionRecordStrings;
    steadyStateSelectionRecordStrings.push_back(users_favorite_steady_state_selection_record);
    rr.setSteadyStateSelections(steadyStateSelectionRecordStrings);
    std::vector<SelectionRecord> steadyStateSelectionRecords = rr.getSteadyStateSelections();
    EXPECT_EQ(steadyStateSelectionRecords.size(), 1);
    EXPECT_EQ(steadyStateSelectionRecords.front().p1, users_favorite_steady_state_selection_record);
}
