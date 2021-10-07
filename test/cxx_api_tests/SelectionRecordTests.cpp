//
// Created by Ciaran on 20/05/2021.
//

#include "gtest/gtest.h"
#include "rrSelectionRecord.h"
#include "rrRoadRunner.h"
#include "TestModelFactory.h"

using namespace rr;
using namespace testing;

/**
 * This is a difficult set of tests to retroactively write.
 *
 */

class SelectionRecordTests: public ::testing::Test {
public:
    SelectionRecordTests() {
        //std::cout << TestModelFactory("SimpleFlux")->str() << std::endl;
    };
};


TEST_F(SelectionRecordTests, TIME) {
    TestModel* testModel = TestModelFactory("SimpleFlux");
    RoadRunner rr(testModel->str());
    SelectionRecord record = rr.createSelection("time");
    ASSERT_EQ(record.selectionType, SelectionRecord::TIME);
    ASSERT_EQ(record.index, -1);
    delete testModel;
}

TEST_F(SelectionRecordTests, CONCENTRATION) {}

TEST_F(SelectionRecordTests, AMOUNT) {}

TEST_F(SelectionRecordTests, RATE) {
    TestModel* testModel = TestModelFactory("SimpleFlux");
    RoadRunner rr(testModel->str());
    SelectionRecord record = rr.createSelection("_J1");
    ASSERT_EQ(record.selectionType, SelectionRecord::REACTION_RATE);
    std::cout << record << std::endl;
    delete testModel;
}

TEST_F(SelectionRecordTests, BOUNDARY) {
    TestModel* testModel = TestModelFactory("SimpleFlux");
    RoadRunner rr(testModel->str());
    rr.setBoundary("S1", true, true);
    SelectionRecord record = rr.createSelection("[S1]");
    ASSERT_EQ(record.selectionType, SelectionRecord::BOUNDARY_CONCENTRATION);
    std::cout << record << std::endl;
    delete testModel;
}

TEST_F(SelectionRecordTests, UNKNOWN) {
    TestModel* testModel = TestModelFactory("SimpleFlux");
    RoadRunner rr(testModel->str());
    ASSERT_THROW(
        SelectionRecord record = rr.createSelection("random");,
        Exception
    );
    delete testModel;
}

//TEST_F(SelectionRecordTests, FLOATING){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::FLOATING);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::FLOATING);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, _COMPARTMENT){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::_COMPARTMENT);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::_COMPARTMENT);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, _GLOBAL_PARAMETER){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::_GLOBAL_PARAMETER);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::_GLOBAL_PARAMETER);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, REACTION){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::REACTION);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::REACTION);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, INITIAL){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::INITIAL);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::INITIAL);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, CURRENT){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::CURRENT);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::CURRENT);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, UNSCALED){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::UNSCALED);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::UNSCALED);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, ELASTICITY){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::ELASTICITY);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::ELASTICITY);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, CONTROL){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::CONTROL);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::CONTROL);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, EIGENVALUE_REAL){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::EIGENVALUE_REAL);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::EIGENVALUE_REAL);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, ELEMENT){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::ELEMENT);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::ELEMENT);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, STOICHIOMETRY){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::STOICHIOMETRY);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::STOICHIOMETRY);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, UNKNOWN){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::UNKNOWN);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::UNKNOWN);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, DEPENDENT){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::DEPENDENT);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::DEPENDENT);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, INDEPENDENT){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::INDEPENDENT);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::INDEPENDENT);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, CONSERVED_MOIETY){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::CONSERVED_MOIETY);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::CONSERVED_MOIETY);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, STATE_VECTOR){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::STATE_VECTOR);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::STATE_VECTOR);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, EVENT){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::EVENT);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::EVENT);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, EIGENVALUE_IMAG){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::EIGENVALUE_IMAG);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::EIGENVALUE_IMAG);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, ALL){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::ALL);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::ALL);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, EIGENVALUE_COMPLEX){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::EIGENVALUE_COMPLEX);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::EIGENVALUE_COMPLEX);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, UNKNOWN_CONCENTRATION){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::UNKNOWN_CONCENTRATION);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::UNKNOWN_CONCENTRATION);
//    std::cout << record << std::endl;
//    delete testModel;
//}

TEST_F(SelectionRecordTests, COMPARTMENT) {
    TestModel* testModel = TestModelFactory("SimpleFlux");
    RoadRunner rr(testModel->str());
    SelectionRecord record = rr.createSelection("default_compartment");
    ASSERT_EQ(record.selectionType, SelectionRecord::COMPARTMENT);
    std::cout << record << std::endl;
    delete testModel;
}

TEST_F(SelectionRecordTests, GLOBAL_PARAMETER) {
    TestModel* testModel = TestModelFactory("SimpleFlux");
    RoadRunner rr(testModel->str());
    SelectionRecord record = rr.createSelection("kf");
    ASSERT_EQ(record.selectionType, SelectionRecord::GLOBAL_PARAMETER);
    std::cout << record << std::endl;
    delete testModel;
}

TEST_F(SelectionRecordTests, FLOATING_AMOUNT) {
    TestModel* testModel = TestModelFactory("SimpleFlux");
    RoadRunner rr(testModel->str());
    SelectionRecord record = rr.createSelection("S2");
    ASSERT_EQ(record.selectionType, SelectionRecord::FLOATING_AMOUNT);
    ASSERT_STREQ(record.to_string().c_str(), "S2");
    ASSERT_EQ(record.index, 1);
    delete testModel;
}
//
//TEST_F(SelectionRecordTests, INDEPENDENT_FLOATING_AMOUNT){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::INDEPENDENT_FLOATING_AMOUNT);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::INDEPENDENT_FLOATING_AMOUNT);
//    delete testModel;
//    std::cout << record << std::endl;
//}
//
//TEST_F(SelectionRecordTests, DEPENDENT_FLOATING_AMOUNT){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::DEPENDENT_FLOATING_AMOUNT);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::DEPENDENT_FLOATING_AMOUNT);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
TEST_F(SelectionRecordTests, BOUNDARY_AMOUNT) {
    TestModel* testModel = TestModelFactory("SimpleFlux");
    RoadRunner rr(testModel->str());
    rr.setBoundary("S1", true, true);
    SelectionRecord record = rr.createSelection("S1");
    ASSERT_EQ(record.selectionType, SelectionRecord::BOUNDARY_AMOUNT);
    std::cout << record << std::endl;
    delete testModel;
}
//
//TEST_F(SelectionRecordTests, BOUNDARY_CONCENTRATION){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::BOUNDARY_CONCENTRATION);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::BOUNDARY_CONCENTRATION);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
TEST_F(SelectionRecordTests, FLOATING_CONCENTRATION) {
    TestModel* testModel = TestModelFactory("SimpleFlux");
    RoadRunner rr(testModel->str());
    SelectionRecord record = rr.createSelection("[S1]");
    std::cout << record << std::endl;
    ASSERT_EQ(record.selectionType, SelectionRecord::FLOATING_CONCENTRATION);
    ASSERT_STREQ(record.to_string().c_str(), "[S1]");
    ASSERT_EQ(record.index, 0);
    delete testModel;
}

//TEST_F(SelectionRecordTests, FLOATING_AMOUNT_RATE){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::FLOATING_AMOUNT_RATE);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::FLOATING_AMOUNT_RATE);
//    std::cout << record << std::endl;
//    delete testModel;
//}

//TEST_F(SelectionRecordTests, FLOATING_CONCENTRATION_RATE){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::FLOATING_CONCENTRATION_RATE);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::FLOATING_CONCENTRATION_RATE);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, GLOBAL_PARAMETER_RATE){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::GLOBAL_PARAMETER_RATE);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::GLOBAL_PARAMETER_RATE);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
TEST_F(SelectionRecordTests, REACTION_RATE) {
    TestModel* testModel = TestModelFactory("SimpleFlux");
    RoadRunner rr(testModel->str());
    SelectionRecord record = rr.createSelection("_J1");
    ASSERT_EQ(record.selectionType, SelectionRecord::REACTION_RATE);
    std::cout << record << std::endl;
    delete testModel;
}
//
//TEST_F(SelectionRecordTests, INITIAL_AMOUNT){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::INITIAL_AMOUNT);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::INITIAL_AMOUNT);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, INITIAL_FLOATING_AMOUNT){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::INITIAL_FLOATING_AMOUNT);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::INITIAL_FLOATING_AMOUNT);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, INITIAL_CONCENTRATION){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::INITIAL_CONCENTRATION);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::INITIAL_CONCENTRATION);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, INITIAL_FLOATING_CONCENTRATION){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::INITIAL_FLOATING_CONCENTRATION);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::INITIAL_FLOATING_CONCENTRATION);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, INITIAL_COMPARTMENT){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::INITIAL_COMPARTMENT);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::INITIAL_COMPARTMENT);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, INITIAL_GLOBAL_PARAMETER){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::INITIAL_GLOBAL_PARAMETER);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::INITIAL_GLOBAL_PARAMETER);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, DEPENDENT_INITIAL_GLOBAL_PARAMETER){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::DEPENDENT_INITIAL_GLOBAL_PARAMETER);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::DEPENDENT_INITIAL_GLOBAL_PARAMETER);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, UNSCALED_ELASTICITY){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::UNSCALED_ELASTICITY);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::UNSCALED_ELASTICITY);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, UNSCALED_CONTROL){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::UNSCALED_CONTROL);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::UNSCALED_CONTROL);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, UNKNOWN_ELEMENT){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::UNKNOWN_ELEMENT);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::UNKNOWN_ELEMENT);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, ALL_INDEPENDENT){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::ALL_INDEPENDENT);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::ALL_INDEPENDENT);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, ALL_DEPENDENT){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::ALL_DEPENDENT);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::ALL_DEPENDENT);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, ALL_INDEPENDENT_AMOUNT){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::ALL_INDEPENDENT_AMOUNT);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::ALL_INDEPENDENT_AMOUNT);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, ALL_DEPENDENT_AMOUNT){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::ALL_DEPENDENT_AMOUNT);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::ALL_DEPENDENT_AMOUNT);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, ALL_INDEPENDENT_CONCENTRATION){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::ALL_INDEPENDENT_CONCENTRATION);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::ALL_INDEPENDENT_CONCENTRATION);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, ALL_DEPENDENT_CONCENTRATION){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::ALL_DEPENDENT_CONCENTRATION);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::ALL_DEPENDENT_CONCENTRATION);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, MODEL_STATE){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::MODEL_STATE);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::MODEL_STATE);
//    std::cout << record << std::endl;
//    delete testModel;
//}
//
//TEST_F(SelectionRecordTests, SBML_INITIALIZE){
//    TestModel* testModel = TestModelFactory("SimpleFlux");
//    RoadRunner rr(testModel->str());
//    SelectionRecord record = rr.createSelection( SelectionRecord::SBML_INITIALIZE);
//    ASSERT_EQ(record.selectionType,         SelectionRecord::SBML_INITIALIZE);
//    std::cout << record << std::endl;
//    delete testModel;
//}
