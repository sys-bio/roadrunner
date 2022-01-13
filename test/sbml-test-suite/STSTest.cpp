//
// Created by Ciaran on 17/12/2021.
//

#include "gtest/gtest.h"
#include "STS.h"

/**
 * A little contrived to test the interface to a test suite,
 * but it needs doing...
 */

class SemanticSTSModelTests : public ::testing::Test {
public:
    SemanticSTSModelTests() = default;
};

TEST_F(SemanticSTSModelTests, getRootDirectory) {
    SemanticSTSModel s(3);
    std::filesystem::path p = s.getRootDirectory();
    ASSERT_TRUE(std::filesystem::exists(p));
}

TEST_F(SemanticSTSModelTests, getModelDescriptionFile) {
    SemanticSTSModel s(3);
    std::filesystem::path p = s.getModelDescriptionFile();
    ASSERT_TRUE(std::filesystem::exists(p));
}

TEST_F(SemanticSTSModelTests, readModelDescriptionFile) {
    std::string expected = "(* \n"
                           "\n"
                           "category:      Test\n"
                           "synopsis:      Basic single forward reaction with two species in one compartment\n"
                           "componentTags: Compartment, Species, Reaction, Parameter \n"
                           "testTags:      Amount\n"
                           "testType:      TimeCourse\n"
                           "levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2\n"
                           "generatedBy:   Analytic\n"
                           "\n"
                           "The model contains one compartment called \"compartment\".  There are two\n"
                           "species called S1 and S2 and one parameter called k1.  The model contains\n"
                           "one reaction:\n"
                           "\n"
                           "[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |\n"
                           "| S1 -> S2 | $k1 * S1 * compartment$ |]\n"
                           "\n"
                           "The model does not contain any rules.\n"
                           "\n"
                           "The initial conditions are as follows:\n"
                           "\n"
                           "[{width:30em,margin: 1em auto}|       |*Value*          |*Units*        |\n"
                           "|Initial amount of S1                |$1.5 \\x 10^-4$  |mole           |\n"
                           "|Initial amount of S2                |$0$              |mole           |\n"
                           "|Value of parameter k1               |$1$              |second^-1^     |\n"
                           "|Volume of compartment \"compartment\" |$1$              |litre          |]\n"
                           "\n"
                           "The species' initial quantities are given in terms of substance units to\n"
                           "make it easier to use the model in a discrete stochastic simulator, but (as\n"
                           "per usual SBML principles) their symbols represent their values in\n"
                           "concentration units where they appear in expressions.\n"
                           "\n"
                           "Note: The test data for this model was generated from an analytical\n"
                           "solution of the system of equations.\n"
                           "\n"
                           "*)\n"
                           "\n"
                           "newcase[ \"00001\" ];\n"
                           "\n"
                           "addCompartment[ compartment, size -> 1 ];\n"
                           "addSpecies[ S1, initialAmount -> 1.5 10^-4 ];\n"
                           "addSpecies[ S2, initialAmount -> 0 ];\n"
                           "addParameter[ k1, value -> 1 ];\n"
                           "addReaction[ S1 -> S2, reversible -> False, \n"
                           " kineticLaw -> k1 * S1 * compartment ];\n"
                           "\n"
                           "makemodel[]\n"
                           "\n";
    SemanticSTSModel s(1);
    ASSERT_STREQ(s.readModelDescriptionFile().c_str(), expected.c_str());
}

TEST_F(SemanticSTSModelTests, getComponentTags){
    SemanticSTSModel s(3);
    std::vector<std::string> expected({"Compartment", "Species", "Reaction", "Parameter"});
    auto actual = s.getComponentTags();
    ASSERT_EQ(expected, actual);
}

TEST_F(SemanticSTSModelTests, getTestTags){
    SemanticSTSModel s(1);
    std::vector<std::string> expected({"Amount"});
    auto actual = s.getTestTags();
    ASSERT_EQ(expected, actual);
}

TEST_F(SemanticSTSModelTests, getTestTypes){
    SemanticSTSModel s(3);
    std::vector<std::string> expected({"TimeCourse"});
    auto actual = s.getTestType();
    ASSERT_EQ(expected, actual);
}

TEST_F(SemanticSTSModelTests, getResultsFile) {
    SemanticSTSModel s(3);
    std::filesystem::path p = s.getResultsFile();
    ASSERT_TRUE(std::filesystem::exists(p));
}

TEST_F(SemanticSTSModelTests, getSettingsFile) {
    SemanticSTSModel s(3);
    std::filesystem::path p = s.getSettingsFile();
    ASSERT_TRUE(std::filesystem::exists(p));
}

TEST_F(SemanticSTSModelTests, getL3V2) {
    SemanticSTSModel s(3);
    std::filesystem::path p = s.getLevelAndVersion(3, 2);
    ASSERT_TRUE(std::filesystem::exists(p));
}

TEST_F(SemanticSTSModelTests, getL3V1) {
    SemanticSTSModel s(3);
    std::filesystem::path p = s.getLevelAndVersion(3, 1);
    ASSERT_TRUE(std::filesystem::exists(p));
}

TEST_F(SemanticSTSModelTests, getL2V5) {
    SemanticSTSModel s(3);
    std::filesystem::path p = s.getLevelAndVersion(2, 5);
    ASSERT_TRUE(std::filesystem::exists(p));
}

TEST_F(SemanticSTSModelTests, getL2V4) {
    SemanticSTSModel s(3);
    std::filesystem::path p = s.getLevelAndVersion(2, 4);
    ASSERT_TRUE(std::filesystem::exists(p));
}

TEST_F(SemanticSTSModelTests, getL2V3) {
    SemanticSTSModel s(3);
    std::filesystem::path p = s.getLevelAndVersion(2, 3);
    ASSERT_TRUE(std::filesystem::exists(p));
}

TEST_F(SemanticSTSModelTests, getL2V2) {
    SemanticSTSModel s(3);
    std::filesystem::path p = s.getLevelAndVersion(2, 2);
    ASSERT_TRUE(std::filesystem::exists(p));
}

class StochasticSTSModelTests : public ::testing::Test {
public:
    StochasticSTSModelTests() = default;
};


TEST_F(StochasticSTSModelTests, getMeanFile) {
    StochasticSTSModel s(3);
    std::filesystem::path p = s.getMeanFile();
    ASSERT_TRUE(std::filesystem::exists(p));
}

TEST_F(StochasticSTSModelTests, getSDFile) {
    StochasticSTSModel s(3);
    std::filesystem::path p = s.getSDFile();
    ASSERT_TRUE(std::filesystem::exists(p));
}

TEST_F(StochasticSTSModelTests, getModFile) {
    StochasticSTSModel s(3);
    std::filesystem::path p = s.getModFile();
    ASSERT_TRUE(std::filesystem::exists(p));
}


class STSTests : public ::testing::Test {
public:
    STSTests() = default;
};

TEST_F(STSTests, getModelNFromSTS){
    STS<SemanticSTSModel> sts;
    SemanticSTSModel stsModel = sts.getModelNFromSTS(5);
    ASSERT_TRUE(std::filesystem::exists(stsModel.getLevelAndVersion(3, 2)));
}

TEST_F(STSTests, getModelsFromSTSVec){
    std::vector<int> v ({6, 3, 2, 66, 33});
    STS<SemanticSTSModel> sts;
    std::vector<std::string> models = sts.getModelsFromSTS(v);
    ASSERT_EQ(v.size(), models.size());
}

TEST_F(STSTests, getModelsFromSTS){
    STS<SemanticSTSModel> sts;
    std::vector<std::string> models = sts.getModelsFromSTS(1, 10);
    ASSERT_EQ(9, models.size());
}

TEST_F(STSTests, getModelsFromSTSAsStrings){
    STS<SemanticSTSModel> sts;
    std::vector<std::string> models = sts.getModelsFromSTSAsStrings(1, 10);
    std::cout << models[0] << std::endl;
    ASSERT_EQ(9, models.size());
}

