//
// Created by Ciaran on 08/07/2021.
//

#include "gtest/gtest.h"
#include "Matrix.h"

using namespace rr;

class MatrixTests : public ::testing::Test {
public:
    MatrixTests() = default;
};


TEST_F(MatrixTests, TestEqualityWithIntsTrue) {
    Matrix<int> first({
                              {1, 2, 3},
                              {4, 5, 6},
                      });
    Matrix<int> second({
                               {1, 2, 3},
                               {4, 5, 6},
                       });
    ASSERT_TRUE(first == second);
}

TEST_F(MatrixTests, TestEqualityWithIntsFalse) {
    Matrix<int> first({
                              {1, 2, 3},
                              {4, 5, 7},
                      });
    Matrix<int> second({
                               {1, 2, 3},
                               {4, 5, 6},
                       });
    ASSERT_TRUE(first != second);
}

TEST_F(MatrixTests, TestAlmostEqualsWhenTrue) {
    Matrix<double> first({
                                 {1.1234, 2.1234, 3.1234},
                                 {4.1234, 5.1234, 7.1234},
                         });
    Matrix<double> second({
                                  {1.1234, 2.1234, 3.1234},
                                  {4.1234, 5.1234, 7.1234},
                          });
    ASSERT_TRUE(first.almostEquals(second, 1e-4));
}


TEST_F(MatrixTests, TestAlmostEqualsWhenFalse) {
    Matrix<double> first({
                                 {1.1245, 2.1234, 3.1234},
                                 {4.1234, 5.1234, 7.1234},
                         });
    Matrix<double> second({
                                  {1.1234, 2.1234, 3.1234},
                                  {4.1234, 5.1234, 7.1234},
                          });
    ASSERT_FALSE(first.almostEquals(second, 1e-5));
}

TEST_F(MatrixTests, TestAlmostEqualsWhenFalseButAcceptableTolerance) {
    Matrix<double> first({
                                 {1.12346, 2.1234, 3.1234},
                                 {4.1234,  5.1234, 7.1234},
                         });
    Matrix<double> second({
                                  {1.1234, 2.1234, 3.1234},
                                  {4.1234, 5.1234, 7.1234},
                          });
    ASSERT_TRUE(first.almostEquals(second, 1e-3));
}


TEST_F(MatrixTests, CheckEmptyWhenTrue) {
    Matrix<double> emptyMatrix;
    ASSERT_TRUE(emptyMatrix.empty());
}

TEST_F(MatrixTests, CheckEmptyWhenTrueWithColnames) {
    Matrix<double> emptyMatrix;
    emptyMatrix.setColNames({"S1", "S2"});
    ASSERT_TRUE(emptyMatrix.empty());
}

TEST_F(MatrixTests, CheckEmptyWhenFalse) {
    Matrix<double> notEmptyMatrix({
                                          {0, 1, 2}
                                  });
    ASSERT_FALSE(notEmptyMatrix.empty());
}

TEST_F(MatrixTests, CheckEqualityOfEmptyMatrices) {
    Matrix<double> first;
    Matrix<double> second;
    bool equal = first == second;
    ASSERT_TRUE(equal);
}

TEST_F(MatrixTests, ConstructorForLsMatrix) {
    ls::Matrix<int> lsMatrix({
                                     {1, 2}
                             });
    rr::Matrix<int> rrMatrix(lsMatrix);
    rr::Matrix<int> expected({{1, 2}});
    ASSERT_TRUE(expected == rrMatrix);
}

TEST_F(MatrixTests, ConstructorForLsMatrixWithRowNames) {
    ls::Matrix<int> lsMatrix({
                                     {1, 2}
                             });
    lsMatrix.setRowNames({"R0", "R1"});
    rr::Matrix<int> rrMatrix(lsMatrix);
    std::vector<std::string> expectedColNames({"R0", "R1"});
    ASSERT_TRUE(expectedColNames == rrMatrix.getRowNames());
}

TEST_F(MatrixTests, ConstructorForLsMatrixWithColNames) {
    ls::Matrix<int> lsMatrix({{1, 2}});
    lsMatrix.setRowNames({"R0", "R1"});
    lsMatrix.setColNames({"C0", "C1"});
    rr::Matrix<int> rrMatrix(lsMatrix);
    std::vector<std::string> expectedColNames({"C0", "C1"});
    ASSERT_TRUE(expectedColNames == rrMatrix.getColNames());
}

TEST_F(MatrixTests, SortRowsByNames2) {
    rr::Matrix<int> mat({
                                {0, 1, 2},
                                {3, 4, 5},
                        });
    mat.setRowNames({"R1", "R0"});
    rr::Matrix<int> expected({
                                     {3, 4, 5},
                                     {0, 1, 2},
                             });
    expected.setRowNames({"R0", "R1"});
    mat.sortRowsByLabels();
    ASSERT_TRUE(expected == mat);
    ASSERT_EQ(mat.rowNames, std::vector<std::string>({"R0", "R1"}));
}

TEST_F(MatrixTests, SortRowsByNames3) {
    rr::Matrix<int> mat({
                                /*R1*/ {0, 1, 2},
                                /*R2*/
                                       {3, 4, 5},
                                /*R0*/
                                       {6, 7, 8},
                        });
    mat.setRowNames({"R1", "R2", "R0"});
    rr::Matrix<int> expected({
                                     /*R0*/{6, 7, 8},
                                     /*R1*/
                                           {0, 1, 2},
                                     /*R2*/
                                           {3, 4, 5},
                             });
    expected.setRowNames({"R0", "R1", "R2"});
    mat.sortRowsByLabels();
    ASSERT_TRUE(expected == mat);
    ASSERT_EQ(mat.rowNames, std::vector<std::string>({"R0", "R1", "R2"}));
}

TEST_F(MatrixTests, SortRowsByNames4) {
    rr::Matrix<int> mat({
                                /*R4*/{0, 1,  2},
                                /*R1*/
                                      {3, 4,  5},
                                /*R2*/
                                      {6, 7,  8},
                                /*R0*/
                                      {9, 10, 11},
                        });
    mat.setRowNames({"R4", "R1", "R2", "R0"});
    rr::Matrix<int> expected({
                                     /*R0*/{9, 10, 11},
                                     /*R1*/
                                           {3, 4,  5},
                                     /*R2*/
                                           {6, 7,  8},
                                     /*R3*/
                                           {0, 1,  2},
                             });
    expected.setRowNames({"R0", "R1", "R2", "R3"});
    mat.sortRowsByLabels();
    std::cout << "expected" << std::endl;
    std::cout << expected << std::endl;
    std::cout << mat << std::endl;
    ASSERT_TRUE(expected == mat);
    ASSERT_EQ(mat.rowNames, std::vector<std::string>({"R0", "R1", "R2", "R4"}));
}


TEST_F(MatrixTests, SortColsByNames2) {
    rr::Matrix<int> mat({
                                {0, 1},
                                {3, 4},
                        });
    mat.setColNames({"C3", "C1"});
    rr::Matrix<int> expected({
                                     {1, 0},
                                     {4, 3},
                             });
    expected.setColNames({"C1", "C3"});
    mat.sortColsByLabels();
    ASSERT_TRUE(expected == mat);
    ASSERT_EQ(mat.colNames, std::vector<std::string>({"C1", "C3"}));
}

TEST_F(MatrixTests, SortColsByNames6) {
    rr::Matrix<int> mat({
                                //6  3  5  1  2  3
                                {6, 3, 5, 1, 2, 4},
                                {3, 4, 4, 4, 4, 4},
                        });
    mat.setColNames({"C6", "C3", "C5", "C1", "C2", "C4"});
    rr::Matrix<int> expected({
                                     {1, 2, 3, 4, 5, 6},
                                     {4, 4, 4, 4, 4, 3},
                             });
    expected.setColNames({"C1", "C2", "C3", "C4", "C5", "C6"});
    mat.sortColsByLabels();
    ASSERT_TRUE(expected == mat);
    ASSERT_EQ(mat.colNames, std::vector<std::string>({"C1", "C2", "C3", "C4", "C5", "C6"}));
}

















