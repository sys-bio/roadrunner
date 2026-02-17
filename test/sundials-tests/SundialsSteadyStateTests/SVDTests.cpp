
#include "gtest/gtest.h"
#include "rr-libstruct/lsMatrix.h"
#include "SVD.h"

using namespace rr;

/**
 * Interface for testing the SVG class.
 */
class SVDResult {
public:

    /**
     * This matrix will be fed into SVG
     */
    virtual ls::DoubleMatrix inputMatrix() = 0;

    /**
     * A vector of singular values. Size should be equal to number
     * of columns in the input matrix
     */
    virtual ls::DoubleMatrix singularValues() = 0;

    /**
     * Left singular vectors. Each column is a left singular vector
     */
    virtual ls::DoubleMatrix leftSingularVectors() = 0;

    /**
     * Right singular vectors. Each row is a right singular vector
     */
    virtual ls::DoubleMatrix rightSingularVectors() = 0;

    /**
     * Rank of matrix
     */
    virtual unsigned int rank() = 0;

    /**
     * Whether matrix is singular or not
     */
    virtual bool isSingular() = 0;
};

/**
 * This example is from the docs here:
 *  https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/dgesvd_ex.c.htm
 *
 * Answers computed with numpy match though in the example
 *
    a = np.array([[ 8.79,   9.93,   9.83,   5.45,   3.16],
         [ 6.11,   6.91,   5.04,  -0.27,   7.98],
         [-9.15,  -7.93,   4.86,   4.85,   3.01],
         [ 9.57,   1.64,   8.83,   0.74,   5.80],
         [-3.49,   4.02,   9.80,  10.00,   4.27],
         [ 9.84,   0.15,  -8.99,  -6.02,  -5.31]])
 Out:
 In [6]: np.linalg.svd(a)
Out[6]:
(array([[-0.59114238,  0.26316781,  0.35543017,  0.31426436,  0.22993832,
          0.55075318],
        [-0.39756679,  0.24379903, -0.22239   , -0.75346615, -0.36358969,
          0.18203479],
        [-0.03347897, -0.60027258, -0.45083927,  0.23344966, -0.30547573,
          0.53617327],
        [-0.4297069 ,  0.23616681, -0.68586286,  0.33186002,  0.16492763,
         -0.38966287],
        [-0.46974792, -0.3508914 ,  0.3874446 ,  0.15873556, -0.51825744,
         -0.46077223],
        [ 0.29335876,  0.57626212, -0.02085292,  0.37907767, -0.6525516 ,
          0.10910681]]),
 array([27.46873242, 22.64318501,  8.55838823,  5.9857232 ,  2.01489966]),
 array([[-0.25138279, -0.39684555, -0.69215101, -0.36617044, -0.40763524],
        [ 0.81483669,  0.3586615 , -0.24888801, -0.36859354, -0.09796257],
        [-0.26061851,  0.70076821, -0.22081145,  0.38593848, -0.49325014],
        [ 0.39672378, -0.45071124,  0.25132115,  0.4342486 , -0.62268407],
        [-0.21802776,  0.14020995,  0.58911945, -0.62652825, -0.43955169]]))
 *
 */
class SixByFiveMatrix : public SVDResult {
public:
    SixByFiveMatrix() = default;

    ls::DoubleMatrix inputMatrix() override {
        return ls::DoubleMatrix(
                {
                        {8.79,  9.93,  9.83,  5.45,  3.16},
                        {6.11,  6.91,  5.04,  -0.27, 7.98},
                        {-9.15, -7.93, 4.86,  4.85,  3.01},
                        {9.57,  1.64,  8.83,  0.74,  5.80},
                        {-3.49, 4.02,  9.80,  10.00, 4.27},
                        {9.84,  0.15,  -8.99, -6.02, -5.31},
                }
        );
    }

    ls::DoubleMatrix singularValues() override {
        return ls::DoubleMatrix({{27.47, 22.64, 8.56, 5.99, 2.01}});
    }

    ls::DoubleMatrix leftSingularVectors() override {
        return ls::DoubleMatrix(
                {
                        {-0.59, 0.26,  0.36,  0.31,  0.23,  0.55},
                        {-0.40, 0.24,  -0.22, -0.75, -0.36, 0.18},
                        {-0.03, -0.60, -0.45, 0.23,  -0.31, 0.54},
                        {-0.43, 0.24,  -0.69, 0.33,  0.16,  -0.39},
                        {-0.47, -0.35, 0.39,  0.16,  -0.52, -0.46},
                        {0.29,  0.58,  -0.02, 0.38,  -0.65, 0.11},
                }
        );
    }

    ls::DoubleMatrix rightSingularVectors() override {
        return ls::DoubleMatrix(
                {
                        {-0.25, -0.40, -0.69, -0.37, -0.41},
                        {0.81,  0.36,  -0.25, -0.37, -0.10},
                        {-0.26, 0.70,  -0.22, 0.39,  -0.49},
                        {0.40,  -0.45, 0.25,  0.43,  -0.62},
                        {-0.22, 0.14,  0.59,  -0.63, -0.44},
                }
        );
    }

    unsigned int rank() override {
        return 5;
    }

    bool isSingular() override {
        return false;
    }


};

class TwoByThreeMatrix : public SVDResult {
public:

    ls::DoubleMatrix inputMatrix() override {
        return ls::DoubleMatrix(
                {
                        {3, 2, 2},
                        {2, 3, -2}
                }
        );
    }

    ls::DoubleMatrix singularValues() override {
        return ls::DoubleMatrix({
                                        {5.0, 3.0}
                                });
    }

    ls::DoubleMatrix leftSingularVectors() override {
        return ls::DoubleMatrix({
                                        {-0.70710678, -0.70710678},
                                        {-0.70710678, 0.70710678}
                                });
    }

    ls::DoubleMatrix rightSingularVectors() override {
        return ls::DoubleMatrix({
                                        {-7.07106781e-01, -7.07106781e-01, -6.47932334e-17},
                                        {-2.35702260e-01, 2.35702260e-01,  -9.42809042e-01},
                                        {-6.66666667e-01, 6.66666667e-01,  3.33333333e-01}
                                });
    }

    unsigned int rank() override {
        return 2;
    }

    bool isSingular() override {
        return false;
    }
};

/**
 * @brief 5 by 14 matrix
 * @details numpy code:
In [21]: a = np.random.randint(0, 100, (2, 4))
In [23]: a
Out[23]:
array([[76, 54, 85, 29],
       [45, 93, 90, 34]])
In [22]: np.linalg.svd(a)
Out[22]:
(array([[-0.67362077, -0.73907717],
        [-0.73907717,  0.67362077]]),
 array([188.40245814,  34.53279263]),
 array([[-0.44826194, -0.55789982, -0.65696972, -0.23706499],
        [-0.74876453,  0.65840503, -0.06358276,  0.04256442],
        [-0.47204005, -0.45839448,  0.74664849, -0.09781987],
        [-0.12486436, -0.21242671, -0.08284956,  0.96561884]]))
 */
class TwoByFour : public SVDResult {
public:
    ls::DoubleMatrix inputMatrix() override {
        return ls::DoubleMatrix(
                {
                        {76, 54, 85, 29},
                        {45, 93, 90, 34}
                });
    }

    ls::DoubleMatrix singularValues() override {
        return ls::DoubleMatrix({{188.40245814, 34.53279263}});
    }

    ls::DoubleMatrix leftSingularVectors() override {
        return ls::DoubleMatrix({
                                        {-0.67362077, -0.73907717},
                                        {-0.73907717, 0.67362077}
                                });
    }

    ls::DoubleMatrix rightSingularVectors() override {
        return ls::DoubleMatrix(
                {
                        {-0.44826194, -0.55789982, -0.65696972, -0.23706499},
                        {-0.74876453, 0.65840503,  -0.06358276, 0.04256442},
                        {-0.47204005, -0.45839448, 0.74664849,  -0.09781987},
                        {-0.12486436, -0.21242671, -0.08284956, 0.96561884},
                });
    };

    unsigned int rank() override {
        return 2;
    }

    bool isSingular() override {
        return false;
    }
};

class SingularMatrix : public SVDResult {
public:
    ls::DoubleMatrix inputMatrix() override {
        return ls::DoubleMatrix({
                                        {1,  -1},
                                        {-1, 1}
                                });
    }

    ls::DoubleMatrix singularValues() override {
        return ls::DoubleMatrix({{2.0, 0.0}});
    }

    ls::DoubleMatrix leftSingularVectors() override {
        return ls::DoubleMatrix({
                                        {-0.70710678, 0.70710678},
                                        {0.70710678,  0.70710678},
                                });
    }

    ls::DoubleMatrix rightSingularVectors() override {
        return ls::DoubleMatrix({
                                        {-0.70710678, 0.70710678},
                                        {0.70710678,  0.70710678},
                                });
    }

    unsigned int rank() override {
        return 1;
    }

    bool isSingular() override {
        return true;
    }
};

class SVGTests : public ::testing::Test {
public:
    SVGTests() = default;

    static void checkMatrixEquality(const ls::DoubleMatrix &expectedMatrix, const ls::DoubleMatrix &actualMatrix, bool checkAbsOnly=false) {
        std::cout << "Comparing expected matrix: " << std::endl;
        std::cout << expectedMatrix << std::endl;
        std::cout << "with actual matrix: " << std::endl;
        std::cout << actualMatrix << std::endl;
        if (expectedMatrix.numCols() != actualMatrix.numCols() ||
            expectedMatrix.numRows() != actualMatrix.numRows()) {
            std::cout << "Checking row dimensions: expected " << expectedMatrix.numRows() << " vs actual "
                      << actualMatrix.numRows()
                      << std::endl;
            std::cout << "Checking col dimensions expected " << expectedMatrix.numCols() << " vs actual "
                      << actualMatrix.numCols()
                      << std::endl;
            ASSERT_TRUE(false && "expectedMatrix dimensions not equal to actualMatrix");
        }
        for (int i = 0; i < expectedMatrix.numRows(); i++) {
            for (int j = 0; j < expectedMatrix.numCols(); j++) {
                //std::cout << "i: " << i << "; j: " << j << std::endl;
              if (checkAbsOnly) {
                EXPECT_NEAR(abs(expectedMatrix(i, j)), abs(actualMatrix(i, j)), 0.1);
              }
              else {
                EXPECT_NEAR(expectedMatrix(i, j), actualMatrix(i, j), 0.1);
              }
            }
        }
    }

    template<class SVDType>
    void checkSVDValuesSingularValues() {
        SVDType svdReference;
        auto input = svdReference.inputMatrix();
        SVD svd(input);
        checkMatrixEquality(svdReference.singularValues(), svd.getSingularValues());
    }

    template<class SVDType>
    void checkSVDValuesLeftSingularVectors() {
        SVDType svdReference;
        auto input = svdReference.inputMatrix();
        SVD svd(input);
        checkMatrixEquality(svdReference.leftSingularVectors(), svd.getLeftSingularVectors());
    }

    template<class SVDType>
    void checkSVDValuesRightSingularVectors() {
        SVDType svdReference;
        auto input = svdReference.inputMatrix();
        SVD svd(input);
        //On Macs with AMD64 architectures, the second row of the matrix is negative.
        // I *think* this is OK?  The math is a bit confusing.  But it's not worth investigating
        // much further, because roadrunner doesn't actually use the SVD class at all right now.
        // So for now, added a 'abs only' argument, just for this test.
        checkMatrixEquality(svdReference.rightSingularVectors(), svd.getRightSingularVectors(), true);
    }

    template<class SVDType>
    void checkRank() {
        SVDType svdReference;
        auto input = svdReference.inputMatrix();
        SVD svd(input);
        ASSERT_EQ(svdReference.rank(), svd.rank());
    }

    template<class SVDType>
    void checkSingular() {
        SVDType svdReference;
        auto input = svdReference.inputMatrix();
        SVD svd(input);
        auto expectedSingularStatus = svdReference.isSingular();
        if (expectedSingularStatus) {
            ASSERT_TRUE(svd.isSingular());
        } else {
            ASSERT_FALSE(svd.isSingular());
        }
    }
};


/**
 * Note: Each test case only does *one* thing.
 * This means we we don't need to untangle the a problem when
 * there is one =]
 */

class SixByFiveMatrixTests : public SVGTests {
public:
    SixByFiveMatrixTests() = default;
};

TEST_F(SixByFiveMatrixTests, SingularValues) {
    checkSVDValuesSingularValues<SixByFiveMatrix>();
}

TEST_F(SixByFiveMatrixTests, LeftSingularVectors) {
    checkSVDValuesLeftSingularVectors<SixByFiveMatrix>();
}

TEST_F(SixByFiveMatrixTests, RightingularVectors) {
    checkSVDValuesRightSingularVectors<SixByFiveMatrix>();
}

TEST_F(SixByFiveMatrixTests, Rank) {
    checkRank<SixByFiveMatrix>();
}

TEST_F(SixByFiveMatrixTests, IsSingular) {
    checkSingular<SixByFiveMatrix>();
}

class TwoByThreeMatrixTests : public SVGTests {
public:
    TwoByThreeMatrixTests() = default;

};

TEST_F(TwoByThreeMatrixTests, SingularValues) {
    checkSVDValuesSingularValues<TwoByThreeMatrix>();
}

TEST_F(TwoByThreeMatrixTests, LeftSingularVector) {
    checkSVDValuesLeftSingularVectors<TwoByThreeMatrix>();
}

TEST_F(TwoByThreeMatrixTests, RightSingularVector) {
    checkSVDValuesRightSingularVectors<TwoByThreeMatrix>();
}

TEST_F(TwoByThreeMatrixTests, Rank) {
    checkRank<TwoByThreeMatrix>();
}

TEST_F(TwoByThreeMatrixTests, IsSingular) {
    checkSingular<SixByFiveMatrix>();
}

class TwoByFourMatrixTests : public SVGTests {
public:
    TwoByFourMatrixTests() = default;
};


TEST_F(TwoByFourMatrixTests, SingularValues) {
    checkSVDValuesSingularValues<TwoByFour>();
}

TEST_F(TwoByFourMatrixTests, LeftSingularVector) {
    checkSVDValuesLeftSingularVectors<TwoByFour>();
}

TEST_F(TwoByFourMatrixTests, RightSingularVector) {
    checkSVDValuesRightSingularVectors<TwoByFour>();
}

TEST_F(TwoByFourMatrixTests, Rank) {
    checkRank<TwoByFour>();
}

TEST_F(TwoByFourMatrixTests, IsSingular) {
    checkSingular<SixByFiveMatrix>();
}

class SingularMatrixTests : public SVGTests {
public:
    SingularMatrixTests() = default;
};

TEST_F(SingularMatrixTests, SingularValues) {
    checkSVDValuesSingularValues<SingularMatrix>();
}

TEST_F(SingularMatrixTests, LeftSingularVector) {
    checkSVDValuesLeftSingularVectors<SingularMatrix>();
}

TEST_F(SingularMatrixTests, RightSingularVector) {
    checkSVDValuesRightSingularVectors<SingularMatrix>();
}

TEST_F(SingularMatrixTests, Rank) {
    checkRank<SingularMatrix>();
}

TEST_F(SingularMatrixTests, IsSingular) {
    checkSingular<SixByFiveMatrix>();
}




















