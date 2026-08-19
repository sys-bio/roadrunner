//
// Created by Ciaran on 08/07/2021.
//

#include "gtest/gtest.h"
#include "Matrix3D.h"
#include <limits>

using namespace rr;

class Matrix3DTests : public ::testing::Test {
public:
    Matrix3DTests() = default;
};


TEST_F(Matrix3DTests, CheckSlice) {
    Matrix3D<double, double> matrix3D(
            {0.0, 1.0},
            {
                    // 0.0
                    {
                            {0.0, 1.0},
                            {0.0, 1.0}
                    },
                    // 1.0
                    {
                            {0.0, 2.0},
                            {0.0, 2.0}
                    }
            }
    );
    Matrix<double> actual = matrix3D.getItem(0.0);
    Matrix<double> expected({
                                    {0.0, 1.0},
                                    {0.0, 1.0}
                            });
    bool passed = expected.almostEquals(actual, 1e-4);
    ASSERT_TRUE(passed);
}

TEST_F(Matrix3DTests, CheckSlice3D) {
    Matrix3D<double, double> matrix3D(
            {0.0, 1.0},
            {
                    // 0.0
                    {
                            {0.0, 1.0},
                            {2.0, 3.0}
                    },
                    // 1.0
                    {
                            {4.0, 5.0},
                            {6.0, 7.0}
                    }
            }
    );
    double actual = matrix3D.slice(0, 1, 1);
    ASSERT_NEAR(3.0, actual, 1e-7);
}

TEST_F(Matrix3DTests, CheckSlice2D) {
    Matrix3D<double, double> matrix3D(
            {0.0, 1.0},
            {
                    // 0.0
                    {
                            {0.0, 1.0},
                            {2.0, 3.0}
                    },
                    // 1.0
                    {
                            {4.0, 5.0},
                            {6.0, 7.0}
                    }
            }
    );
    std::vector<double> actual = matrix3D.slice(0, 1);
    ASSERT_NEAR(2.0, actual[0], 1e-7);
    ASSERT_NEAR(3.0, actual[1], 1e-7);
}

TEST_F(Matrix3DTests, CheckSlice1D) {
    Matrix3D<double, double> matrix3D(
            {0.0, 1.0},
            {
                    // 0.0
                    {
                            {0.0, 1.0},
                            {2.0, 3.0}
                    },
                    // 1.0
                    {
                            {4.0, 5.0},
                            {6.0, 7.0}
                    }
            }
    );
    rr::Matrix<double> actual = matrix3D.slice(0);
    rr::Matrix<double> expected(
            {
                    {0.0, 1.0},
                    {2.0, 3.0},
            });
    ASSERT_TRUE(expected.almostEquals(expected, 1e-7));
}

TEST_F(Matrix3DTests, CheckSlice3D3x4x2) {
    Matrix3D<double, double> matrix3D(
            {0, 6, 12},
            {
                    {
                            {0,  1},
                            {2,  3},
                            {4,  5},
                            {6,  7},
                    },
                    {
                            {8,  9},
                            {10, 11},
                            {12, 13},
                            {14, 15},
                    },
                    {
                            {16, 17},
                            {18, 19},
                            {20, 21},
                            {22, 23},
                    },
            }
    );
    ASSERT_NEAR(0.0, matrix3D.slice(0, 0, 0), 1e-7);
    ASSERT_NEAR(1.0, matrix3D.slice(0, 0, 1), 1e-7);
    ASSERT_NEAR(22.0, matrix3D.slice(2, 3, 0), 1e-7);
    ASSERT_NEAR(23.0, matrix3D.slice(2, 3, 1), 1e-7);
}

TEST_F(Matrix3DTests, NumRows) {
    Matrix3D<double, double> matrix3D(
            {0.0, 1.0},
            {
                    // 0.0
                    {
                            {0.0, 1.0},
                            {0.0, 1.0},
                            {0.0, 1.0}
                    },
                    // 1.0
                    {
                            {0.0, 2.0},
                            {0.0, 2.0},
                            {0.0, 2.0}
                    }
            }
    );
    ASSERT_EQ(matrix3D.numRows(), 3);
}

TEST_F(Matrix3DTests, NumCols) {
    Matrix3D<double, double> matrix3D(
            {0.0, 1.0},
            {
                    // 0.0
                    {
                            {0.0, 1.0},
                            {0.0, 1.0},
                            {0.0, 1.0}
                    },
                    // 1.0
                    {
                            {0.0, 2.0},
                            {0.0, 2.0},
                            {0.0, 2.0}
                    }
            }
    );
    ASSERT_EQ(matrix3D.numCols(), 2);
}

TEST_F(Matrix3DTests, NumZ) {
    Matrix3D<double, double> matrix3D(
            {0.0, 1.0},
            {
                    // 0.0
                    {
                            {0.0, 1.0},
                            {0.0, 1.0},
                            {0.0, 1.0}
                    },
                    // 1.0
                    {
                            {0.0, 2.0},
                            {0.0, 2.0},
                            {0.0, 2.0}
                    }
            }
    );
    ASSERT_EQ(matrix3D.numZ(), 2);
}

TEST_F(Matrix3DTests, InsertEmptyCheckNumRows) {
    // needs to set x, y and z
    Matrix3D<int, double> m;
    m.insert(0, {
            {0, 1, 2},
            {3, 4, 5},
    });
    ASSERT_EQ(2, m.numRows());
}

TEST_F(Matrix3DTests, InsertEmptyCheckNumCols) {
    // needs to set x, y and z
    Matrix3D<int, double> m;
    m.insert(0, {
            {0, 1, 2},
            {3, 4, 5},
    });
    ASSERT_EQ(3, m.numCols());
}

TEST_F(Matrix3DTests, InsertEmptyCheckNumZ) {
    // needs to set x, y and z
    Matrix3D<int, double> m;
    m.insert(0, {
            {0, 1, 2},
            {3, 4, 5},
    });
    ASSERT_EQ(1, m.numZ());
}

TEST_F(Matrix3DTests, EqualsWhenTrue) {
    Matrix3D<int, int> first(
            {0, 1},
            {
                    // 0
                    {
                            {0, 1},
                            {0, 1},
                            {0, 1}
                    },
                    // 1
                    {
                            {0, 2},
                            {0, 2},
                            {0, 2}
                    }
            }
    );
    Matrix3D<int, int> second(
            {0, 1},
            {
                    // 0
                    {
                            {0, 1},
                            {0, 1},
                            {0, 1}
                    },
                    // 1
                    {
                            {0, 2},
                            {0, 2},
                            {0, 2}
                    }
            }
    );
    ASSERT_TRUE(first == second);
}

TEST_F(Matrix3DTests, EqualsWhenFalseBecauseOfDifferentIdx) {
    Matrix3D<int, int> first(
            {0, 3},
            {
                    // 0
                    {
                            {0, 1},
                            {0, 1},
                            {0, 1}
                    },
                    // 1
                    {
                            {0, 2},
                            {0, 2},
                            {0, 2}
                    }
            }
    );
    Matrix3D<int, int> second(
            {0, 1},
            {
                    // 0
                    {
                            {0, 1},
                            {0, 1},
                            {0, 1}
                    },
                    // 1
                    {
                            {0, 2},
                            {0, 2},
                            {0, 2}
                    }
            }
    );
    ASSERT_FALSE(first == second);
}

TEST_F(Matrix3DTests, EqualsWhenFalseBecauseOfDifferentData) {
    Matrix3D<int, int> first(
            {0, 3},
            {
                    // 0
                    {
                            {0, 4},
                            {0, 1},
                            {0, 1}
                    },
                    // 1
                    {
                            {0, 2},
                            {0, 2},
                            {0, 2}
                    }
            }
    );
    Matrix3D<int, int> second(
            {0, 1},
            {
                    // 0
                    {
                            {0, 1},
                            {0, 1},
                            {0, 1}
                    },
                    // 1
                    {
                            {0, 2},
                            {0, 2},
                            {0, 2}
                    }
            }
    );
    ASSERT_FALSE(first == second);
}

TEST_F(Matrix3DTests, AlmostEqualsWhenTrue) {
    Matrix3D<double, double> first(
            {0, 1},
            {
                    // 0
                    {
                            {0.2, 4.2},
                            {0.2, 1.2},
                            {0.2, 1.2}
                    },
                    // 1
                    {
                            {0.2, 2.2},
                            {0.2, 2.2},
                            {0.2, 2.2}
                    }
            }
    );
    Matrix3D<double, double> second(
            {0, 1},
            {
                    // 0
                    {
                            {0.2, 4.2},
                            {0.2, 1.2},
                            {0.2, 1.2}
                    },
                    // 1
                    {
                            {0.2, 2.2},
                            {0.2, 2.2},
                            {0.2, 2.2}
                    }
            }
    );
    ASSERT_TRUE(first.almostEquals(second, 1e-4));
}

TEST_F(Matrix3DTests, AlmostEqualsWhenFalse) {
    Matrix3D<double, double> first(
            {0.2, 3.6},
            {
                    // 0
                    {
                            {0.2, 4.2},
                            {0.2, 1.2},
                            {0.2, 1.2}
                    },
                    // 1
                    {
                            {0.2, 2.2},
                            {0.2, 2.2},
                            {0.2, 2.2}
                    }
            }
    );
    Matrix3D<double, double> second(
            {0.2, 1.2},
            {
                    // 0
                    {
                            {0.2, 1.2},
                            {0.2, 1.2},
                            {0.2, 1.2}
                    },
                    // 1
                    {
                            {0.2, 2.2},
                            {0.2, 2.2},
                            {0.2, 2.2}
                    }
            }
    );
    ASSERT_FALSE(first.almostEquals(second, 1e-4));
}

TEST_F(Matrix3DTests, AlmostEqualsWhenFalseReverseDirection) {
    // almostEquals must be symmetric. The one-sided check
    // (index_[i] - other.index_[i]) > tol only caught the case where the
    // callee's index was smaller than the argument's.
    Matrix3D<double, double> small(
            {0.0, 1.0},
            {
                    {{0.0}},
                    {{0.0}}
            }
    );
    Matrix3D<double, double> large(
            {0.0, 1000.0},
            {
                    {{0.0}},
                    {{0.0}}
            }
    );
    ASSERT_FALSE(small.almostEquals(large, 1e-4));
    ASSERT_FALSE(large.almostEquals(small, 1e-4));
}

TEST_F(Matrix3DTests, AlmostEqualsWithNaNIndexDifferenceIsFalse) {
    // NaN comparisons are always false, so the old check treated a NaN
    // difference between index values as "not greater than tolerance",
    // silently reporting equality.
    double nan = std::numeric_limits<double>::quiet_NaN();
    Matrix3D<double, double> first(
            {nan, 1.0},
            {
                    {{0.0}},
                    {{0.0}}
            }
    );
    Matrix3D<double, double> second(
            {2.0, 1.0},
            {
                    {{0.0}},
                    {{0.0}}
            }
    );
    ASSERT_FALSE(first.almostEquals(second, 1e-4));
}

TEST_F(Matrix3DTests, CheckRowNames) {
    Matrix3D<double, double> matrix3D(
            {0.0, 1.0},
            {
                    // 0.0
                    {
                            {0.0, 1.0},
                            {2.0, 3.0}
                    },
                    // 1.0
                    {
                            {4.0, 5.0},
                            {6.0, 7.0}
                    }
            }
    );
    matrix3D.setRowNames({"First", "Second"});
    ASSERT_EQ(matrix3D.getRowNames(), std::vector<std::string>({"First", "Second"}));
}

TEST_F(Matrix3DTests, CheckColNames) {
    Matrix3D<double, double> matrix3D(
            {0.0, 1.0},
            {
                    // 0.0
                    {
                            {0.0, 1.0},
                            {2.0, 3.0}
                    },
                    // 1.0
                    {
                            {4.0, 5.0},
                            {6.0, 7.0}
                    }
            }
    );
    matrix3D.setColNames({"First", "Second"});
    ASSERT_EQ(matrix3D.getColNames(), std::vector<std::string>({"First", "Second"}));
}

TEST_F(Matrix3DTests, CheckDeleteRow0UsingIndex) {
    Matrix3D<double, double> matrix3D(
            {0.0, 1.0},
            {
                    // 0.0
                    {
                            {0.0, 1.0,  2.0},
                            {3.0,  4.0,  5.0},
                            {6.0,  7.0,  8.0}
                    },
                    // 1.0
                    {
                            {9.0, 10.0, 11.0},
                            {12.0, 13.0, 14.0},
                            {15.0, 16.0, 17.0},
                    }
            }
    );
    matrix3D.setColNames({"C1", "C2", "C3"});
    matrix3D.setRowNames({"R1", "R2", "R3"});

    matrix3D.deleteRow(0);

    Matrix3D<double, double> expected(
            {0.0, 1.0},
            {
                    // 0.0
                    {
                            {3.0,  4.0,  5.0},
                            {6.0,  7.0,  8.0}
                    },
                    // 1.0
                    {
                            {12.0, 13.0, 14.0},
                            {15.0, 16.0, 17.0},
                    }
            }
    );
    ASSERT_TRUE(matrix3D == expected);
    ASSERT_TRUE(matrix3D.getRowNames() == std::vector<std::string>({"R2", "R3"}));
    ASSERT_TRUE(matrix3D.getColNames() == std::vector<std::string>({"C1", "C2", "C3"}));
}
TEST_F(Matrix3DTests, CheckDeleteRow1UsingIndex) {
    Matrix3D<double, double> matrix3D(
            {0.0, 1.0},
            {
                    // 0.0
                    {
                            {0.0, 1.0,  2.0},
                            {3.0,  4.0,  5.0},
                            {6.0,  7.0,  8.0}
                    },
                    // 1.0
                    {
                            {9.0, 10.0, 11.0},
                            {12.0, 13.0, 14.0},
                            {15.0, 16.0, 17.0},
                    }
            }
    );
    matrix3D.setColNames({"C1", "C2", "C3"});
    matrix3D.setRowNames({"R1", "R2", "R3"});

    matrix3D.deleteRow(1);

    Matrix3D<double, double> expected(
            {0.0, 1.0},
            {
                    // 0.0
                    {
                            {0.0,  1.0,  2.0},
                            {6.0,  7.0,  8.0}
                    },
                    // 1.0
                    {
                            {9.0, 10.0, 11.0},
                            {15.0, 16.0, 17.0},
                    }
            }
    );
    ASSERT_TRUE(matrix3D == expected);
    ASSERT_TRUE(matrix3D.getRowNames() == std::vector<std::string>({"R1", "R3"}));
    ASSERT_TRUE(matrix3D.getColNames() == std::vector<std::string>({"C1", "C2", "C3"}));
}
TEST_F(Matrix3DTests, CheckDeleteRow2UsingIndex) {
    Matrix3D<double, double> matrix3D(
            {0.0, 1.0},
            {
                    // 0.0
                    {
                            {0.0, 1.0,  2.0},
                            {3.0,  4.0,  5.0},
                            {6.0,  7.0,  8.0}
                    },
                    // 1.0
                    {
                            {9.0, 10.0, 11.0},
                            {12.0, 13.0, 14.0},
                            {15.0, 16.0, 17.0},
                    }
            }
    );
    matrix3D.setColNames({"C1", "C2", "C3"});
    matrix3D.setRowNames({"R1", "R2", "R3"});

    matrix3D.deleteRow(2);

    Matrix3D<double, double> expected(
            {0.0, 1.0},
            {
                    // 0.0
                    {
                            {0.0,  1.0,  2.0},
                            {3.0,  4.0,  5.0},
                    },
                    // 1.0
                    {
                            {9.0, 10.0, 11.0},
                            {12.0, 13.0, 14.0},
                    }
            }
    );
    ASSERT_TRUE(matrix3D == expected);
    ASSERT_TRUE(matrix3D.getRowNames() == std::vector<std::string>({"R1", "R2"}));
    ASSERT_TRUE(matrix3D.getColNames() == std::vector<std::string>({"C1", "C2", "C3"}));
}

TEST_F(Matrix3DTests, CheckDeleteCol0UsingIndex) {
    Matrix3D<double, double> matrix3D(
            {0.0, 1.0},
            {
                    // 0.0
                    {
                            {0.0, 1.0,  2.0},
                            {3.0,  4.0,  5.0},
                            {6.0,  7.0,  8.0}
                    },
                    // 1.0
                    {
                            {9.0, 10.0, 11.0},
                            {12.0, 13.0, 14.0},
                            {15.0, 16.0, 17.0},
                    }
            }
    );
    matrix3D.setColNames({"C1", "C2", "C3"});
    matrix3D.setRowNames({"R1", "R2", "R3"});

    matrix3D.deleteCol(0);

    Matrix3D<double, double> expected(
            {0.0, 1.0},
            {
                    // 0.0
                    {
                            { 1.0,  2.0},
                            { 4.0,  5.0},
                            { 7.0,  8.0}
                    },
                    // 1.0
                    {
                            {10.0,   11.0},
                            {13.0,   14.0},
                            {16.0,   17.0},
                    }
            }
    );
    std::cout << expected << std::endl;
    std::cout << matrix3D << std::endl;
    ASSERT_TRUE(matrix3D == expected);
    ASSERT_TRUE(matrix3D.getRowNames() == std::vector<std::string>({"R1", "R2", "R3"}));
    ASSERT_TRUE(matrix3D.getColNames() == std::vector<std::string>({"C2", "C3"}));
}
TEST_F(Matrix3DTests, CheckDeleteCol1UsingIndex) {
    Matrix3D<double, double> matrix3D(
            {0.0, 1.0},
            {
                    // 0.0
                    {
                            {0.0, 1.0,  2.0},
                            {3.0,  4.0,  5.0},
                            {6.0,  7.0,  8.0}
                    },
                    // 1.0
                    {
                            {9.0, 10.0, 11.0},
                            {12.0, 13.0, 14.0},
                            {15.0, 16.0, 17.0},
                    }
            }
    );
    matrix3D.setColNames({"C1", "C2", "C3"});
    matrix3D.setRowNames({"R1", "R2", "R3"});

    matrix3D.deleteCol(1);

    Matrix3D<double, double> expected(
            {0.0, 1.0},
            {
                    // 0.0
                    {
                            {0.0,  2.0},
                            {3.0,  5.0},
                            {6.0,  8.0}
                    },
                    // 1.0
                    {
                            {9.0,   11.0},
                            {12.0,  14.0},
                            {15.0,  17.0},
                    }
            }
    );
    ASSERT_TRUE(matrix3D == expected);
    ASSERT_TRUE(matrix3D.getRowNames() == std::vector<std::string>({"R1", "R2", "R3"}));
    ASSERT_TRUE(matrix3D.getColNames() == std::vector<std::string>({"C1", "C3"}));
}
TEST_F(Matrix3DTests, CheckDeleteCol2UsingIndex) {
    Matrix3D<double, double> matrix3D(
            {0.0, 1.0},
            {
                    // 0.0
                    {
                            {0.0, 1.0,  2.0},
                            {3.0,  4.0,  5.0},
                            {6.0,  7.0,  8.0}
                    },
                    // 1.0
                    {
                            {9.0, 10.0, 11.0},
                            {12.0, 13.0, 14.0},
                            {15.0, 16.0, 17.0},
                    }
            }
    );
    matrix3D.setColNames({"C1", "C2", "C3"});
    matrix3D.setRowNames({"R1", "R2", "R3"});

    matrix3D.deleteCol(2);

    Matrix3D<double, double> expected(
            {0.0, 1.0},
            {
                    // 0.0
                    {
                            {0.0,  1.0},
                            {3.0,  4.0},
                            {6.0,  7.0}
                    },
                    // 1.0
                    {
                            {9.0, 10.0},
                            {12.0, 13.0},
                            {15.0, 16.0},
                    }
            }
    );
    ASSERT_TRUE(matrix3D == expected);
    ASSERT_TRUE(matrix3D.getRowNames() == std::vector<std::string>({"R1", "R2", "R3"}));
    ASSERT_TRUE(matrix3D.getColNames() == std::vector<std::string>({"C1", "C2"}));
}


TEST_F(Matrix3DTests, CheckDeleteRow0UsingString) {
    Matrix3D<double, double> matrix3D(
            {0.0, 1.0},
            {
                    // 0.0
                    {
                            {0.0, 1.0,  2.0},
                            {3.0,  4.0,  5.0},
                            {6.0,  7.0,  8.0}
                    },
                    // 1.0
                    {
                            {9.0, 10.0, 11.0},
                            {12.0, 13.0, 14.0},
                            {15.0, 16.0, 17.0},
                    }
            }
    );
    matrix3D.setColNames({"C1", "C2", "C3"});
    matrix3D.setRowNames({"R1", "R2", "R3"});

    matrix3D.deleteRow("R1");

    Matrix3D<double, double> expected(
            {0.0, 1.0},
            {
                    // 0.0
                    {
                            {3.0,  4.0,  5.0},
                            {6.0,  7.0,  8.0}
                    },
                    // 1.0
                    {
                            {12.0, 13.0, 14.0},
                            {15.0, 16.0, 17.0},
                    }
            }
    );
    ASSERT_TRUE(matrix3D == expected);
    ASSERT_TRUE(matrix3D.getRowNames() == std::vector<std::string>({"R2", "R3"}));
    ASSERT_TRUE(matrix3D.getColNames() == std::vector<std::string>({"C1", "C2", "C3"}));
}
TEST_F(Matrix3DTests, CheckDeleteRow1UsingString) {
    Matrix3D<double, double> matrix3D(
            {0.0, 1.0},
            {
                    // 0.0
                    {
                            {0.0, 1.0,  2.0},
                            {3.0,  4.0,  5.0},
                            {6.0,  7.0,  8.0}
                    },
                    // 1.0
                    {
                            {9.0, 10.0, 11.0},
                            {12.0, 13.0, 14.0},
                            {15.0, 16.0, 17.0},
                    }
            }
    );
    matrix3D.setColNames({"C1", "C2", "C3"});
    matrix3D.setRowNames({"R1", "R2", "R3"});

    matrix3D.deleteRow("R2");

    Matrix3D<double, double> expected(
            {0.0, 1.0},
            {
                    // 0.0
                    {
                            {0.0,  1.0,  2.0},
                            {6.0,  7.0,  8.0}
                    },
                    // 1.0
                    {
                            {9.0, 10.0, 11.0},
                            {15.0, 16.0, 17.0},
                    }
            }
    );
    ASSERT_TRUE(matrix3D == expected);
    ASSERT_TRUE(matrix3D.getRowNames() == std::vector<std::string>({"R1", "R3"}));
    ASSERT_TRUE(matrix3D.getColNames() == std::vector<std::string>({"C1", "C2", "C3"}));
}
TEST_F(Matrix3DTests, CheckDeleteRow2UsingString) {
    Matrix3D<double, double> matrix3D(
            {0.0, 1.0},
            {
                    // 0.0
                    {
                            {0.0, 1.0,  2.0},
                            {3.0,  4.0,  5.0},
                            {6.0,  7.0,  8.0}
                    },
                    // 1.0
                    {
                            {9.0, 10.0, 11.0},
                            {12.0, 13.0, 14.0},
                            {15.0, 16.0, 17.0},
                    }
            }
    );
    matrix3D.setColNames({"C1", "C2", "C3"});
    matrix3D.setRowNames({"R1", "R2", "R3"});

    matrix3D.deleteRow("R3");

    Matrix3D<double, double> expected(
            {0.0, 1.0},
            {
                    // 0.0
                    {
                            {0.0,  1.0,  2.0},
                            {3.0,  4.0,  5.0},
                    },
                    // 1.0
                    {
                            {9.0, 10.0, 11.0},
                            {12.0, 13.0, 14.0},
                    }
            }
    );
    ASSERT_TRUE(matrix3D == expected);
    ASSERT_TRUE(matrix3D.getRowNames() == std::vector<std::string>({"R1", "R2"}));
    ASSERT_TRUE(matrix3D.getColNames() == std::vector<std::string>({"C1", "C2", "C3"}));
}

TEST_F(Matrix3DTests, CheckDeleteCol0UsingString) {
    Matrix3D<double, double> matrix3D(
            {0.0, 1.0},
            {
                    // 0.0
                    {
                            {0.0, 1.0,  2.0},
                            {3.0,  4.0,  5.0},
                            {6.0,  7.0,  8.0}
                    },
                    // 1.0
                    {
                            {9.0, 10.0, 11.0},
                            {12.0, 13.0, 14.0},
                            {15.0, 16.0, 17.0},
                    }
            }
    );
    matrix3D.setColNames({"C1", "C2", "C3"});
    matrix3D.setRowNames({"R1", "R2", "R3"});

    matrix3D.deleteCol("C1");

    Matrix3D<double, double> expected(
            {0.0, 1.0},
            {
                    // 0.0
                    {
                            { 1.0,  2.0},
                            { 4.0,  5.0},
                            { 7.0,  8.0}
                    },
                    // 1.0
                    {
                            {10.0,   11.0},
                            {13.0,   14.0},
                            {16.0,   17.0},
                    }
            }
    );
    std::cout << expected << std::endl;
    std::cout << matrix3D << std::endl;
    ASSERT_TRUE(matrix3D == expected);
    ASSERT_TRUE(matrix3D.getRowNames() == std::vector<std::string>({"R1", "R2", "R3"}));
    ASSERT_TRUE(matrix3D.getColNames() == std::vector<std::string>({"C2", "C3"}));
}
TEST_F(Matrix3DTests, CheckDeleteCol1UsingString) {
    Matrix3D<double, double> matrix3D(
            {0.0, 1.0},
            {
                    // 0.0
                    {
                            {0.0, 1.0,  2.0},
                            {3.0,  4.0,  5.0},
                            {6.0,  7.0,  8.0}
                    },
                    // 1.0
                    {
                            {9.0, 10.0, 11.0},
                            {12.0, 13.0, 14.0},
                            {15.0, 16.0, 17.0},
                    }
            }
    );
    matrix3D.setColNames({"C1", "C2", "C3"});
    matrix3D.setRowNames({"R1", "R2", "R3"});

    matrix3D.deleteCol("C2");

    Matrix3D<double, double> expected(
            {0.0, 1.0},
            {
                    // 0.0
                    {
                            {0.0,  2.0},
                            {3.0,  5.0},
                            {6.0,  8.0}
                    },
                    // 1.0
                    {
                            {9.0,   11.0},
                            {12.0,  14.0},
                            {15.0,  17.0},
                    }
            }
    );
    ASSERT_TRUE(matrix3D == expected);
    ASSERT_TRUE(matrix3D.getRowNames() == std::vector<std::string>({"R1", "R2", "R3"}));
    ASSERT_TRUE(matrix3D.getColNames() == std::vector<std::string>({"C1", "C3"}));
}
TEST_F(Matrix3DTests, CheckDeleteCol2UsingString) {
    Matrix3D<double, double> matrix3D(
            {0.0, 1.0},
            {
                    // 0.0
                    {
                            {0.0, 1.0,  2.0},
                            {3.0,  4.0,  5.0},
                            {6.0,  7.0,  8.0}
                    },
                    // 1.0
                    {
                            {9.0, 10.0, 11.0},
                            {12.0, 13.0, 14.0},
                            {15.0, 16.0, 17.0},
                    }
            }
    );
    matrix3D.setColNames({"C1", "C2", "C3"});
    matrix3D.setRowNames({"R1", "R2", "R3"});

    matrix3D.deleteCol("C3");

    Matrix3D<double, double> expected(
            {0.0, 1.0},
            {
                    // 0.0
                    {
                            {0.0,  1.0},
                            {3.0,  4.0},
                            {6.0,  7.0}
                    },
                    // 1.0
                    {
                            {9.0, 10.0},
                            {12.0, 13.0},
                            {15.0, 16.0},
                    }
            }
    );
    ASSERT_TRUE(matrix3D == expected);
    ASSERT_TRUE(matrix3D.getRowNames() == std::vector<std::string>({"R1", "R2", "R3"}));
    ASSERT_TRUE(matrix3D.getColNames() == std::vector<std::string>({"C1", "C2"}));
}









