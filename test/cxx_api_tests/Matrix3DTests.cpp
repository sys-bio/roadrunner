//
// Created by Ciaran on 08/07/2021.
//

#include "gtest/gtest.h"
#include "Matrix3D.h"

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











