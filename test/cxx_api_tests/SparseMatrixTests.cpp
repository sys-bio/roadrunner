#include "gtest/gtest.h"
#include "rrSparse.h"
#include <cmath>

using namespace rr;

class SparseMatrixTests : public ::testing::Test {
public:
    SparseMatrixTests() = default;
};

TEST_F(SparseMatrixTests, GetNzValidIndexReturnsStoredValue) {
    csr_matrix *mat = csr_matrix_new(1, 1, {0}, {0}, {42.0});
    ASSERT_NEAR(42.0, csr_matrix_get_nz(mat, 0, 0), 1e-9);
    csr_matrix_delete(mat);
}

/**
 * Regression tests for a strict-bounds bug: csr_matrix_get_nz/set_nz
 * accepted row == m or col == n -- one past the last valid index -- because
 * they checked row <= m / col <= n instead of strict less-than. rowptr is
 * only allocated with m + 1 entries, so row == m reads rowptr[m + 1], one
 * past the allocation: an ASan-detectable heap-buffer-overflow on a 1x1
 * matrix. Under a non-sanitized build this may not crash and could return
 * whatever garbage follows the allocation instead of NaN/false, so these
 * are most decisive when run under ASan or valgrind.
 */
TEST_F(SparseMatrixTests, GetNzRowEqualToDimensionIsRejected) {
    csr_matrix *mat = csr_matrix_new(1, 1, {0}, {0}, {42.0});
    double result = csr_matrix_get_nz(mat, 1, 0); // row == m
    ASSERT_TRUE(std::isnan(result));
    csr_matrix_delete(mat);
}

TEST_F(SparseMatrixTests, GetNzColEqualToDimensionIsRejected) {
    csr_matrix *mat = csr_matrix_new(1, 1, {0}, {0}, {42.0});
    double result = csr_matrix_get_nz(mat, 0, 1); // col == n
    ASSERT_TRUE(std::isnan(result));
    csr_matrix_delete(mat);
}

TEST_F(SparseMatrixTests, SetNzRowEqualToDimensionIsRejected) {
    csr_matrix *mat = csr_matrix_new(1, 1, {0}, {0}, {0.0});
    bool wrote = csr_matrix_set_nz(mat, 1, 0, 7.0); // row == m
    ASSERT_FALSE(wrote);
    csr_matrix_delete(mat);
}

TEST_F(SparseMatrixTests, SetNzColEqualToDimensionIsRejected) {
    csr_matrix *mat = csr_matrix_new(1, 1, {0}, {0}, {0.0});
    bool wrote = csr_matrix_set_nz(mat, 0, 1, 7.0); // col == n
    ASSERT_FALSE(wrote);
    csr_matrix_delete(mat);
}
