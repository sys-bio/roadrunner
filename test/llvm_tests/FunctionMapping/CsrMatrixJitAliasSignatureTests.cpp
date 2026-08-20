//
// Regression tests for the JIT sparse-matrix function-pointer aliases.
//
// csr_matrix_get_nz_FnTy / csr_matrix_set_nz_FnTy (declared in Jit.h) exist
// so that code can safely call through a JIT-looked-up function address
// for csr_matrix_get_nz/csr_matrix_set_nz using these aliases. Before the
// fix the aliases didn't match the real functions at all -- wrong return
// type, wrong parameter count and types -- so calling through them was
// undefined behavior. The existing tests in MapFunctionsToJitSymbolsTests.h
// only checked that the looked-up address was non-null, which can't catch
// a signature mismatch.
//
// This file intentionally calls through the aliases with the real
// arguments (a csr_matrix*, plus row/col indices). Pre-fix, this does not
// even compile -- the alias only accepts two plain ints and returns a
// csr_matrix* -- so a compile failure here is the expected pre-patch
// signal, not a sign these tests are broken. Post-fix it compiles and the
// calls behave correctly.
//

#include "gtest/gtest.h"
#include "llvm/LLJit.h"
#include "llvm/MCJit.h"
#include "Jit.h"
#include "rrSparse.h"
#include "rrRoadRunnerOptions.h"

using namespace rr;
using namespace rrllvm;

TEST(CsrMatrixJitAliasSignatureTests, LLJitGetAndSetNzInvokeCorrectly) {
    LLJit llJit(LoadSBMLOptions().modelGeneratorOpt);

    csr_matrix_get_nz_FnTy getNz =
            (csr_matrix_get_nz_FnTy) llJit.lookupFunctionAddress("csr_matrix_get_nz");
    csr_matrix_set_nz_FnTy setNz =
            (csr_matrix_set_nz_FnTy) llJit.lookupFunctionAddress("csr_matrix_set_nz");
    ASSERT_FALSE(getNz == nullptr);
    ASSERT_FALSE(setNz == nullptr);

    csr_matrix *mat = csr_matrix_new(1, 1, {0}, {0}, {0.0});
    ASSERT_TRUE(setNz(mat, 0, 0, 7.5));
    ASSERT_NEAR(7.5, getNz(mat, 0, 0), 1e-9);
    csr_matrix_delete(mat);
}

TEST(CsrMatrixJitAliasSignatureTests, MCJitGetAndSetNzInvokeCorrectly) {
    MCJit mcJit(LoadSBMLOptions().modelGeneratorOpt);

    csr_matrix_get_nz_FnTy getNz =
            (csr_matrix_get_nz_FnTy) mcJit.lookupFunctionAddress("csr_matrix_get_nz");
    csr_matrix_set_nz_FnTy setNz =
            (csr_matrix_set_nz_FnTy) mcJit.lookupFunctionAddress("csr_matrix_set_nz");
    ASSERT_FALSE(getNz == nullptr);
    ASSERT_FALSE(setNz == nullptr);

    csr_matrix *mat = csr_matrix_new(1, 1, {0}, {0}, {0.0});
    ASSERT_TRUE(setNz(mat, 0, 0, 7.5));
    ASSERT_NEAR(7.5, getNz(mat, 0, 0), 1e-9);
    csr_matrix_delete(mat);
}
