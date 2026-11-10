//
// Created by Ciaran on 27/10/2021.
//

#include "llvm/LLJit.h"
#include "gtest/gtest.h"
#include "TestModelFactory.h"
#include "rrLogger.h"
#include "rrRoadRunnerOptions.h"
#include "MapFunctionsToJitSymbolsTests.h"

using namespace rr;
using namespace rrllvm;


class LLJitMapFunctionsToJitSymbolsTests : public MapFunctionsToJitSymbolsTests {
public:
    LLJitMapFunctionsToJitSymbolsTests()
        : MapFunctionsToJitSymbolsTests(std::make_unique<LLJit>(LoadSBMLOptions().modelGeneratorOpt)){
        //Logger::setLevel(Logger::LOG_DEBUG);
    };
};

TEST_F(LLJitMapFunctionsToJitSymbolsTests, CheckLibFuncPow) {
    LLJit* llJit = (LLJit*)jit.get();
    std::cout << llJit->emitToString() << std::endl;
    llJit->addIRModule();
    CheckLibFuncPow();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, CheckLibFuncFabs){
    CheckLibFuncFabs();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, CheckLibFuncAcos){
    CheckLibFuncAcos();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, CheckLibFuncAsin){
    CheckLibFuncAsin();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, CheckLibFuncAtan){
    CheckLibFuncAtan();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, CheckLibFuncCeil){
    CheckLibFuncCeil();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, CheckLibFuncCos){
    CheckLibFuncCos();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, CheckLibFuncCosh){
    CheckLibFuncCosh();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, CheckLibFuncExp){
    CheckLibFuncExp();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, CheckLibFuncFloor){
    CheckLibFuncFloor();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, CheckLibFuncLog){
    CheckLibFuncLog();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, CheckLibFuncLog10){
    CheckLibFuncLog10();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, CheckLibFuncSin){
    CheckLibFuncSin();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, CheckLibFuncSinh){
    CheckLibFuncSinh();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, CheckLibFuncTan){
    CheckLibFuncTan();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, CheckLibFuncTanh){
    CheckLibFuncTanh();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, CheckLibFuncFmod){
    CheckLibFuncFmod();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, Check_arccotFnTy){
    Check_arccotFnTy();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, Check_rr_arccot_negzeroFnTy){
    Check_rr_arccot_negzeroFnTy();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, Check_arccothFnTy){
    Check_arccothFnTy();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, Check_arccscFnTy){
    Check_arccscFnTy();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, Check_arccschFnTy){
    Check_arccschFnTy();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, Check_arcsecFnTy){
    Check_arcsecFnTy();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, Check_arcsechFnTy){
    Check_arcsechFnTy();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, Check_cotFnTy){
    Check_cotFnTy();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, Check_cothFnTy){
    Check_cothFnTy();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, Check_cscFnTy){
    Check_cscFnTy();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, Check_cschFnTy){
    Check_cschFnTy();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, Check_rr_factorialiFnTy){
    Check_rr_factorialiFnTy();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, Check_rr_factorialdFnTy){
    Check_rr_factorialdFnTy();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, Check_rr_logdFnTy){
    Check_rr_logdFnTy();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, Check_rr_rootdFnTy){
    Check_rr_rootdFnTy();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, Check_secFnTy){
    Check_secFnTy();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, Check_sechFnTy){
    Check_sechFnTy();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, Check_arccoshFnTy){
    Check_arccoshFnTy();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, Check_arcsinhFnTy){
    Check_arcsinhFnTy();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, Check_arctanhFnTy){
    Check_arctanhFnTy();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, Check_quotientFnTy){
    Check_quotientFnTy();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, Check_rr_maxFnTy){
    Check_rr_maxFnTy();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, Check_rr_minFnTy){
    Check_rr_minFnTy();
}


TEST_F(LLJitMapFunctionsToJitSymbolsTests, Check_rr_csr_matrix_get_nz) {
    Check_rr_csr_matrix_get_nz();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, Check_rr_csr_matrix_set_nz) {
    Check_rr_csr_matrix_set_nz();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_pow_ViaModule){
    check_pow_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_fabs_ViaModule){
    check_fabs_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_acos_ViaModule){
    check_acos_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_asin_ViaModule){
    check_asin_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_atan_ViaModule){
    check_atan_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_ceil_ViaModule){
    check_ceil_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_cos_ViaModule){
    check_cos_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_cosh_ViaModule){
    check_cosh_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_exp_ViaModule){
    check_exp_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_floor_ViaModule){
    check_floor_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_log_ViaModule){
    check_log_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_log10_ViaModule){
    check_log10_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_sin_ViaModule){
    check_sin_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_sinh_ViaModule){
    check_sinh_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_tan_ViaModule){
    check_tan_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_tanh_ViaModule){
    check_tanh_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_fmod_ViaModule){
    check_fmod_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_arccot_ViaModule){
    check_arccot_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_rr_arccot_negzero_ViaModule){
    check_rr_arccot_negzero_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_arccoth_ViaModule){
    check_arccoth_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_arccsc_ViaModule){
    check_arccsc_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_arccsch_ViaModule){
    check_arccsch_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_arcsec_ViaModule){
    check_arcsec_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_arcsech_ViaModule){
    check_arcsech_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_cot_ViaModule){
    check_cot_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_coth_ViaModule){
    check_coth_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_csc_ViaModule){
    check_csc_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_csch_ViaModule){
    check_csch_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_rr_factoriali_ViaModule){
    check_rr_factoriali_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_rr_factoriald_ViaModule){
    check_rr_factoriald_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_rr_logd_ViaModule){
    check_rr_logd_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_rr_rootd_ViaModule){
    check_rr_rootd_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_sec_ViaModule){
    check_sec_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_sech_ViaModule){
    check_sech_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_arccosh_ViaModule){
    check_arccosh_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_arcsinh_ViaModule){
    check_arcsinh_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_arctanh_ViaModule){
    check_arctanh_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_quotient_ViaModule){
    check_quotient_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_rr_max_ViaModule){
    check_rr_max_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_rr_min_ViaModule){
    check_rr_min_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_csr_matrix_get_nz_ViaModule){
    check_csr_matrix_get_nz_ViaModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_csr_matrix_set_nz_ViaModule){
    check_csr_matrix_set_nz_ViaModule();
}

#ifdef LIBSBML_HAS_PACKAGE_DISTRIB

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_rr_distrib_uniform_LoadsFromModule){
    check_rr_distrib_uniform_LoadsFromModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_rr_distrib_normal_LoadsFromModule){
    check_rr_distrib_normal_LoadsFromModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_rr_distrib_normal_four_LoadsFromModule){
    check_rr_distrib_normal_four_LoadsFromModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_rr_distrib_bernoulli_LoadsFromModule){
    check_rr_distrib_bernoulli_LoadsFromModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_rr_distrib_binomial_LoadsFromModule){
    check_rr_distrib_binomial_LoadsFromModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_rr_distrib_binomial_four_LoadsFromModule){
    check_rr_distrib_binomial_four_LoadsFromModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_rr_distrib_cauchy_LoadsFromModule){
    check_rr_distrib_cauchy_LoadsFromModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_rr_distrib_cauchy_one_LoadsFromModule){
    check_rr_distrib_cauchy_one_LoadsFromModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_rr_distrib_cauchy_four_LoadsFromModule){
    check_rr_distrib_cauchy_four_LoadsFromModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_rr_distrib_chisquare_LoadsFromModule){
    check_rr_distrib_chisquare_LoadsFromModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_rr_distrib_chisquare_three_LoadsFromModule){
    check_rr_distrib_chisquare_three_LoadsFromModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_rr_distrib_exponential_LoadsFromModule){
    check_rr_distrib_exponential_LoadsFromModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_rr_distrib_exponential_three_LoadsFromModule){
    check_rr_distrib_exponential_three_LoadsFromModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_rr_distrib_gamma_LoadsFromModule){
    check_rr_distrib_gamma_LoadsFromModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_rr_distrib_gamma_four_LoadsFromModule){
    check_rr_distrib_gamma_four_LoadsFromModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_rr_distrib_laplace_LoadsFromModule){
    check_rr_distrib_laplace_LoadsFromModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_rr_distrib_laplace_one_LoadsFromModule){
    check_rr_distrib_laplace_one_LoadsFromModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_rr_distrib_laplace_four_LoadsFromModule){
    check_rr_distrib_laplace_four_LoadsFromModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_rr_distrib_lognormal_LoadsFromModule){
    check_rr_distrib_lognormal_LoadsFromModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_rr_distrib_lognormal_four_LoadsFromModule){
    check_rr_distrib_lognormal_four_LoadsFromModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_rr_distrib_poisson_LoadsFromModule){
    check_rr_distrib_poisson_LoadsFromModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_rr_distrib_poisson_three_LoadsFromModule){
    check_rr_distrib_poisson_three_LoadsFromModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_rr_distrib_rayleigh_LoadsFromModule){
    check_rr_distrib_rayleigh_LoadsFromModule();
}

TEST_F(LLJitMapFunctionsToJitSymbolsTests, check_rr_distrib_rayleigh_three_LoadsFromModule){
    check_rr_distrib_rayleigh_three_LoadsFromModule();
}



#endif