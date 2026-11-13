//
// Created by Ciaran on 01/11/2021.
//

#include <ModelGeneratorContext.h>
#include "llvm/MCJit.h"
#include "gtest/gtest.h"
#include "TestModelFactory.h"
#include "rrLogger.h"
#include "rrRoadRunnerOptions.h"

#ifdef LIBSBML_HAS_PACKAGE_DISTRIB
#   include "Random.h"
#endif

#include "LLJit.h"

using namespace rr;
using namespace rrllvm;

/**
 * Clearly, these tests are stochastic in nature and so we never test that a particular value is exactly reproducible.
 * It is useful however to have unit tests to check that these functions are properly loaded and at least return a
 * numerical value.
 */
class RandomTests : public ::testing::Test {
public:
    std::unique_ptr<Random> random;
    std::unique_ptr<Jit> jit;
    ModelGeneratorContext ctx;

    RandomTests(std::unique_ptr<Jit> jit_)
            : jit(std::move(jit_)) {
        random = std::move(std::make_unique<Random>(ctx));
        random->setRandomSeed(5);
    };

    void checkDistrib_uniform() const {
#ifdef LIBSBML_HAS_PACKAGE_DISTRIB
        distrib_uniform_FnTy fn = reinterpret_cast<distrib_uniform_FnTy>(jit->lookupFunctionAddress(
                "rr_distrib_uniform"));
        double ans = fn(random.get(), 5, 6);
        ASSERT_LE(ans, 6);
        ASSERT_GE(ans, 5);
#endif
    }

    void checkDistrib_normal() const {
#ifdef LIBSBML_HAS_PACKAGE_DISTRIB
        distrib_normal_FnTy fn = reinterpret_cast<distrib_normal_FnTy>(jit->lookupFunctionAddress("rr_distrib_normal"));
        double val = fn(random.get(), 1, 0.1);
        std::cout << val << std::endl;
        ASSERT_TRUE(typeid(val) == typeid(double));

#endif
    }

    void checkDistrib_normal_four() const {
#ifdef LIBSBML_HAS_PACKAGE_DISTRIB
        distrib_normal_four_FnTy fn = reinterpret_cast<distrib_normal_four_FnTy>(jit->lookupFunctionAddress(
                "rr_distrib_normal_four"));
        double val = fn(random.get(), 10, 1, 8, 11);
        std::cout << val << std::endl;
        ASSERT_TRUE(typeid(val) == typeid(double));
#endif
    }

    void checkDistrib_bernoulli() const {
#ifdef LIBSBML_HAS_PACKAGE_DISTRIB
        distrib_bernoulli_FnTy fn = reinterpret_cast<distrib_bernoulli_FnTy>(jit->lookupFunctionAddress(
                "rr_distrib_bernoulli"));
        double val = fn(random.get(), 0.4);
        std::cout << val << std::endl;
        
        ASSERT_TRUE(typeid(val) == typeid(double));
#endif
    }

    void checkDistrib_binomial() const {
#ifdef LIBSBML_HAS_PACKAGE_DISTRIB
        distrib_binomial_FnTy fn = reinterpret_cast<distrib_binomial_FnTy>(jit->lookupFunctionAddress(
                "rr_distrib_binomial"));
        double val = fn(random.get(), 10, 0.8);
        std::cout << val << std::endl;
        ASSERT_TRUE(typeid(val) == typeid(double));
#endif
    }

    void checkDistrib_binomial_four() const {
#ifdef LIBSBML_HAS_PACKAGE_DISTRIB
        distrib_binomial_four_FnTy fn = reinterpret_cast<distrib_binomial_four_FnTy>(jit->lookupFunctionAddress(
                "rr_distrib_binomial_four"));
        double val = fn(random.get(), 10, 0.8, 3, 20);
        std::cout << val << std::endl;
        ASSERT_TRUE(typeid(val) == typeid(double));
#endif
    }

    void checkDistrib_cauchy() const {
#ifdef LIBSBML_HAS_PACKAGE_DISTRIB
        distrib_cauchy_FnTy fn = reinterpret_cast<distrib_cauchy_FnTy>(jit->lookupFunctionAddress("rr_distrib_cauchy"));
        double val = fn(random.get(), 1, 0.1);
        std::cout << val << std::endl;
        ASSERT_TRUE(typeid(val) == typeid(double));
#endif
    }

    void checkDistrib_cauchy_one() const {
#ifdef LIBSBML_HAS_PACKAGE_DISTRIB
        distrib_cauchy_one_FnTy fn = reinterpret_cast<distrib_cauchy_one_FnTy>(jit->lookupFunctionAddress(
                "rr_distrib_cauchy_one"));
        double val = fn(random.get(), 1);
        std::cout << val << std::endl;
        ASSERT_TRUE(typeid(val) == typeid(double));
#endif
    }

    void checkDistrib_cauchy_four() const {
#ifdef LIBSBML_HAS_PACKAGE_DISTRIB
        distrib_cauchy_four_FnTy fn = reinterpret_cast<distrib_cauchy_four_FnTy>(jit->lookupFunctionAddress(
                "rr_distrib_cauchy_four"));
        double val = fn(random.get(), 10, 0.1, 9, 11);
        std::cout << val << std::endl;
        ASSERT_TRUE(typeid(val) == typeid(double));
#endif
    }

    void checkDistrib_chisquare() const {
#ifdef LIBSBML_HAS_PACKAGE_DISTRIB
        distrib_chisquare_FnTy fn = reinterpret_cast<distrib_chisquare_FnTy>(jit->lookupFunctionAddress(
                "rr_distrib_chisquare"));
        double val = fn(random.get(), 6);
        std::cout << val << std::endl;
        ASSERT_TRUE(typeid(val) == typeid(double));
#endif
    }

    void checkDistrib_chisquare_three() const {
#ifdef LIBSBML_HAS_PACKAGE_DISTRIB
        distrib_chisquare_three_FnTy fn = reinterpret_cast<distrib_chisquare_three_FnTy>(jit->lookupFunctionAddress(
                "rr_distrib_chisquare_three"));
        double val = fn(random.get(), 6, 5, 7);
        std::cout << val << std::endl;
        ASSERT_TRUE(typeid(val) == typeid(double));
#endif
    }

    void checkDistrib_exponential() const {
#ifdef LIBSBML_HAS_PACKAGE_DISTRIB
        distrib_exponential_FnTy fn = reinterpret_cast<distrib_exponential_FnTy>(jit->lookupFunctionAddress(
                "rr_distrib_exponential"));
        double val = fn(random.get(), 4.5);
        std::cout << val << std::endl;
        ASSERT_TRUE(typeid(val) == typeid(double));
#endif
    }

    void checkDistrib_exponential_three() const {
#ifdef LIBSBML_HAS_PACKAGE_DISTRIB
        distrib_exponential_three_FnTy fn = reinterpret_cast<distrib_exponential_three_FnTy>(jit->lookupFunctionAddress(
                "rr_distrib_exponential_three"));
        double val = fn(random.get(), 4.5, 1, 9);
        std::cout << val << std::endl;
        ASSERT_TRUE(typeid(val) == typeid(double));
#endif
    }

    void checkDistrib_gamma() const {
#ifdef LIBSBML_HAS_PACKAGE_DISTRIB
        distrib_gamma_FnTy fn = reinterpret_cast<distrib_gamma_FnTy>(jit->lookupFunctionAddress("rr_distrib_gamma"));
        double val = fn(random.get(), 1, 0.1);
        std::cout << val << std::endl;
        ASSERT_TRUE(typeid(val) == typeid(double));
#endif
    }

    void checkDistrib_gamma_four() const {
#ifdef LIBSBML_HAS_PACKAGE_DISTRIB
        distrib_gamma_four_FnTy fn = reinterpret_cast<distrib_gamma_four_FnTy>(jit->lookupFunctionAddress(
                "rr_distrib_gamma_four"));
        double val = fn(random.get(), 6, 5, 5, 7);
        std::cout << val << std::endl;
        ASSERT_TRUE(typeid(val) == typeid(double));
#endif
    }

    void checkDistrib_laplace() const {
#ifdef LIBSBML_HAS_PACKAGE_DISTRIB
        distrib_laplace_FnTy fn = reinterpret_cast<distrib_laplace_FnTy>(jit->lookupFunctionAddress(
                "rr_distrib_laplace"));
        double val = fn(random.get(), 1, 0.1);
        std::cout << val << std::endl;
        ASSERT_TRUE(typeid(val) == typeid(double));

#endif
    }

    void checkDistrib_laplace_one() const {
#ifdef LIBSBML_HAS_PACKAGE_DISTRIB
        distrib_laplace_one_FnTy fn = reinterpret_cast<distrib_laplace_one_FnTy>(jit->lookupFunctionAddress(
                "rr_distrib_laplace_one"));
        double val = fn(random.get(), 12);
        std::cout << val << std::endl;
        ASSERT_TRUE(typeid(val) == typeid(double));
#endif
    }

    void checkDistrib_laplace_four() const {
#ifdef LIBSBML_HAS_PACKAGE_DISTRIB
        distrib_laplace_four_FnTy fn = reinterpret_cast<distrib_laplace_four_FnTy>(jit->lookupFunctionAddress(
                "rr_distrib_laplace_four"));
        double val = fn(random.get(), 12, 6, 3, 18);
        std::cout << val << std::endl;
        ASSERT_TRUE(typeid(val) == typeid(double));
#endif
    }

    void checkDistrib_lognormal() const {
#ifdef LIBSBML_HAS_PACKAGE_DISTRIB
        distrib_lognormal_FnTy fn = reinterpret_cast<distrib_lognormal_FnTy>(jit->lookupFunctionAddress(
                "rr_distrib_lognormal"));
        double val = fn(random.get(), 20, 10);
        std::cout << val << std::endl;
        ASSERT_TRUE(typeid(val) == typeid(double));
#endif
    }

    void checkDistrib_lognormal_four() const {
#ifdef LIBSBML_HAS_PACKAGE_DISTRIB
        distrib_lognormal_four_FnTy fn = reinterpret_cast<distrib_lognormal_four_FnTy>(jit->lookupFunctionAddress(
                "rr_distrib_lognormal_four"));
        double val = fn(random.get(), 20, 3, 15, 24);
        std::cout << val << std::endl;
        ASSERT_TRUE(typeid(val) == typeid(double));
#endif
    }

    void checkDistrib_poisson() const {
#ifdef LIBSBML_HAS_PACKAGE_DISTRIB
        distrib_poisson_FnTy fn = reinterpret_cast<distrib_poisson_FnTy>(jit->lookupFunctionAddress(
                "rr_distrib_poisson"));
        double val = fn(random.get(), 2);
        std::cout << val << std::endl;
        ASSERT_TRUE(typeid(val) == typeid(double));
#endif
    }

    void checkDistrib_poisson_three() const {
#ifdef LIBSBML_HAS_PACKAGE_DISTRIB
        distrib_poisson_three_FnTy fn = reinterpret_cast<distrib_poisson_three_FnTy>(jit->lookupFunctionAddress(
                "rr_distrib_poisson_three"));
        double val = fn(random.get(), 2, 1, 5);
        std::cout << val << std::endl;
        ASSERT_TRUE(typeid(val) == typeid(double));
#endif
    }

    void checkDistrib_rayleigh() const {
#ifdef LIBSBML_HAS_PACKAGE_DISTRIB
        distrib_rayleigh_FnTy fn = reinterpret_cast<distrib_rayleigh_FnTy>(jit->lookupFunctionAddress(
                "rr_distrib_rayleigh"));
        double val = fn(random.get(), 5);
        std::cout << val << std::endl;
        ASSERT_TRUE(typeid(val) == typeid(double));
#endif
    }

    void checkDistrib_rayleigh_three() const {
#ifdef LIBSBML_HAS_PACKAGE_DISTRIB
        distrib_rayleigh_three_FnTy fn = reinterpret_cast<distrib_rayleigh_three_FnTy>(jit->lookupFunctionAddress(
                "rr_distrib_rayleigh_three"));
        double val = fn(random.get(), 5, 4, 6);
        std::cout << val << std::endl;
        ASSERT_TRUE(typeid(val) == typeid(double));
#endif
    }

};

class RandomTestsMCJit : public RandomTests {
public:
    RandomTestsMCJit()
            : RandomTests(std::make_unique<MCJit>(LoadSBMLOptions().modelGeneratorOpt)) {}
};

TEST_F(RandomTestsMCJit, checkDistrib_uniform){
    checkDistrib_uniform();
}

TEST_F(RandomTestsMCJit, checkDistrib_normal){
    checkDistrib_normal();
}

TEST_F(RandomTestsMCJit, checkDistrib_normal_four){
    checkDistrib_normal_four();
}

TEST_F(RandomTestsMCJit, checkDistrib_bernoulli){
    checkDistrib_bernoulli();
}

TEST_F(RandomTestsMCJit, checkDistrib_binomial){
    checkDistrib_binomial();
}

TEST_F(RandomTestsMCJit, checkDistrib_binomial_four){
    checkDistrib_binomial_four();
}

TEST_F(RandomTestsMCJit, checkDistrib_cauchy){
    checkDistrib_cauchy();
}

TEST_F(RandomTestsMCJit, checkDistrib_cauchy_one){
    checkDistrib_cauchy_one();
}

TEST_F(RandomTestsMCJit, checkDistrib_cauchy_four){
    checkDistrib_cauchy_four();
}

TEST_F(RandomTestsMCJit, checkDistrib_chisquare){
    checkDistrib_chisquare();
}

TEST_F(RandomTestsMCJit, checkDistrib_chisquare_three){
    checkDistrib_chisquare_three();
}

TEST_F(RandomTestsMCJit, checkDistrib_exponential){
    checkDistrib_exponential();
}

TEST_F(RandomTestsMCJit, checkDistrib_exponential_three){
    checkDistrib_exponential_three();
}

TEST_F(RandomTestsMCJit, checkDistrib_gamma){
    checkDistrib_gamma();
}

TEST_F(RandomTestsMCJit, checkDistrib_gamma_four){
    checkDistrib_gamma_four();
}

TEST_F(RandomTestsMCJit, checkDistrib_laplace){
    checkDistrib_laplace();
}

TEST_F(RandomTestsMCJit, checkDistrib_laplace_one){
    checkDistrib_laplace_one();
}

TEST_F(RandomTestsMCJit, checkDistrib_laplace_four){
    checkDistrib_laplace_four();
}

TEST_F(RandomTestsMCJit, checkDistrib_lognormal){
    checkDistrib_lognormal();
}

TEST_F(RandomTestsMCJit, checkDistrib_lognormal_four){
    checkDistrib_lognormal_four();
}

TEST_F(RandomTestsMCJit, checkDistrib_poisson){
    checkDistrib_poisson();
}

TEST_F(RandomTestsMCJit, checkDistrib_poisson_three){
    checkDistrib_poisson_three();
}

TEST_F(RandomTestsMCJit, checkDistrib_rayleigh){
    checkDistrib_rayleigh();
}

TEST_F(RandomTestsMCJit, checkDistrib_rayleigh_three){
    checkDistrib_rayleigh_three();
}


class RandomTestsRRLLJit : public RandomTests {
public:
    RandomTestsRRLLJit()
            : RandomTests(std::make_unique<LLJit>(LoadSBMLOptions().modelGeneratorOpt)) {}
};

TEST_F(RandomTestsRRLLJit, checkDistrib_uniform){
    checkDistrib_uniform();
}

TEST_F(RandomTestsRRLLJit, checkDistrib_normal){
    checkDistrib_normal();
}

TEST_F(RandomTestsRRLLJit, checkDistrib_normal_four){
    checkDistrib_normal_four();
}

TEST_F(RandomTestsRRLLJit, checkDistrib_bernoulli){
    checkDistrib_bernoulli();
}

TEST_F(RandomTestsRRLLJit, checkDistrib_binomial){
    checkDistrib_binomial();
}

TEST_F(RandomTestsRRLLJit, checkDistrib_binomial_four){
    checkDistrib_binomial_four();
}

TEST_F(RandomTestsRRLLJit, checkDistrib_cauchy){
    checkDistrib_cauchy();
}

TEST_F(RandomTestsRRLLJit, checkDistrib_cauchy_one){
    checkDistrib_cauchy_one();
}

TEST_F(RandomTestsRRLLJit, checkDistrib_cauchy_four){
    checkDistrib_cauchy_four();
}

TEST_F(RandomTestsRRLLJit, checkDistrib_chisquare){
    checkDistrib_chisquare();
}

TEST_F(RandomTestsRRLLJit, checkDistrib_chisquare_three){
    checkDistrib_chisquare_three();
}

TEST_F(RandomTestsRRLLJit, checkDistrib_exponential){
    checkDistrib_exponential();
}

TEST_F(RandomTestsRRLLJit, checkDistrib_exponential_three){
    checkDistrib_exponential_three();
}

TEST_F(RandomTestsRRLLJit, checkDistrib_gamma){
    checkDistrib_gamma();
}

TEST_F(RandomTestsRRLLJit, checkDistrib_gamma_four){
    checkDistrib_gamma_four();
}

TEST_F(RandomTestsRRLLJit, checkDistrib_laplace){
    checkDistrib_laplace();
}

TEST_F(RandomTestsRRLLJit, checkDistrib_laplace_one){
    checkDistrib_laplace_one();
}

TEST_F(RandomTestsRRLLJit, checkDistrib_laplace_four){
    checkDistrib_laplace_four();
}

TEST_F(RandomTestsRRLLJit, checkDistrib_lognormal){
    checkDistrib_lognormal();
}

TEST_F(RandomTestsRRLLJit, checkDistrib_lognormal_four){
    checkDistrib_lognormal_four();
}

TEST_F(RandomTestsRRLLJit, checkDistrib_poisson){
    checkDistrib_poisson();
}

TEST_F(RandomTestsRRLLJit, checkDistrib_poisson_three){
    checkDistrib_poisson_three();
}

TEST_F(RandomTestsRRLLJit, checkDistrib_rayleigh){
    checkDistrib_rayleigh();
}

TEST_F(RandomTestsRRLLJit, checkDistrib_rayleigh_three){
    checkDistrib_rayleigh_three();
}
