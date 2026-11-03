// Tests of the exact Jacobian computed by Enzyme on the generated LLVM IR
// (load option ENZYME_JACOBIAN, CMake option RR_USE_ENZYME).

#include "gtest/gtest.h"
#include "rrRoadRunner.h"
#include "rrConfig.h"
#include "rrExecutableModel.h"
#include "llvm/LLVMExecutableModel.h"

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

using namespace rr;

namespace {

    /** Directory of the benchmark models, RR_ENZYME_MODELS or the default checkouts. */
    std::string modelPath(const std::string &envDefault, const std::string &file) {
        const char *env = std::getenv("RR_ENZYME_MODELS");
        std::string dir = env ? std::string(env) : std::string(std::getenv("HOME")) + envDefault;
        return dir + "/" + file;
    }

    /** True if a and b agree within the global tolerance of the prototype. */
    bool close(double a, double b, double scale) {
        return std::fabs(a - b) <= 1e-6 * std::fabs(b) + 1e-8 * scale;
    }

    class EnzymeJacobianTests : public ::testing::Test {
    public:
        EnzymeJacobianTests() {
            Config::setValue(Config::LLVM_ENZYME_JACOBIAN, true);
        }

        ~EnzymeJacobianTests() override {
            Config::setValue(Config::LLVM_ENZYME_JACOBIAN, false);
        }
    };
}

/**
 * Go/no-go gate: the Enzyme tangent of evalReactionRates in direction e_j equals the
 * central difference of the reaction rates in the state y_j.
 */
TEST_F(EnzymeJacobianTests, GateHctzDirectionalDerivative) {
    RoadRunner rr(modelPath("/git/hctz-model/models", "hctz_body_flat.xml"));
    rr.setValue("PODOSE_hctz", 25.0);
    rr.simulate(0, 60, 61);

    auto *model = dynamic_cast<rrllvm::LLVMExecutableModel *>(rr.getModel());
    ASSERT_NE(model, nullptr);

    int n = model->getStateVector(nullptr);
    int nr = model->getNumReactions();
    int nRateRules = n - model->getNumFloatingSpecies();
    ASSERT_GT(nr, 0);

    std::vector<double> y(n);
    model->getStateVector(y.data());

    int nfs = model->getNumFloatingSpecies();
    for (int j: {nRateRules, nRateRules + nfs / 2, n - 1}) {
        std::vector<double> seed(n, 0.0), dv(nr, 0.0);
        seed[j] = 1.0;
        model->setStateVector(y.data());
        model->evalReactionRatesTangent(seed.data(), dv.data());

        double h = 1e-6 * std::max(std::fabs(y[j]), 1e-6);
        std::vector<double> yp(y), ym(y), vp(nr), vm(nr);
        yp[j] += h;
        ym[j] -= h;
        model->setStateVector(yp.data());
        model->getReactionRates(nr, nullptr, vp.data());
        model->setStateVector(ym.data());
        model->getReactionRates(nr, nullptr, vm.data());

        std::vector<double> fd(nr);
        double scale = 0.0;
        for (int r = 0; r < nr; ++r) {
            fd[r] = (vp[r] - vm[r]) / (2 * h);
            scale = std::max(scale, std::fabs(fd[r]));
        }
        int nonzero = 0;
        for (int r = 0; r < nr; ++r) {
            EXPECT_TRUE(close(dv[r], fd[r], scale))
                                << "state " << j << " reaction " << r << ": enzyme " << dv[r] << " fd " << fd[r];
            nonzero += dv[r] != 0.0;
        }
        EXPECT_GT(nonzero, 0) << "state " << j << " has an all-zero column";
    }
    model->setStateVector(y.data());
}
