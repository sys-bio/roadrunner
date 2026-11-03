// Tests of the exact Jacobian computed by Enzyme on the generated LLVM IR
// (load option ENZYME_JACOBIAN, CMake option RR_USE_ENZYME).

#include "gtest/gtest.h"
#include "rrRoadRunner.h"
#include "rrConfig.h"
#include "rrExecutableModel.h"
#include "llvm/LLVMExecutableModel.h"
#include "SbmlModelBuilder.h"

#include <algorithm>
#include <cmath>
#include <functional>
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

namespace {

    /**
     * Central, forward and backward difference Jacobians of the state vector rate at the
     * current state, row major. At a kink (max, min, piecewise switching exactly at the state)
     * the exact derivative is the one of the active branch, which is one of the one-sided
     * differences, while the central difference averages the two.
     */
    struct FDJacobians {
        std::vector<double> central, forward, backward;
    };

    FDJacobians fdJacobians(ExecutableModel *model, double floor) {
        int n = model->getStateVector(nullptr);
        std::vector<double> y(n), f0(n), fp(n), fm(n);
        FDJacobians J{std::vector<double>(n * n), std::vector<double>(n * n), std::vector<double>(n * n)};
        model->getStateVector(y.data());
        double t = model->getTime();
        model->getStateVectorRate(t, y.data(), f0.data());
        for (int j = 0; j < n; ++j) {
            double h = 1e-6 * std::max(std::fabs(y[j]), floor);
            std::vector<double> yp(y), ym(y);
            yp[j] += h;
            ym[j] -= h;
            model->getStateVectorRate(t, yp.data(), fp.data());
            model->getStateVectorRate(t, ym.data(), fm.data());
            for (int i = 0; i < n; ++i) {
                J.central[i * n + j] = (fp[i] - fm[i]) / (2 * h);
                J.forward[i * n + j] = (fp[i] - f0[i]) / h;
                J.backward[i * n + j] = (f0[i] - fm[i]) / h;
            }
        }
        model->getStateVectorRate(t, y.data(), fp.data());
        return J;
    }

    /** Compare the Enzyme Jacobian of rr with finite differences at the current state. */
    void expectJacobianMatchesFD(RoadRunner &rr, const std::string &label) {
        ExecutableModel *model = rr.getModel();
        ASSERT_TRUE(model->hasStateVectorJacobian()) << label << ": " << model->getStateVectorJacobianUnavailableReason();
        ls::DoubleMatrix J = rr.getStateVectorJacobian();
        ls::DoubleMatrix P = rr.getStateVectorJacobianPattern();
        int n = model->getStateVector(nullptr);
        ASSERT_EQ((int) J.numRows(), n);
        // The step 1e-6 * max(|y_j|, 1e-6) is below the rounding of the other terms of a rate
        // when y_j is zero and the rate is large; a second reference with the floor 1 covers that.
        FDJacobians fdj = fdJacobians(model, 1e-6);
        FDJacobians fdj1 = fdJacobians(model, 1.0);
        const std::vector<double> &fd = fdj.central;
        for (int i = 0; i < n; ++i) {
            double scale = 0.0;
            for (int j = 0; j < n; ++j) {
                scale = std::max(scale, std::fabs(fd[i * n + j]));
            }
            for (int j = 0; j < n; ++j) {
                double f = fd[i * n + j];
                // one-sided differences are accurate to about sqrt(eps)
                auto oneSided = [&](double g) {
                    return std::fabs(J(i, j) - g) <= 1e-4 * std::fabs(g) + 1e-6 * scale;
                };
                EXPECT_TRUE(close(J(i, j), f, scale) || oneSided(fdj.forward[i * n + j]) ||
                            oneSided(fdj.backward[i * n + j]) || close(J(i, j), fdj1.central[i * n + j], scale) ||
                            oneSided(fdj1.forward[i * n + j]) || oneSided(fdj1.backward[i * n + j]))
                                    << label << " J(" << i << "," << j << "): enzyme " << J(i, j) << " fd " << f
                                    << " forward " << fdj.forward[i * n + j] << " backward "
                                    << fdj.backward[i * n + j] << " central (floor 1) " << fdj1.central[i * n + j];
                if (std::fabs(f) > 1e-8 * scale && std::fabs(f) > 1e-12) {
                    EXPECT_EQ(P(i, j), 1.0) << label << " fd nonzero (" << i << "," << j << ") = " << f
                                            << " outside the pattern";
                }
            }
        }
    }

    std::vector<std::pair<std::string, std::string>> smallModels() {
        std::vector<std::pair<std::string, std::string>> models;
        {
            ModelBuilder b;
            b.compartment("c").species("S1").species("S2").parameter("k", 0.1)
                    .reaction("R1", {"S1"}, {"S2"}, "k*S1").reaction("R2", {"S2"}, {"S1"}, "0.3*S2*S2");
            models.push_back({"mass action", b.sbml()});
        }
        {
            ModelBuilder b;
            b.compartment("c").species("S1").species("S2").species("S3").parameter("p", 0.0, false)
                    .assignmentRule("p", "2*S3*S3").reaction("R1", {"S1"}, {"S2"}, "p*S1")
                    .reaction("R2", {"S3"}, {}, "0.1*S3");
            models.push_back({"assignment rule", b.sbml()});
        }
        {
            ModelBuilder b;
            b.compartment("c").species("S1").species("S2").parameter("k", 0.1)
                    .function("f", "lambda(x, y, x*y/(1+x))").reaction("R1", {"S1"}, {"S2"}, "f(S1, k)");
            models.push_back({"function definition", b.sbml()});
        }
        {
            ModelBuilder b;
            b.compartment("c").species("S1").species("S2").parameter("k", 0.1)
                    .reaction("R1", {"S2"}, {"S1"}, "piecewise(S1*S2, S1 > 0.5, k*S2)");
            models.push_back({"piecewise", b.sbml()});
        }
        {
            ModelBuilder b;
            b.compartment("c", false).species("S1", "c", false).species("S2", "c", false).parameter("k", 0.1)
                    .rateRule("c", "0.01*S2").reaction("R1", {"S1"}, {"S2"}, "k*S1*c");
            models.push_back({"variable compartment", b.sbml()});
        }
        {
            ModelBuilder b;
            b.compartment("c").species("S1").species("S2").parameter("x", 1.0, false)
                    .rateRule("x", "-0.2*x*S1 + R1").reaction("R1", {"S1"}, {"S2"}, "x*S1");
            models.push_back({"rate rule reading a reaction", b.sbml()});
        }
        return models;
    }
}

TEST_F(EnzymeJacobianTests, FullJacobianMatchesFD) {
    for (const auto &[name, sbml]: smallModels()) {
        RoadRunner rr(sbml);
        rr.simulate(0, 2, 3);
        expectJacobianMatchesFD(rr, name);
    }
    struct Case {
        const char *dir, *file, *dose;
        double value;
    };
    for (const Case &c: {Case{"/git/hctz-model/models", "hctz_body_flat.xml", "PODOSE_hctz", 25.0},
                         Case{"/git/glimepiride-model/models", "glimepiride_body_flat.xml", "PODOSE_gli", 4.0}}) {
        RoadRunner rr(modelPath(c.dir, c.file));
        rr.setValue(c.dose, c.value);
        double t0 = 0;
        for (double t: {0.0, 60.0, 1440.0}) {
            if (t > t0) {
                rr.simulate(t0, t, 11);
                t0 = t;
            }
            expectJacobianMatchesFD(rr, std::string(c.file) + " t=" + std::to_string(t));
        }
    }
}

TEST_F(EnzymeJacobianTests, HelperDerivatives) {
    ModelBuilder b;
    b.compartment("c").species("S1").species("S2").parameter("k", 1.0)
            .reaction("R1", {"S2"}, {}, "max(S1, k)*S2")
            .reaction("R2", {"S2"}, {}, "min(S1, k)*S2")
            .reaction("R3", {"S2"}, {}, "log(10, S1)*S2")
            .reaction("R4", {"S2"}, {}, "root(3, S1)*S2")
            .reaction("R5", {"S2"}, {}, "arccot(S1)*S2");
    RoadRunner rr(b.sbml());
    for (double s1: {0.5, 2.0}) {
        rr.setValue("S1", s1);
        expectJacobianMatchesFD(rr, "S1=" + std::to_string(s1));
    }
}

TEST_F(EnzymeJacobianTests, FallbackReasons) {
    auto reason = [](RoadRunner &rr) {
        EXPECT_FALSE(rr.getModel()->hasStateVectorJacobian());
        return rr.getModel()->getStateVectorJacobianUnavailableReason();
    };
    {
        ModelBuilder b;
        b.compartment("c").species("S1").species("S2").reaction("R1", {"S1"}, {"S2"}, "0.1*S1");
        Config::setValue(Config::LOADSBMLOPTIONS_CONSERVED_MOIETIES, true);
        RoadRunner rr(b.sbml());
        Config::setValue(Config::LOADSBMLOPTIONS_CONSERVED_MOIETIES, false);
        EXPECT_NE(reason(rr).find("conserved moiet"), std::string::npos) << reason(rr);
    }
    {
        ModelBuilder b;
        b.compartment("c").species("S1").species("S2").reaction("R1", {"S1"}, {"S2"}, "0.1*S1");
        auto *ref = b.model->getReaction("R1")->getReactant(0);
        ref->setId("st");
        ref->setConstant(false);
        b.assignmentRule("st", "1 + S2");
        RoadRunner rr(b.sbml());
        EXPECT_NE(reason(rr).find("stoichiometry"), std::string::npos) << reason(rr);
    }
    {
        ModelBuilder b;
        b.compartment("c").species("S1").species("S2").parameter("cf", 1.0, false)
                .assignmentRule("cf", "1 + S1").conversionFactor("cf")
                .reaction("R1", {"S1"}, {"S2"}, "0.1*S1");
        RoadRunner rr(b.sbml());
        EXPECT_NE(reason(rr).find("conversion factor"), std::string::npos) << reason(rr);
    }
    {
        ModelBuilder b;
        b.compartment("c").species("S1").species("S2").reaction("R1", {"S1"}, {"S2"}, "factorial(S1)");
        RoadRunner rr(b.sbml());
        EXPECT_NE(reason(rr).find("rr_factoriald"), std::string::npos) << reason(rr);
        rr.simulate(0, 1, 2);  // the model itself still works
    }
}

TEST_F(EnzymeJacobianTests, JacobianFollowsSetValue) {
    ModelBuilder b;
    b.compartment("c").species("S1", "c", false).species("S2", "c", false).parameter("k", 0.1)
            .reaction("R1", {"S1"}, {"S2"}, "k*S1*S2");
    RoadRunner rr(b.sbml());
    ls::DoubleMatrix before = rr.getStateVectorJacobian();
    rr.setValue("k", 0.2);
    expectJacobianMatchesFD(rr, "after k");
    ls::DoubleMatrix after = rr.getStateVectorJacobian();
    EXPECT_NEAR(after(0, 0), 2 * before(0, 0), 1e-12);
    rr.setValue("c", 2.0);
    expectJacobianMatchesFD(rr, "after volume");
}

TEST_F(EnzymeJacobianTests, CacheKeySeparatesEnzyme) {
    ModelBuilder b;
    b.compartment("c").species("S1").species("S2").reaction("R1", {"S1"}, {"S2"}, "0.1*S1");
    Config::setValue(Config::LLVM_ENZYME_JACOBIAN, false);
    RoadRunner plain(b.sbml());
    Config::setValue(Config::LLVM_ENZYME_JACOBIAN, true);
    RoadRunner enzyme(b.sbml());
    EXPECT_FALSE(plain.getModel()->hasStateVectorJacobian());
    EXPECT_NE(plain.getModel()->getStateVectorJacobianUnavailableReason().find("LLVM_ENZYME_JACOBIAN"),
              std::string::npos);
    EXPECT_TRUE(enzyme.getModel()->hasStateVectorJacobian());
}

TEST_F(EnzymeJacobianTests, DegenerateModels) {
    {
        ModelBuilder b;
        b.compartment("c").parameter("x", 1.0, false).parameter("y", 2.0, false)
                .rateRule("x", "-0.5*x*y").rateRule("y", "x");
        RoadRunner rr(b.sbml());
        expectJacobianMatchesFD(rr, "rate rules only");
    }
    {
        ModelBuilder b;
        b.compartment("c").parameter("k", 1.0);
        RoadRunner rr(b.sbml());
        ASSERT_TRUE(rr.getModel()->hasStateVectorJacobian()) << rr.getModel()->getStateVectorJacobianUnavailableReason();
        EXPECT_EQ(rr.getStateVectorJacobian().numRows(), 0u);
        rr.simulate(0, 1, 2);
    }
}
