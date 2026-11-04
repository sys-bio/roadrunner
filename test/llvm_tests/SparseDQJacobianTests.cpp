// Tests of the sparse difference quotient Jacobian and of CVODE with it and with KLU.

#include "gtest/gtest.h"
#include "rrRoadRunner.h"
#include "rrConfig.h"
#include "rrExecutableModel.h"
#include "rrLogger.h"
#include "CVODEIntegrator.h"
#include "SbmlModelBuilder.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

using namespace rr;

namespace {

    /** Runs every test with both JIT backends of roadrunner. */
    class SparseDQJacobianTests : public ::testing::TestWithParam<Config::LLVM_BACKEND_VALUES> {
    public:
        SparseDQJacobianTests() : backend_(Config::getValue(Config::LLVM_BACKEND).getAs<int>()) {
            Config::setValue(Config::LLVM_BACKEND, static_cast<int>(GetParam()));
        }

        ~SparseDQJacobianTests() override {
            Config::setValue(Config::LLVM_BACKEND, backend_);
        }

    private:
        int backend_;
    };

    std::vector<std::pair<std::string, std::string>> smallModels() {
        std::vector<std::pair<std::string, std::string>> models;
        {
            ModelBuilder b;
            b.compartment("c").species("S1").species("S2").species("S3").parameter("k", 0.1)
                    .reaction("R1", {"S1"}, {"S2"}, "k*S1").reaction("R2", {"S2"}, {"S1"}, "0.3*S2*S2")
                    .reaction("R3", {"S2"}, {"S3"}, "0.2*S2");
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
        {
            // a function without a derivative for Enzyme is no problem for differences
            ModelBuilder b;
            b.compartment("c").species("S1").species("S2").reaction("R1", {"S1"}, {"S2"}, "0.1*S1 + 0.01*factorial(3)");
            models.push_back({"factorial", b.sbml()});
        }
        return models;
    }

    std::string fmt(double x) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.17g", x);
        return buf;
    }

    /**
     * Compares the sparse difference quotient of the model at its current state with the dense
     * forward difference of the state vector rate with the same increments: they must agree
     * up to rounding, and the dense forward difference must have no nonzero outside the
     * pattern. Returns the first mismatch, or an empty string.
     */
    std::string compareSparseDQWithDenseDQ(ExecutableModel *model) {
        const CscPattern *p = model->getStateVectorJacobianPattern();
        if (!p) {
            return "no pattern: " + model->getStateVectorJacobianUnavailableReason();
        }
        int n = model->getStateVector(nullptr);
        if (p->nCols != n || p->nRows != n) {
            return "the pattern is " + std::to_string(p->nRows) + "x" + std::to_string(p->nCols) +
                   ", the state vector has " + std::to_string(n) + " entries";
        }
        if (n > 0 && model->getStateVectorJacobianNumColors() <= 0) {
            return "no colors";
        }
        double t = model->getTime();
        std::vector<double> y(n), inc(n), values(p->nnz());
        model->getStateVector(y.data());
        for (int j = 0; j < n; ++j) {
            inc[j] = 1e-7 * std::max(std::fabs(y[j]), 1.0);
        }
        model->getStateVectorJacobianDQ(t, y.data(), inc.data(), values.data());

        // the size of the rates, for the rounding error of the differences
        std::vector<double> v(model->getNumReactions());
        model->getReactionRates(v.size(), nullptr, v.data());
        double rates = 0;
        for (double r: v) {
            rates += std::fabs(r);
        }

        std::vector<double> f0(n), f1(n), yj(y);
        model->getStateVectorRate(t, y.data(), f0.data());
        for (int j = 0; j < n; ++j) {
            yj[j] = y[j] + inc[j];
            model->getStateVectorRate(t, yj.data(), f1.data());
            double h = yj[j] - y[j];
            yj[j] = y[j];
            std::vector<double> column(n);
            double scale = 0;
            for (int i = 0; i < n; ++i) {
                column[i] = (f1[i] - f0[i]) / h;
                scale = std::max(scale, std::fabs(column[i]));
            }
            for (int k = p->colPtr[j]; k < p->colPtr[j + 1]; ++k) {
                int i = p->rowIdx[k];
                // The two differ only by rounding: the dense difference subtracts the sums of the
                // reaction rates, the sparse one the rates. The rounding error of either is
                // about eps times the size of the rates divided by the increment.
                const double eps = std::numeric_limits<double>::epsilon();
                double rounding = 100 * eps * (rates + std::fabs(f0[i]) + std::fabs(f1[i])) / h;
                if (!(std::fabs(values[k] - column[i]) <= 1e-8 * (std::fabs(column[i]) + scale) + rounding + 1e-14)) {
                    return "J(" + std::to_string(i) + "," + std::to_string(j) + "): sparse " +
                           fmt(values[k]) + ", dense " + fmt(column[i]) + ", scale " + fmt(scale) + ", h " + fmt(h) + ", |f0| " + fmt(std::fabs(f0[i]));
                }
                column[i] = 0;
            }
            for (int i = 0; i < n; ++i) {
                if (column[i] != 0.0) {
                    return "J(" + std::to_string(i) + "," + std::to_string(j) + ") = " +
                           std::to_string(column[i]) + " is outside the pattern";
                }
            }
        }
        return "";
    }

    void expectSparseDQMatchesDenseDQ(ExecutableModel *model, const std::string &label) {
        EXPECT_EQ(compareSparseDQWithDenseDQ(model), "") << label;
    }

    /** Simulate with the given Jacobian and linear solver and return the result. */
    ls::DoubleMatrix simulate(RoadRunner &rr, const std::string &jacobian, const std::string &solver,
                              double end = 10) {
        rr.getIntegrator()->setValue("jacobian", Setting(jacobian));
        rr.getIntegrator()->setValue("linear_solver", Setting(solver));
        rr.getIntegrator()->setValue("relative_tolerance", Setting(1e-10));
        rr.getIntegrator()->setValue("absolute_tolerance", Setting(1e-12));
        rr.reset(SelectionRecord::ALL);
        return *rr.simulate(0, end, 21);
    }

    void expectSameResult(const ls::DoubleMatrix &a, const ls::DoubleMatrix &b, const std::string &label) {
        ASSERT_EQ(a.numRows(), b.numRows());
        ASSERT_EQ(a.numCols(), b.numCols());
        for (unsigned i = 0; i < a.numRows(); ++i) {
            for (unsigned j = 0; j < a.numCols(); ++j) {
                EXPECT_NEAR(a(i, j), b(i, j), 1e-6 * std::fabs(a(i, j)) + 1e-9) << label << " (" << i << "," << j << ")";
            }
        }
    }
}

TEST_P(SparseDQJacobianTests, SparseDQMatchesDenseDQ) {
    for (const auto &[name, sbml]: smallModels()) {
        RoadRunner rr(sbml);
        expectSparseDQMatchesDenseDQ(rr.getModel(), name + " t=0");
        rr.simulate(0, 2, 3);
        expectSparseDQMatchesDenseDQ(rr.getModel(), name + " t=2");
    }
}

/** Every column of a chain of reactions has its own row in two reactions, so it needs three colors. */
TEST_P(SparseDQJacobianTests, ChainNeedsFewColors) {
    ModelBuilder b;
    b.compartment("c");
    int n = 50;
    for (int i = 0; i < n; ++i) {
        b.species("S" + std::to_string(i));
    }
    for (int i = 0; i + 1 < n; ++i) {
        b.reaction("R" + std::to_string(i), {"S" + std::to_string(i)}, {"S" + std::to_string(i + 1)},
                   "0.1*S" + std::to_string(i));
    }
    RoadRunner rr(b.sbml());
    EXPECT_LE(rr.getModel()->getStateVectorJacobianNumColors(), 3);
    expectSparseDQMatchesDenseDQ(rr.getModel(), "chain");
}

TEST_P(SparseDQJacobianTests, FallbackReasons) {
    auto reason = [](RoadRunner &rr) {
        EXPECT_EQ(rr.getModel()->getStateVectorJacobianPattern(), nullptr);
        EXPECT_EQ(rr.getModel()->getStateVectorJacobianNumColors(), 0);
        return rr.getModel()->getStateVectorJacobianUnavailableReason();
    };
    {
        ModelBuilder b;
        b.compartment("c").species("S1").species("S2").reaction("R1", {"S1"}, {"S2"}, "0.1*S1");
        Config::setValue(Config::LOADSBMLOPTIONS_CONSERVED_MOIETIES, true);
        RoadRunner rr(b.sbml());
        Config::setValue(Config::LOADSBMLOPTIONS_CONSERVED_MOIETIES, false);
        EXPECT_NE(reason(rr).find("conserved moiet"), std::string::npos) << reason(rr);
        // CVODE falls back to the dense difference quotient
        ls::DoubleMatrix sparse = simulate(rr, "sparse_difference_quotient", "klu");
        ls::DoubleMatrix dense = simulate(rr, "difference_quotient", "dense");
        expectSameResult(dense, sparse, "conserved moieties");
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
}

/**
 * CVODE with the sparse difference quotient, with the dense solver and with KLU, gives the
 * result of CVODE's own dense difference quotient, also across events.
 */
TEST_P(SparseDQJacobianTests, CvodeSimulatesTheSame) {
    std::vector<std::pair<std::string, std::string>> models = smallModels();
    {
        ModelBuilder b;
        b.compartment("c").species("S1").species("S2").parameter("k", 0.5)
                .reaction("R1", {"S1"}, {"S2"}, "k*S1").reaction("R2", {"S2"}, {"S1"}, "0.1*S2");
        libsbml::Event *e = b.model->createEvent();
        e->setId("E");
        e->setUseValuesFromTriggerTime(true);
        libsbml::Trigger *trigger = e->createTrigger();
        trigger->setInitialValue(false);
        trigger->setPersistent(true);
        trigger->setMath(libsbml::SBML_parseL3Formula("S1 < 0.5"));
        libsbml::EventAssignment *ea = e->createEventAssignment();
        ea->setVariable("S1");
        ea->setMath(libsbml::SBML_parseL3Formula("S1 + 2"));
        models.push_back({"event", b.sbml()});
    }
    for (const auto &[name, sbml]: models) {
        RoadRunner rr(sbml);
        ls::DoubleMatrix reference = simulate(rr, "difference_quotient", "dense");
        ls::DoubleMatrix sparseDense = simulate(rr, "sparse_difference_quotient", "dense");
        std::unordered_map<std::string, Setting> stats = rr.getIntegrator()->getStatistics();
        EXPECT_FALSE(stats["sparse_linear_solver"].getAs<bool>()) << name;
        EXPECT_GT(stats["num_jac_colors"].getAs<int>(), 0) << name;
        EXPECT_EQ(stats["num_lin_rhs_evals"].getAs<std::int64_t>(), 0) << name;
        expectSameResult(reference, sparseDense, name + ", dense solver");
#ifdef RR_USE_KLU
        ls::DoubleMatrix sparseKlu = simulate(rr, "sparse_difference_quotient", "klu");
        stats = rr.getIntegrator()->getStatistics();
        EXPECT_TRUE(stats["sparse_linear_solver"].getAs<bool>()) << name;
        expectSameResult(reference, sparseKlu, name + ", KLU");
#endif
    }
}

/** The settings can be changed in any order, and an invalid value changes nothing. */
TEST_P(SparseDQJacobianTests, CvodeSettings) {
    RoadRunner rr(smallModels()[0].second);
    Integrator *integrator = rr.getIntegrator();
    ASSERT_THROW(integrator->setValue("jacobian", Setting(std::string("enzyme"))), std::invalid_argument);
    ASSERT_THROW(integrator->setValue("linear_solver", Setting(std::string("lapack"))), std::invalid_argument);
    EXPECT_EQ(integrator->getValue("jacobian").getAs<std::string>(), "difference_quotient");
    EXPECT_EQ(integrator->getValue("linear_solver").getAs<std::string>(), "dense");
    // KLU without the sparse difference quotient falls back to the dense solver
    integrator->setValue("linear_solver", Setting(std::string("klu")));
    rr.simulate(0, 1, 2);
    EXPECT_FALSE(integrator->getStatistics()["sparse_linear_solver"].getAs<bool>());
    integrator->setValue("jacobian", Setting(std::string("sparse_difference_quotient")));
    rr.simulate(0, 1, 2);
#ifdef RR_USE_KLU
    EXPECT_TRUE(integrator->getStatistics()["sparse_linear_solver"].getAs<bool>());
#endif
    // the non-stiff solver has no linear solver
    integrator->setValue("stiff", Setting(false));
    rr.simulate(0, 1, 2);
    EXPECT_FALSE(integrator->getStatistics()["sparse_linear_solver"].getAs<bool>());
}

/**
 * Validation on the semantic cases of the SBML test suite in the directory RR_SBML_TEST_SUITE
 * (e.g. test/sbml-test-suite/semantic); skipped without it. Every case that roadrunner loads
 * and that has a pattern is compared at its start and in the middle of its timecourse.
 */
TEST_P(SparseDQJacobianTests, SbmlTestSuite) {
    const char *dir = std::getenv("RR_SBML_TEST_SUITE");
    if (!dir) {
        GTEST_SKIP() << "RR_SBML_TEST_SUITE is not set";
    }
    Logger::setLevel(Logger::LOG_FATAL);
    int compared = 0, notLoaded = 0;
    std::map<std::string, int> fallbacks;
    for (int c = 1; c <= 1900; ++c) {
        char id[8];
        std::snprintf(id, sizeof(id), "%05d", c);
        std::string path;
        for (const char *level: {"l3v2", "l3v1", "l2v5", "l2v4"}) {
            std::string candidate = std::string(dir) + "/" + id + "/" + id + "-sbml-" + level + ".xml";
            if (std::filesystem::exists(candidate)) {
                path = candidate;
                break;
            }
        }
        if (path.empty()) {
            continue;
        }
        std::unique_ptr<RoadRunner> rr;
        try {
            rr = std::make_unique<RoadRunner>(path);
        } catch (const std::exception &) {
            notLoaded++;
            continue;
        }
        ExecutableModel *model = rr->getModel();
        if (!model || !model->getStateVectorJacobianPattern()) {
            std::string reason = model ? model->getStateVectorJacobianUnavailableReason() : "no model";
            fallbacks[reason.substr(0, reason.find('\''))]++;
            continue;
        }
        compared++;
        EXPECT_EQ(compareSparseDQWithDenseDQ(model), "") << id << " at the start";
        try {
            rr->simulate(0, 2, 3);
        } catch (const std::exception &) {
            continue;
        }
        EXPECT_EQ(compareSparseDQWithDenseDQ(model), "") << id << " at t = 2";
    }
    std::cout << "compared " << compared << ", not loaded " << notLoaded << std::endl;
    for (const auto &[reason, count]: fallbacks) {
        std::cout << "fallback: " << count << " x " << reason << std::endl;
    }
    EXPECT_GT(compared, 1000);
}

INSTANTIATE_TEST_SUITE_P(Backends, SparseDQJacobianTests,
                         ::testing::Values(Config::MCJIT, Config::LLJIT),
                         [](const ::testing::TestParamInfo<Config::LLVM_BACKEND_VALUES> &info) {
                             return std::string(info.param == Config::MCJIT ? "MCJit" : "LLJit");
                         });
