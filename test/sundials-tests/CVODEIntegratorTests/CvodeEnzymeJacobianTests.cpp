// CVODE with the exact Jacobian computed by Enzyme (settings "jacobian" and "linear_solver").

#include "gtest/gtest.h"
#include "rrRoadRunner.h"
#include "rrConfig.h"
#include "CVODEIntegrator.h"
#include "../../llvm_tests/SbmlModelBuilder.h"

#include <cmath>
#include <cstdlib>
#include <string>

using namespace rr;

namespace {

    std::string modelPath(const std::string &dirInHome, const std::string &file) {
        const char *env = std::getenv("RR_ENZYME_MODELS");
        std::string dir = env ? std::string(env) : std::string(std::getenv("HOME")) + dirInHome;
        return dir + "/" + file;
    }

    std::string hctz() { return modelPath("/git/hctz-model/models", "hctz_body_flat.xml"); }

    std::string glimepiride() { return modelPath("/git/glimepiride-model/models", "glimepiride_body_flat.xml"); }

    CVODEIntegrator *cvode(RoadRunner &rr) {
        auto *integrator = dynamic_cast<CVODEIntegrator *>(rr.getIntegrator());
        EXPECT_NE(integrator, nullptr);
        return integrator;
    }

    /** Reset, dose and simulate with the given settings; tolerances as in the benchmark. */
    ls::DoubleMatrix run(RoadRunner &rr, const std::string &dose, double value, const std::string &jacobian,
                         const std::string &linearSolver = "dense") {
        rr.reset();
        rr.getIntegrator()->setValue("jacobian", Setting(jacobian));
        rr.getIntegrator()->setValue("linear_solver", Setting(linearSolver));
        rr.getIntegrator()->setValue("relative_tolerance", Setting(1e-10));
        rr.getIntegrator()->setValue("absolute_tolerance", Setting(1e-10));
        if (!dose.empty()) {
            rr.setValue(dose, value);
        }
        return *rr.simulate(0, 1440, 201);
    }

    void expectSameResults(const ls::DoubleMatrix &a, const ls::DoubleMatrix &b, const std::string &label) {
        ASSERT_EQ(a.numRows(), b.numRows());
        ASSERT_EQ(a.numCols(), b.numCols());
        int bad = 0;
        for (unsigned i = 0; i < a.numRows(); ++i) {
            for (unsigned j = 0; j < a.numCols(); ++j) {
                if (std::fabs(a(i, j) - b(i, j)) > 1e-6 * std::fabs(b(i, j)) + 1e-10) {
                    if (bad++ < 5) {
                        ADD_FAILURE() << label << " row " << i << " col " << j << ": " << a(i, j) << " vs " << b(i, j);
                    }
                }
            }
        }
        EXPECT_EQ(bad, 0) << label;
    }

    class CvodeEnzymeJacobianTests : public ::testing::Test {
    public:
        CvodeEnzymeJacobianTests() {
            Config::setValue(Config::LLVM_ENZYME_JACOBIAN, true);
        }

        ~CvodeEnzymeJacobianTests() override {
            Config::setValue(Config::LLVM_ENZYME_JACOBIAN, false);
            Config::setValue(Config::LOADSBMLOPTIONS_CONSERVED_MOIETIES, false);
        }
    };
}

TEST_F(CvodeEnzymeJacobianTests, EnzymeDenseMatchesDefault) {
    RoadRunner rr(hctz());
    ls::DoubleMatrix reference = run(rr, "PODOSE_hctz", 25.0, "difference_quotient");
    auto dq = cvode(rr)->getSolverStatistics();
    EXPECT_GT(dq["rhs_evals_jac"], 0);

    ls::DoubleMatrix enzyme = run(rr, "PODOSE_hctz", 25.0, "enzyme");
    auto stats = cvode(rr)->getSolverStatistics();
    EXPECT_EQ(stats["sparse_linear_solver"], 0);
    EXPECT_GT(stats["jac_evals"], 0);
    EXPECT_EQ(stats["rhs_evals_jac"], 0);
    EXPECT_GT(stats["steps"], 0);
    expectSameResults(enzyme, reference, "hctz enzyme vs difference quotient");
}

TEST_F(CvodeEnzymeJacobianTests, EnzymeWithoutLoadOptionIsError) {
    Config::setValue(Config::LLVM_ENZYME_JACOBIAN, false);
    RoadRunner rr(hctz());
    try {
        rr.getIntegrator()->setValue("jacobian", Setting("enzyme"));
        FAIL() << "setting jacobian = enzyme on a model without the Enzyme Jacobian did not throw";
    } catch (const std::exception &e) {
        EXPECT_NE(std::string(e.what()).find("LLVM_ENZYME_JACOBIAN"), std::string::npos) << e.what();
    }
}

TEST_F(CvodeEnzymeJacobianTests, InvalidSettingValueIsError) {
    RoadRunner rr(hctz());
    EXPECT_THROW(rr.getIntegrator()->setValue("jacobian", Setting("exact")), std::invalid_argument);
    EXPECT_THROW(rr.getIntegrator()->setValue("linear_solver", Setting("lu")), std::invalid_argument);
}

TEST_F(CvodeEnzymeJacobianTests, FallbackWarns) {
    ModelBuilder b;
    b.compartment("c").species("S1").species("S2").parameter("k", 0.1)
            .reaction("R1", {"S1"}, {"S2"}, "k*S1").reaction("R2", {"S2"}, {"S1"}, "0.05*S2*S2");
    Config::setValue(Config::LOADSBMLOPTIONS_CONSERVED_MOIETIES, true);
    RoadRunner rr(b.sbml());
    ASSERT_FALSE(rr.getModel()->hasStateVectorJacobian());
    ls::DoubleMatrix fallback = run(rr, "", 0.0, "enzyme");
    EXPECT_GT(cvode(rr)->getSolverStatistics()["rhs_evals_jac"], 0);
    ls::DoubleMatrix reference = run(rr, "", 0.0, "difference_quotient");
    expectSameResults(fallback, reference, "fallback");
}

TEST_F(CvodeEnzymeJacobianTests, SwitchSettingsBetweenSimulations) {
    RoadRunner rr(glimepiride());
    ls::DoubleMatrix a = run(rr, "PODOSE_gli", 4.0, "difference_quotient");
    ls::DoubleMatrix b = run(rr, "PODOSE_gli", 4.0, "enzyme");
    EXPECT_EQ(cvode(rr)->getSolverStatistics()["rhs_evals_jac"], 0);
    ls::DoubleMatrix c = run(rr, "PODOSE_gli", 4.0, "difference_quotient");
    EXPECT_GT(cvode(rr)->getSolverStatistics()["rhs_evals_jac"], 0);
    expectSameResults(b, a, "enzyme after difference quotient");
    expectSameResults(c, a, "difference quotient after enzyme");
}

TEST_F(CvodeEnzymeJacobianTests, KluMatchesDense) {
    struct Case {
        std::string path, dose;
        double value;
    };
    for (const Case &c: {Case{hctz(), "PODOSE_hctz", 25.0}, Case{glimepiride(), "PODOSE_gli", 4.0}}) {
        RoadRunner rr(c.path);
        ls::DoubleMatrix reference = run(rr, c.dose, c.value, "difference_quotient");
        ls::DoubleMatrix klu = run(rr, c.dose, c.value, "enzyme", "klu");
        auto stats = cvode(rr)->getSolverStatistics();
        EXPECT_EQ(stats["sparse_linear_solver"], 1);
        EXPECT_GT(stats["jac_evals"], 0);
        EXPECT_EQ(stats["rhs_evals_jac"], 0);
        expectSameResults(klu, reference, c.path + " klu vs dense difference quotient");
    }
}

TEST_F(CvodeEnzymeJacobianTests, KluRequiresEnzyme) {
    RoadRunner rr(hctz());
    EXPECT_THROW(rr.getIntegrator()->setValue("linear_solver", Setting("klu")), std::invalid_argument);
    // the integrator still works with its previous settings
    EXPECT_EQ(rr.getIntegrator()->getValue("linear_solver").getAs<std::string>(), "dense");
    rr.simulate(0, 10, 11);
}

TEST_F(CvodeEnzymeJacobianTests, KluSurvivesEventsAndReset) {
    ModelBuilder b;
    b.compartment("c").species("S1").species("S2").parameter("k", 0.5)
            .reaction("R1", {"S1"}, {"S2"}, "k*S1").reaction("R2", {"S2"}, {"S1"}, "0.1*S2*S2");
    auto *event = b.model->createEvent();
    event->setId("E1");
    event->setUseValuesFromTriggerTime(true);
    auto *trigger = event->createTrigger();
    std::unique_ptr<libsbml::ASTNode> cond(libsbml::SBML_parseL3Formula("time > 3"));
    trigger->setMath(cond.get());
    trigger->setInitialValue(false);
    trigger->setPersistent(true);
    auto *assignment = event->createEventAssignment();
    assignment->setVariable("S1");
    std::unique_ptr<libsbml::ASTNode> value(libsbml::SBML_parseL3Formula("S1 + 2"));
    assignment->setMath(value.get());

    RoadRunner rr(b.sbml());
    ls::DoubleMatrix reference = run(rr, "", 0.0, "difference_quotient");
    ls::DoubleMatrix first = run(rr, "", 0.0, "enzyme", "klu");
    EXPECT_EQ(cvode(rr)->getSolverStatistics()["sparse_linear_solver"], 1);
    ls::DoubleMatrix second = run(rr, "", 0.0, "enzyme", "klu");
    expectSameResults(first, reference, "klu with an event");
    expectSameResults(second, reference, "klu after reset");
}
