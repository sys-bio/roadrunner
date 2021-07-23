#include <rrRoadRunner.h>
#include "gtest/gtest.h"

#include "RoadRunnerTest.h"
#include "TestModelFactory.h"
#include "GillespieIntegrator.h"
#include "rrConfig.h"
#include "Matrix.h"
#include "SteadyStateSolver.h"

using namespace rr;

class JacobianTests : public RoadRunnerTest {

public:
    JacobianTests() = default;

    void checkJacobianFullAmt(const std::string &modelName, double tol) {
        checkJacFull(modelName, tol, Config::ROADRUNNER_JACOBIAN_MODE_AMOUNTS);
    }

    void checkJacobianReducedAmt(const std::string &modelName, double tol) {
        checkJacReduced(modelName, tol, Config::ROADRUNNER_JACOBIAN_MODE_AMOUNTS);
    }

    void checkJacobianFullConc(const std::string &modelName, double tol) {
        checkJacFull(modelName, tol, Config::ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS);
    }

    void checkJacobianReducedConc(const std::string &modelName, double tol) {
        checkJacReduced(modelName, tol, Config::ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS);
    }


private:

    /**
     * @brief check the values of TestModel jacobian interface against those computed by the current model
     * @param modelName A valid test model name (See TestModelFactory::getAvailableTestModelNames()
     * @param tol degree of tolerance, allowed error
     * @param rrJacMode either Config::ROADRUNNER_JACOBIAN_MODE_AMOUNTS or Config::ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS
     * @param full true if we are looking at full jacobian, false if were looking at reduced.
     * @details if full is set to false, then we turn on moiety conservation
     */
    static void
    checkJacFull(const std::string &modelName, double tol, Config::ROADRUNNER_JACOBIAN_MODE_VALUES rrJacMode) {
        TestModel *testModel = TestModelFactory(modelName);
        JacobianResultFull *jacobianResultTestModel = dynamic_cast<JacobianResultFull *>(testModel);

        RoadRunner rr(testModel->str());
        Config::setValue(Config::ROADRUNNER_JACOBIAN_MODE, rrJacMode);

        ls::Matrix<double> expectedJac;

        if (rrJacMode == Config::ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS) {
            expectedJac = jacobianResultTestModel->fullJacobianConc();
        } else if (rrJacMode == Config::ROADRUNNER_JACOBIAN_MODE_AMOUNTS) {
            expectedJac = jacobianResultTestModel->fullJacobianAmt();
        } else {
            throw std::logic_error("Should never reach here. Check your input.");
        }

        auto jacSettings = jacobianResultTestModel->jacobianSettings();

        // apply steady state if flag true in settings
        if (jacSettings.find("steady_state") != jacSettings.end()) {
            if (jacSettings["steady_state"].getAs<bool>()) {
                rr.steadyState();
            }
        }

        if (jacSettings.find("selections") != jacSettings.end()) {
            std::vector<std::string> sel = jacSettings["selections"].get<std::vector<std::string>>();
            // note : this does not work - cannot reorder jacobian.
            //rr.setSelections(sel);
        }

        ls::Matrix<double> actualJac = rr.getFullJacobian();

        // these matrix types have equality operators
        rr::Matrix<double> expectedJacMatrix(expectedJac);
        rr::Matrix<double> actualJacMatrix(actualJac);

        // sort matrices, in case ordering is messed up
        expectedJacMatrix.sortColsByLabels();
        expectedJacMatrix.sortRowsByLabels();
        actualJacMatrix.sortColsByLabels();
        actualJacMatrix.sortRowsByLabels();

        // true if these matrices are equal to tolerance tol
        bool passed = expectedJacMatrix.almostEquals(actualJacMatrix, tol);

        if (!passed) {
            std::cout << "expectedJacMatrix" << std::endl;
            std::cout << expectedJacMatrix << std::endl;
            std::cout << "actualJacMatrix" << std::endl;
            std::cout << actualJacMatrix << std::endl;
        }
        ASSERT_TRUE(passed);
        delete testModel;
    }

    /**
     * @brief check the values of TestModel jacobian interface against those computed by the current model
     * @param modelName A valid test model name (See TestModelFactory::getAvailableTestModelNames()
     * @param tol degree of tolerance, allowed error
     * @param rrJacMode either Config::ROADRUNNER_JACOBIAN_MODE_AMOUNTS or Config::ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS
     * @param full true if we are looking at full jacobian, false if were looking at reduced.
     * @details if full is set to false, then we turn on moiety conservation
     */
    static void
    checkJacReduced(const std::string &modelName, double tol, Config::ROADRUNNER_JACOBIAN_MODE_VALUES rrJacMode) {
        TestModel *testModel = TestModelFactory(modelName);
        JacobianResultReduced *jacobianResultTestModel = dynamic_cast<JacobianResultReduced *>(testModel);

        RoadRunner rr(testModel->str());
        rr.getSteadyStateSolver()->setValue("auto_moiety_analysis", false);
        Config::setValue(Config::ROADRUNNER_JACOBIAN_MODE, rrJacMode);

        ls::Matrix<double> expectedJac;

        if (rrJacMode == Config::ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS) {
            expectedJac = jacobianResultTestModel->reducedJacobianConc();
        } else if (rrJacMode == Config::ROADRUNNER_JACOBIAN_MODE_AMOUNTS) {
            expectedJac = jacobianResultTestModel->reducedJacobianAmt();
        } else {
            throw std::logic_error("Should never reach here. Check your input.");
        }

        rr.setConservedMoietyAnalysis(true);

        auto jacSettings = jacobianResultTestModel->jacobianSettings();

        // apply steady state if flag true in settings
        if (jacSettings.find("steady_state") != jacSettings.end()) {
            if (jacSettings["steady_state"].getAs<bool>()) {
                rr.steadyState();
            }
        }
        ls::Matrix<double> actualJac = rr.getReducedJacobian();

        // these matrix types have equality operators
        rr::Matrix<double> expectedJacMatrix(expectedJac);
        rr::Matrix<double> actualJacMatrix(actualJac);

        // sort matrices, in case ordering is messed up
        expectedJacMatrix.sortColsByLabels();
        expectedJacMatrix.sortRowsByLabels();
        actualJacMatrix.sortColsByLabels();
        actualJacMatrix.sortRowsByLabels();

        // true if these matrices are equal to tolerance tol
        bool passed = expectedJacMatrix.almostEquals(actualJacMatrix, tol);

        if (!passed) {
            std::cout << "expectedJacMatrix" << std::endl;
            std::cout << expectedJacMatrix << std::endl;
            std::cout << "actualJacMatrix" << std::endl;
            std::cout << actualJacMatrix << std::endl;
        }
        ASSERT_TRUE(passed);
        delete testModel;
    }


};


TEST_F(JacobianTests, BimolecularEndFullAmt) {
    checkJacobianFullAmt("BimolecularEnd", 1e-4);
}

TEST_F(JacobianTests, BimolecularEndFullConc) {
    checkJacobianFullConc("BimolecularEnd", 1e-4);
}

TEST_F(JacobianTests, SimpleFluxFullAmt) {
    checkJacobianFullAmt("SimpleFlux", 1e-4);
}

TEST_F(JacobianTests, SimpleFluxFullConc) {
    checkJacobianFullConc("SimpleFlux", 1e-4);
}

TEST_F(JacobianTests, SimpleFluxReducedAmt) {
    checkJacobianReducedAmt("SimpleFlux", 1e-4);
}

TEST_F(JacobianTests, SimpleFluxReducedConc) {
    checkJacobianReducedConc("SimpleFlux", 1e-4);
}












