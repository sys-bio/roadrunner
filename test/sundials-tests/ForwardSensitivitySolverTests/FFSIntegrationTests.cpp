//
// Created by Ciaran Welsh on 09/06/2021.
//

#include "gtest/gtest.h"
#include "rrRoadRunner.h"
#include "TestModelFactory.h"
#include "ForwardSensitivitySolver.h"
#include "ForwardSensitivitySolver.h"
#include "../CVODEIntegratorTests/CvodeIntegrationTest.h"

using namespace rr;

class FFSIntegrationTests : public CVODEIntegratorTests {
public:
    FFSIntegrationTests() = default;

    /**
     * @brief Integrate the model and compute sensitivities. Compare
     * results against expected values
     */
    template<class SolverType = ForwardSensitivitySolver>
    void checkTimeSeriesSensitivities(TestModel *testModel, double tol=1e-4) {

        // load model
        RoadRunner r(testModel->str());

        auto *timeSeriesSensResult = dynamic_cast<TimeSeriesSensitivityResult *>(testModel);

        ExecutableModel *model = r.getModel();

        SolverType solver(model);
        solver.setValue("stiff", false);

        // get handle on the *known true values.
        auto expectedResults = timeSeriesSensResult->timeSeriesSensitivityResult();

        // grab integrator settings.
        auto settings = timeSeriesSensResult->timeSeriesSensitivityResultSettings();

        double start = settings["start"];
        double stop = settings["duration"];
        int    steps = settings["steps"];

        std::cout << "start: " << start << ": stop: " << stop << "; steps: "<< steps << std::endl;

        Matrix3D<double, double> actualResults = solver.solveSensitivities(start, stop, steps);

        bool passed = expectedResults.almostEquals(actualResults, tol);

        if (!passed) {
            std::cout << "Expected result: " << std::endl;
            std::cout << expectedResults << std::endl;
            std::cout << "Actual result: " << std::endl;
            std::cout << actualResults << std::endl;
        }

        ASSERT_TRUE(passed);
    }
};


TEST_F(FFSIntegrationTests, CheckTimeSeriesAccurateTestSimpleFlux) {
    SimpleFlux testModel;
    checkModelIntegrates<ForwardSensitivitySolver>(&testModel);
}

TEST_F(FFSIntegrationTests, CheckTimeSeriesAccurateOpenLinearFlux) {
    OpenLinearFlux testModel;
    checkModelIntegrates<ForwardSensitivitySolver>(&testModel);
}

TEST_F(FFSIntegrationTests, CheckTimeSeriesAccurateTestModel269) {
    Model269 testModel;
    checkModelIntegrates<ForwardSensitivitySolver>(&testModel);
}

TEST_F(FFSIntegrationTests, CheckTimeSeriesAccurateTestModel28) {
    Model28 testModel;
    checkModelIntegrates<ForwardSensitivitySolver>(&testModel);
}

/**
 * Note: the following tests pass, but with quite a high tolerance. This may be due to
 * the fact that these sensitivities are copasi generated at each time point. They
 * are close, but not identical
 */

TEST_F(FFSIntegrationTests, CheckTimeSeriesSensAccurateTestSimpleFlux) {
    SimpleFlux testModel;
    checkTimeSeriesSensitivities<ForwardSensitivitySolver>(&testModel, 0.2);
}


TEST_F(FFSIntegrationTests, CheckTimeSeriesSensAccurateTestModel269) {
    Model269 testModel;
    checkTimeSeriesSensitivities<ForwardSensitivitySolver>(&testModel, 0.1);
}

TEST_F(FFSIntegrationTests, CheckTimeSeriesSensAccurateTestModel28) {
    Model28 testModel;
    checkTimeSeriesSensitivities<ForwardSensitivitySolver>(&testModel, 0.1);
}

TEST_F(FFSIntegrationTests, CheckTimeSeriesSensAccurateOpenLinearFlux) {
    OpenLinearFlux testModel;
    checkTimeSeriesSensitivities<ForwardSensitivitySolver>(&testModel, 0.1);
}






















