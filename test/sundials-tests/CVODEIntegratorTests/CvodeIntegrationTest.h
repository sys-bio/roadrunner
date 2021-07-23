//
// Created by Ciaran on 08/07/2021.
//

#ifndef ROADRUNNER_CVODEINTEGRATIONTEST_H
#define ROADRUNNER_CVODEINTEGRATIONTEST_H

#include "Matrix.h"
#include "rrRoadRunner.h"
#include "LLVMExecutableModel.h"

using namespace rr;

/**
 * Parameterised fixture for passing models through a series of tests.
 * The template parameter is a string and is used as input into TestModelFactory.
 */
class CVODEIntegratorTests : public ::testing::Test {
public:

    CVODEIntegratorTests() = default;

    /**
     * Test to check that a sbml model can simulate using CVODEIntegrator
     * and not throw an error.
     *
     */
    void checkModelSimulatesWithRoadRunner(const std::string &modelName) {
        TestModel *testModelPtr = TestModelFactory(modelName);
        auto *tsTestModel = dynamic_cast<TimeSeriesResult *>(testModelPtr);

        // load model
        RoadRunner r(testModelPtr->str());
        r.getIntegrator()->setValue("stiff", false);

        // get handle on the *known true values.
        auto trueValues = tsTestModel->timeSeriesResult();

        // grab integrator settings.
        const auto &settings = tsTestModel->timeSeriesSettings();

        // apply settings
        SimulateOptions opt;
        for (const auto &setting: settings) {
            // opt.setItem(setting.first, setting.second); Surprisingly this strategy does not work. Suspected bug.
            // instead we'll have to manually unpack the simulation variables.
            if (setting.first == "reset_model") {
                opt.reset_model = setting.second;
            } else if (setting.first == "structured_result") {
                opt.structured_result = setting.second;
            } else if (setting.first == "copy_result") {
                opt.copy_result = setting.second;
            } else if (setting.first == "steps") {
                opt.steps = setting.second;
            } else if (setting.first == "start") {
                opt.start = setting.second;
            } else if (setting.first == "duration") {
                opt.duration = setting.second;
            }

            // then try applying integration level settings.
            try {
                r.getIntegrator()->setValue(setting.first, setting.second);
            } catch (std::exception &e) {
                // its okay to have a setting that doesn't belong here
                continue;
            }
        }

        const ls::DoubleMatrix *tmp = r.simulate(&opt);
        Matrix<double> result(*tmp);
        bool passed = result.almostEquals(trueValues, 1e-4);
        if (!passed) {
            std::cout << "Expected result: " << std::endl;
            std::cout << trueValues << std::endl;
            std::cout << "Actual result: " << std::endl;
            std::cout << result << std::endl;
        }

        ASSERT_TRUE(passed);
        delete testModelPtr;
//        got weird problem here try valgrind
    }

    /**
     * @brief check that the model integrates with
     * solver SolverType.
     * @details The model should integrate without
     * the RoadRunner object, though the main RR object
     * needs to be used to create the ExecutableModel
     */
    template<class SolverType = CVODEIntegrator>
    void checkModelIntegrates(TestModel *testModel) {

        // load model
        RoadRunner r(testModel->str());

        auto *timeseriesTestModel = dynamic_cast<TimeSeriesResult *>(testModel);

        ExecutableModel *model = r.getModel();

        SolverType solver(model);

        solver.setValue("stiff", false);

        // get handle on the *known true values.
        auto trueValues = timeseriesTestModel->timeSeriesResult();

        // grab integrator settings.
        auto settings = timeseriesTestModel->timeSeriesSettings();
        double start = settings["start"];
        double duration = settings["duration"];
        double steps = settings["steps"];
        double stepSize = (duration - start) / steps ;

        Matrix<double> simulationResults(trueValues.numRows(), trueValues.numCols());
        int numStates = model->getStateVector(nullptr);
        double *stateVector = new double[numStates];

        // collect initial state
        simulationResults(0, 0) = model->getTime();
        model->getStateVector(stateVector);
        for (int j = 1; j < numStates + 1; j++) {
            simulationResults(0, j) = stateVector[j - 1];
        }

        double t = 0.0;
        for (int i = 1; i < steps + 1; i++) {
            t = solver.integrate(t, stepSize);
            model->getStateVector(stateVector);
            simulationResults(i, 0) = model->getTime();
            for (int j = 1; j < numStates + 1; j++) {
                simulationResults(i, j) = stateVector[j - 1];
            }
        }

        bool passed = trueValues.almostEquals(simulationResults, 1e-4);

        if (!passed) {
            std::cout << "Expected result: " << std::endl;
            std::cout << trueValues << std::endl;
            std::cout << "Actual result: " << std::endl;
            std::cout << simulationResults << std::endl;
        }

        ASSERT_TRUE(passed);

        delete[] stateVector;
    }

};

#endif //ROADRUNNER_CVODEINTEGRATIONTEST_H
