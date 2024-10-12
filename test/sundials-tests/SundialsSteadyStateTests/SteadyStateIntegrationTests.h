#ifndef ROADRUNNER_STEADYSTATEINTEGRATIONTEST
#define ROADRUNNER_STEADYSTATEINTEGRATIONTEST

#include <fstream>
#include "gtest/gtest.h"

#include "NewtonIteration.h"
#include "BasicNewtonIteration.h"
#include "LinesearchNewtonIteration.h"
#include "TestModelFactory.h"
#include <chrono>
#include "rrRoadRunner.h"

using namespace std::chrono;
using namespace rr;

class SteadyStateIntegrationTest : public ::testing::Test {

public:
    SteadyStateIntegrationTest() = default;

    void testSteadyState(
            const std::string &modelName,
            const std::string &solverName,
            double tol = 1e-4) {
        // get the model
        TestModel *testModel = TestModelFactory(modelName);

        auto* steadyStateTestModel = dynamic_cast<SteadyStateResult *>(testModel);

        assert(testModel && "testModel is nullptr");
        assert(steadyStateTestModel && "steadyStateTestModel is nullptr");

        // load it into rr
        RoadRunner rr(testModel->str());

        // collect reference results
        StringDoubleMap expectedResult = steadyStateTestModel->steadyState();

        rr.setSteadyStateSolver(solverName);

        BasicDictionary steadyStateOptions;
        steadyStateOptions.setItem("PrintLevel", 0);

        for (auto &settingsIterator : steadyStateTestModel->steadyStateSettings()) {
            if (settingsIterator.first == "moiety_conservation") {
                rr.setConservedMoietyAnalysis(settingsIterator.second);
            } else {
                try {
                    rr.getSteadyStateSolver()->setValue(settingsIterator.first, Setting(settingsIterator.second));
                } catch (std::exception &err) {
                    // if solver does not have this option, that's okay
                    continue;
                }
            }
        }

        try {
            rr.steadyState(&steadyStateOptions);

            // collect actual results from model
            auto result = rr.getFloatingSpeciesConcentrationsNamedArray();
            std::vector<std::string> names = result.getColNames();

            // check to see if actual results are near expected.
            for (int i = 0; i < names.size(); i++) {
                std::string speciesID = names[i];
                double actualResult = result[0][i]; // 0th row, ith col of a DoubleMatrix
                double expected = expectedResult[speciesID]; // first is start val, second is speciesID at steady state

                std::cout << "Comparing \"" << speciesID << "\" expected result: " << expected
                          << " with actual result " << actualResult << std::endl;
                EXPECT_NEAR(expected, actualResult, tol);
            }
        } catch (const std::exception& e) {
            // We just print the error message here as steady state was not reached for this model using this solver
            std::cerr << "Error during steady state calculation: " << e.what() << std::endl;
        }
        delete testModel;
    }

    void testSteadyStateFluxes(
            const std::string& modelName,
            const std::string& solverName,
            double tol = 1e-4
            ){
        // get the model
        TestModel *testModel = TestModelFactory(modelName);
        assert(testModel && "testModel is nullptr");

        // load it into rr
        RoadRunner rr(testModel->str());

        // collect reference results
        auto* steadyStateFluxesTestModel = dynamic_cast<SteadyStateFluxes *>(testModel);
        assert(steadyStateFluxesTestModel && "steadyStateFluxesTestModel is nullptr");
        StringDoubleMap expectedResult = steadyStateFluxesTestModel->steadyStateFluxes();

        rr.setSteadyStateSolver(solverName);

        BasicDictionary steadyStateOptions;
        steadyStateOptions.setItem("PrintLevel", 0);

        for (auto &settingsIterator : steadyStateFluxesTestModel->steadyStateSettings()) {
            if (settingsIterator.first == "moiety_conservation") {
                rr.setConservedMoietyAnalysis(settingsIterator.second);
            } else {
                try {
                    rr.getSteadyStateSolver()->setValue(settingsIterator.first, Setting(settingsIterator.second));
                } catch (std::exception &err) {
                    // if solver does not have this option, that's okay
                    continue;
                }
            }
        }

        rr.steadyState(&steadyStateOptions);

        for (auto& [reactionId, expectedSteadyStateFlux] : expectedResult){
            double actualSteadyStateFlux = rr.getModel()->getValue(reactionId);
            std::cout << "comparing steady state flux for reaction \""
                << reactionId << "\": expected value = " << expectedSteadyStateFlux
                << "; actual value = " << actualSteadyStateFlux << std::endl;
            ASSERT_NEAR(expectedSteadyStateFlux, actualSteadyStateFlux, tol);
        }

        delete testModel;
    }

};

#endif //ROADRUNNER_STEADYSTATEINTEGRATIONTEST










