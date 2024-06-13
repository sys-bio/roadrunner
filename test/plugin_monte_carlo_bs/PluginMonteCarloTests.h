#ifndef ROADRUNNER_PLUGIN_MONTE_CARLO_TESTS_H
#define ROADRUNNER_PLUGIN_MONTE_CARLO_TESTS_H

#include "gtest/gtest.h"
#include <filesystem>
#include "RoadRunnerTest.h"

using std::filesystem::path;

class PluginMonteCarloTests : public RoadRunnerTest {
public:
    path pluginsModelsDir;

    PluginMonteCarloTests();
};

#endif
