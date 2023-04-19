#ifndef ROADRUNNER_PLUGIN_LEVENBERG_MARQUARDT_TESTS_H
#define ROADRUNNER_PLUGIN_LEVENBERG_MARQUARDT_TESTS_H

#include "gtest/gtest.h"
#include <filesystem>
#include "RoadRunnerTest.h"

using std::filesystem::path;

class PluginLevenbergMarquardtTests : public RoadRunnerTest {
public:
    path pluginsModelsDir;

    PluginLevenbergMarquardtTests();
};

#endif
