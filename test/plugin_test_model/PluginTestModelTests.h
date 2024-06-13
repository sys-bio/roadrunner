#ifndef ROADRUNNER_PLUGIN_TEST_MODEL_TESTS_H
#define ROADRUNNER_PLUGIN_TEST_MODEL_TESTS_H

#include "gtest/gtest.h"
#include <filesystem>
#include "RoadRunnerTest.h"

using std::filesystem::path;

class PluginTestModelTests : public RoadRunnerTest {
public:
    path pluginsModelsDir;

    PluginTestModelTests();
};

#endif
