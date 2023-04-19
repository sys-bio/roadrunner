#ifndef ROADRUNNER_PLUGIN_AUTO_2000_TESTS_H
#define ROADRUNNER_PLUGIN_AUTO_2000_TESTS_H

#include "gtest/gtest.h"
#include <filesystem>
#include "RoadRunnerTest.h"

using std::filesystem::path;

class PluginAuto2000Tests : public RoadRunnerTest {
public:
    PluginAuto2000Tests();

    path pluginsModelsDir;
};

#endif
