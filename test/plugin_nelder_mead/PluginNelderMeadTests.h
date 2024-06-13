#ifndef ROADRUNNER_PLUGIN_NELDER_MEAD_TESTS_H
#define ROADRUNNER_PLUGIN_NELDER_MEAD_TESTS_H

#include "gtest/gtest.h"
#include <filesystem>
#include "RoadRunnerTest.h"

using std::filesystem::path;

class PluginNelderMeadTests : public RoadRunnerTest {
public:
    path pluginsModelsDir;

    PluginNelderMeadTests();
};


#endif
