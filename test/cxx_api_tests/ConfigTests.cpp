//
// Created by Ciaran on 26/05/2021.
//

#include "gtest/gtest.h"
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include "rrConfig.h"
#include "Poco/Path.h" // for Poco::Path::home, which is used in rrConfig.cpp

using namespace rr;

class ConfigTests : public ::testing::Test {
public:
    std::filesystem::path fname = std::filesystem::current_path() / "rr.config";

    ConfigTests() {
        std::ofstream of(fname);
        of << "LOADSBMLOPTIONS_OPTIMIZE_GVN: true" << std::endl;
        of << "SIMULATEOPTIONS_MINIMUM_TIMESTEP: 1e-4" << std::endl;
        of << "STEADYSTATE_APPROX: true" << std::endl;
        of.close();
    }

    ~ConfigTests() {
        if (std::filesystem::exists(fname)) {
            std::filesystem::remove(fname);
        }
    }
};

TEST_F(ConfigTests, getString) {
    std::filesystem::path fname = std::filesystem::current_path() / "here.log";
    Config::setValue(Config::LOGGER_LOG_FILE_PATH, fname.string());
    std::string stringRep = Config::getString(Config::LOGGER_LOG_FILE_PATH);
    ASSERT_STREQ(("'" + fname.string() + "'").c_str(), stringRep.c_str());
}

TEST_F(ConfigTests, getInt) {
    int x = Config::getInt(Config::STEADYSTATE_APPROX_MAX_STEPS);
    ASSERT_EQ(x, 10000);
}

TEST_F(ConfigTests, getBool) {
    bool validate = Config::getBool(Config::VALIDATION_IN_REGENERATION);
    ASSERT_TRUE(validate);
}

TEST_F(ConfigTests, getDouble) {
    double value = Config::getDouble(Config::STEADYSTATE_MINIMUM_DAMPING);
    ASSERT_NEAR(1e-20, value, 1e-16);
}

// not cross platform
//TEST_F(ConfigTests, getConfigFilePathFromEnvVariable) {
//    putenv(("ROADRUNNER_CONFIG=" + fname.string()).c_str());
//    std::string fp = Config::getConfigFilePath();
//    ASSERT_STREQ(fname.string().c_str(), fp.c_str());
//}

TEST_F(ConfigTests, getConfigFilePathFromHOME) {
    std::filesystem::path conf = Poco::Path::home(); // get home directory. There is no standard C++ for this
    conf = conf / "roadrunner.conf";
    // if this file already exists on disk, we don't want to overwrite it
    if (!std::filesystem::exists(conf)) {
        std::ofstream of(conf); // write empty file so it exists
        of.close(); // make sure file is closed
    }
    std::string fp = Config::getConfigFilePath();
    ASSERT_STREQ(fp.c_str(), conf.string().c_str());
    if (std::filesystem::exists(conf)) {
        std::filesystem::remove(conf);
    }
}

TEST_F(ConfigTests, getAndSetValue) {
    Config::setValue(Config::ROADRUNNER_JACOBIAN_STEP_SIZE, 0.1);
    ASSERT_NEAR(0.1, Config::getValue(Config::ROADRUNNER_JACOBIAN_STEP_SIZE), 1e-7);
}

TEST_F(ConfigTests, readConfigFile) {
    Config::readConfigFile(fname.string());
    ASSERT_TRUE((bool) Config::getValue(Config::LOADSBMLOPTIONS_OPTIMIZE_GVN));
    ASSERT_NEAR(Config::getValue(Config::SIMULATEOPTIONS_MINIMUM_TIMESTEP), 1e-4, 1e-3);
    ASSERT_TRUE((bool) Config::getValue(Config::STEADYSTATE_APPROX));
}

TEST_F(ConfigTests, writeConfigFile) {
    std::filesystem::path config = std::filesystem::current_path() / "roadrunner.conf";
    Config::setValue(Config::STEADYSTATE_APPROX_TIME, 50.5);
    Config::writeConfigFile(config.string());
    ASSERT_TRUE(std::filesystem::exists(config));
    // now read the config
    std::ifstream inStream(config);
    if (!inStream) {
        throw std::logic_error("No file found at " + config.string());
    }
    bool found = false;
    std::string line;
    while (std::getline(inStream, line)) {
        if (line.find("STEADYSTATE_APPROX_TIME: 50.5") != std::string::npos){
            found = true;
        }
    }
    ASSERT_TRUE(found);
    inStream.close();
    // clean up after ourselves.
    if(std::filesystem::exists(config)){
        std::filesystem::remove(config);
    }
}

TEST_F(ConfigTests, stringToKey){
    ASSERT_EQ(0, Config::stringToKey("LOADSBMLOPTIONS_CONSERVED_MOIETIES"));

}
