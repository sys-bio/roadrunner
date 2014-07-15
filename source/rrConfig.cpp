/*
 * Config.cpp
 *
 *  Created on: Mar 24, 2014
 *      Author: andy
 */

#include "rrUtils.h"
#include "rrConfig.h"
#include "rrLogger.h"

#if (__cplusplus >= 201103L) || defined(_MSC_VER)
#include <memory>
#include <unordered_map>
#define cxx11_ns std
#else
#include <tr1/memory>
#include <tr1/unordered_map>
#define cxx11_ns std::tr1
#endif

#include <stdexcept>
#include <cctype>
#include <cstdlib>
#include <algorithm>
#include <fstream>      // std::ofstream
#include <assert.h>

// TODO When we have gcc 4.4 as minimal compiler, drop poco and use C++ standard regex
#include <Poco/RegularExpression.h>
#include <Poco/Path.h>

// somebody will likely call this multithread and then bitch and moan if
// there is an issue, so lock it.
#include <Poco/Mutex.h>

// default values of sbml consistency check
#include <sbml/SBMLDocument.h>

// default values of model reset
#include <rrSelectionRecord.h>

using Poco::Mutex;
using Poco::RegularExpression;
using std::string;


namespace rr
{


typedef cxx11_ns::unordered_map<std::string, int> StringIntMap;

/**
 * check range of key
 */
#define CHECK_RANGE(key) {                                 \
        if (key < 0 || key >= rr::Config::CONFIG_END) {    \
            throw std::out_of_range("invalid Config key"); \
        }                                                  \
    }

/**
 * strip any leading or trailing whitespace
 */
static std::string strip(const std::string& in)
{
    std::string out;
    std::string::const_iterator b = in.begin(), e = in.end();

    // skipping leading spaces
    while (std::isspace(*b)){
        ++b;
    }

    if (b != e){
        // skipping trailing spaces
        while (std::isspace(*(e-1))){
            --e;
        }
    }

    out.assign(b, e);

    return out;
}



static Variant values[] =  {
    Variant(false),    // LOADSBMLOPTIONS_CONSERVED_MOIETIES
    Variant(false),    // LOADSBMLOPTIONS_RECOMPILE
    Variant(false),    // LOADSBMLOPTIONS_READ_ONLY
    Variant(true),     // LOADSBMLOPTIONS_MUTABLE_INITIAL_CONDITIONS
    Variant(false),    // LOADSBMLOPTIONS_OPTIMIZE_GVN
    Variant(false),    // LOADSBMLOPTIONS_OPTIMIZE_CFG_SIMPLIFICATION
    Variant(false),    // LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_COMBINING
    Variant(false),    // LOADSBMLOPTIONS_OPTIMIZE_DEAD_INST_ELIMINATION
    Variant(false),    // LOADSBMLOPTIONS_OPTIMIZE_DEAD_CODE_ELIMINATION
    Variant(false),    // LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_SIMPLIFIER
    Variant(false),    // LOADSBMLOPTIONS_USE_MCJIT
    Variant(50),       // SIMULATEOPTIONS_STEPS,
    Variant(5),        // SIMULATEOPTIONS_DURATION,
    Variant(1.e-10),   // SIMULATEOPTIONS_ABSOLUTE,
    Variant(1.e-5),    // SIMULATEOPTIONS_RELATIVE,
    Variant(false),    // SIMULATEOPTIONS_STRUCTURED_RESULT,
    Variant(false),    // SIMULATEOPTIONS_STIFF,
    Variant(false),    // SIMULATEOPTIONS_MULTI_STEP,
    Variant(false),    // SIMULATEOPTIONS_DETERMINISTIC_VARIABLE_STEP,
    Variant(true),     // SIMULATEOPTIONS_STOCHASTIC_VARIABLE_STEP,
    Variant(std::string("CVODE")), // SIMULATEOPTIONS_INTEGRATOR
    Variant(-1),       // SIMULATEOPTIONS_INITIAL_TIMESTEP,
    Variant(-1),       // SIMULATEOPTIONS_MINIMUM_TIMESTEP,
    Variant(-1),       // SIMULATEOPTIONS_MAXIMUM_TIMESTEP,
    Variant(-1),       // SIMULATEOPTIONS_MAXIMUM_NUM_STEPS
    Variant(0),        // ROADRUNNER_DISABLE_WARNINGS
    Variant(false),    // ROADRUNNER_DISABLE_PYTHON_DYNAMIC_PROPERTIES
    Variant(int(AllChecksON & UnitsCheckOFF)),          //SBML_APPLICABLEVALIDATORS
    Variant(0.00001),   // ROADRUNNER_JACOBIAN_STEP_SIZE
    Variant((int)(SelectionRecord::TIME
            | SelectionRecord::RATE
            | SelectionRecord::FLOATING
            | SelectionRecord::CONSREVED_MOIETY)),       // MODEL_RESET
    Variant(1.e-10),   // CVODE_MIN_ABSOLUTE
    Variant(1.e-5),    // CVODE_MIN_RELATIVE
    Variant(true),     // SIMULATEOPTIONS_COPY_RESULT
    Variant(1.e-4),    // STEADYSTATE_RELATIVE
    Variant(100),      // STEADYSTATE_MAXIMUM_NUM_STEPS
    Variant(1.e-16)    // STEADYSTATE_MINIMUM_DAMPING
};

static bool initialized = false;
static Mutex configMutex;

static void readDefaultConfig() {
    Mutex::ScopedLock lock(configMutex);

    if(!initialized) {
        assert(rr::Config::CONFIG_END == sizeof(values) / sizeof(Variant) &&
                "values array size different than CONFIG_END");

        string confPath = rr::Config::getConfigFilePath();

        try {
            if(confPath.size() > 0) {
                rr::Config::readConfigFile(confPath);
            }
        }
        catch(std::exception& e) {
            Log(rr::Logger::LOG_WARNING) << "error reading configuration file: "
                    << confPath << ", " << e.what();
        }
        initialized = true;
    }
}

/**
 * load the names of the keys and values into a map
 */
static void getKeyNames(StringIntMap& keys)
{
    keys["LOADSBMLOPTIONS_CONSERVED_MOIETIES"] = rr::Config::LOADSBMLOPTIONS_CONSERVED_MOIETIES;
    keys["LOADSBMLOPTIONS_RECOMPILE"] = rr::Config::LOADSBMLOPTIONS_RECOMPILE;
    keys["LOADSBMLOPTIONS_READ_ONLY"] = rr::Config::LOADSBMLOPTIONS_READ_ONLY;
    keys["LOADSBMLOPTIONS_MUTABLE_INITIAL_CONDITIONS"] = rr::Config::LOADSBMLOPTIONS_MUTABLE_INITIAL_CONDITIONS;
    keys["LOADSBMLOPTIONS_OPTIMIZE_GVN"] = rr::Config::LOADSBMLOPTIONS_OPTIMIZE_GVN;
    keys["LOADSBMLOPTIONS_OPTIMIZE_CFG_SIMPLIFICATION"] = rr::Config::LOADSBMLOPTIONS_OPTIMIZE_CFG_SIMPLIFICATION;
    keys["LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_COMBINING"] = rr::Config::LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_COMBINING;
    keys["LOADSBMLOPTIONS_OPTIMIZE_DEAD_INST_ELIMINATION"] = rr::Config::LOADSBMLOPTIONS_OPTIMIZE_DEAD_INST_ELIMINATION;
    keys["LOADSBMLOPTIONS_OPTIMIZE_DEAD_CODE_ELIMINATION"] = rr::Config::LOADSBMLOPTIONS_OPTIMIZE_DEAD_CODE_ELIMINATION;
    keys["LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_SIMPLIFIER"] = rr::Config::LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_SIMPLIFIER;
    keys["LOADSBMLOPTIONS_USE_MCJIT"] = rr::Config::LOADSBMLOPTIONS_USE_MCJIT;
    keys["SIMULATEOPTIONS_STEPS"] = rr::Config::SIMULATEOPTIONS_STEPS;
    keys["SIMULATEOPTIONS_DURATION"] = rr::Config::SIMULATEOPTIONS_DURATION;
    keys["SIMULATEOPTIONS_ABSOLUTE"] = rr::Config::SIMULATEOPTIONS_ABSOLUTE;
    keys["SIMULATEOPTIONS_RELATIVE"] = rr::Config::SIMULATEOPTIONS_RELATIVE;
    keys["SIMULATEOPTIONS_STRUCTURED_RESULT"] = rr::Config::SIMULATEOPTIONS_STRUCTURED_RESULT;
    keys["SIMULATEOPTIONS_STIFF"] = rr::Config::SIMULATEOPTIONS_STIFF;
    keys["SIMULATEOPTIONS_MULTI_STEP"] = rr::Config::SIMULATEOPTIONS_MULTI_STEP;
    keys["SIMULATEOPTIONS_DETERMINISTIC_VARIABLE_STEP"] = rr::Config::SIMULATEOPTIONS_DETERMINISTIC_VARIABLE_STEP;
    keys["SIMULATEOPTIONS_STOCHASTIC_VARIABLE_STEP"] = rr::Config::SIMULATEOPTIONS_STOCHASTIC_VARIABLE_STEP;
    keys["SIMULATEOPTIONS_INTEGRATOR"] = rr::Config::SIMULATEOPTIONS_INTEGRATOR;
    keys["SIMULATEOPTIONS_INITIAL_TIMESTEP"] = rr::Config::SIMULATEOPTIONS_INITIAL_TIMESTEP;
    keys["SIMULATEOPTIONS_MINIMUM_TIMESTEP"] = rr::Config::SIMULATEOPTIONS_MINIMUM_TIMESTEP;
    keys["SIMULATEOPTIONS_MAXIMUM_TIMESTEP"] = rr::Config::SIMULATEOPTIONS_MAXIMUM_TIMESTEP;
    keys["SIMULATEOPTIONS_MAXIMUM_NUM_STEPS"] = rr::Config::SIMULATEOPTIONS_MAXIMUM_NUM_STEPS;
    keys["ROADRUNNER_DISABLE_WARNINGS"] = rr::Config::ROADRUNNER_DISABLE_WARNINGS;
    keys["ROADRUNNER_DISABLE_PYTHON_DYNAMIC_PROPERTIES"] = rr::Config::ROADRUNNER_DISABLE_PYTHON_DYNAMIC_PROPERTIES;
    keys["SBML_APPLICABLEVALIDATORS"] = rr::Config::SBML_APPLICABLEVALIDATORS;
    keys["ROADRUNNER_JACOBIAN_STEP_SIZE"] = rr::Config::ROADRUNNER_JACOBIAN_STEP_SIZE;
    keys["MODEL_RESET"] = rr::Config::MODEL_RESET;
    keys["CVODE_MIN_ABSOLUTE"] = rr::Config::CVODE_MIN_ABSOLUTE;
    keys["CVODE_MIN_RELATIVE"] = rr::Config::CVODE_MIN_RELATIVE;
    keys["SIMULATEOPTIONS_COPY_RESULT"] = rr::Config::SIMULATEOPTIONS_COPY_RESULT;
    keys["STEADYSTATE_RELATIVE"] = rr::Config::STEADYSTATE_RELATIVE;
    keys["STEADYSTATE_MAXIMUM_NUM_STEPS"] = rr::Config::STEADYSTATE_MAXIMUM_NUM_STEPS;
    keys["STEADYSTATE_MINIMUM_DAMPING"] = rr::Config::STEADYSTATE_MINIMUM_DAMPING;

    assert(rr::Config::CONFIG_END == sizeof(values) / sizeof(Variant) &&
            "values array size different than CONFIG_END");
    assert(rr::Config::CONFIG_END == keys.size()  &&
            "number of keys in map does not match static values");
}


std::string Config::getString(Keys key)
{
    readDefaultConfig();
    CHECK_RANGE(key);
    return values[key].convert<std::string>();
}

int Config::getInt(Keys key)
{
    readDefaultConfig();
    CHECK_RANGE(key);
    return values[key].convert<int>();
}

double Config::getDouble(Keys key)
{
    readDefaultConfig();
    CHECK_RANGE(key);
    return values[key].convert<double>();
}

std::string Config::getConfigFilePath()
{
    // check env var
    const char* env = std::getenv("ROADRUNNER_CONFIG");
    std::string path;
    Poco::Path ppath;

    Log(rr::Logger::LOG_DEBUG) << "trying config file from ROADRUNNER_CONFIG "
        << (env ? env : "NULL");

    if (env && rr::fileExists(env, 4))
    {
        return env;
    }

    // check home dir
    ppath.assign(Poco::Path::home());
    ppath.setFileName("roadrunner.conf");
    path = ppath.toString();
    Log(rr::Logger::LOG_DEBUG) << "trying config file " << path;
    if (rr::fileExists(path, 4))
    {
        return path;
    }

    ppath.setFileName(".roadrunner.conf");
    path = ppath.toString();
    Log(rr::Logger::LOG_DEBUG) << "trying config file " << path;
    if (rr::fileExists(path, 4))
    {
        return path;
    }

    // this could be an empty string if we are in a statically
    // linked executable, if so, Poco::Path will puke if popDir is called
    string chkDir = rr::getCurrentSharedLibDir();
    if (chkDir.empty())
    {
        chkDir = rr::getCurrentExeFolder();
    }

    assert(!chkDir.empty() && "could not get either shared lib or exe dir");

    // check in library dir
    ppath.assign(chkDir);
    ppath.setFileName("roadrunner.conf");
    path = ppath.toString();
    Log(rr::Logger::LOG_DEBUG) << "trying config file " << path;
    if (rr::fileExists(path, 4))
    {
        return path;
    }

    // check one level up
    ppath.assign(chkDir);
    ppath.popDirectory();
    ppath.setFileName("roadrunner.conf");
    path = ppath.toString();
    Log(rr::Logger::LOG_DEBUG) << "trying config file " << path;
    if (rr::fileExists(path, 4))
    {
        return path;
    }

    Log(rr::Logger::LOG_DEBUG) << "no config file found; using built-in defaults";
    return "";
}

void Config::setValue(Keys key, const Variant& value)
{
    readDefaultConfig();
    CHECK_RANGE(key);
    values[key] = value;
}

void Config::readConfigFile(const std::string& path)
{
    Mutex::ScopedLock lock(configMutex);

    const Poco::RegularExpression re("^\\s*(\\w*)\\s*:\\s*(.*)\\s*$", RegularExpression::RE_CASELESS);
    StringIntMap keys;
    std::ifstream in(path.c_str());

    if(!in) {
        throw std::ifstream::failure("could not open " + path + " for reading");
    }

    getKeyNames(keys);

    std::string line;

    while(std::getline(in, line)) {
        std::vector<std::string> matches;

        int nmatch = re.split(line, matches);

        if (nmatch == 3)
        {
            StringIntMap::const_iterator i = keys.find(matches[1]);
            if(i != keys.end()) {
                values[i->second] = Variant::parse((matches[2]));
                Log(Logger::LOG_INFORMATION) << "read key " << i->first << " with value: " << values[i->second].toString();
            } else {
                Log(Logger::LOG_WARNING) << "invalid key: \"" << matches[1] << "\" in " << path;
            }
        }
    }

    initialized = true;
}

const Variant& Config::getValue(Keys key)
{
    readDefaultConfig();
    CHECK_RANGE(key);
    return values[key];
}

bool Config::getBool(Keys key)
{
    readDefaultConfig();
    CHECK_RANGE(key);
    return values[key].convert<bool>();
}

void Config::writeConfigFile(const std::string& path)
{
    Mutex::ScopedLock lock(configMutex);

    std::ofstream out(path.c_str());

    if(!out) {
        throw std::ofstream::failure("could not open " + path + " for writing");
    }

    StringIntMap keys;
    std::ifstream in(path.c_str());

    getKeyNames(keys);

    for (StringIntMap::const_iterator i = keys.begin(); i != keys.end(); ++i) {
        out << i->first << ": " << values[i->second].toString() << std::endl;
    }
}

}
