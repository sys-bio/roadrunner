/*
 * Config.cpp
 *
 *  Created on: Mar 24, 2014
 *      Author: andy
 */

#include "rrUtils.h"
#include "rrConfig.h"
#include "rrLogger.h"

#include "tr1proxy/rr_memory.h"
#include "tr1proxy/rr_unordered_map.h"

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
    Variant(false),     // LOADSBMLOPTIONS_CONSERVED_MOIETIES
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
    Variant(true),     // SIMULATEOPTIONS_STIFF,
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
    Variant(int(AllChecksON & UnitsCheckOFF)),           //SBML_APPLICABLEVALIDATORS
    Variant(0.00001),   // ROADRUNNER_JACOBIAN_STEP_SIZE
    Variant((int)(SelectionRecord::TIME
            | SelectionRecord::RATE
            | SelectionRecord::FLOATING)),        // MODEL_RESET
    Variant(1.e-12),   // CVODE_MIN_ABSOLUTE
    Variant(1.e-6),    // CVODE_MIN_RELATIVE
    Variant(true),     // SIMULATEOPTIONS_COPY_RESULT
    Variant(true),      // STEADYSTATE_APPROX_DEFAULT
    Variant(1.e-12),    // STEADYSTATE_APPROX_TOL
    Variant(10000),      // STEADYSTATE_APPROX_MAX_STEPS
    Variant(10000),      // STEADYSTATE_APPROX_TIME
    Variant(1e-16),      // STEADYSTATE_RELATIVE
    Variant(100),      // STEADYSTATE_MAXIMUM_NUM_STEPS
    Variant(1.e-20),   // STEADYSTATE_MINIMUM_DAMPING
    Variant(0),        // STEADYSTATE_BROYDEN
    Variant(3),        // STEADYSTATE_LINEARITY
    Variant((int)Config::ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS), // ROADRUNNER_JACOBIAN_MODE
    Variant(std::string(".")),                           // TEMP_DIR_PATH,
    Variant(std::string("")),                            // LOGGER_LOG_FILE_PATH,
    Variant(-1),                              // RANDOM_SEED
    Variant(true),      // PYTHON_ENABLE_NAMED_MATRIX
    Variant(true),      // LLVM_SYMBOL_CACHE
    Variant(true),      // OPTIMIZE_REACTION_RATE_SELECTION
    Variant(true),     // LOADSBMLOPTIONS_PERMISSIVE
    Variant(20000)      // MAX_OUTPUT_ROWS
    // add space after develop keys to clean up merging























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
    keys["STEADYSTATE_APPROX_DEFAULT"] = rr::Config::STEADYSTATE_APPROX_DEFAULT;
    keys["STEADYSTATE_APPROX_TOL"] = rr::Config::STEADYSTATE_APPROX_TOL;
    keys["STEADYSTATE_APPROX_MAX_STEPS"] = rr::Config::STEADYSTATE_APPROX_MAX_STEPS;
    keys["STEADYSTATE_APPROX_TIME"] = rr::Config::STEADYSTATE_APPROX_TIME;
    keys["STEADYSTATE_RELATIVE"] = rr::Config::STEADYSTATE_RELATIVE;
    keys["STEADYSTATE_MAXIMUM_NUM_STEPS"] = rr::Config::STEADYSTATE_MAXIMUM_NUM_STEPS;
    keys["STEADYSTATE_MINIMUM_DAMPING"] = rr::Config::STEADYSTATE_MINIMUM_DAMPING;
    keys["STEADYSTATE_BROYDEN"] = rr::Config::STEADYSTATE_BROYDEN;
    keys["STEADYSTATE_LINEARITY"] = rr::Config::STEADYSTATE_LINEARITY;
    keys["ROADRUNNER_JACOBIAN_MODE"] = rr::Config::ROADRUNNER_JACOBIAN_MODE;
    keys["TEMP_DIR_PATH"] = rr::Config::TEMP_DIR_PATH;
    keys["LOGGER_LOG_FILE_PATH"] = rr::Config::LOGGER_LOG_FILE_PATH;
    keys["RANDOM_SEED"] = rr::Config::RANDOM_SEED;
    keys["PYTHON_ENABLE_NAMED_MATRIX"] = rr::Config::PYTHON_ENABLE_NAMED_MATRIX;
    keys["LLVM_SYMBOL_CACHE"] = rr::Config::LLVM_SYMBOL_CACHE;
    keys["OPTIMIZE_REACTION_RATE_SELECTION"] = rr::Config::OPTIMIZE_REACTION_RATE_SELECTION;
    keys["LOADSBMLOPTIONS_PERMISSIVE"] = rr::Config::LOADSBMLOPTIONS_PERMISSIVE;
    keys["MAX_OUTPUT_ROWS"] = rr::Config::MAX_OUTPUT_ROWS;



    // add space after develop keys to clean up merging.


























    assert(rr::Config::CONFIG_END == sizeof(values) / sizeof(Variant) &&
            "values array size different than CONFIG_END");
    assert(rr::Config::CONFIG_END == keys.size()  &&
            "number of keys in map does not match static values");
}

static std::string reverseLookup(StringIntMap& keys, Config::Keys k) {
    for (StringIntMap::iterator i=keys.begin(); i!= keys.end(); ++i) {
        if (i->second == k)
            return i->first;
    }
    throw std::runtime_error("No such key");
}

std::vector<string> Config::getKeyList() {
    std::vector<string> result;
    StringIntMap m;

    getKeyNames(m);

    for(int n=0; n<CONFIG_END;++n) {
        try {
            std::string key_str = reverseLookup(m,(Config::Keys)n);
            result.push_back(key_str);
        } catch(std::runtime_error) {}
    }

    return result;
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

Config::Keys Config::stringToKey(const std::string& key) {
    if (key == "LOADSBMLOPTIONS_CONSERVED_MOIETIES")
        return Config::LOADSBMLOPTIONS_CONSERVED_MOIETIES;
    else if (key == "LOADSBMLOPTIONS_RECOMPILE")
        return Config::LOADSBMLOPTIONS_RECOMPILE;
    else if (key == "LOADSBMLOPTIONS_READ_ONLY")
        return Config::LOADSBMLOPTIONS_READ_ONLY;
    else if (key == "LOADSBMLOPTIONS_MUTABLE_INITIAL_CONDITIONS")
        return Config::LOADSBMLOPTIONS_MUTABLE_INITIAL_CONDITIONS;
    else if (key == "LOADSBMLOPTIONS_OPTIMIZE_GVN")
        return Config::LOADSBMLOPTIONS_OPTIMIZE_GVN;
    else if (key == "LOADSBMLOPTIONS_OPTIMIZE_CFG_SIMPLIFICATION")
        return Config::LOADSBMLOPTIONS_OPTIMIZE_CFG_SIMPLIFICATION;
    else if (key == "LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_COMBINING")
        return Config::LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_COMBINING;
    else if (key == "LOADSBMLOPTIONS_OPTIMIZE_DEAD_INST_ELIMINATION")
        return Config::LOADSBMLOPTIONS_OPTIMIZE_DEAD_INST_ELIMINATION;
    else if (key == "LOADSBMLOPTIONS_OPTIMIZE_DEAD_CODE_ELIMINATION")
        return Config::LOADSBMLOPTIONS_OPTIMIZE_DEAD_CODE_ELIMINATION;
    else if (key == "LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_SIMPLIFIER")
        return Config::LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_SIMPLIFIER;
    else if (key == "LOADSBMLOPTIONS_USE_MCJIT")
        return Config::LOADSBMLOPTIONS_USE_MCJIT;
    else if (key == "SIMULATEOPTIONS_STEPS")
        return Config::SIMULATEOPTIONS_STEPS;
    else if (key == "SIMULATEOPTIONS_DURATION")
        return Config::SIMULATEOPTIONS_DURATION;
    else if (key == "SIMULATEOPTIONS_ABSOLUTE")
        return Config::SIMULATEOPTIONS_ABSOLUTE;
    else if (key == "SIMULATEOPTIONS_RELATIVE")
        return Config::SIMULATEOPTIONS_RELATIVE;
    else if (key == "SIMULATEOPTIONS_STRUCTURED_RESULT")
        return Config::SIMULATEOPTIONS_STRUCTURED_RESULT;
    else if (key == "SIMULATEOPTIONS_STIFF")
        return Config::SIMULATEOPTIONS_STIFF;
    else if (key == "SIMULATEOPTIONS_MULTI_STEP")
        return Config::SIMULATEOPTIONS_MULTI_STEP;
    else if (key == "SIMULATEOPTIONS_DETERMINISTIC_VARIABLE_STEP")
        return Config::SIMULATEOPTIONS_DETERMINISTIC_VARIABLE_STEP;
    else if (key == "SIMULATEOPTIONS_STOCHASTIC_VARIABLE_STEP")
        return Config::SIMULATEOPTIONS_STOCHASTIC_VARIABLE_STEP;
    else if (key == "SIMULATEOPTIONS_INTEGRATOR")
        return Config::SIMULATEOPTIONS_INTEGRATOR;
    else if (key == "SIMULATEOPTIONS_INITIAL_TIMESTEP")
        return Config::SIMULATEOPTIONS_INITIAL_TIMESTEP;
    else if (key == "SIMULATEOPTIONS_MINIMUM_TIMESTEP")
        return Config::SIMULATEOPTIONS_MINIMUM_TIMESTEP;
    else if (key == "SIMULATEOPTIONS_MAXIMUM_TIMESTEP")
        return Config::SIMULATEOPTIONS_MAXIMUM_TIMESTEP;
    else if (key == "SIMULATEOPTIONS_MAXIMUM_NUM_STEPS")
        return Config::SIMULATEOPTIONS_MAXIMUM_NUM_STEPS;
    else if (key == "ROADRUNNER_DISABLE_WARNINGS")
        return Config::ROADRUNNER_DISABLE_WARNINGS;
    else if (key == "ROADRUNNER_DISABLE_PYTHON_DYNAMIC_PROPERTIES")
        return Config::ROADRUNNER_DISABLE_PYTHON_DYNAMIC_PROPERTIES;
    else if (key == "SBML_APPLICABLEVALIDATORS")
        return Config::SBML_APPLICABLEVALIDATORS;
    else if (key == "ROADRUNNER_JACOBIAN_STEP_SIZE")
        return Config::ROADRUNNER_JACOBIAN_STEP_SIZE;
    else if (key == "MODEL_RESET")
        return Config::MODEL_RESET;
    else if (key == "CVODE_MIN_ABSOLUTE")
        return Config::CVODE_MIN_ABSOLUTE;
    else if (key == "CVODE_MIN_RELATIVE")
        return Config::CVODE_MIN_RELATIVE;
    else if (key == "SIMULATEOPTIONS_COPY_RESULT")
        return Config::SIMULATEOPTIONS_COPY_RESULT;
    else if (key == "STEADYSTATE_APPROX_DEFAULT")
        return Config::STEADYSTATE_APPROX_DEFAULT;
    else if (key == "STEADYSTATE_APPROX_TOL")
        return Config::STEADYSTATE_APPROX_TOL;
    else if (key == "STEADYSTATE_APPROX_MAX_STEPS")
        return Config::STEADYSTATE_APPROX_MAX_STEPS;
    else if (key == "STEADYSTATE_APPROX_TIME")
        return Config::STEADYSTATE_APPROX_TIME;
    else if (key == "STEADYSTATE_RELATIVE")
        return Config::STEADYSTATE_RELATIVE;
    else if (key == "STEADYSTATE_MAXIMUM_NUM_STEPS")
        return Config::STEADYSTATE_MAXIMUM_NUM_STEPS;
    else if (key == "STEADYSTATE_MINIMUM_DAMPING")
        return Config::STEADYSTATE_MINIMUM_DAMPING;
    else if (key == "STEADYSTATE_BROYDEN")
        return Config::STEADYSTATE_BROYDEN;
    else if (key == "STEADYSTATE_LINEARITY")
        return Config::STEADYSTATE_LINEARITY;
    else if (key == "ROADRUNNER_JACOBIAN_MODE")
        return Config::ROADRUNNER_JACOBIAN_MODE;
    else if (key == "TEMP_DIR_PATH")
        return Config::TEMP_DIR_PATH;
    else if (key == "LOGGER_LOG_FILE_PATH")
        return Config::LOGGER_LOG_FILE_PATH;
    else if (key == "RANDOM_SEED")
        return Config::RANDOM_SEED;
    else if (key == "PYTHON_ENABLE_NAMED_MATRIX")
        return Config::PYTHON_ENABLE_NAMED_MATRIX;
    else if (key == "LLVM_SYMBOL_CACHE")
        return Config::LLVM_SYMBOL_CACHE;
    else if (key == "OPTIMIZE_REACTION_RATE_SELECTION")
        return Config::OPTIMIZE_REACTION_RATE_SELECTION;
    else if (key == "LOADSBMLOPTIONS_PERMISSIVE")
        return Config::LOADSBMLOPTIONS_PERMISSIVE;
    else if (key == "MAX_OUTPUT_ROWS")
        return Config::MAX_OUTPUT_ROWS;
    else
        throw std::runtime_error("No such config key: '" + key + "'");
}

}