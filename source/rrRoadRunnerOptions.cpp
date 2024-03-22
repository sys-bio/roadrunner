/*
*
*  Created on: Sep 9, 2013
*      Author: andy
*/
#pragma hdrstop

#include "rrRoadRunnerOptions.h"
#include "rrLogger.h"
#include "rrUtils.h"
#include "rrSelectionRecord.h"
#include "rrConfig.h"

#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <assert.h>
#include <bitset>
#include <sstream>

namespace rr {

    LoadSBMLOptions::LoadSBMLOptions(const Dictionary *dict) {
        defaultInit();

        const LoadSBMLOptions *opt = dynamic_cast<const LoadSBMLOptions *>(dict);

        if (opt) {
            version = opt->version;
            size = opt->size;
            modelGeneratorOpt = opt->modelGeneratorOpt;
            loadFlags = opt->loadFlags;
            this->items = opt->items;
        } else if (dict) {
            std::vector<std::string> keys = dict->getKeys();
            for (std::vector<std::string>::const_iterator k = keys.begin(); k != keys.end(); ++k) {
                setItem(*k, dict->getItem(*k));
            }
        }
    }


    LoadSBMLOptions::LoadSBMLOptions() {
        defaultInit();
    }


    SimulateOptions::SimulateOptions()
        : reset_model(false)
        , structured_result(Config::getBool(Config::SIMULATEOPTIONS_STRUCTURED_RESULT))
        , copy_result(Config::getBool(Config::SIMULATEOPTIONS_COPY_RESULT))
        , steps(Config::getInt(Config::SIMULATEOPTIONS_STEPS))
        , start(0)
        , duration(Config::getDouble(Config::SIMULATEOPTIONS_DURATION))
        , times()
        , hstep(0) 
    {
    }

    void SimulateOptions::reset()
    {
        reset_model = false;
        structured_result = Config::getBool(Config::SIMULATEOPTIONS_STRUCTURED_RESULT);
        copy_result = Config::getBool(Config::SIMULATEOPTIONS_COPY_RESULT);
        steps = Config::getInt(Config::SIMULATEOPTIONS_STEPS);
        start = 0;
        duration = Config::getDouble(Config::SIMULATEOPTIONS_DURATION);
        times.clear();
        hstep = 0;
    }

    void SimulateOptions::loadSBMLSettings(const std::string &fname) {
        if (!fname.size()) {
            rrLog(Logger::LOG_ERROR) << "Empty file name for setings file";
        } else {
            std::map<std::string, std::string> settings;
            std::map<std::string, std::string>::iterator it;
            //Read each line in the settings file
            std::vector<std::string> lines = getLinesInFile(fname);
            for (int i = 0; i < lines.size(); i++) {
                std::vector<std::string> line = splitString(lines[i], ":");
                if (line.size() == 2) {
                    settings.insert(std::pair<std::string, std::string>(line[0], line[1]));
                } else {
                    rrLog(lDebug2) << "Empty line in settings file: " << lines[i];
                }
            }

            rrLog(lDebug3) << "Settings File =============";
            for (it = settings.begin(); it != settings.end(); it++) {
                rrLog(lDebug) << (*it).first << " => " << (*it).second;
            }
            rrLog(lDebug) << "===========================";

            //Assign values
            it = settings.find("start");
            start = (it != settings.end()) ? std::abs(toDouble((*it).second)) : 0;

            it = settings.find("duration");
            duration = (it != settings.end()) ? std::abs(toDouble((*it).second)) : 0;

            it = settings.find("steps");
            steps = (it != settings.end()) ? std::abs(toInt((*it).second)) : 50;

            it = settings.find("variables");
            if (it != settings.end()) {
                std::vector<std::string> vars = splitString((*it).second, ",");
                for (int i = 0; i < vars.size(); i++) {
                    variables.push_back(trim(vars[i]));
                }
            }

            //NOTE:  The 'amount' setting is never used anywhere: if a species appears
            // in the 'concentration' list, its ID is changed to "[ID]", and it's assumed
            // that if it does not, it should remain "ID".
            it = settings.find("amount");
            if (it != settings.end()) {
                std::vector<std::string> vars = splitString((*it).second, ",");
                for (int i = 0; i < vars.size(); i++) {
                    std::string rec = trim(vars[i]);
                    if (rec.size()) {
                        amounts.push_back(rec);
                    }
                }
            }

            it = settings.find("concentration");
            if (it != settings.end()) {
                std::vector<std::string> vars = splitString((*it).second, ",");
                for (int i = 0; i < vars.size(); i++) {
                    std::string rec = trim(vars[i]);
                    if (rec.size()) {
                        concentrations.push_back(rec);
                    }
                }
            }

            it = settings.find("output_file");
            output_file = (it != settings.end()) ? it->second : "";
        }
    }

    RoadRunnerOptions::RoadRunnerOptions() :
            flags(0) {
        if (Config::getBool(Config::ROADRUNNER_DISABLE_PYTHON_DYNAMIC_PROPERTIES)) {
            flags |= RoadRunnerOptions::DISABLE_PYTHON_DYNAMIC_PROPERTIES;
        }

        jacobianStepSize = Config::getDouble(Config::ROADRUNNER_JACOBIAN_STEP_SIZE);
        diffStepSize = Config::getDouble(Config::METABOLIC_CONTROL_ANALYSIS_DIFFERENTIAL_STEP_SIZE);
        steadyStateThreshold = Config::getDouble(Config::METABOLIC_CONTROL_ANALYSIS_STEADY_STATE_THRESHOLD);
        fluxThreshold = Config::getDouble(Config::METABOLIC_CONTROL_ANALYSIS_FLUX_THRESHOLD);
    }


    std::string SimulateOptions::toString() const {
        std::stringstream ss;

        ss << "< roadrunner.SimulateOptions() " << std::endl << "{ "
           << std::endl << "'this' : " << (void *) this << ", " << std::endl;

        ss << "'reset' : " << reset_model << "," << std::endl;

        ss << "'structuredResult' : " << structured_result << "," << std::endl;

        ss << "'copyResult' : " << copy_result << "," << std::endl;

        ss << "'steps' : " << steps << "," << std::endl;

        ss << "'start' : " << start << "," << std::endl;

        ss << "'duration' : " << duration << std::endl;

        ss << "'output_file' : " << output_file;

        //std::vector<std::string> keys = getKeys();

        //if (keys.size() > 0) {
        //    ss << "," << std::endl;
        //}

        //for (std::vector<std::string>::iterator i = keys.begin(); i != keys.end(); ++i) {
        //    ss << "'" << *i << "' : ";
        //    ss << getItem(*i).get<std::string>();

        //    if (std::distance(i, keys.end()) > 1) {
        //        ss << ", " << std::endl;
        //    }
        //}

        ss << std::endl << "}>";

        return ss.str();
    }

    std::string SimulateOptions::toRepr() const {
        std::stringstream ss;
        ss << "< roadrunner.SimulateOptions() { 'this' : "
           << (void *) this << " }>";
        return ss.str();
    }

    //void SimulateOptions::setItem(const std::string &key, const rr::Setting &value) {
    //    //BasicDictionary::setItem(key, value);
    //    if ((key == "reset" || key=="reset_model") && value.isBool() && value.getAs<bool>())
    //    {
    //        reset_model = true;
    //    }
    //    else if (key == "reset_model"
    //        || key == "structured_result"
    //        || key == "copy_result"
    //        || key == "steps"
    //        || key == "start"
    //        || key == "duration")
    //    {
    //        throw std::invalid_argument("The option '" + key + "' cannot be set by using 'setItem'.  Intead, set the member variable directly (i.e. 'opt." + key + " = <value>.");
    //    }
    //}

    void SimulateOptions::initialize() {
        if (times.empty())//!hasKey("times"))
        {
            hstep = duration / steps;
        }
        else {
            if (steps != times.size() - 1) {
                if (steps == Config::getInt(Config::SIMULATEOPTIONS_STEPS)) {
                    steps = times.size() - 1;
                }
                else {
                    std::stringstream err;
                    err << "If the 'times' and the 'steps' settings are both used, the number of steps must equal the length of the 'times' vector, minus one.  The length of the 'times' vector is "
                        << times.size() << ", and the 'steps' setting is " << steps << ".";
                    throw std::invalid_argument(err.str());
                }
            }
            if (times.size() <= 1) {
                throw std::invalid_argument("The 'times' setting must be a vector of at least two values, as the first value is the time at the initial state of the model, and the second (and subsequent) times are the times the simulation progresses to.");
            }
            if (times[0] != start) {
                if (start == 0) //The default.
                {
                    start = times[0];
                }
                else {
                    std::stringstream err;
                    err << "If the 'times' and the 'start' settings are both used, the first value of 'times' must equal the value of 'start'.  Instead, 'start' is "
                        << start << ", and the first value of 'times' is " << times[0] << ".";
                    throw std::invalid_argument(err.str());
                }
            }
            double prev = start;
            for (size_t tv = 1; tv < times.size(); tv++) {
                double hstep = times[tv] - prev;
                if (hstep < 0) {
                    std::stringstream err;
                    err << "The 'times' setting must be a vector of time values that start at the time value at the initial state of the model and increase along the vector.  The value "
                        << times[tv] << " is less than the previous value of " << prev << ".";
                    throw std::invalid_argument(err.str());
                }
                prev = times[tv];
            }
        }
    }

    double SimulateOptions::getNext(size_t step) {
        if (hstep) {
            return start + step * hstep;
        }
        if (step > times.size()) {
            std::stringstream err;
            err << "Cannot get the time step " << step << " because there are only " << times.size()
                << " set for the output.";
            throw std::invalid_argument(err.str());
        }
        return times[step];
    }

    void LoadSBMLOptions::setItem(const std::string &key, const rr::Setting &value) {
        BasicDictionary::setItem(key, value);
    }

    Setting LoadSBMLOptions::getItem(const std::string &key) const {
        return BasicDictionary::getItem(key);
    }

    bool LoadSBMLOptions::hasKey(const std::string &key) const {
        return BasicDictionary::hasKey(key);
    }

    size_t LoadSBMLOptions::deleteItem(const std::string &key) {
        return BasicDictionary::deleteItem(key);
    }

    std::vector<std::string> LoadSBMLOptions::getKeys() const {
        return BasicDictionary::getKeys();
    }

    LoadSBMLOptions::~LoadSBMLOptions() {
    }

    void LoadSBMLOptions::defaultInit() {
        version = 0;
        size = sizeof(LoadSBMLOptions);
        modelGeneratorOpt = 0;

        if (Config::getBool(Config::LOADSBMLOPTIONS_CONSERVED_MOIETIES))
            modelGeneratorOpt |= LoadSBMLOptions::CONSERVED_MOIETIES;

        if (Config::getBool(Config::LOADSBMLOPTIONS_RECOMPILE))
            modelGeneratorOpt |= LoadSBMLOptions::RECOMPILE;

        if (Config::getBool(Config::LOADSBMLOPTIONS_READ_ONLY))
            modelGeneratorOpt |= LoadSBMLOptions::READ_ONLY;

        if (Config::getBool(Config::LOADSBMLOPTIONS_MUTABLE_INITIAL_CONDITIONS))
            modelGeneratorOpt |= LoadSBMLOptions::MUTABLE_INITIAL_CONDITIONS;

        if (Config::getBool(Config::LOADSBMLOPTIONS_OPTIMIZE_GVN))
            modelGeneratorOpt |= LoadSBMLOptions::OPTIMIZE_GVN;

        if (Config::getBool(Config::LOADSBMLOPTIONS_OPTIMIZE_CFG_SIMPLIFICATION))
            modelGeneratorOpt |= LoadSBMLOptions::OPTIMIZE_CFG_SIMPLIFICATION;

        if (Config::getBool(Config::LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_COMBINING))
            modelGeneratorOpt |= LoadSBMLOptions::OPTIMIZE_INSTRUCTION_COMBINING;

        if (Config::getBool(Config::LOADSBMLOPTIONS_OPTIMIZE_DEAD_INST_ELIMINATION))
            modelGeneratorOpt |= LoadSBMLOptions::OPTIMIZE_DEAD_INST_ELIMINATION;

        if (Config::getBool(Config::LOADSBMLOPTIONS_OPTIMIZE_DEAD_CODE_ELIMINATION))
            modelGeneratorOpt |= LoadSBMLOptions::OPTIMIZE_DEAD_CODE_ELIMINATION;

        if (Config::getBool(Config::LOADSBMLOPTIONS_OPTIMIZE_INSTRUCTION_SIMPLIFIER))
            modelGeneratorOpt |= LoadSBMLOptions::OPTIMIZE_INSTRUCTION_SIMPLIFIER;

        if (Config::getBool(Config::LLVM_SYMBOL_CACHE))
            modelGeneratorOpt |= LoadSBMLOptions::LLVM_SYMBOL_CACHE;

        // todo delete this options and use the below system instead
        if (Config::getBool(Config::LOADSBMLOPTIONS_USE_MCJIT))
            modelGeneratorOpt |= LoadSBMLOptions::USE_MCJIT;

        // use the values in Config to update values in Options
        Config::LLVM_BACKEND_VALUES whichBackend =
                (Config::LLVM_BACKEND_VALUES) Config::getValue(Config::LLVM_BACKEND).getAs<int>();

        // updates modelGeneratorOpt inside setLLVMBackend...
        switch (whichBackend) {
            case Config::MCJIT:
                setLLVMBackend(LoadSBMLOptions::MCJIT);
                break;
            case Config::LLJIT:
                setLLVMBackend(LoadSBMLOptions::LLJIT);
                break;
            default:
                std::string err = "Compiler option is invalid";
                rrLogWarn << err;
                throw std::invalid_argument(err);
        }

        Config::LLJIT_OPTIMIZATION_LEVELS whichOptLevel
                = (Config::LLJIT_OPTIMIZATION_LEVELS) Config::getValue(Config::LLJIT_OPTIMIZATION_LEVEL).getAs<int>();

        switch (whichOptLevel) {
            case Config::NONE:
                setLLJitOptimizationLevel(LoadSBMLOptions::LLJIT_OPTIMIZATION_LEVELS::NONE);
                break;
            case Config::LESS:
                setLLJitOptimizationLevel(LoadSBMLOptions::LLJIT_OPTIMIZATION_LEVELS::LESS);
                break;
            case Config::DEFAULT:
                setLLJitOptimizationLevel(LoadSBMLOptions::LLJIT_OPTIMIZATION_LEVELS::DEFAULT);
                break;
            case Config::AGGRESSIVE:
                setLLJitOptimizationLevel(LoadSBMLOptions::LLJIT_OPTIMIZATION_LEVELS::AGGRESSIVE);
                break;
        }

        setItem("tempDir", Setting(std::string()));
        setItem("compiler", Setting("LLVM"));
        setItem("supportCodeDir", Setting(std::string()));
        loadFlags = 0;
    }

    void LoadSBMLOptions::setLLVMBackend(LoadSBMLOptions::LLVM_BACKEND_VALUES val) {
        // backend options are multiple choice. So iterate over all and turn them off
        for (auto backendValue: getAllLLVMBackendValues()) {
            modelGeneratorOpt = modelGeneratorOpt & ~backendValue;
        }

        // before turning the right one on
        switch (val) {
            case MCJIT:
                modelGeneratorOpt = modelGeneratorOpt | LLVM_BACKEND_VALUES::MCJIT;
                break;
            case LLJIT:
                modelGeneratorOpt = modelGeneratorOpt | LLVM_BACKEND_VALUES::LLJIT;
                break;
        }
    }

    void LoadSBMLOptions::setLLJitOptimizationLevel(LoadSBMLOptions::LLJIT_OPTIMIZATION_LEVELS level) {
        // turn off all options
        for (auto lljitOptValue: getAllLLJitOptimizationValues()) {
            modelGeneratorOpt = modelGeneratorOpt & ~lljitOptValue;
        }

        // now turn just the one on
        switch (level) {
            case NONE:
                modelGeneratorOpt = modelGeneratorOpt | LLJIT_OPTIMIZATION_LEVELS::NONE;
                break;
            case LESS:
                modelGeneratorOpt = modelGeneratorOpt | LLJIT_OPTIMIZATION_LEVELS::LESS;
                break;
            case DEFAULT:
                modelGeneratorOpt = modelGeneratorOpt | LLJIT_OPTIMIZATION_LEVELS::DEFAULT;
                break;
            case AGGRESSIVE:
                modelGeneratorOpt = modelGeneratorOpt | LLJIT_OPTIMIZATION_LEVELS::AGGRESSIVE;
                break;
        }
    }

} /* namespace rr */

