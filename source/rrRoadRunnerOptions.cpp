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
            : reset_model(false), structured_result(Config::getBool(Config::SIMULATEOPTIONS_STRUCTURED_RESULT)),
              copy_result(Config::getBool(Config::SIMULATEOPTIONS_COPY_RESULT)),
              steps(Config::getInt(Config::SIMULATEOPTIONS_STEPS)), start(0),
              duration(Config::getDouble(Config::SIMULATEOPTIONS_DURATION)) {
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

        std::vector<std::string> keys = getKeys();

        if (keys.size() > 0) {
            ss << "," << std::endl;
        }

        for (std::vector<std::string>::iterator i = keys.begin(); i != keys.end(); ++i) {
            ss << "'" << *i << "' : ";
            ss << getItem(*i).get<std::string>();

            if (std::distance(i, keys.end()) > 1) {
                ss << ", " << std::endl;
            }
        }

        ss << std::endl << "}>";

        return ss.str();
    }

    std::string SimulateOptions::toRepr() const {
        std::stringstream ss;
        ss << "< roadrunner.SimulateOptions() { 'this' : "
           << (void *) this << " }>";
        return ss.str();
    }

    void SimulateOptions::setItem(const std::string &key, const rr::Setting &value) {
        BasicDictionary::setItem(key, value);
        if (key == "reset")
            reset_model = true;
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

        if (Config::getBool(Config::LOADSBMLOPTIONS_USE_MCJIT))
            modelGeneratorOpt |= LoadSBMLOptions::USE_MCJIT;

        if (Config::getBool(Config::LLVM_SYMBOL_CACHE))
            modelGeneratorOpt |= LoadSBMLOptions::LLVM_SYMBOL_CACHE;


        setItem("tempDir", Setting(std::string()));
        setItem("compiler", Setting("LLVM"));
        setItem("supportCodeDir", Setting(std::string()));

        loadFlags = 0;
    }

} /* namespace rr */

