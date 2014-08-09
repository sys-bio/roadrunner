/*
 * rrRoadRunnerOptions.cpp
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

using namespace std;

namespace rr
{

LoadSBMLOptions::LoadSBMLOptions()
{
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

    if (Config::getInt(Config::LOADSBMLOPTIONS_OPTIMIZE_CFG_SIMPLIFICATION))
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

    if (Config::getBool(Config::LOADSBMLOPTIONS_USE_GPUSIM))
        modelGeneratorOpt |= LoadSBMLOptions::USE_GPUSIM;

    loadFlags = 0;
}


static void getConfigValues(SimulateOptions *s)
{
    if (Config::getBool(Config::SIMULATEOPTIONS_STRUCTURED_RESULT)) {
        s->flags |=  SimulateOptions::STRUCTURED_RESULT;
    } else {
        s->flags &= ~SimulateOptions::STRUCTURED_RESULT;
    }


    if (Config::getBool(Config::SIMULATEOPTIONS_STIFF)) {
        s->integratorFlags |=  SimulateOptions::STIFF;
    } else {
        s->integratorFlags &= ~SimulateOptions::STIFF;
    }

    if (Config::getBool(Config::SIMULATEOPTIONS_MULTI_STEP)) {
        s->integratorFlags |= SimulateOptions::MULTI_STEP;
    } else {
        s->integratorFlags &= ~SimulateOptions::MULTI_STEP;
    }

    // set the variable step based on if we are using a stochastic or deterministic
    // integrator
    if (Config::getString(Config::SIMULATEOPTIONS_INTEGRATOR) == "CVODE") {
        s->integrator = SimulateOptions::CVODE;
    }
    else if (Config::getString(Config::SIMULATEOPTIONS_INTEGRATOR) == "GILLESPIE") {
        s->integrator = SimulateOptions::GILLESPIE;
    }
    else {
        Log(Logger::LOG_WARNING) << "Invalid integrator specified in configuration: "
                << Config::getString(Config::SIMULATEOPTIONS_INTEGRATOR)
        << std::endl << "Defaulting to CVODE";
        s->integrator = SimulateOptions::CVODE;
    }

    bool vs = false;

    if (SimulateOptions::getIntegratorType(s->integrator) == SimulateOptions::STOCHASTIC) {
        vs = Config::getBool(rr::Config::SIMULATEOPTIONS_STOCHASTIC_VARIABLE_STEP);
    }

    else if (SimulateOptions::getIntegratorType(s->integrator) == SimulateOptions::DETERMINISTIC) {
        vs = rr::Config::getBool(rr::Config::SIMULATEOPTIONS_DETERMINISTIC_VARIABLE_STEP);
    }

    if (vs) {
        s->integratorFlags |= rr::SimulateOptions::VARIABLE_STEP;
    } else {
        s->integratorFlags &= ~rr::SimulateOptions::VARIABLE_STEP;
    }

    if (rr::Config::getBool(rr::Config::SIMULATEOPTIONS_COPY_RESULT)) {
        s->flags |= SimulateOptions::COPY_RESULT;
    } else {
        s->flags &= ~SimulateOptions::COPY_RESULT;
    }
}



SimulateOptions::SimulateOptions()
:
steps(Config::getInt(Config::SIMULATEOPTIONS_STEPS)),
start(0),
duration(Config::getDouble(Config::SIMULATEOPTIONS_DURATION)),
absolute(Config::getDouble(Config::SIMULATEOPTIONS_ABSOLUTE)),
relative(Config::getDouble(Config::SIMULATEOPTIONS_RELATIVE)),
flags(0),
integrator(CVODE),
integratorFlags(0),
initialTimeStep(Config::getDouble(Config::SIMULATEOPTIONS_INITIAL_TIMESTEP)),
minimumTimeStep(Config::getDouble(Config::SIMULATEOPTIONS_MINIMUM_TIMESTEP)),
maximumTimeStep(Config::getDouble(Config::SIMULATEOPTIONS_MAXIMUM_TIMESTEP)),
maximumNumSteps(Config::getInt(Config::SIMULATEOPTIONS_MAXIMUM_NUM_STEPS))
{
    getConfigValues(this);
}

SimulateOptions::SimulateOptions(const std::string &fname)
:
steps(Config::getInt(Config::SIMULATEOPTIONS_STEPS)),
start(0),
duration(Config::getDouble(Config::SIMULATEOPTIONS_DURATION)),
absolute(Config::getDouble(Config::SIMULATEOPTIONS_ABSOLUTE)),
relative(Config::getDouble(Config::SIMULATEOPTIONS_RELATIVE)),
flags(0),
integrator(CVODE),
integratorFlags(0),
initialTimeStep(Config::getDouble(Config::SIMULATEOPTIONS_INITIAL_TIMESTEP)),
minimumTimeStep(Config::getDouble(Config::SIMULATEOPTIONS_MINIMUM_TIMESTEP)),
maximumTimeStep(Config::getDouble(Config::SIMULATEOPTIONS_MAXIMUM_TIMESTEP)),
maximumNumSteps(Config::getInt(Config::SIMULATEOPTIONS_MAXIMUM_NUM_STEPS))
{
    getConfigValues(this);

    if(!fname.size())
    {
        Log(Logger::LOG_ERROR)<<"Empty file name for setings file";
    }
    else
    {
        double minAbs = Config::getDouble(Config::CVODE_MIN_ABSOLUTE);
        double minRel = Config::getDouble(Config::CVODE_MIN_RELATIVE);


        map<string, string> settings;
        map<string, string>::iterator it;
        //Read each line in the settings file
        vector<string> lines = getLinesInFile(fname);
        for(int i = 0; i < lines.size(); i++)
        {
            vector<string> line = splitString(lines[i], ":");
            if(line.size() == 2)
            {
                settings.insert( pair<string, string>(line[0], line[1]));
            }
            else
            {
                Log(lDebug2)<<"Empty line in settings file: "<<lines[i];
            }
        }

        Log(lDebug3)<<"Settings File =============";
        for (it = settings.begin() ; it != settings.end(); it++ )
        {
            Log(lDebug) << (*it).first << " => " << (*it).second;
        }
        Log(lDebug)<<"===========================";



        //Assign values
        it = settings.find("start");
        start = (it != settings.end())       ? std::abs(toDouble((*it).second)): 0;

        it = settings.find("duration");
        duration = (it != settings.end())    ? std::abs(toDouble((*it).second)) : 0;

        it = settings.find("steps");
        steps = (it != settings.end())       ? std::abs(toInt((*it).second)) : 50;

        it = settings.find("absolute");
        absolute = (it != settings.end())    ? std::abs(toDouble((*it).second)) : minAbs;

        it = settings.find("relative");
        relative = (it != settings.end())    ? std::abs(toDouble((*it).second)) : minRel;



        it = settings.find("variables");
        if(it != settings.end())
        {
            vector<string> vars = splitString((*it).second, ",");
            for(int i=0; i < vars.size(); i++)
            {
                variables.push_back(trim(vars[i]));
            }
        }

        it = settings.find("amount");
        if(it != settings.end())
        {
            vector<string> vars = splitString((*it).second, ",");
            for(int i=0; i < vars.size(); i++)
            {
                string rec = trim(vars[i]);
                if(rec.size())
                {
                    amounts.push_back(rec);
                }
            }
        }

        it = settings.find("concentration");
        if(it != settings.end())
        {
            vector<string> vars = splitString((*it).second, ",");
            for(int i=0; i < vars.size(); i++)
            {
                string rec = trim(vars[i]);
                if(rec.size())
                {
                    concentrations.push_back(rec);
                }
            }
        }
    }
}

void SimulateOptions::setValue(const std::string& name,
        const rr::Variant& value)
{
    values[name] = value;
}

const Variant& SimulateOptions::getValue(const std::string& name) const
{
    VariantMap::const_iterator i = values.find(name);
    if (i != values.end())
    {
        return i->second;
    }

    throw std::invalid_argument("key not found: " + name);
}


RoadRunnerOptions::RoadRunnerOptions() :
    flags(0)
{
    if (Config::getBool(Config::ROADRUNNER_DISABLE_PYTHON_DYNAMIC_PROPERTIES))
    {
        flags |= RoadRunnerOptions::DISABLE_PYTHON_DYNAMIC_PROPERTIES;
    }

    jacobianStepSize = Config::getDouble(Config::ROADRUNNER_JACOBIAN_STEP_SIZE);
}

bool SimulateOptions::hasKey(const std::string& key) const
{
    VariantMap::const_iterator i = values.find(key);
    return i != values.end();
}

SimulateOptions::IntegratorType SimulateOptions::getIntegratorType(Integrator i)
{
    if (i == CVODE) {
        return DETERMINISTIC;
    } else {
        return STOCHASTIC;
    }
}

std::vector<std::string> SimulateOptions::getKeys() const
{
    std::vector<std::string> keys(values.size());
    int j = 0;

    for (VariantMap::const_iterator i = values.begin(); i != values.end(); ++i) {
        keys[j++] = i->first;
    }

    return keys;
}

int SimulateOptions::deleteValue(const std::string& key)
{
    return values.erase(key);
}


#define BITFIELD2STR(bf) rr::toString((bool)((bf) ? true : false))


std::string SimulateOptions::toString() const
{
    std::stringstream ss;

    ss << "integrator: ";
    if (integrator == CVODE) {
        ss << "cvode" << std::endl;
    }

    else if (integrator == GILLESPIE ) {
        ss << "gillespie" << std::endl;
    }

    else {
        ss << "unknown" << std::endl;
    }

    ss << "stiff: " << BITFIELD2STR(integratorFlags & STIFF) << std::endl;

    ss << "multiStep: " << BITFIELD2STR(integratorFlags & MULTI_STEP) << std::endl;

    ss << "variableStep: " << BITFIELD2STR(integratorFlags & VARIABLE_STEP) << std::endl;

    ss << "reset: " << BITFIELD2STR(flags & RESET_MODEL) << std::endl;

    ss << "structuredResult: " << BITFIELD2STR(flags & STRUCTURED_RESULT) << std::endl;

    ss << "copyResult: " << BITFIELD2STR(flags & COPY_RESULT) << std::endl;

    ss << "steps: " << steps << std::endl;

    ss << "start: " << start << std::endl;

    ss << "duration: " << duration << std::endl;

    ss << "relative: " << relative << std::endl;

    ss << "absolute: " << absolute << std::endl;

    ss << "initialTimeStep: " << initialTimeStep << std::endl;

    ss << "minimumTimeStep: " << minimumTimeStep << std::endl;

    ss << "maximumTimeStep: " << maximumTimeStep << std::endl;

    ss << "maximumNumSteps: " << maximumNumSteps << std::endl;

    return ss.str();
}

void SimulateOptions::tweakTolerances()
{
    if (integrator == CVODE)
    {
        double minAbs = Config::getDouble(Config::CVODE_MIN_ABSOLUTE);
        double minRel = Config::getDouble(Config::CVODE_MIN_RELATIVE);

        absolute = std::min(absolute, minAbs);
        relative = std::min(relative, minRel);

        Log(Logger::LOG_INFORMATION) << "tweaking CVODE tolerances to abs=" << absolute
                << ", rel=" << relative;
    }
}

} /* namespace rr */

