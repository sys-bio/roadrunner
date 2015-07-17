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

using namespace std;

namespace rr
{

	
	


	LoadSBMLOptions::LoadSBMLOptions(const Dictionary* dict)
	{
		defaultInit();

		const LoadSBMLOptions *opt = dynamic_cast<const LoadSBMLOptions*>(dict);

		if (opt) {
			version = opt->version;
			size = opt->size;
			modelGeneratorOpt = opt->modelGeneratorOpt;
			loadFlags = opt->loadFlags;
			this->items = opt->items;
		}
		else if (dict) {
			std::vector<std::string> keys = dict->getKeys();
			for (std::vector<std::string>::const_iterator k = keys.begin(); k != keys.end(); ++k) {
				setItem(*k, dict->getItem(*k));
			}
		}
	}



	LoadSBMLOptions::LoadSBMLOptions()
	{
		defaultInit();
	}


	static void getConfigValues(SimulateOptions *s)
	{
		if (Config::getBool(Config::SIMULATEOPTIONS_STRUCTURED_RESULT)) 
		{
			s->structured_result = true;
		}
		else 
		{
			s->structured_result = false;
		}

		// set the variable step based on if we are using a stochastic or deterministic
		// integrator
		if (Config::getString(Config::SIMULATEOPTIONS_INTEGRATOR) == "CVODE") {
			s->integrator = "cvode";
		}
		//else if (Config::getString(Config::SIMULATEOPTIONS_INTEGRATOR) == "GILLESPIE") {
		//	s->integrator = Integrator::GILLESPIE;
		//}
    else if (Config::getString(Config::SIMULATEOPTIONS_INTEGRATOR) == "GPUSIM") {
        s->integrator = SimulateOptions::GPUSIM;
    }
		else {
			Log(Logger::LOG_WARNING) << "Invalid integrator specified in configuration: "
				<< Config::getString(Config::SIMULATEOPTIONS_INTEGRATOR)
				<< std::endl << "Defaulting to CVODE";
			s->integrator = "cvode";
		}

		if (rr::Config::getBool(rr::Config::SIMULATEOPTIONS_COPY_RESULT)) {
			s->copy_result = true;
		}
		else {
			s->copy_result = false;
		}
	}
	

	SimulateOptions::SimulateOptions()
		:
		steps(Config::getInt(Config::SIMULATEOPTIONS_STEPS)),
		start(0),
		duration(Config::getDouble(Config::SIMULATEOPTIONS_DURATION)),
		integrator("cvode"),
		initialTimeStep(Config::getDouble(Config::SIMULATEOPTIONS_INITIAL_TIMESTEP)),
		minimumTimeStep(Config::getDouble(Config::SIMULATEOPTIONS_MINIMUM_TIMESTEP)),
		maximumTimeStep(Config::getDouble(Config::SIMULATEOPTIONS_MAXIMUM_TIMESTEP)),
		maximumNumSteps(Config::getInt(Config::SIMULATEOPTIONS_MAXIMUM_NUM_STEPS))
	{
		getConfigValues(this);
	}

	void SimulateOptions::loadSBMLSettings(const std::string &fname)
	{
		if (!fname.size())
		{
			Log(Logger::LOG_ERROR) << "Empty file name for setings file";
		}
		else
		{
			map<string, string> settings;
			map<string, string>::iterator it;
			//Read each line in the settings file
			vector<string> lines = getLinesInFile(fname);
			for (int i = 0; i < lines.size(); i++)
			{
				vector<string> line = splitString(lines[i], ":");
				if (line.size() == 2)
				{
					settings.insert(pair<string, string>(line[0], line[1]));
				}
				else
				{
					Log(lDebug2) << "Empty line in settings file: " << lines[i];
				}
			}

			Log(lDebug3) << "Settings File =============";
			for (it = settings.begin(); it != settings.end(); it++)
			{
				Log(lDebug) << (*it).first << " => " << (*it).second;
			}
			Log(lDebug) << "===========================";

			//Assign values
			it = settings.find("start");
			start = (it != settings.end()) ? std::abs(toDouble((*it).second)) : 0;

			it = settings.find("duration");
			duration = (it != settings.end()) ? std::abs(toDouble((*it).second)) : 0;

			it = settings.find("steps");
			steps = (it != settings.end()) ? std::abs(toInt((*it).second)) : 50;

			it = settings.find("variables");
			if (it != settings.end())
			{
				vector<string> vars = splitString((*it).second, ",");
				for (int i = 0; i < vars.size(); i++)
				{
					variables.push_back(trim(vars[i]));
				}
			}

			it = settings.find("amount");
			if (it != settings.end())
			{
				vector<string> vars = splitString((*it).second, ",");
				for (int i = 0; i < vars.size(); i++)
				{
					string rec = trim(vars[i]);
					if (rec.size())
					{
						amounts.push_back(rec);
					}
				}
			}

			it = settings.find("concentration");
			if (it != settings.end())
			{
				vector<string> vars = splitString((*it).second, ",");
				for (int i = 0; i < vars.size(); i++)
				{
					string rec = trim(vars[i]);
					if (rec.size())
					{
						concentrations.push_back(rec);
					}
				}
			}
		}
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



#define BITFIELD2STR(bf) rr::toString((bool)((bf) ? true : false))


	std::string SimulateOptions::toString() const
	{
		std::stringstream ss;

		ss << "< roadrunner.SimulateOptions() " << endl << "{ "
			<< endl << "'this' : " << (void*)this << ", " << std::endl;

		ss << "integrator: ";
		if (integrator == "cvode") {
			ss << "\"cvode\"," << std::endl;
		}

		else if (integrator == "gillespie") {
			ss << "\"gillespie\"," << std::endl;
		}

    else if (integrator == GPUSIM ) {
        ss << "\"gpusim\"," << std::endl;
    }

		else {
			ss << "\"unknown\"," << std::endl;
		}

		ss << "'reset' : " << reset_model << "," << std::endl;

		ss << "'structuredResult' : " << structured_result << "," << std::endl;

		ss << "'copyResult' : " << copy_result << "," << std::endl;

		ss << "'steps' : " << steps << "," << std::endl;

		ss << "'start' : " << start << "," << std::endl;

		ss << "'duration' : " << duration << "," << std::endl;

		ss << "'initialTimeStep' : " << initialTimeStep << "," << std::endl;

		ss << "'minimumTimeStep' : " << minimumTimeStep << "," << std::endl;

		ss << "'maximumTimeStep' : " << maximumTimeStep << "," << std::endl;

		ss << "'maximumNumSteps' : " << maximumNumSteps;

		std::vector<std::string> keys = getKeys();

		if (keys.size() > 0) {
			ss << "," << std::endl;
		}

		for (std::vector<std::string>::iterator i = keys.begin(); i != keys.end(); ++i)
		{
			ss << "'" << *i << "' : ";
			ss << getItem(*i).toString();

			if (std::distance(i, keys.end()) > 1) {
				ss << ", " << std::endl;
			}
		}

		ss << std::endl << "}>";

		return ss.str();
	}

	std::string SimulateOptions::toRepr() const
	{
		std::stringstream ss;
		ss << "< roadrunner.SimulateOptions() { 'this' : "
			<< (void*)this << " }>";
		return ss.str();
	}

	


	void LoadSBMLOptions::setItem(const std::string& key, const rr::Variant& value)
	{
		BasicDictionary::setItem(key, value);
	}

	Variant LoadSBMLOptions::getItem(const std::string& key) const
	{
		return BasicDictionary::getItem(key);
	}

	bool LoadSBMLOptions::hasKey(const std::string& key) const
	{
		return BasicDictionary::hasKey(key);
	}

	int LoadSBMLOptions::deleteItem(const std::string& key)
	{
		return BasicDictionary::deleteItem(key);
	}

	std::vector<std::string> LoadSBMLOptions::getKeys() const
	{
		return BasicDictionary::getKeys();
	}

	LoadSBMLOptions::~LoadSBMLOptions()
	{
	}

	void LoadSBMLOptions::defaultInit()
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

		if (Config::getBool(Config::LLVM_SYMBOL_CACHE))
			modelGeneratorOpt |= LoadSBMLOptions::LLVM_SYMBOL_CACHE;


		setItem("tempDir", "");
		setItem("compiler", "LLVM");
		setItem("supportCodeDir", "");

		loadFlags = 0;
	}

} /* namespace rr */

