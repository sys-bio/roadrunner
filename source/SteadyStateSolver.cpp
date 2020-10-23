// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

// == FILEDOC =================================================

/** @file SteadyStateSolver.cpp
* @author ETS, WBC, JKM
* @date Apr 25, 2014
* @copyright Apache License, Version 2.0
* @brief Contains the base class for RoadRunner SteadyStateSolvers
**/

// == INCLUDES ================================================

#include "SteadyStateSolver.h"
#include "rrStringUtils.h"
#include "rrConfig.h"
#include "rrUtils.h"
#include <typeinfo>
#include <iomanip>

// == CODE ====================================================

using namespace std;
namespace rr
{
	/*------------------------------------------------------------------------------------------
		SteadyStateSolver
	  ------------------------------------------------------------------------------------------*/

	void SteadyStateSolver::loadConfigSettings()
	{
		// empty
	}

	SteadyStateSolverRegistrar::~SteadyStateSolverRegistrar() {}

    /********************************************************************************************
    * SteadyStateSolver FACTORY
    ********************************************************************************************/

    SteadyStateSolverFactory::~SteadyStateSolverFactory() {
        for (SteadyStateSolverRegistrars::const_iterator it(mRegisteredSteadyStateSolvers.begin()); it != mRegisteredSteadyStateSolvers.end(); ++it) {
            delete *it;
        }
    }

    SteadyStateSolver* SteadyStateSolverFactory::New(std::string name, ExecutableModel* m) const {
        for (SteadyStateSolverRegistrars::const_iterator it(mRegisteredSteadyStateSolvers.begin()); it != mRegisteredSteadyStateSolvers.end(); ++it) {
            if ((*it)->getName() == name) {
                return (*it)->construct(m);
            }
        }
        Log(Logger::LOG_ERROR) << "No such SteadyStateSolver '" << name << "'";
        throw InvalidKeyException("No such SteadyStateSolver: " + name);
    }

    void SteadyStateSolverFactory::registerSteadyStateSolver(SteadyStateSolverRegistrar* i) {
        if (!i)
            throw CoreException("Registrar is null");
        mRegisteredSteadyStateSolvers.push_back(i);
    }

    SteadyStateSolverFactory& SteadyStateSolverFactory::getInstance() {
        // FIXME: not thread safe -- JKM, July 24, 2015.
        static SteadyStateSolverFactory factory;
        return factory;
    }

    std::size_t SteadyStateSolverFactory::getNumSteadyStateSolvers() const {
        return mRegisteredSteadyStateSolvers.size();
    }

	std::vector<std::string> SteadyStateSolverFactory::getListSteadyStateSolverNames()
	{
		std::vector<std::string> intgNames;
		int numIntgs = static_cast<int>(SteadyStateSolverFactory::getInstance().mRegisteredSteadyStateSolvers.size());
		for (int i = 0; i < numIntgs; ++i)
		{
			std::size_t n = static_cast<std::size_t>(i);
			intgNames.push_back(SteadyStateSolverFactory::getInstance().getSteadyStateSolverName(n));
		}
		return intgNames;
	}

    std::string SteadyStateSolverFactory::getSteadyStateSolverName(std::size_t n) const {
        return mRegisteredSteadyStateSolvers.at(n)->getName();
    }

    std::string SteadyStateSolverFactory::getSteadyStateSolverHint(std::size_t n) const {
        return mRegisteredSteadyStateSolvers.at(n)->getHint();
    }

    std::string SteadyStateSolverFactory::getSteadyStateSolverDescription(std::size_t n) const {
        return mRegisteredSteadyStateSolvers.at(n)->getDescription();
    }

	std::string SteadyStateSolver::getSettingsRepr() const
	{
		std::stringstream ss;
		for (size_t n = 0; n<getNumParams(); ++n)
			ss << "    " << std::setw(20) << getParamName(n) << ": " << getValue(getParamName(n)).toString() << "\n";
		return ss.str();
	}

	std::string SteadyStateSolver::settingsPyDictRepr() const
	{
		std::stringstream ss;
		for (size_t n = 0; n<getNumParams(); ++n)
			ss << (n ? ", " : "") << "'" << getParamName(n) << "': " << getValue(getParamName(n)).pythonRepr();
		return ss.str();
	}

	std::string SteadyStateSolver::toString() const
	{
		std::stringstream ss;
		ss << "< roadrunner.SteadyStateSolver() >\n";
		ss << "  name: " << getName() << "\n";
		ss << "  settings:\n";
		ss << getSettingsRepr();
		return ss.str();
	}

	std::string SteadyStateSolver::toRepr() const
	{
		std::stringstream ss;
		ss << "< roadrunner.SteadyStateSolver() \"" << getName() << "\" " << settingsPyDictRepr() << " >\n";
		return ss.str();
	}
}