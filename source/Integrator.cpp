// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

// == FILEDOC =================================================

/** @file Integrator.cpp
* @author ETS, WBC, JKM
* @date Apr 25, 2014
* @copyright Apache License, Version 2.0
* @brief Contains the base class for RoadRunner integrators
**/

// == INCLUDES ================================================

#include "Integrator.h"
// #include "CVODEIntegrator.h"
// #include "GillespieIntegrator.h"
// #include "RK4Integrator.h"
// #include "EulerIntegrator.h"
#include "rrExecutableModel.h"
#include "rrStringUtils.h"
#include "rrConfig.h"
#include "rrUtils.h"
#include <typeinfo>

// == CODE ====================================================

using namespace std;
namespace rr
{
	/*------------------------------------------------------------------------------------------
		INTEGRATOR
	  ------------------------------------------------------------------------------------------*/

    void Integrator::syncWithModel(ExecutableModel* m) {}

    void Integrator::loadConfigSettings() {}

    void Integrator::loadSBMLSettings(const std::string& filename)
    {
        // Stub for loading SBML settings (can override in derived classes).
    }

    /* TODO: Create getType() method. */

    std::string Integrator::toString() const
    {
        std::stringstream ss;
        ss << "< roadrunner.Integrator() >\n";
        ss << "  name: " << getName() << "\n";
        ss << "  settings:\n";
        ss << getSettingsRepr();
        return ss.str();

    }

    std::string Integrator::toRepr() const
    {
        std::stringstream ss;
        ss << "< roadrunner.Integrator() \"" << getName() << "\" " << settingsPyDictRepr() << " >\n";
        return ss.str();
    }

	
    void Integrator::tweakTolerances() {}
	
	// since vector tolerance is only used in CVODEIntegrator
	// the functions below will not be implemented here
	void Integrator::setIndividualTolerance(string sid, double value) {}

	void Integrator::setConcentrationTolerance(const Variant& value) {}

	std::vector<double> Integrator::getConcentrationTolerance() {
		vector<double> v;
		return v;
	}

    IntegratorRegistrar::~IntegratorRegistrar() {}

    /********************************************************************************************
    * INTEGRATOR FACTORY
    ********************************************************************************************/

    IntegratorFactory::~IntegratorFactory() {
        for (IntegratorRegistrars::const_iterator it(mRegisteredIntegrators.begin()); it != mRegisteredIntegrators.end(); ++it) {
            delete *it;
        }
    }

    Integrator* IntegratorFactory::New(std::string name, ExecutableModel* m) const {
        for (IntegratorRegistrars::const_iterator it(mRegisteredIntegrators.begin()); it != mRegisteredIntegrators.end(); ++it) {
            if ((*it)->getName() == name) {
                return (*it)->construct(m);
            }
        }
        throw InvalidKeyException("No such integrator: " + name);
    }

    void IntegratorFactory::registerIntegrator(IntegratorRegistrar* i) {
        if (!i)
            throw CoreException("Registrar is null");
        mRegisteredIntegrators.push_back(i);
    }

    IntegratorFactory& IntegratorFactory::getInstance() {
        // FIXME: not thread safe -- JKM, July 24, 2015.
        static IntegratorFactory factory;
        return factory;
    }

    std::size_t IntegratorFactory::getNumIntegrators() const {
        return mRegisteredIntegrators.size();
    }

    std::string IntegratorFactory::getIntegratorName(std::size_t n) const {
        return mRegisteredIntegrators.at(n)->getName();
    }

    std::string IntegratorFactory::getIntegratorHint(std::size_t n) const {
        return mRegisteredIntegrators.at(n)->getHint();
    }

    std::string IntegratorFactory::getIntegratorDescription(std::size_t n) const {
        return mRegisteredIntegrators.at(n)->getDescription();
    }

}
