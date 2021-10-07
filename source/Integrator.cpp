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
#include "CVODEIntegrator.h"
#include "GillespieIntegrator.h"
#include "RK4Integrator.h"
#include "EulerIntegrator.h"
#include "RK45Integrator.h"

#include "rrExecutableModel.h"
#include "rrConfig.h"

// == CODE ====================================================


namespace rr
{
	/*------------------------------------------------------------------------------------------
		INTEGRATOR
	  ------------------------------------------------------------------------------------------*/
	Integrator::Integrator(ExecutableModel *model)
        : Solver(model)
        , mIntegrationStartTime(0.0)
    {
    }

    Integrator::Integrator()
        : Solver()
        , mIntegrationStartTime(0.0)
    {
    }

    void Integrator::syncWithModel(ExecutableModel* m) {}

    void Integrator::loadConfigSettings() {}

    void Integrator::loadSBMLSettings(const std::string& filename)
    {
        // Stub for loading SBML settings (can override in derived classes).
    }

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
	
	// since std::vector tolerance is only used in CVODEIntegrator
	// the functions below will not be implemented here
	void Integrator::setIndividualTolerance(std::string sid, double value) {}

	void Integrator::setConcentrationTolerance(Setting value) {}

	std::vector<double> Integrator::getConcentrationTolerance() {
		std::vector<double> v;
		return v;
	}

    void Integrator::setIntegrationStartTime(double time)
    {
        mIntegrationStartTime = time;
    }


}
