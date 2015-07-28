/*
* Integrator.cpp
*
*  Created on: Apr 25, 2014
*      Author: andy
 *      Author: jkm
*/

#include "Integrator.h"
#include "CVODEIntegrator.h"
#include "GillespieIntegrator.h"
#include "RK4Integrator.h"
#include "gpu/GPUSimIntegratorProxy.h"

#include <cassert>
#include <sstream>
#include "EulerIntegrator.h"
#include "rrStringUtils.h"
#include "rrConfig.h"
#include "rrUtils.h"
#include <typeinfo>

using namespace std;
namespace rr
{
	/*------------------------------------------------------------------------------------------
		INTEGRATOR
	  ------------------------------------------------------------------------------------------*/

	void Integrator::addSetting(string name, Variant val, string hint, string description)
	{
		settings.insert({ name, val });
		hints.insert({ name, hint });
		descriptions.insert({ name, description });
	}

	void Integrator::loadConfigSettings()
	{
		//	VARIABLE STEP SIZE
		/*	Need better error handling.
			(1) What happens if variable_step_size does not exist in settings?
		*/
		bool bVal = false;
		if (getIntegrationMethod() == Integrator::IntegrationMethod::Deterministic)
		{
			bVal = Config::getBool(Config::SIMULATEOPTIONS_DETERMINISTIC_VARIABLE_STEP);
			Integrator::setValue("variable_step_size", bVal);
		}
		else if (getIntegrationMethod() == Integrator::IntegrationMethod::Stochastic)
		{
			bVal = Config::getBool(Config::SIMULATEOPTIONS_STOCHASTIC_VARIABLE_STEP);
			Integrator::setValue("variable_step_size", bVal);
		}

		// STIFFNESS
		bVal = Config::getBool(Config::SIMULATEOPTIONS_STIFF);
		Integrator::setValue("stiff", bVal);

		// MULTIPLE STEPS
		bVal = Config::getBool(Config::SIMULATEOPTIONS_MULTI_STEP);
		Integrator::setValue("multiple_steps", bVal);

		// ABSOLUTE TOLERANCE
		Integrator::setValue("absolute_tolerance", Config::getDouble(Config::SIMULATEOPTIONS_ABSOLUTE));
		Integrator::setValue("relative_tolerance", Config::getDouble(Config::SIMULATEOPTIONS_RELATIVE));
	}

	void Integrator::loadSBMLSettings(const std::string& filename)
	{
		// Stub for loading SBML settings (can override in derived classes).
	}

	std::vector<string> Integrator::getSettings()
	{
		std::vector<string> keys;
		for (auto entry : settings)
		{
			keys.push_back(entry.first);
		}
		return keys;
	}
// -- TimecourseIntegrationParameters --

TimecourseIntegrationParameters::~TimecourseIntegrationParameters() {

}

void TimecourseIntegrationParameters::addTimevalue(double t) {
    t_.push_back(t);
}

void TimecourseIntegrationParameters::setPrecision(Precision p) {
    prec_ = p;
}

TimecourseIntegrationParameters::Precision TimecourseIntegrationParameters::getPrecision() const {
    return prec_;
}

TimecourseIntegrationParameters::size_type TimecourseIntegrationParameters::getTimevalueCount() const {
    return t_.size();
}

double TimecourseIntegrationParameters::getTimevalue(size_type i) const {
    return t_.at(i);
}

double* TimecourseIntegrationParameters::getTimevaluesHeapArrayDbl() const {
    double* result = (double*)malloc(getTimevalueCount()*sizeof(double));
    assert(result && "No memory");
    for (size_type i=0; i<getTimevalueCount(); ++i)
        result[i] = getTimevalue(i);
    return result;
}

float* TimecourseIntegrationParameters::getTimevaluesHeapArrayFlt() const {
    float* result = (float*)malloc(getTimevalueCount()*sizeof(float));
    assert(result && "No memory");
    for (size_type i=0; i<getTimevalueCount(); ++i)
        result[i] = (float)getTimevalue(i);
    return result;
}

// -- TimecourseIntegrationResultsRealVector --

TimecourseIntegrationResultsRealVector::size_type TimecourseIntegrationResultsRealVector::getTimevalueCount() const {
    return ntval_;
}

void TimecourseIntegrationResultsRealVector::setTimevalueCount(size_type n) {
    ntval_ = n;
    rebuild();
}

	Variant Integrator::getValue(std::string key)
	{
		std::unordered_map<string, Variant>::const_iterator option = settings.find(key);
		if (option == settings.end())
		{
			throw std::invalid_argument("invalid key: " + key);
		}
		return option->second;
	}

	int Integrator::getValueAsInt(std::string key)
	{
		return getValue(key).convert<int>();
	}

	unsigned int Integrator::getValueAsUInt(std::string key)
	{
		return getValue(key).convert<unsigned int>();
	}

	long Integrator::getValueAsLong(std::string key)
	{
		return getValue(key).convert<long>();
	}

	unsigned long Integrator::getValueAsULong(std::string key)
	{
		return getValue(key).convert<unsigned long>();
	}

void TimecourseIntegrationResultsRealVector::setVectorLength(size_type n) {
    veclen_ = n;
    rebuild();
}

TimecourseIntegrationResultsRealVector::size_type TimecourseIntegrationResultsRealVector::getVectorLength() const {
    return veclen_;
}

VariableValue TimecourseIntegrationResultsRealVector::getValue(size_type ti, size_type i) const {
    return val_.at(ti).at(i);
}

void TimecourseIntegrationResultsRealVector::setValue(size_type ti, size_type i, double val) {
    if (ti > val_.size() || i > val_.at(ti).size()) {
        assert(getTimevalueCount() ==  val_.size());
        std::stringstream ss;
        ss << "Tried to set value " << ti << "x" << i << " in a container of size " << getTimevalueCount() << "x" << getVectorLength();
        throw std::runtime_error(ss.str());
    }
    val_.at(ti).at(i) = val;
}

void TimecourseIntegrationResultsRealVector::rebuild() {
    val_.resize(ntval_);
    for (ValMatrix::iterator i=val_.begin(); i != val_.end(); ++i)
        i->resize(veclen_);
}

// -- Integrator --

	float Integrator::getValueAsFloat(std::string key)
	{
		return getValue(key).convert<float>();
	}

	double Integrator::getValueAsDouble(std::string key)
	{
		return getValue(key).convert<double>();
	}

	char Integrator::getValueAsChar(std::string key)
	{
		return getValue(key).convert<char>();
	}

	unsigned char Integrator::getValueAsUChar(std::string key)
	{
		return getValue(key).convert<unsigned char>();
	}

	std::string Integrator::getValueAsString(std::string key)
	{
		return getValue(key).convert<std::string>();
	}

	bool Integrator::getValueAsBool(std::string key)
	{
		return getValue(key).convert<bool>();
	}

	void Integrator::setValue(std::string key, const Variant& value)
	{
		settings.insert({ key, value });
	}

	const std::string& Integrator::getHint(std::string key) const
	{
		HintMap::const_iterator option = Integrator::hints.find(key);
		if (option == hints.end())
		{
			throw std::invalid_argument("invalid key: " + key);
		}
		return option->second;
	}

	const std::string& Integrator::getDescription(std::string key) const
	{
		DescriptionMap::const_iterator option = Integrator::descriptions.find(key);
		if (option == descriptions.end())
		{
			throw std::invalid_argument("invalid key: " + key);
		}
		return option->second;
	}

	const Variant::TypeId Integrator::getType(std::string key)
	{
		return getValue(key).type();
	}

	/* TODO: Create getType() method. */

	std::string Integrator::toString() const
	{
		std::stringstream ss;
		ss << "< roadrunner.CVODEIntegrator() >" << endl;
		return ss.str();
	}

	/********************************************************************************************
	*	INTEGRATOR FACTORY
	********************************************************************************************/

	Integrator* IntegratorFactory::New(std::string name, ExecutableModel* m)
	{
		Integrator *result = 0;

		for (std::vector<IntegratorRegistrar>::iterator it(mRegisteredIntegrators.begin()); it != mRegisteredIntegrators.end(); ++it)
		{
			if (it->getName() == name)
			{
				return it->construct(m);
			}
		}

		if (name == "cvode")
		{
			result = new CVODEIntegrator(m);
		}
		else if (name == "gillespie")
		{
			result = new GillespieIntegrator(m);
		}
		else
		{
			throw std::invalid_argument("invalid integrator name was requested: " + name);
		}

		return result;
	}

	int IntegratorFactory::registerIntegrator(const IntegratorRegistrar& i)
	{
		mRegisteredIntegrators.push_back(i);
		return 0;
	}

	std::vector<std::string> IntegratorFactory::getRegisteredIntegratorNames()
	{
		std::vector<std::string> names;
		for (std::vector<IntegratorRegistrar>::iterator it(mRegisteredIntegrators.begin()); it != mRegisteredIntegrators.end(); ++it)
		{
			names.push_back(it->getName());
		}
		return names;
	}

}