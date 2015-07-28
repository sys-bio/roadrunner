/*
 * GillespieIntegrator.cpp
 *
 *  Created on: Apr 23, 2014
 *      Author: andy
 */

#include "GillespieIntegrator.h"
#include "rrUtils.h"
#include "rrLogger.h"
#include "rrConfig.h"

#include <cstring>
#include <assert.h>
#include <iostream>
#include <exception>
#include <ctime>
#include <limits>
#include <sstream>

using namespace std;

// min and max macros on windows interfer with max method of engine.
#undef max
#undef min

namespace rr
{
	/* Store the name and description as global variables to ensure consistency between
	* between IntegratorRegistrar and internal integrator methods (ie. getIntegratorName()). */
	const char* gGillespieName = "gillespie";
	const char* gGillespieDesc = "Gillespie description.";
	const char* gGillespieHint = "Gillespie hint.";

	// Creates a new Gillespie integrator object
	static Integrator* makeGillespieIntegrator(ExecutableModel *m)
	{
		return new GillespieIntegrator(m);
	}

	// Register with factory
	static int gillespie_result = IntegratorFactory::getInstance().registerIntegrator(IntegratorRegistrar(gGillespieName, gGillespieDesc, gGillespieHint, makeGillespieIntegrator));

	static unsigned long defaultSeed()
	{
		int64_t seed = Config::getValue(Config::RANDOM_SEED).convert<int>();
		if (seed < 0)
		{
			// system time in mirsoseconds since 1970
			seed = getMicroSeconds();
		}

		unsigned long maxl = std::numeric_limits<unsigned long>::max() - 2;

		seed = seed % maxl;

		return (unsigned long)seed;
	}

	GillespieIntegrator::GillespieIntegrator(ExecutableModel* m) :
			model(m),
			timeScale(1.0),
			stoichScale(1.0),
			stoichRows(0),
			stoichCols(0),
			stoichData(0)
	{
		// Set default integrator settings.
		addSetting("seed", defaultSeed(), "stiff hint.", "stiff description.");
		addSetting("variable_step_size", false, "Perform a variable time step simulation.", "Enabling this setting will allow the integrator to adapt the size of each time step. This will result in a non-uniform time column.");
		addSetting("initial_time_step", 0.0, "initial time step hint.", "initial time step description.");
		addSetting("minimum_time_step", 0.0, "minimum time step hint.", "minimum time step description.");
		addSetting("maximum_time_step", 0.0, "maximum time step hint.", "maximum time step description.");


		nReactions = model->getNumReactions();
		reactionRates = new double[nReactions];
		reactionRatesBuffer = new double[nReactions];
		stateVectorSize = model->getStateVector(0);
		stateVector = new double[stateVectorSize];
		stateVectorRate = new double[stateVectorSize];

		floatingSpeciesStart = stateVectorSize - model->getNumIndFloatingSpecies();

		assert(floatingSpeciesStart >= 0);

		// get rows and columns
		model->getStoichiometryMatrix(&stoichRows, &stoichCols, 0);
		stoichData = new double[stoichRows * stoichCols];

		// fill stoichData
		model->getStoichiometryMatrix(&stoichRows, &stoichCols, &stoichData);

		setEngineSeed(getValue("seed").convert<unsigned long>());
	}

	GillespieIntegrator::~GillespieIntegrator()
	{
		delete[] reactionRates;
		delete[] reactionRatesBuffer;
		delete[] stateVector;
		delete[] stateVectorRate;
		delete[] stoichData;
	}

	std::string GillespieIntegrator::getIntegratorName() const
	{
		return gGillespieName;
	}

	std::string GillespieIntegrator::getIntegratorDescription() const
	{
		return gGillespieDesc;
	}

	std::string GillespieIntegrator::getIntegratorHint() const
	{
		return gGillespieHint;
	}

	Integrator::IntegrationMethod GillespieIntegrator::getIntegrationMethod() const
	{
		return Integrator::IntegrationMethod::Deterministic;
	}

	void GillespieIntegrator::setValue(string key, const Variant& val)
	{
		Integrator::setValue(key, val);

		/*	In addition to typically value-setting behavior, some settings require further changes
		within CVODE. */
		if (key == "seed")
		{
			try
			{
				unsigned long seed = val.convert<unsigned long>();
				setEngineSeed(seed);
			}
			catch (std::exception& e)
			{
				std::stringstream ss;
				ss << "Could not convert the value \"" << val.toString();
				ss << "\" to an unsigned long integer. " << endl;
				ss << "The seed must be a number between 0 and ";
				ss << std::numeric_limits<unsigned long>::max();
				ss << "; error message: " << e.what() << ".";
				throw std::invalid_argument(ss.str());
			}
		}
	}

	double GillespieIntegrator::integrate(double t, double hstep)
	{
		double tf = 0;
		bool singleStep;

		assert(hstep > 0 && "hstep must be > 0");

		if (getValue("variable_step_size").convert<bool>())
		{
			if (getValue("minimum_time_step").convert<double>() > 0.0)
			{
				tf = t + getValue("minimum_time_step").convert<double>();
				singleStep = false;
			}
			else
			{
				tf = t + hstep;
				singleStep = true;
			}
		}
		else
		{
			tf = t + hstep;
			singleStep = false;
		}

		Log(Logger::LOG_DEBUG) << "ssa(" << t << ", " << tf << ")";

		// get the initial state vector
		model->setTime(t);
		model->getStateVector(stateVector);

		while (t < tf)
		{
			// random uniform numbers
			double r1 = urand();
			double r2 = urand();

			assert(r1 > 0 && r1 <= 1 && r2 >= 0 && r2 <= 1);

			// sum of propensities
			double s = 0;

			// next time
			double tau = 0;

			// get the 'propensity' -- reaction rates
			model->getReactionRates(nReactions, 0, reactionRates);

			// sum the propensity
			for (int k = 0; k < nReactions; k++)
			{
				Log(Logger::LOG_DEBUG) << "reac rate: " << k << ": "
					<< reactionRates[k];

				// if reaction rate is negative, that means reaction goes in reverse,
				// this is fine, we just have to reverse the stoichiometry,
				// but still need to sum the absolute value of the propensities
				// to get tau.
				s += std::abs(reactionRates[k]);
			}

			// sample tau
			if (s > 0)
			{
				tau = -log(r1) / s;
			}
			else
			{
				// no reaction occurs
				return std::numeric_limits<double>::infinity();
			}

			t = t + tau;

			// select reaction
			int reaction = -1;
			double sp = 0.0;

			r2 = r2 * s;
			for (int i = 0; i < nReactions; ++i)
			{
				sp += std::abs(reactionRates[i]);
				if (r2 < sp)
				{
					reaction = i;
					break;
				}
			}

			assert(reaction >= 0 && reaction < nReactions);

			// update chemical species
			// if rate is negative, means reaction goes in reverse, so
			// multiply by sign
			double sign = (reactionRates[reaction] > 0)
				- (reactionRates[reaction] < 0);


			for (int i = floatingSpeciesStart; i < stateVectorSize; ++i)
			{
				stateVector[i] = stateVector[i]
					+ getStoich(i - floatingSpeciesStart, reaction)
					* stoichScale * sign;

				if (stateVector[i] < 0.0) {
					Log(Logger::LOG_WARNING) << "Error, negative value of "
						<< stateVector[i]
						<< " encountred for floating species "
						<< model->getFloatingSpeciesId(i - floatingSpeciesStart);
					t = std::numeric_limits<double>::infinity();
				}
			}

			// rates could be time dependent
			model->setTime(t);
			model->setStateVector(stateVector);

			if (singleStep)
			{
				return t;
			}
		}

		return t;
	}

	void GillespieIntegrator::restart(double t0)
	{
	}

	void GillespieIntegrator::setListener(IntegratorListenerPtr)
	{
	}

	IntegratorListenerPtr GillespieIntegrator::getListener()
	{
		return IntegratorListenerPtr();
	}

	double GillespieIntegrator::urand()
	{
		return (double)engine() / (double)engine.max();
	}

	void GillespieIntegrator::setEngineSeed(unsigned long seed)
	{
		Log(Logger::LOG_INFORMATION) << "Using user specified seed value: " << seed;

		// MSVC needs an explicit cast, fail to compile otherwise.
		engine.seed((unsigned long)seed);
	}

	} /* namespace rr */


