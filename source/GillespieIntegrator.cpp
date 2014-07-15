/*
 * GillespieIntegrator.cpp
 *
 *  Created on: Apr 23, 2014
 *      Author: andy
 */

#include "GillespieIntegrator.h"
#include "rrUtils.h"
#include "rrLogger.h"

#include <cstring>
#include <assert.h>
#include <iostream>
#include <exception>
#include <ctime>
#include <limits>

using namespace std;

// min and max macros on windows interfer with max method of engine.
#undef max
#undef min

namespace rr
{

GillespieIntegrator::GillespieIntegrator(ExecutableModel* m,
        const SimulateOptions* o) :
        model(m),
        timeScale(1.0),
        stoichScale(1.0),
        stoichRows(0),
        stoichCols(0),
        stoichData(0)
{
    if (o)
    {
        this->options = *o;
    }

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


}

GillespieIntegrator::~GillespieIntegrator()
{
    delete[] reactionRates;
    delete[] reactionRatesBuffer;
    delete[] stateVector;
    delete[] stateVectorRate;
    delete[] stoichData;
}

void GillespieIntegrator::setSimulateOptions(const SimulateOptions* o)
{
    if (o)
    {
        options = *o;

        if(options.hasKey("stoichScale"))
        {
            Log(Logger::LOG_NOTICE) << "getting stoichScale";
            stoichScale = options.getValue("stoichScale").convert<double>();
            Log(Logger::LOG_NOTICE) << "new stoichScale: " << stoichScale;
        }

        if(options.hasKey("timeScale"))
        {
            Log(Logger::LOG_NOTICE) << "getting timeScale";
            timeScale = options.getValue("timeScale").convert<double>();
            Log(Logger::LOG_NOTICE) << "new timeScale: " << timeScale;
        }

        if(options.hasKey("seed") && !options.getValue("seed").isEmpty())
        {
            // variant at the moment can only do signed numbers.
            unsigned long seed = options.getValue("seed").convert<unsigned long>();
            setSeed(seed);
        }
    }
}



double GillespieIntegrator::integrate(double t, double hstep)
{
    double tf = 0;
    bool singleStep;

    assert(hstep > 0 && "hstep must be > 0");

    if (options.integratorFlags & SimulateOptions::VARIABLE_STEP)
    {
        if (options.minimumTimeStep > 0.0)
        {
            tf = t + options.minimumTimeStep;
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
#ifdef RR_CXX_RANDOM
    return (double) engine() / (double) engine.max();
#else
    return drand48();
#endif
}


void GillespieIntegrator::setValue(const std::string& key,
        const rr::Variant& value)
{
    if (key == "seed")
    {
        if (value.isNumeric())
        {
            unsigned long seed = value.convert<unsigned long>();
            setSeed(seed);
        }
        else
        {
            Log(Logger::LOG_WARNING) << "could not set randon number seed with value: "
                    << value.toString();
        }

        options.setValue("seed", value);
    }
}

Variant GillespieIntegrator::getValue(const std::string& key) const
{
    if (key == "seed")
    {
        return Variant(getSeed());
    }

    else if(key == "rand")
    {
        // cheating
        GillespieIntegrator *pthis = const_cast<GillespieIntegrator*>(this);
        return Variant(pthis->urand());
    }

    return Variant();
}

bool GillespieIntegrator::hasKey(const std::string& key) const
{
    return key == "seed" || key == "rand";
}

int GillespieIntegrator::deleteValue(const std::string& key)
{
    return -1;
}

std::vector<std::string> GillespieIntegrator::getKeys() const
{
    std::vector<std::string> result;
    result.push_back("seed");
    result.push_back("rand");
    return result;
}

unsigned long GillespieIntegrator::getSeed() const
{
    try
    {
        if (options.hasKey("seed"))
        {
            return options.getValue("seed").convert<unsigned long>();
        }
    }
    catch(std::exception& e)
    {
        Log(Logger::LOG_INFORMATION) << "failed to get seed from dict: "
                << e.what() << ", returning default value";
    }
    // default value for mersene twister
    return 5489UL;
}

void GillespieIntegrator::setSeed(unsigned long seed)
{
    Log(Logger::LOG_INFORMATION) << "Using user specified seed value: " << seed;

#ifdef RR_CXX_RANDOM
    // MSVC needs an explicit cast, fail to compile otherwise.
    engine.seed((unsigned long)seed);
#else
    srand48(seed);
#endif
}

} /* namespace rr */


