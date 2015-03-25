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

GillespieIntegrator::GillespieIntegrator(ExecutableModel* m,
        const SimulateOptions* o) :
        model(m),
        timeScale(1.0),
        stoichScale(1.0),
        stoichRows(0),
        stoichCols(0),
        stoichData(0),
        seed(defaultSeed()) // default value for mersene twister
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

    setEngineSeed(seed);
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
            stoichScale = options.getItem("stoichScale").convert<double>();
            Log(Logger::LOG_NOTICE) << "new stoichScale: " << stoichScale;
        }

        if(options.hasKey("timeScale"))
        {
            Log(Logger::LOG_NOTICE) << "getting timeScale";
            timeScale = options.getItem("timeScale").convert<double>();
            Log(Logger::LOG_NOTICE) << "new timeScale: " << timeScale;
        }

        if(options.hasKey("seed") && !options.getItem("seed").isEmpty())
        {
            setSeed(options.getItem("seed"));
        }
    }
}



double GillespieIntegrator::integrate(double t, double hstep)
{
    double tf = 0;
    bool singleStep;

    assert(hstep > 0 && "hstep must be > 0");

    if (options.integratorFlags & VARIABLE_STEP)
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

            if(stateVector[i] < 0.0) {
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
    return (double) engine() / (double) engine.max();
}


void GillespieIntegrator::setSeed(const rr::Variant& value)
{
    try
    {
        unsigned long seed = value.convert<unsigned long>();
        this->seed = seed;
        setEngineSeed(seed);
    }
    catch(std::exception& e)
    {
        std::stringstream ss;
        ss << "Could not convert the value \"" << value.toString();
        ss << "\" to an unsigned long integer. " << endl;
        ss << "The seed must be a number between 0 and ";
        ss << std::numeric_limits<unsigned long>::max();
        ss << "; error message: " << e.what() << ".";
        throw std::invalid_argument(ss.str());
    }
}

void GillespieIntegrator::setItem(const std::string& key,
        const rr::Variant& value)
{
    if (key == "seed")
    {
        setSeed(value);
    }
    else if (key == "rand")
    {
        std::invalid_argument("'rand' is a read only value");
    }
    else
    {
        std::string err = "invalid key: \"";
        err += key;
        err += "\"";
        throw std::invalid_argument(err);
    }
}

Variant GillespieIntegrator::getItem(const std::string& key) const
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

    std::string err = "invalid key: \"";
    err += key;
    err += "\"";
    throw std::invalid_argument(err);
}

bool GillespieIntegrator::hasKey(const std::string& key) const
{
    return key == "seed" || key == "rand";
}

int GillespieIntegrator::deleteItem(const std::string& key)
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
    return seed;
}

void GillespieIntegrator::setEngineSeed(unsigned long seed)
{
    Log(Logger::LOG_INFORMATION) << "Using user specified seed value: " << seed;

    // MSVC needs an explicit cast, fail to compile otherwise.
    engine.seed((unsigned long)seed);
}

std::string GillespieIntegrator::toString() const
{
    std::stringstream ss;
    ss << "< roadrunner.GillespieIntegrator() " << endl << "{ "
            << endl << "'this' : " << (void*)this << ", " << std::endl;

    std::vector<std::string> keys = getKeys();

    for(std::vector<std::string>::iterator i = keys.begin(); i != keys.end(); ++i)
    {
        ss << "'" << *i << "' : ";
        ss << getItem(*i).toString();

        if (i + 1 < keys.end()) {
            ss << ", " << std::endl;
        }
    }

    ss << endl << "}>";

    return ss.str();
}

std::string GillespieIntegrator::toRepr() const
{
    std::stringstream ss;
    ss << "< roadrunner.GillespieIntegrator() { 'this' : "
            << (void*)this << " }>";
    return ss.str();
}


std::string GillespieIntegrator::getName() const
{
    return "gillespie";
}


const Dictionary* GillespieIntegrator::getIntegratorOptions()
{
    // static instance
    static SimulateOptions opt;

    // defaults could have changed, so re-load them.
    opt = SimulateOptions();

    opt.setItem("integrator", "gillespie");
    opt.setItem("seed", Config::getValue(Config::RANDOM_SEED).convert<int>());
    opt.setItem("seed.description", "random number seed value used for random number generator");
    opt.setItem("seed.hint", "random number seed");

    return &opt;
}

} /* namespace rr */


