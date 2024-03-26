/*
 * GillespieIntegrator.cpp
 *
 *  Created on: Apr 23, 2014
 *      Author: andy
 */

#include "GillespieIntegrator.h"
#include "rrUtils.h"
#include "rrConfig.h"

#include <cstring>
#include <cassert>
#include <climits>
#include <iostream>
#include <exception>
#include <limits>
#include <chrono>



// min and max macros on windows interfer with max method of engine.
#undef max
#undef min

namespace rr
{

    static std::uint64_t seedValue(Setting seedSetting) {
        std::uint64_t seed;
        if (auto int32Val = seedSetting.get_if<std::int32_t>()) {
            seed = (std::uint64_t)*int32Val;
        }
        else if (auto uInt32Val = seedSetting.get_if<std::uint32_t>()) {
            seed = (std::uint64_t)*uInt32Val;
        }
        else if (auto int64Val = seedSetting.get_if<std::int64_t>()) {
            seed = (std::uint64_t)*int64Val;
        }
        else if (auto uInt64Val = seedSetting.get_if<std::uint64_t>()) {
            seed = *uInt64Val;
        }
        else {
            throw std::invalid_argument("seedSetting is of incorrect type.");
        }

        return seed;
    }

    static std::uint64_t defaultSeed() {
        return seedValue(Config::getValue(Config::RANDOM_SEED));
    }

    void GillespieIntegrator::initializeFromModel() {
        nReactions = mModel->getNumReactions();
        reactionRates = new double[nReactions];
        reactionRatesBuffer = new double[nReactions];
        stateVectorSize = mModel->getStateVector(nullptr);
        stateVector = new double[stateVectorSize];
        stateVectorRate = new double[stateVectorSize];

        eventStatus = std::vector<unsigned char>(mModel->getEventTriggers(0, nullptr, nullptr), false);
        previousEventStatus = std::vector<unsigned char>(mModel->getEventTriggers(0, nullptr, nullptr), false);

        floatingSpeciesStart = stateVectorSize - mModel->getNumIndFloatingSpecies();

        assert(floatingSpeciesStart >= 0);

        setEngineSeed(getValue("seed"));
    }

    GillespieIntegrator::GillespieIntegrator(ExecutableModel* m)
        : Integrator(m),
        timeScale(1.0),
        stoichScale(1.0),
        reactionRates(nullptr),
        reactionRatesBuffer(nullptr),
        stateVector(nullptr),
        stateVectorRate(nullptr)
    {
        GillespieIntegrator::resetSettings();

        if (mModel)
            initializeFromModel();
    }

    GillespieIntegrator::~GillespieIntegrator()
    {
        if (mModel) {
            // like in the constructor, we only delete these components
            // if the model was present on construction.
            delete[] reactionRates;
            delete[] reactionRatesBuffer;
            delete[] stateVector;
            delete[] stateVectorRate;
            reactionRates = nullptr;
            reactionRatesBuffer = nullptr;
            stateVector = nullptr;
            stateVectorRate = nullptr;
        }
    }

    void GillespieIntegrator::syncWithModel(ExecutableModel* m)
    {
        resetSettings();
        delete[] reactionRates;
        delete[] reactionRatesBuffer;
        delete[] stateVector;
        delete[] stateVectorRate;
        reactionRates = nullptr;
        reactionRatesBuffer = nullptr;
        stateVector = nullptr;
        stateVectorRate = nullptr;

        mModel = m;
        mModel->reset();

        nReactions = 0;
        stateVectorSize = 0;

        timeScale = 1.;
        stoichScale = 1.;

        initializeFromModel();
    }

    std::string GillespieIntegrator::getName() const {
        return "gillespie";
    }

    std::string GillespieIntegrator::getDescription() const {
        return "RoadRunner's implementation of the standard Gillespie Direct "
            "Method SSA. The granularity of this simulator is individual "
            "molecules and kinetic processes are stochastic. "
            "Results will, in general, be different in each run, but "
            "a sufficiently large ensemble of runs should be statistically correct.";
    }

    std::string GillespieIntegrator::getHint() const {
        return "Gillespie Direct Method SSA";
    }

    Integrator::IntegrationMethod GillespieIntegrator::getIntegrationMethod() const
    {
        return Integrator::Deterministic;
    }

    void GillespieIntegrator::setValue(const std::string& key, Setting val)
    {
        Integrator::setValue(key, val);

        /*	In addition to typically value-setting behavior, some settings require further changes
        within CVODE. */
        if (key == "seed")
        {
            try
            {
                setEngineSeed(val);
            }
            catch (std::exception& e)
            {
                std::stringstream ss;
                ss << "Could not convert the value \"" << val.get<std::string>();
                ss << "\" to an unsigned long integer. " << std::endl;
                ss << "The seed must be a number between 0 and ";
                ss << std::numeric_limits<unsigned long>::max();
                ss << "; " << std::endl << "error message: " << e.what() << ".";
                throw std::invalid_argument(ss.str());
            }
        }
    }

    void GillespieIntegrator::resetSettings()
    {
        Solver::resetSettings();

        // Set default integrator settings.
        addSetting("seed",                  std::uint64_t(defaultSeed()), "Seed", "Set the seed into the random engine. (ulong)", "(ulong) Set the seed into the random engine.");
        addSetting("variable_step_size",    Setting(true), "Variable Step Size", "Perform a variable time step simulation. (bool)", "(bool) Enabling this setting will allow the integrator to adapt the size of each time step. This will result in a non-uniform time column.  The number of steps or points will be ignored, and the max number of output rows will be used instead.");
        //addSetting("initial_time_step",     Setting(0.0),   "Initial Time Step", "Specifies the initial time step size. (double)", "(double) Specifies the initial time step size.");
        addSetting("minimum_time_step",     Setting(0.0),   "Minimum Time Step", "Specifies the minimum absolute value of step size allowed. (double)", "(double) The minimum absolute value of step size allowed.");
        addSetting("maximum_time_step",     Setting(0.0),   "Maximum Time Step", "Specifies the maximum absolute value of step size allowed. (double)", "(double) The maximum absolute value of step size allowed.");
        addSetting("nonnegative",           Setting(false), "Non-negative species only", "Prevents species amounts from going negative during a simulation. (bool)", "(bool) Enforce non-negative species constraint.");
        addSetting("max_output_rows",       Setting(Config::getInt(Config::MAX_OUTPUT_ROWS)), "Maximum Output Rows", "For variable step size simulations, the maximum number of output rows produced (int).", "(int) This will set the maximum number of output rows for variable step size integration.  This may truncate some simulations that may not reach the desired end time, but prevents massive output for simulations where the variable step size ends up decreasing too much.  This setting is ignored when the variable_step_size is false, and is also ignored when the output is being written directly to a file.");
        addSetting("maximum_num_steps", Setting(0), "Maximum Number of Steps",
            "Specifies the maximum number of steps to be taken by the Gillespie solver before reaching the next reporting time. (int)",
            "(int) Maximum number of steps to be taken by the Gillespie solver before reaching the next reporting time.");
    }

    double GillespieIntegrator::integrate(double t, double hstep)
    {
        double tf;
        bool singleStep;
        bool varStep = getValue("variable_step_size").get<bool>();
        auto minTimeStep = getValue("minimum_time_step").get<double>();
        auto maxTimeStep = getValue("maximum_time_step").get<double>();
        int maxNumSteps = getValue("maximum_num_steps").get<int>();
        if (maxTimeStep > minTimeStep)
        {
            hstep = std::min(hstep, maxTimeStep);
        }

        if (varStep)
        {
            if (minTimeStep > 0.0)
            {
                if (minTimeStep > hstep) // no step bigger than hstep
                {
                    minTimeStep = hstep;
                }
                tf = t + minTimeStep;
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

        assert(hstep > 0 || singleStep && "hstep must be > 0 or we must be taking a single step.");

        rrLog(Logger::LOG_DEBUG) << "ssa(" << t << ", " << tf << ")";

        // get the initial state std::vector
        mModel->setTime(t);
        mModel->getStateVector(stateVector);
        int step = 0;

        while (singleStep || (t < tf && (maxNumSteps == 0 || step < maxNumSteps)))
        {
            step++;
            // random uniform numbers
            double r1 = urand();
            double r2 = urand();

            assert(r1 > 0 && r1 <= 1 && r2 >= 0 && r2 <= 1);

            // sum of propensities
            double s = 0;

            // next time
            double tau;

            // get the 'propensity' -- reaction rates
            mModel->getReactionRates(nReactions, nullptr, reactionRates);

            // sum the propensity
            for (int k = 0; k < nReactions; k++)
            {
                rrLog(Logger::LOG_DEBUG) << "reac rate: " << k << ": "
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
                return tf;
            }
            if (!singleStep && t + tau > tf)        // if time exhausted, don't allow reaction to proceed
            {
                return tf;
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

            bool skip = false;

            if ((bool)getValue("nonnegative")) {
                // skip reactions which cause species amts to become negative
                for (int i = floatingSpeciesStart; i < stateVectorSize; ++i) {
                    if (stateVector[i]
                        + getStoich(i - floatingSpeciesStart, reaction)
                        * stoichScale * sign < 0.0) {
                        skip = true;
                        break;
                    }
                }
            }

            if (!skip) {
                for (int i = floatingSpeciesStart; i < stateVectorSize; ++i)
                {
                    stateVector[i] = stateVector[i]
                        + getStoich(i - floatingSpeciesStart, reaction)
                        * stoichScale * sign;

                    if (stateVector[i] < 0.0) {
                        rrLog(Logger::LOG_WARNING) << "Error, negative value of "
                            << stateVector[i]
                            << " encountred for floating species "
                            << mModel->getFloatingSpeciesId(i - floatingSpeciesStart);
                        t = std::numeric_limits<double>::infinity();
                    }
                }
            }

            // rates could be time dependent
            mModel->setTime(t);
            mModel->setStateVector(stateVector);

            // events
            bool triggered = false;

            mModel->getEventTriggers(eventStatus.size(), nullptr, !eventStatus.empty() ? &eventStatus[0] : nullptr);
            for (unsigned char eventStatu : eventStatus) {
                if (eventStatu)
                    triggered = true;
            }

            if (triggered) {
                applyEvents(t, previousEventStatus);
            }

            if (!eventStatus.empty())
                memcpy(&previousEventStatus[0], &eventStatus[0], eventStatus.size() * sizeof(unsigned char));


            if (singleStep)
            {
                return t;
            }
        }
        if (t < tf && maxNumSteps > 0 && step >= maxNumSteps)
        {
            std::stringstream msg;
            msg << "GillespieIntegrator::integrate failed:  max number of steps ("
                << maxNumSteps << ") reached before desired output at time " << tf << ".";
            throw std::runtime_error(msg.str());
        }
        return t;
    }

    void GillespieIntegrator::testRootsAtInitialTime()
    {
        std::vector<unsigned char> initialEventStatus(mModel->getEventTriggers(0, nullptr, nullptr), false);
        mModel->getEventTriggers(initialEventStatus.size(), nullptr, initialEventStatus.empty() ? nullptr : &initialEventStatus[0]);
        applyEvents(mIntegrationStartTime, initialEventStatus);
    }

    void GillespieIntegrator::applyEvents(double timeEnd, std::vector<unsigned char>& prevEventStatus)
    {
        mModel->applyEvents(timeEnd, prevEventStatus.empty() ? nullptr : &prevEventStatus[0], stateVector, stateVector);
    }

    void GillespieIntegrator::restart(double t0)
    {
        if (!mModel) {
            return;
        }

        if (stateVector)
        {
            mModel->getStateVector(stateVector);
        }

        testRootsAtInitialTime();

        mModel->setTime(t0);

        // copy state std::vector into memory
        if (stateVector)
        {
            mModel->getStateVector(stateVector);
        }
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
        return (double)engine() / (double)std::mt19937::max();
    }

    void GillespieIntegrator::setEngineSeed(Setting seedSetting)
    {
        std::uint64_t seed = seedValue(seedSetting);
        rrLog(Logger::LOG_INFORMATION) << "Using user specified seed value: " << seed;

        // Checks if seed is not equal to -1 (the value which is considered as the random seed value)
        if (seed != -1) {
            engine.seed(seed);
            // MSVC needs an explicit cast, fail to compile otherwise.
        }
        else {
            seed = (std::uint64_t)getMicroSeconds();
            engine.seed(seed);
            Integrator::setValue("seed", seed);
            rrLog(Logger::LOG_INFORMATION) << "Using seed value from the clock: " << seed;
        }
    }

    Solver* GillespieIntegrator::construct(ExecutableModel* executableModel) const {
        return new GillespieIntegrator(executableModel);
    }

} /* namespace rr */
