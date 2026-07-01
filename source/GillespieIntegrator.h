// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

// == FILEDOC =================================================

/** @file Integrator.h
* @author ETS, WBC, JKM
* @date Apr 23, 2014
* @copyright Apache License, Version 2.0
* @brief RoadRunner's Gillespie SSA integrator
**/

#ifndef GILLESPIEINTEGRATOR_H_
#define GILLESPIEINTEGRATOR_H_

// == INCLUDES ================================================

#include "Integrator.h"
#include "rrRoadRunnerOptions.h"
#include "rrExecutableModel.h"
#include "tr1proxy/rr_random.h"

// == CODE ====================================================

namespace rr
{

    class ExecutableModel;

    /**
     * @author WBC, ETS
     * @brief RoadRunner's implementation of the Gillespie SSA
     */
    class GillespieIntegrator: public Integrator
    {
    public:
        using Integrator::Integrator;

        GillespieIntegrator(ExecutableModel* model);

        ~GillespieIntegrator() override;

        /**
        * @author JKM
        * @brief Called whenever a new model is loaded to allow integrator
        * to reset internal state
        */
        void syncWithModel(ExecutableModel* m) override;

        // ** Meta Info ********************************************************

        /**
         * @author WBC
         * @brief Get the name for this integrator
         * @note Delegates to @ref getName
         */
        std::string getName() const override;

        /**
         * @author WBC
         * @brief Get the description for this integrator
         * @note Delegates to @ref getDescription
         */
        std::string getDescription() const override;

        /**
         * @author WBC
         * @brief Get the hint for this integrator
         * @note Delegates to @ref getHint
         */
        std::string getHint() const override;

        Solver* construct(ExecutableModel* executableModel) const override;

        // ** Getters / Setters ************************************************

        /**
         * @author WBC, ETS
         * @brief Always stochastic for Gillespie
         */
        IntegrationMethod getIntegrationMethod() const override;

        /**
         * @author WBC, ETS
         * @brief Sets the value of an integrator setting (e.g. absolute_tolerance)
         */
        void setValue(const std::string& setting, Setting value) override;

        /**
        * @author JKM
        * @brief Reset all integrator settings to their respective default values
        */
        void resetSettings() override;

        // ** Integration Routines *********************************************

        /**
         * @author WBC, ETS
         * @brief Main integration routine
         */
        double integrate(double t0, double tf) override;

        /**
         * @author WBC, ETS
         * @brief Reset time to zero and reinitialize model
         */
        void restart(double timeStart) override;

        // ** Listeners ********************************************************

        /**
         * @author WBC, ETS
         * @brief Gets the integrator listener
         */
        IntegratorListenerPtr getListener() override;

        /**
         * @author WBC, ETS
         * @brief Sets the integrator listener
         */
        void setListener(IntegratorListenerPtr) override;

    private:
        std::mt19937 engine;
        //unsigned long seed;
        double timeScale;
        double stoichScale;
        int nReactions;
        int floatingSpeciesStart;		// starting index of floating species
        double* reactionRates;
        double* reactionRatesBuffer;
        int stateVectorSize;
        double* stateVector;
        double* stateVectorRate;
        std::vector<unsigned char> eventStatus;
        std::vector<unsigned char> previousEventStatus;

        /**
         * @brief Whether any reaction rate depends explicitly on time.
         * @details -1 = not yet determined, 0 = no, 1 = yes.  When a rate law is
         * an explicit function of time (directly, or through a time-dependent
         * assignment rule it reads) the propensity is not constant between
         * reaction events, so the standard direct method (which samples the
         * waiting time from a frozen propensity) is biased.  When this is 1 the
         * integrator instead integrates the propensity over time (see
         * @ref integrate).  Determined once, lazily, on the first @ref integrate
         * call so the (common) time-homogeneous case keeps its exact, unchanged
         * code path.  Rate rules are not covered: Gillespie holds the state
         * vector fixed between reaction events and never integrates a rate-rule
         * variable, so advancing the clock alone does not change its value -- it
         * is neither detected here nor tracked by the propensity integral.
         */
        int timeDependentRates;

        void testRootsAtInitialTime();
        void applyEvents(double timeEnd, std::vector<unsigned char> &prevEventStatus);

        double urand();
        void setEngineSeed(Setting seedSetting);

        /**
         * @brief Sum of the absolute reaction rates (the total propensity) with
         * the model evaluated at time @p tEval and the current state vector.
         * @details Leaves @ref reactionRates holding the individual rates at
         * @p tEval so the caller can use them for reaction selection.  Species
         * amounts are not modified, so this re-evaluates only the time- and
         * parameter-dependent parts of each rate law.
         */
        double totalPropensity(double tEval);

        /**
         * @brief Detect whether any reaction rate changes with time alone.
         * @details Compares the reaction rates at the current state for several
         * times spanning @p probeSpan.  If any rate differs the propensity is
         * time-dependent.  Restores the model time to @p t before returning.
         *
         * This is a value-based heuristic: it can in principle miss a rate that
         * is flat at the sampled times but varies elsewhere.  Explicit time
         * dependence is a structural property, so inspecting the rate-law
         * expression trees for the SBML time symbol (following the assignment-rule
         * graph, so it also catches a rate that reads a time-dependent parameter,
         * which the value-based check gets for free) would be more robust than
         * sampling values.
         */
        bool detectTimeDependentRates(double t, double probeSpan);

        /**
         * @brief Relative change in the total propensity tolerated across a
         * single panel when integrating a time-dependent propensity.  Smaller
         * values track a fast-varying rate more accurately at the cost of more
         * rate-law evaluations.
         */
        static constexpr double timeDependentRelTol = 0.05;

        inline double getStoich(uint species, uint reaction)
        {
            return mModel->getStoichiometry(species, reaction);
        }

        /**
        * @author JKM
        * @brief Initialize model-specific variables
        * @details Called whenever a model is loaded or a Gillespie
        * integrator is constructed
        */
        void initializeFromModel();
    };
} /* namespace rr */

#endif /* GILLESPIEINTEGRATOR_H_ */
