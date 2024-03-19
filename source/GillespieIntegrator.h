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

        void testRootsAtInitialTime();
        void applyEvents(double timeEnd, std::vector<unsigned char> &prevEventStatus);

        double urand();
        void setEngineSeed(Setting seedSetting);

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
