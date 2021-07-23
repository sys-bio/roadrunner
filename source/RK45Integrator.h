// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

// == FILEDOC =================================================

/** @file RK45Integrator.h
* @author KC
* @date 10/02/2015
* @copyright Apache License, Version 2.0
* @brief A Runge-Kutta Fehlberg method for roadrunner.
**/

#ifndef RK45Integrator_H_
#define RK45Integrator_H_

#include <Integrator.h>
#include <rrRoadRunnerOptions.h>
#include "Registrable.h"

namespace rr
{


    /**
     * @author KC
     * @brief A Runge-Kutta Fehlberg method for roadrunner
     * @details Uses the Fehlberg method, an adaptive step
     * method, to integrate models.
     */
    class RK45Integrator: public Integrator
    {
    public:

        using Integrator::Integrator;

        /**
         * @author KC
         * @brief Constructor: takes an executable model, does not own the pointer
         */
        RK45Integrator(ExecutableModel *m);

        /**
         * @author KC
         * @brief Destructor
         */
        virtual ~RK45Integrator();

        /**
        * @author JKM
        * @brief Called whenever a new model is loaded to allow integrator
        * to reset internal state
        */
        virtual void syncWithModel(ExecutableModel* m);


        /**
        * implement Integrator interface
        */
    public:

        /**
         * @author CC
         * @brief Integrates the model from t to t + h.
         * @details Attempts to find the state std::vector at
         * t + h, and returns time t + h if successful.
         * If the integrator does not find a convergent
         * solution, the state std::vector is not updated and
         * t is returned.
         * Also calculates a new timestep and saves it 
         * to a member variable for future use.
         */
        virtual double integrate(double t, double h);

        /**
         * @brief Restarts the integrator
         */
        virtual void restart(double t0);

        // ** Meta Info ********************************************************

        /**
         * @author JKM
         * @brief Get the name for this integrator
         * @note Delegates to @ref getName
         */
        std::string getName() const;

        /**
         * @author JKM
         * @brief Get the name for this integrator
         */
        static std::string getRK45Name();

        /**
         * @author JKM
         * @brief Get the description for this integrator
         * @note Delegates to @ref getDescription
         */
        std::string getDescription() const;

        /**
         * @author JKM
         * @brief Get the description for this integrator
         */
        static std::string getRK45Description();

        /**
         * @author WBC
         * @brief Get the hint for this integrator
         * @note Delegates to @ref getHint
         */
        std::string getHint() const;

        /**
         * @author JKM
         * @brief Get the hint for this integrator
         */
        static std::string getRK45Hint();

        Solver* construct(ExecutableModel* executableModel) const;

        // ** Getters / Setters ************************************************

        virtual Setting getValue(std::string key);

        /**
         * @author JKM
         * @brief Always deterministic for RK45
         */
        IntegrationMethod getIntegrationMethod() const;

        /**
        * @author JKM
        * @brief Reset all integrator settings to their respective default values
        */
        void resetSettings();

        // ** Listeners ********************************************************

        /**
        * the integrator can hold a single listener. If clients require multicast,
        * they can create a multi-cast listener.
        */
        virtual void setListener(IntegratorListenerPtr);

        /**
        * get the integrator listener
        */
        virtual IntegratorListenerPtr getListener();


    private:

        unsigned stateVectorSize;

        /**
        * arrays to store function eval values.
        */
        double *k1, *k2, *k3, *k4, *y, *ytmp;

        // new scalars for RK45
        double hCurrent, hmin, hmax;

        double *k5, *k6, *err;

        void testRootsAtInitialTime();
        void applyEvents(double timeEnd, std::vector<unsigned char> &previousEventStatus);

    };

} /* namespace rr */

#endif /* RK45Integrator_H_ */
