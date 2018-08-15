/*
 * EulerIntegrator.h
 *
 *  Created on: Dec 17, 2014
 *      Author: andy
 */

#ifndef EULERINTEGRATOR_H_
#define EULERINTEGRATOR_H_

#include "Integrator.h"
#include "rrExecutableModel.h"
#include "rrRoadRunnerOptions.h"

#include <string>
#include <stdexcept>
#include <sstream>
#include <iostream>

namespace rr
{
    using std::string;
    using std::invalid_argument;
    using std::stringstream;

    /**
    * An example class intended to show how to create
    * an Integrator. This class implements the most
    * basic possible integration algorithm: the forward Euler method.
    *
    * This integrator should not really be used in practice as the
    * forward Euler algorithm is highly unstable and will seldomly
    * yield numerically correct values.
    *
    * This class also has two dummy variables, exampleParameter1 and
    * exampleParameter2. These serve as examples of how to use the Dictionary
    * interface to get/set various tuning paramters that an Integrator
    * might have. These values will be displayed when the toString method
    * is called, typically by Python.
    *
    * An example of calling simulate in Python, and the results of specifying
    * the parameters as keyword arguments is
    *
    * @code
    * r.simulate(integrator='euler', exampleParameter1=123456, exampleParameter2='some value');
    *
    * print(r.integrator)
    * < roadrunner.EulerIntegrator()
    * { 'this' : 0x101f28350
    * 'exampleParameter1' : 123456
    * 'exampleParameter2' : some value
    * }>
    * @endcode
    *
    */
    class EulerIntegrator: public Integrator
    {
    public:

        /**
        * Creates a new EulerIntegrator.
        *
        * The IntegratorFactory is the ONLY object that creates integrators.
        *
        * Integrators are created when the IntegratorFactory::New method is called,
        * typically by the top level RoadRunner object.
        *
        * The integrator will hold onto the ExecutableModel pointer, m, and when the
        * integrate method is called, will advance the model object forward in time.
        *
        * @param m: a borrowed reference to an existing ExecutableModel object.
        * @param o: a reference to a SimulatOptions object where the configuration
        * parameters will be read from.
        */
        EulerIntegrator(ExecutableModel *m)
			:	eventStatus(std::vector<unsigned char>(m->getNumEvents(),false)),
				previousEventStatus(std::vector<unsigned char>(m->getNumEvents(), false)) {
            resetSettings();

            model = m;
            exampleParameter1 = 3.14;
            exampleParameter2 = "hello";
			Log(Logger::LOG_WARNING) << "Euler integrator is inaccurate";
			//std::cerr << "Number of event triggers: " << m->getEventTriggers(0, 0, 0) << std::endl;

            if(model) {
                // calling the getStateVector with a NULL argument returns
                // the size of teh state vector.
                stateVectorSize = model->getStateVector(NULL);
                rateBuffer = new double[stateVectorSize];
                stateBufferBegin = new double[stateVectorSize];
                stateBufferEnd = new double[stateVectorSize];
            } else {
                rateBuffer = NULL;
                stateBufferBegin = NULL;
                stateBufferEnd = NULL;
            }
        }

        /**
        * delete any memory we allocated
        */
        virtual ~EulerIntegrator() {
            delete[] rateBuffer;
            delete[] stateBufferBegin;
            delete[] stateBufferEnd;
        };

        /**
        * integrates the model from t0 to t0 + hstep
        *
        * In this implementation, this performs a basic forward Euler step.
        * This method also demonstrates how to notify the listener
        * that a timestep has completed.
        *
        * @return the end time.
        */
        virtual double integrate(double t0, double h) {
			int internal_steps = getValueAsInt("subdivision_steps");
			if (model == (rr::ExecutableModel*)NULL) return 0;

            double finalTimeEnd;

			h /= internal_steps;
			for (int subdiv = 0; subdiv < internal_steps; ++subdiv) {
				// evaluate and copy the rate of change of the state vector
				// rate into the local buffer. If the 2nd argument is NULL,
				// the current model state is used to evaluate the
				// state vector rate.
				model->getStateVectorRate(t0, NULL, rateBuffer);

				// copy the current state vector into a local buffer
				model->getStateVector(stateBufferBegin);

				// perform the Euler integration step, i.e.
				// y_{n+1} = y_{n} + h * y'_{n}
				for (int i = 0; i < stateVectorSize; ++i) {
					stateBufferEnd[i] = stateBufferBegin[i] + h * rateBuffer[i];
				}

				// set the model state to the newly calculated state
				model->setStateVector(stateBufferEnd);

				// update the model time to the new time
				double timeEnd = t0 + h;
				model->setTime(timeEnd);

				// if we have a client, notify them that we have taken
				// a time step
				if (listener) {
					listener->onTimeStep(this, model, timeEnd);
				}

				// events
				bool triggered = false;

				model->getEventTriggers(eventStatus.size(), NULL, eventStatus.size() ? &eventStatus[0] : NULL);
				for (int k_ = 0; k_ < eventStatus.size(); ++k_) {
					if (eventStatus.at(k_)) {
						triggered = true;
						//std::cerr << "Triggered" << std::endl;
					}
				}

				if (triggered) {
					// applyEvents takes the list of events which were previously triggered
					//std::cerr << "An event was triggered at " << t0 << std::endl;
					applyEvents(timeEnd, previousEventStatus);
				}

				if (eventStatus.size()) {
					previousEventStatus = eventStatus;
				}

                finalTimeEnd = timeEnd;
			}

            return finalTimeEnd;
        }

		void applyEvents(double timeEnd, std::vector<unsigned char> &previousEventStatus) {
			//std::cerr << "Size of previous events: " << previousEventStatus.size() << std::endl;
			// If we pass in the events including the ones just triggered, they won't be applied, so use previousEventStatus
			model->applyEvents(timeEnd, previousEventStatus.size() == 0 ? NULL : &previousEventStatus[0], stateBufferEnd, NULL);
			// AHu: jk I think that model->applyEvents does update the mode's state vector
			// The previous statement loaded the result into the final stateBufferEnd, so now update the model's state vector
			//model->setStateVector(stateBufferEnd);
		}

        /**
        * This simple integrator has nothing to reset, so do nothing here
        */
        virtual void restart(double t0) {}

        /**
        * Clients may register a listener to listen for
        * sbml events and time step events.
        */
        virtual void setListener(IntegratorListenerPtr p) {
            listener = p;
        }

        /**
        * get the integrator listener
        */
        virtual IntegratorListenerPtr getListener() {
            return listener;
        }

        /**
        * get a description of this object, compatable with python __str__
        */
        virtual std::string toString() const {
            std::stringstream ss;
            ss << "< roadrunner.EulerIntegrator() " << std::endl;
            ss << "{ 'this' : " << (void*)this << std::endl;
            ss << "'exampleParameter1' : " << exampleParameter1 << std::endl;
            ss << "'exampleParameter2' : " << exampleParameter2 << std::endl;
            ss << "}>";
            return ss.str();
        }

        /**
        * get a short descriptions of this object, compatable with python __repr__.
        */
        virtual std::string toRepr() const {
            std::stringstream ss;
            ss << "< roadrunner.EulerIntegrator() { 'this' : "
                << (void*)this << " }>";
            return ss.str();
        }

        /**
        * get the name of this integrator
        */
        virtual std::string getName() const {
            return getEulerName();
        }

        /**
         * @author JKM
         * @brief Get the name for this integrator
         */
        static std::string getEulerName() {
            return "euler";
        }

        /**
         * @author JKM
         * @brief Get the description for this integrator
         * @note Delegates to @ref getDescription
         */
        std::string getDescription() const {
            return getEulerDescription();
        }

        /**
         * @author JKM
         * @brief Get the description for this integrator
         */
        static std::string getEulerDescription() {
            return "The Euler method is one of the simplest approaches to "
                "solving a first order ODE. Given the rate of change of "
                "function f at time t, it computes the new value of f as "
                "f(t+h) = f(t) + h*f'(t), where h is the time step. "
                "Euler's method is rarely used in practice due to poor "
                "numerical robustness.";
        }

        /**
         * @author JKM
         * @brief Get the hint for this integrator
         * @note Delegates to @ref getHint
         */
        std::string getHint() const {
            return getEulerHint();
        }

        /**
         * @author JKM
         * @brief Get the hint for this integrator
         */
        static std::string getEulerHint() {
            return "A simple Euler integrator";
        }

        /**
         * @author JKM
         * @brief Always deterministic for Euler
         */
        IntegrationMethod getIntegrationMethod() const {
            return Integrator::Deterministic;
        }

        /**
        * sets the value for a key.
        *
        * This integrator only supports 2 values, so those are the
        * only two valid items to set.
        */
        virtual void setItem(const std::string& key, const rr::Variant& value) {
            if (key == "exampleParameter1") {
                exampleParameter1 = value;
                return;
            }

            if(key == "exampleParameter2") {
				// Ahu: Why is this cast here, and is this a static or dynamic cast?
                exampleParameter2 = (string)value;
				return;
            }

            // they did not give a valid key, so throw an exception.
            throw invalid_argument("Error, attempt to set invalid key: " + key);
        }

        /**
        * Get a value.
        *
        * This integrator only supports two parameters, those are the
        * only valid ones to get.
        */
        virtual Variant getItem(const std::string& key) const {
            if (key == "exampleParameter1") {
                return exampleParameter1;
            }

            if(key == "exampleParameter2") {
                return exampleParameter2;
            }

            // they did not give a valid key, so throw an exception.
            throw invalid_argument("Error, attempt to read invalid key: " + key);
        }

        /**
        * is there a key matching this name.
        */
        virtual bool hasKey(const std::string& key) const {
            if (key == "exampleParameter1" || key == "exampleParameter2") {
                return true;
            } else {
                return false;
            }
        }

        /**
        * remove a value, this example class does not support deleting
        * keys, so just raise an exception if someone tries to do so.
        */
        virtual int deleteItem(const std::string& key) {
            throw invalid_argument(
                    "Error, the EulerIntegrator does not support deleting keys");
        }

        /**
        * list of keys in this object.
        */
        virtual std::vector<std::string> getKeys() const {
            std::vector<std::string> keys;
            keys.push_back("exampleParameter1");
            keys.push_back("exampleParameter2");
            return keys;
        }

		void resetSettings() {
			Solver::resetSettings();

			// Set default integrator settings.
            addSetting("subdivision_steps", 1,
                "Subdivision Steps",
                "The number of subdivisions of the Euler step size (int).",
                "(int) For each point, up to this many extra steps will be taken as smaller steps within each step, although their values are not saved");
        }

    private:
        /**
        * a parameter which does nothing
        */
        double exampleParameter1;

        /**
        * another parameter which does nothing
        */
        string exampleParameter2;

        /**
        * a stolen reference to an existing model object.
        */
        ExecutableModel *model;

        /**
        * two buffers to store the state vector rate, and
        * new state vector
        */
        double *rateBuffer, *stateBufferBegin, *stateBufferEnd;

        /**
        * size of state vector
        */
        int stateVectorSize;

		std::vector<unsigned char> eventStatus;
		std::vector<unsigned char> previousEventStatus;

        /**
        * Clients may register a listener to listen for time steps, or
        * sbml events. Time step events are much more usefull for variable
        * time step integrators where they may take long time steps and
        * clients may be notified when these occur.
        *
        * This is a smart pointer to a object which implements the
        * IntegratorListener interface.
        */
        IntegratorListenerPtr listener;
    };


    // ** Registration *********************************************************


    class EulerIntegratorRegistrar : public IntegratorRegistrar {
        public:
            /**
            * @author JKM
            * @brief Gets the name associated with this integrator type
            */
            virtual std::string getName() const {
                return EulerIntegrator::getEulerName();
            }

            /**
            * @author JKM
            * @brief Gets the description associated with this integrator type
            */
            virtual std::string getDescription() const {
                return EulerIntegrator::getEulerDescription();
            }

            /**
            * @author JKM
            * @brief Gets the hint associated with this integrator type
            */
            virtual std::string getHint() const {
                return EulerIntegrator::getEulerHint();
            }

            /**
            * @author JKM
            * @brief Constructs a new integrator of a given type
            */
            virtual Integrator* construct(ExecutableModel *model) const {
                return new EulerIntegrator(model);
            }
    };

} /* namespace rr */

#endif /* EULERINTEGRATOR_H_ */
