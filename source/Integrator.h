/*
 * Integrator.h
 *
 *  Created on: Sep 7, 2013
 *      Author: andy
 */

#ifndef INTEGRATOR_H_
#define INTEGRATOR_H_

#include "rrLogger.h"
#include "rrOSSpecifics.h"
#include "rrRoadRunnerOptions.h"
#include "Dictionary.h"
#include <stdexcept>

#if (__cplusplus >= 201103L) || defined(_MSC_VER)
#include <memory>
#define cxx11_ns std
#else
#include <tr1/memory>
#define cxx11_ns std::tr1
#endif

namespace rr
{

class Integrator;
class ExecutableModel;

/**
 * Listen for integrator events.
 *
 * These are called after the event occured or was processed.
 *
 * The internal integrator typically iterates for many internal time
 * steps, at variable step size for each external time step. So, if
 * RoadRunner::oneStep is called with a step size of say 10 time units, the internal
 * integrator may potentially integrate for 1, 5, 10 100 or some value. The
 * onTimeStep method is called for each of these internal time steps.
 *
 * The return values are currently a place holder and are ignored.
 */
class IntegratorListener
{
public:

    /**
     * is called after the internal integrator completes each internal time step.
     */
    virtual uint onTimeStep(Integrator* integrator, ExecutableModel* model, double time) = 0;

    /**
     * whenever model event occurs and after it is procesed.
     */
    virtual uint onEvent(Integrator* integrator, ExecutableModel* model, double time) = 0;

    virtual ~IntegratorListener() {};
};

/**
 * listeners are shared objects, so use std smart pointers
 * to manage them.
 */
typedef cxx11_ns::shared_ptr<IntegratorListener> IntegratorListenerPtr;

/**
 * Interface to a class which advances a model forward in time.
 *
 * The Integrator is only valid if attached to a model.
 */
class RR_DECLSPEC Integrator : public Dictionary
{
public:

    /**
     * Set the configuration parameters the integrator uses.
     */
    virtual void setSimulateOptions(const SimulateOptions* options) = 0;

    /**
     * integrates the model from t0 to tf.
     */
    virtual double integrate(double t0, double tf) = 0;

    /**
     * copies the state vector out of the model and into cvode vector,
     * re-initializes cvode.
     */
    virtual void restart(double t0) = 0;

    /**
     * the integrator can hold a single listener. If clients require multicast,
     * they can create a multi-cast listener.
     */
    virtual void setListener(IntegratorListenerPtr) = 0;

    /**
     * get the integrator listener
     */
    virtual IntegratorListenerPtr getListener() = 0;

    virtual ~Integrator() {};

    /**
     * create a new integrator based on the settings in the
     * options class.
     *
     * The new integrator borrows a reference to an ExecutableModel object.
     */
    static Integrator* New(const SimulateOptions *o, ExecutableModel *m);
};


class IntegratorException: public std::runtime_error
{
public:
    explicit IntegratorException(const std::string& what) :
            std::runtime_error(what)
    {
        Log(rr::Logger::LOG_ERROR) << __FUNC__ << "what: " << what;
    }

    explicit IntegratorException(const std::string& what, const std::string &where) :
            std::runtime_error(what + "; In " + where)
    {
        Log(rr::Logger::LOG_ERROR) << __FUNC__ << "what: " << what << ", where: " << where;
    }
};

}

#endif /* INTEGRATOR_H_ */
