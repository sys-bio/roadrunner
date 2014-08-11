/*
 * RK4Integrator.h
 *
 *  Created on: Jul 22, 2014
 *      Author: andy
 */

#ifndef RK4Integrator_H_
#define RK4Integrator_H_

#include <Integrator.h>

namespace rr
{


/**
 * A super basic 4'th order fixed step integrator.
 *
 * The RungeKuttaIntegrator will be the more sophisticated,
 * general purpose Runge-Kutta integrator which will support
 * different orders and adaptive time stepping.
 *
 * TODO: SBML event handling.
 */
class RK4Integrator: public Integrator
{
public:
    RK4Integrator(ExecutableModel *m, const SimulateOptions *o);

    virtual ~RK4Integrator();


    /**
     * implement Integrator interface
     */
public:

    /**
     * Set the configuration parameters the integrator uses.
     */
    virtual void setSimulateOptions(const SimulateOptions* options);

    /**
     * integrates the model from t0 to tf.
     */
    virtual double integrate(double t0, double tf);

    /**
     * copies the state vector out of the model and into cvode vector,
     * re-initializes cvode.
     */
    virtual void restart(double t0);

    /**
     * the integrator can hold a single listener. If clients require multicast,
     * they can create a multi-cast listener.
     */
    virtual void setListener(IntegratorListenerPtr);

    /**
     * get the integrator listener
     */
    virtual IntegratorListenerPtr getListener();

    /**
     * get a description of this object, compatable with python __str__
     */
    virtual std::string toString() const;

    /**
     * get a short descriptions of this object, compatable with python __repr__.
     */
    virtual std::string toRepr() const;

    /**
     * get the name of this integrator
     */
    virtual std::string getName() const;


    /**
     * Implement Dictionary Interface
     */
public:

    /**
     * set an arbitrary key
     */
    virtual void setValue(const std::string& key, const rr::Variant& value);

    /**
     * get a value. Variants are POD.
     */
    virtual Variant getValue(const std::string& key) const;

    /**
     * is there a key matching this name.
     */
    virtual bool hasKey(const std::string& key) const;

    /**
     * remove a value
     */
    virtual int deleteValue(const std::string& key);

    /**
     * list of keys in this object.
     */
    virtual std::vector<std::string> getKeys() const;


private:
    ExecutableModel *model;

    SimulateOptions opt;

    unsigned stateVectorSize;

    /**
     * arrays to store function eval values.
     */
    double *k1, *k2, *k3, *k4, *y, *ytmp;

};

} /* namespace rr */

#endif /* RK4Integrator_H_ */
