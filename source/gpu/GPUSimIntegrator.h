// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

// == FILEDOC =================================================

/** @file GPUSimIntegrator.h
  * @brief GPUSim integrator
**/

// == FILEINFO ================================================

/*
 * GPUSimIntegrator.h
 *
 *  Created on: Aug 14, 2014
 *
 * Author: JKM
 */

#ifndef rrGPUSimIntegratorH
#define rrGPUSimIntegratorH

// == INCLUDES ================================================

#include "Integrator.h"
#include "rrRoadRunnerOptions.h"
#include "GPUSimExecutableModel.h"
#include "Configurable.h"

#include <string>
#include <vector>

// == CODE ====================================================

/**
 * CVode vector struct
 */
typedef struct _generic_N_Vector *N_Vector;

namespace rr
{

using std::string;

class ExecutableModel;
class RoadRunner;

namespace rrgpu
{

/**
 * @author JKM
 * @internal
 * @brief GPU-based integrator
 */
class GPUSimIntegrator : public Integrator
{
public:
    GPUSimIntegrator(ExecutableModel* oModel, const SimulateOptions* options);

    virtual ~GPUSimIntegrator();

    /**
     * creates a new xml element that represent the current state of this
     * Configurable object and all if its child objects.
     */
    virtual _xmlNode *createConfigNode();

    /**
     * Given an xml element, the Configurable object should pick its needed
     * values that are stored in the element and use them to set its
     * internal configuration state.
     */
    virtual void loadConfig(const _xmlDoc* doc);

    virtual double integrate(double t0, double tf);

    /**
     * copies the state vector out of the model and into the integrator vector,
     * re-initializes the integrator.
     */
    virtual void restart(double timeStart);

    /**
     * set the options the integrator will use.
     */
    virtual void setSimulateOptions(const SimulateOptions* options);

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
     * implement dictionary interface
     */
    virtual void setValue(const std::string& key, const rr::Variant& value);

    virtual Variant getValue(const std::string& key) const;

    virtual bool hasKey(const std::string& key) const;

    virtual int deleteValue(const std::string& key);

    virtual std::vector<std::string> getKeys() const;

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

    /** @brief Wrapper for model evaluation
     * @param[in] time The time coordinate at which to evaluate the model
     * @param[in] y The state vector (consists of species concentrations)
     * @param[out] dydt The computed rates
     * @sa ExecutableModel::getStateVectorRate
     */
    void evalRate(double time, const double *y, double* dydt=0);

    void evalRatef(float time, const float *y, float* dydt=0);

    /// Gets the size of the state vector
    int getStateVectorSize() const;

private:
    /// Non-owning
    GPUSimExecutableModel* model_;
    int mOneStepCount=0;
};

} // namespace rrgpu

} // namespace rr

#endif
