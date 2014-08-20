// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

// == FILEDOC =================================================

/** @file GPUSimIntegratorInt.h
  * @brief Simple integrator wrapper
  * @details Interface for communication between device and host
  * when running the integrator
**/

/*
 * GPUSimIntegratorInt.h
 *
 *  Created on: Aug 20, 2014
 *
 * Author: JKM
 */

#ifndef rrGPUSimIntegratorIntH
#define rrGPUSimIntegratorIntH

// == INCLUDES ================================================

// == CODE ====================================================


namespace rr
{

namespace rrgpu
{

class GPUSimIntegrator;

/**
 * @author JKM
 * @brief Interface for calling host integrator functions
 * @details Called by device
 */
class GPUSimIntegratorInt
{
public:
    GPUSimIntegratorInt(GPUSimIntegrator* parent)
      : parent_(parent) {}

    /** @brief Wrapper for model evaluation
     * @param[in] time The time coordinate at which to evaluate the model
     * @param[in] y The state vector (consists of species concentrations)
     * @param[out] dydt The computed rates
     * @sa ExecutableModel::getStateVectorRate
     */
    void evalRate(double time, const double *y, double* dydt=0);

    /// eval rate using floats
    void evalRatef(float time, const float *y, float* dydt=0);

    /// Gets the size of the state vector
    int getStateVectorSize() const;

private:
    GPUSimIntegrator* parent_;
};

} // namespace rrgpu

} // namespace rr

#endif
