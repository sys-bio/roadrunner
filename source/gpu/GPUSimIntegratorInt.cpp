// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README


/*
 * GPUSimIntegratorInt.cpp
 *
 *  Created on: Aug 20, 2014
 *
 * Author: JKM
 */

// == INCLUDES ================================================

#pragma hdrstop
#include "GPUSimIntegrator.h"
#include "GPUSimIntegratorInt.h"

// == CODE ====================================================

namespace rr
{

namespace rrgpu
{

void GPUSimIntegratorInt::evalRate(double time, const double *y, double* dydt) {
    parent_->evalRate(time, y, dydt);
}

void GPUSimIntegratorInt::evalRatef(float time, const float *y, float* dydt) {
    parent_->evalRatef(time, y, dydt);
}

int GPUSimIntegratorInt::getStateVectorSize() const {
    return parent_->getStateVectorSize();
}

} // namespace rrgpu

} // namespace rr
