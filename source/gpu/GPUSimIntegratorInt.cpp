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

} // namespace rrgpu

} // namespace rr
