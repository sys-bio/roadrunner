// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

// == FILEDOC =================================================

/** @file GPUSimIntegratorProxy.h
  * @brief Proxy for creating a GPUSim integrator
  * @details Needed because rr core may be built with different
  * flags than GPUSim
**/

// == FILEINFO ================================================

/*
 * GPUSimIntegratorProxy.h
 *
 *  Created on: Aug 18, 2014
 *
 * Author: JKM
 */

#ifndef rrGPUSimIntegratorProxyH
#define rrGPUSimIntegratorProxyH

// == INCLUDES ================================================

#include "Integrator.h"
#include "rrRoadRunnerOptions.h"
#include "Configurable.h"

#include <string>
#include <vector>

// == CODE ====================================================

namespace rr
{

namespace rrgpu
{

Integrator* CreateGPUSimIntegrator(ExecutableModel* oModel, const SimulateOptions* options);

} // namespace rrgpu

} // namespace rr

#endif
