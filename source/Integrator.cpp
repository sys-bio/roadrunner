/*
 * Integrator.cpp
 *
 *  Created on: Apr 25, 2014
 *      Author: andy
 */

#include "Integrator.h"
#include "CVODEIntegrator.h"
#include "GillespieIntegrator.h"
#include "RK4Integrator.h"
#include "gpu/GPUSimIntegratorProxy.h"

namespace rr
{

Integrator* Integrator::New(const SimulateOptions* opt, ExecutableModel* m)
{
    Integrator *result = 0;

    if (opt->integrator == SimulateOptions::GILLESPIE)
    {
        result = new GillespieIntegrator(m, opt);
    }
    else if(opt->integrator == SimulateOptions::RK4)
    {
        result = new RK4Integrator(m, opt);
    }
    else if(opt->integrator == SimulateOptions::GPUSIM)
    {
        result = rrgpu::CreateGPUSimIntegrator(m, opt);
    }
    else
    {
        result = new CVODEIntegrator(m, opt);
    }

    return result;
}

}
