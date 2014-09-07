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

#include <cassert>

namespace rr
{

// -- TimecourseIntegrationParameters --

TimecourseIntegrationParameters::~TimecourseIntegrationParameters() {

}

void TimecourseIntegrationParameters::addTimevalue(double t) {
    t_.push_back(t);
}

TimecourseIntegrationParameters::size_type TimecourseIntegrationParameters::getTimevalueCount() const {
    return t_.size();
}

double TimecourseIntegrationParameters::getTimevalue(size_type i) const {
    return t_.at(i);
}

double* TimecourseIntegrationParameters::getTimevaluesHeapArrayDbl() const {
    double* result = (double*)malloc(getTimevalueCount()*sizeof(double));
    assert(result && "No memory");
    for (size_type i=0; i<getTimevalueCount(); ++i)
        result[i] = getTimevalue(i);
    return result;
}

float* TimecourseIntegrationParameters::getTimevaluesHeapArrayFlt() const {
    float* result = (float*)malloc(getTimevalueCount()*sizeof(float));
    assert(result && "No memory");
    for (size_type i=0; i<getTimevalueCount(); ++i)
        result[i] = (float)getTimevalue(i);
    return result;
}

// -- TimecourseIntegrationResultsRealVector --

TimecourseIntegrationResultsRealVector::size_type TimecourseIntegrationResultsRealVector::getTimevalueCount() const {
    return ntval_;
}

void TimecourseIntegrationResultsRealVector::setTimevalueCount(size_type n) {
    ntval_ = n;
    rebuild();
}

void TimecourseIntegrationResultsRealVector::setVectorLength(size_type n) {
    veclen_ = n;
    rebuild();
}

TimecourseIntegrationResultsRealVector::size_type TimecourseIntegrationResultsRealVector::getVectorLength() const {
    return veclen_;
}

VariableValue TimecourseIntegrationResultsRealVector::getValue(size_type ti, size_type i) const {
    return val_.at(ti).at(i);
}

void TimecourseIntegrationResultsRealVector::setValue(size_type ti, size_type i, double val) {
    val_.at(ti).at(i) = val;
}

void TimecourseIntegrationResultsRealVector::rebuild() {
    val_.resize(ntval_);
    for (ValMatrix::iterator i=val_.begin(); i != val_.end(); ++i)
        i->resize(veclen_);
}

// -- Integrator --

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
#if defined(BUILD_GPUSIM)
    else if(opt->integrator == SimulateOptions::GPUSIM)
    {
        result = rrgpu::CreateGPUSimIntegrator(m, opt);
    }
#endif
    else
    {
        result = new CVODEIntegrator(m, opt);
    }

    return result;
}

}
