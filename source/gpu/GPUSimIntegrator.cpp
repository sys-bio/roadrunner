/*
 * GPUSimIntegrator.cpp
 *
 *  Created on: Aug 13, 2014
 *      Author: JKM
 */

#pragma hdrstop
#include "GPUSimIntegrator.h"
#include "GPUSimIntegratorInt.h"
#include "GPUSimException.h"
#include "rrExecutableModel.h"
#include "rrException.h"
#include "rrLogger.h"
#include "rrStringUtils.h"
#include "rrException.h"
#include "rrUtils.h"

#include <cvode/cvode.h>
#include <cvode/cvode_dense.h>
#include <nvector/nvector_serial.h>
#include <cstring>
#include <iomanip>
#include <math.h>
#include <map>
#include <algorithm>
#include <assert.h>
#include <Poco/Logger.h>

void GPUIntMESerial(rr::rrgpu::GPUSimIntegratorInt& intf);


using namespace std;
namespace rr
{
namespace rrgpu
{

void GPUSimIntegrator::setListener(IntegratorListenerPtr p)
{
    throw_gpusim_exception("not supported");
}

IntegratorListenerPtr GPUSimIntegrator::getListener()
{
    throw_gpusim_exception("not supported");
}

GPUSimIntegrator::GPUSimIntegrator(ExecutableModel *model, const SimulateOptions* options)
{
    Log(Logger::LOG_INFORMATION) << "creating GPUSimIntegrator";
    model_ = dynamic_cast<GPUSimExecutableModel*>(model);
    if (!model_)
        throw_gpusim_exception("Wrong model type (expected GPUSimExecutableModel)");
}

GPUSimIntegrator::~GPUSimIntegrator()
{
}

void GPUSimIntegrator::setSimulateOptions(const SimulateOptions* o)
{
}

double GPUSimIntegrator::integrate(double t0, double tf) {
    throw_gpusim_exception("Use other sig");
}

TimecourseIntegrationResultsPtr GPUSimIntegrator::integrate(const TimecourseIntegrationParameters& p)
{
    TimecourseIntegrationResultsPtr results(new TimecourseIntegrationResultsRealVector());
    TimecourseIntegrationResultsRealVector* realvec = (TimecourseIntegrationResultsRealVector*)results.get();

//     GPUSimIntegratorInt intf(this);

//     GPUIntMESerial(intf);
    std::size_t elt_size = sizeof(float);
    void* values = malloc(p.getTimevalueCount()*model_->getNumIndepFloatingSpecies()*elt_size);
    model_->refresh();
    float* tval = p.getTimevaluesHeapArrayFlt();

//     model_->getEntryPoint()(tval, (float*)values);

    realvec->setTimevalueCount(p.getTimevalueCount());
    realvec->setVectorLength(model_->getNumIndepFloatingSpecies());
    for (TimecourseIntegrationResultsRealVector::size_type i=0; i<realvec->getTimevalueCount(); ++i)
        for (TimecourseIntegrationResultsRealVector::size_type j=0; j<realvec->getVectorLength(); ++j)
            realvec->setValue(i, j, i+j);

    free(values);
    free(tval);

    return results;
}

void GPUSimIntegrator::restart(double time)
{
}

_xmlNode* GPUSimIntegrator::createConfigNode()
{
    throw_gpusim_exception("not supported");
}

void GPUSimIntegrator::loadConfig(const _xmlDoc* doc)
{
    throw_gpusim_exception("not supported");
}

void GPUSimIntegrator::setValue(const std::string& key, const rr::Variant& value)
{
    throw_gpusim_exception("not supported");
}

Variant GPUSimIntegrator::getValue(const std::string& key) const
{
    throw_gpusim_exception("not supported");
}

bool GPUSimIntegrator::hasKey(const std::string& key) const
{
    throw_gpusim_exception("not supported");
}

int GPUSimIntegrator::deleteValue(const std::string& key)
{
    throw_gpusim_exception("not supported");
}

std::vector<std::string> GPUSimIntegrator::getKeys() const
{
    throw_gpusim_exception("not supported");
}


std::string GPUSimIntegrator::toString() const
{
    std::stringstream ss;
    ss << "< roadrunner.GPUSimIntegrator() >\n";

    return ss.str();
}

std::string GPUSimIntegrator::toRepr() const
{
    std::stringstream ss;
    ss << "< roadrunner.GPUSimIntegrator() >\n";
    return ss.str();
}

std::string GPUSimIntegrator::getName() const
{
    return "gpusim";
}

void GPUSimIntegrator::evalRate(double time, const double *y, double* dydt) {
    model_->getStateVectorRate(time, y, dydt);
}

void GPUSimIntegrator::evalRatef(float time, const float *y, float* dydt) {
    // thank you for rejecting N3820, C++ committee, and making
    // this implementation exponentially more complicated
    double* yy = new double[getStateVectorSize()];
    double* dyy = new double[getStateVectorSize()];
    evalRate(time, yy, dyy);
    delete yy;
    delete dyy;
}

int GPUSimIntegrator::getStateVectorSize() const {
    return model_->getStateVector(NULL);
}

Integrator* CreateGPUSimIntegrator(ExecutableModel* oModel, const SimulateOptions* options) {
    return new GPUSimIntegrator(oModel, options);
}

} // namespace rrgpu

} // namespace rr
