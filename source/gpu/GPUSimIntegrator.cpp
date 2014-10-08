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

#include <chrono>

void GPUIntMESerial(rr::rrgpu::GPUSimIntegratorInt& intf);


using namespace std;
namespace rr
{
namespace rrgpu
{

/**
 * @brief Dispatches integration runs in chunks
 */
// TODO: rename SegmentIntegrator
class GPUSimIntSlicer {
public:
    GPUSimIntSlicer(const TimecourseIntegrationParameters& p, GPUSimExecutableModel& model)
      : params_(p), model_(model) {
          initialize();
      }

    ~GPUSimIntSlicer() {
        reset();
    }

    TimecourseIntegrationResultsPtr integrateAll() {
        int N = (int)realvec_->getTimevalueCount();

        auto integration_start = std::chrono::high_resolution_clock::now();

        int n = N;
        if (N*realvec_->getVectorLength() > 5000)
            n = N/realvec_->getVectorLength();

        if (params_.getPrecision() == TimecourseIntegrationParameters::SINGLE) {
            float* values = svalues_;
            float* tval = stval_;
            for (;N; values += n, tval += n, N = (N>n ? N-n : (n = N, N)))
                integrateSingle_(n, values, tval);
        } else {
            double* values = dvalues_;
            double* tval = dtval_;
            for (;N; values += n, tval += n, N = (N>n ? N-n : (n = N, N)))
                integrateSingle_(n, values, tval);
        }

        auto integration_finish = std::chrono::high_resolution_clock::now();
        Log(Logger::LOG_INFORMATION) << "Integration took " << std::chrono::duration_cast<std::chrono::milliseconds>(integration_finish - integration_start).count() << " ms";

        return std::move(results_);
    }

protected:
    void initialize() {
        results_.reset(new TimecourseIntegrationResultsRealVector());
        realvec_ = (TimecourseIntegrationResultsRealVector*)results_.get();
        gen_ = 0;

        realvec_->setTimevalueCount(params_.getTimevalueCount());
        realvec_->setVectorLength(model_.getNumIndepFloatingSpecies());

        if (params_.getPrecision() == TimecourseIntegrationParameters::SINGLE) {
            svalues_ = (float*)malloc(realvec_->getTimevalueCount()*realvec_->getVectorLength()*sizeof(float));
            model_.refresh();
            stval_ = params_.getTimevaluesHeapArrayFlt();
        } else {
            dvalues_ = (double*)malloc(realvec_->getTimevalueCount()*realvec_->getVectorLength()*sizeof(double));
            model_.refresh();
            dtval_ = params_.getTimevaluesHeapArrayDbl();
        }
    }

    void reset() {
        free(svalues_);
        free(dvalues_);
        free(stval_);
        free(dtval_);

        svalues_ = nullptr;
        dvalues_ = nullptr;
        stval_   = nullptr;
        dtval_   = nullptr;

        results_.reset();
        realvec_ = nullptr;
    }

    void integrateSingle_(int n, float* values, float* tval) {
        assert(values && tval && "No values set");

//         auto integration_start = std::chrono::high_resolution_clock::now();

        model_.getEntryPoint()(n, tval, values);

//         auto integration_finish = std::chrono::high_resolution_clock::now();
//         Log(Logger::LOG_INFORMATION) << "Integration took " << std::chrono::duration_cast<std::chrono::milliseconds>(integration_finish - integration_start).count() << " ms";

        for (TimecourseIntegrationResultsRealVector::size_type i=0; i<n; ++i)
            for (TimecourseIntegrationResultsRealVector::size_type j=0; j<realvec_->getVectorLength(); ++j)
                realvec_->setValue(i+gen_, j, values[i*realvec_->getVectorLength() + j]);

        gen_ +=  n;
    }

    void integrateSingle_(int n, double* values, double* tval) {
        assert(values && tval && "No values set");

//         auto integration_start = std::chrono::high_resolution_clock::now();

        model_.getEntryPoint()(n, tval, values);

//         auto integration_finish = std::chrono::high_resolution_clock::now();
//         Log(Logger::LOG_INFORMATION) << "Integration took " << std::chrono::duration_cast<std::chrono::milliseconds>(integration_finish - integration_start).count() << " ms";

        for (TimecourseIntegrationResultsRealVector::size_type i=0; i<n; ++i)
            for (TimecourseIntegrationResultsRealVector::size_type j=0; j<realvec_->getVectorLength(); ++j)
                realvec_->setValue(i+gen_, j, values[i*realvec_->getVectorLength() + j]);

        gen_ +=  n;
    }

    const TimecourseIntegrationParameters& params_;
    GPUSimExecutableModel& model_;

    float*  svalues_ = nullptr;
    double* dvalues_ = nullptr;
    float*  stval_   = nullptr;
    double* dtval_   = nullptr;

    TimecourseIntegrationResultsPtr results_;
    TimecourseIntegrationResultsRealVector* realvec_ = nullptr;
    int gen_ = 0;
};

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

TimecourseIntegrationResultsPtr GPUSimIntegrator::integrate(const TimecourseIntegrationParameters& p) {
    GPUSimIntSlicer slicer(p, *model_);
    return slicer.integrateAll();
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
