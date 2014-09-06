/*
 * RK4Integrator.cpp
 *
 *  Created on: Jul 22, 2014
 *      Author: andy
 */

#include <RK4Integrator.h>
#include <rrExecutableModel.h>

extern "C" {
#include <clapack/f2c.h>
#include <clapack/clapack.h>
}

namespace rr
{

RK4Integrator::RK4Integrator(ExecutableModel *m, const SimulateOptions *o)
{
    Log(Logger::LOG_NOTICE) << "creating runge-kutta integrator";

    model = m;
    if (o) {
        opt = *o;
    }

    if (model) {
        stateVectorSize = model->getStateVector(NULL);
        k1 = new double[stateVectorSize];
        k2 = new double[stateVectorSize];
        k3 = new double[stateVectorSize];
        k4 = new double[stateVectorSize];
        y = new double[stateVectorSize];
        ytmp = new double[stateVectorSize];
    } else {
        stateVectorSize = 0;
        k1 = k2 = k3 = k4 = y = NULL;
    }
}

RK4Integrator::~RK4Integrator()
{
    delete []k1;
    delete []k2;
    delete []k3;
    delete []k4;
    delete []y;
    delete []ytmp;
}

void RK4Integrator::setSimulateOptions(const SimulateOptions* options)
{
}

double RK4Integrator::integrate(double t0, double h)
{
    if (!model) {
        return -1;
    }

    Log(Logger::LOG_DEBUG) <<
            "RK4Integrator::integrate(" << t0 << ", " << h << ")";

    // blas daxpy: y -> y + \alpha x
    integer n = stateVectorSize;
    integer inc = 1;
    double alpha = 0;

    model->setTime(t0);

    model->getStateVector(y);

    // k1 = f(t_n, y_n)
    model->getStateVectorRate(t0, y, k1);

    // k2 = f(t_n + h/2, y_n + (h/2) * k_1)
    alpha = h/2.;
    dcopy_(&n, y, &inc, ytmp, &inc);
    daxpy_(&n, &alpha, k1, &inc, ytmp, &inc);
    model->getStateVectorRate(t0 + alpha, ytmp, k2);

    // k3 = f(t_n + h/2, y_n + (h/2) * k_2)
    alpha = h/2.;
    dcopy_(&n, y, &inc, ytmp, &inc);
    daxpy_(&n, &alpha, k2, &inc, ytmp, &inc);
    model->getStateVectorRate(t0 + alpha, ytmp, k3);

    // k4 = f(t_n + h, y_n + (h) * k_3)
    alpha = h;
    dcopy_(&n, y, &inc, ytmp, &inc);
    daxpy_(&n, &alpha, k3, &inc, ytmp, &inc);
    model->getStateVectorRate(t0 + alpha, ytmp, k4);

    // k_1 = k_1 + 2 k_2
    alpha = 2.;
    daxpy_(&n, &alpha, k2, &inc, k1, &inc);

    // k_1 = (k_1 + 2 k_2) + 2 k_3
    alpha = 2.;
    daxpy_(&n, &alpha, k3, &inc, k1, &inc);

    // k_1 = (k_1 + 2 k_2 + 2 k_3) + k_4
    alpha = 1.;
    daxpy_(&n, &alpha, k4, &inc, k1, &inc);

    // y_{n+1} = (h/6)(k_1 + 2 k_2 + 2 k_3 + k_4);
    alpha = h/6.;

    daxpy_(&n, &alpha, k1, &inc, y, &inc);

    model->setTime(t0 + h);
    model->setStateVector(y);

    return t0 + h;
}

void RK4Integrator::restart(double t0)
{
}

void RK4Integrator::setListener(IntegratorListenerPtr)
{
}

IntegratorListenerPtr RK4Integrator::getListener()
{
    return IntegratorListenerPtr();
}

std::string RK4Integrator::toString() const
{
    return toRepr();
}

std::string RK4Integrator::toRepr() const
{
    std::stringstream ss;
    ss << "< roadrunner.RK4Integrator() { 'this' : "
            << (void*)this << " }>";
    return ss.str();
}

std::string RK4Integrator::getName() const
{
    return "rk4";
}

void RK4Integrator::setValue(const std::string& key,
        const rr::Variant& value)
{
    throw std::invalid_argument("invalid key");
}

Variant RK4Integrator::getValue(const std::string& key) const
{
    throw std::invalid_argument("invalid key");
}

bool RK4Integrator::hasKey(const std::string& key) const
{
    return false;
}

int RK4Integrator::deleteValue(const std::string& key)
{
    return -1;
}

std::vector<std::string> RK4Integrator::getKeys() const
{
    return std::vector<std::string>();
}

} /* namespace rr */
