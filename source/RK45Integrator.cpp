/*
 * RK45Integrator.cpp
 *
 *  Created on: Jul 22, 2014
 *      Author: andy
 */

#include <RK45Integrator.h>
#include <rrExecutableModel.h>

#include <cassert>
#include <math.h>

extern "C" {
#include <clapack/f2c.h>
#include <clapack/clapack.h>
}

namespace rr
{

    RK45Integrator::RK45Integrator(ExecutableModel *m)
    {
        Log(Logger::LOG_NOTICE) << "Creating Runge-Kutta-Fehlberg integrator";
        resetSettings();
        stateVectorSize = hmin = hmax = 0;
        k1 = k2 = k3 = k4 = k5 = k6 = err = y = ytmp = NULL;
        syncWithModel(m);
    }

    void RK45Integrator::syncWithModel(ExecutableModel* m)
    {
        // free existing memory
        delete []k1;
        delete []k2;
        delete []k3;
        delete []k4;
        delete []k5;
        delete []k6;
        delete []err;
        delete []y;
        delete []ytmp;

        model = m;

        resetSettings();

        if (model) {
            stateVectorSize = model->getStateVector(NULL);
            k1 = new double[stateVectorSize];
            k2 = new double[stateVectorSize];
            k3 = new double[stateVectorSize];
            k4 = new double[stateVectorSize];
            k5 = new double[stateVectorSize];
            k6 = new double[stateVectorSize];
            err = new double[stateVectorSize];
            y = new double[stateVectorSize];
            ytmp = new double[stateVectorSize];
            hmin = getValueAsDouble("minimum_time_step"); // TODO: replace hmin with getValueAsDouble("minimum_time_step")
            hmax = getValueAsDouble("maximum_time_step"); // TODO: replace hmax with getValueAsDouble("maximum_time_step")
        } else {
            stateVectorSize = hmin = hmax = 0;
            k1 = k2 = k3 = k4 = k5 = k6 = err = y = ytmp = NULL;
        }
    }

    RK45Integrator::~RK45Integrator()
    {
        delete []k1;
        delete []k2;
        delete []k3;
        delete []k4;
        delete []k5;
        delete []k6;
        delete []err;
        delete []y;
        delete []ytmp;
    }

    double RK45Integrator::integrate(double t, double tDiff)
    {
        double h = getValueAsDouble("maximum_time_step");

        if (!model) {
            throw std::runtime_error("RK45Integrator::integrate: No model");
        }

        Log(Logger::LOG_DEBUG) <<
                "RK45Integrator::integrate(" << t << ", " << h << ")";

        integer n = stateVectorSize;
        integer inc = 1;
        integer i;
        double alpha = 0;
        double error, q;

        do {
          assert(h > 1e-13);

          model->setTime(t);

          model->getStateVector(y);

          // k1 = f(t_n, y_n)
          model->getStateVectorRate(t, y, k1);

          // k2 = f(t_n + h/4, y_n + (h/4) * k_1)
          alpha = h/4.;
          dcopy_(&n, y, &inc, ytmp, &inc);
          daxpy_(&n, &alpha, k1, &inc, ytmp, &inc);
          model->getStateVectorRate(t + alpha, ytmp, k2);

          // k3 = f(t_n + 3*h/8, y_n + (3*h/32) * k_1 + (9*h/32) * k_2)
          alpha = 3*h/32.;
          dcopy_(&n, y, &inc, ytmp, &inc);
          daxpy_(&n, &alpha, k1, &inc, ytmp, &inc);
          alpha = 9*h/32.;
          daxpy_(&n, &alpha, k2, &inc, ytmp, &inc);
          alpha = 3*h/8.;
          model->getStateVectorRate(t + alpha, ytmp, k3);

          // k4 = f(t_n + 12*h/13, y_n + (1932*h/2197) * k_1 - (7200*h/2197) * k_2 + (7296*h/2197) * k_3)
          alpha = 1932*h/2197.;
          dcopy_(&n, y, &inc, ytmp, &inc);
          daxpy_(&n, &alpha, k1, &inc, ytmp, &inc);
          alpha = -7200*h/2197.;
          daxpy_(&n, &alpha, k2, &inc, ytmp, &inc);
          alpha = 7296*h/2197.;
          daxpy_(&n, &alpha, k3, &inc, ytmp, &inc);
          alpha = 12*h/13.;
          model->getStateVectorRate(t + alpha, ytmp, k4);

          // k5 = f(t_n + h, y_n + (439*h/216) * k_1 - (8*h) * k_2 + (3680*h/513) * k_3 - (845*h/4104) * k_4)
          alpha = 439*h/216.;
          dcopy_(&n, y, &inc, ytmp, &inc);
          daxpy_(&n, &alpha, k1, &inc, ytmp, &inc);
          alpha = -8.*h;
          daxpy_(&n, &alpha, k2, &inc, ytmp, &inc);
          alpha = 3680*h/513.;
          daxpy_(&n, &alpha, k3, &inc, ytmp, &inc);
          alpha = -845*h/4104.;
          daxpy_(&n, &alpha, k4, &inc, ytmp, &inc);
          alpha = h;
          model->getStateVectorRate(t + alpha, ytmp, k5);

          // k6 = f(t_n + h/2, y_n - (8*h/27) * k_1 + (2*h) * k_2 - (3544*h/2565) * k_3 + (1859*h/4104) * k_4 - (11*h/40) * k_5)
          alpha = -8*h/27.;
          dcopy_(&n, y, &inc, ytmp, &inc);
          daxpy_(&n, &alpha, k1, &inc, ytmp, &inc);
          alpha = 2.*h;
          daxpy_(&n, &alpha, k2, &inc, ytmp, &inc);
          alpha = -3544*h/2565.;
          daxpy_(&n, &alpha, k3, &inc, ytmp, &inc);
          alpha = 1859*h/4104.;
          daxpy_(&n, &alpha, k4, &inc, ytmp, &inc);
          alpha = -11*h/40.;
          daxpy_(&n, &alpha, k5, &inc, ytmp, &inc);
          alpha = h/2.;
          model->getStateVectorRate(t + alpha, ytmp, k6);

          // E = abs(k1/360 - (128/4275)*k3 - (2197/75240)*k4 + (1/50)*k5 + (2/55)*k6)
          for (i = 0; i < stateVectorSize; i++) {
            err[i] = 0.;
	  }
          alpha = 1./360;
          daxpy_(&n, &alpha, k1, &inc, err, &inc);
          alpha = -128./4275;
          daxpy_(&n, &alpha, k3, &inc, err, &inc);
          alpha = -2197./75240;
          daxpy_(&n, &alpha, k4, &inc, err, &inc);
          alpha = 1./50;
          daxpy_(&n, &alpha, k5, &inc, err, &inc);
          alpha = 2./55;
          daxpy_(&n, &alpha, k6, &inc, err, &inc);
          error = dnrm2_(&n, err, &inc);
          q = 0.84*pow(getValueAsDouble("epsilon")/error, 0.25);

          Log(Logger::LOG_DEBUG) <<
	    "RK45 step: t = " << t << ", error = " << error << ", epsilon = " << getValueAsDouble("epsilon") << ", h = " << h;
          if (q <= 0.1) {
            h = 0.1*h;
          } else if (q >= 4) {
            h = 4*h;
          } else {
            h = q*h;
          }

          if (h > hmax) { 
              h = hmax; 
          }

          if (t > tDiff + t) {
              return tDiff + t;
          }
          else if (h > tDiff) {
              h = tDiff;
          } else if (h < hmin) {
              h = hmin;
              //throw std::runtime_error("RK45Integrator::integrate: Stepsize became smaller than specified minimum.");
          }

        } while ( error > getValueAsDouble("epsilon"));

        Log(Logger::LOG_DEBUG) << "RK45: Update state vector";

        // y = y + (1408*h/2565) k_3
        alpha = 1408.*h / 2565;
        daxpy_(&n, &alpha, k3, &inc, y, &inc);

        // y = y + (1408/2565)*h k_3) + (2197/4104)*h k_4
        alpha = (2197. / 4104)*h;
        daxpy_(&n, &alpha, k4, &inc, y, &inc);

        // y = (y + (1408/2565)*h k_3 + (2197/4104)*h k_4) - (1/5)*h k_5
        alpha = (-1. / 5)*h;
        daxpy_(&n, &alpha, k5, &inc, y, &inc);

        // y = y + (25/216)*h k_1 + (1408/2565)*h k_3 + (2197/4104)*h k_4 - (1/5)*h k_5
        alpha = (25. / 216)*h;
        daxpy_(&n, &alpha, k1, &inc, y, &inc);

        model->setTime(t + h);
        model->setStateVector(y);

        for (int i = 0; i<stateVectorSize; ++i) {
            Log(Logger::LOG_DEBUG) << "  " << y[i];
        }

        Log(Logger::LOG_DEBUG) <<
                "RK45: end of step";

        return t + h;
    }

    void RK45Integrator::testRootsAtInitialTime()
    {
        std::vector<unsigned char> initialEventStatus(model->getEventTriggers(0, 0, 0), false);
        model->getEventTriggers(initialEventStatus.size(), 0, initialEventStatus.size() == 0 ? NULL : &initialEventStatus[0]);
        applyEvents(0, initialEventStatus);
    }

    void RK45Integrator::applyEvents(double timeEnd, std::vector<unsigned char> &previousEventStatus)
    {
        model->applyEvents(timeEnd, previousEventStatus.size() == 0 ? NULL : &previousEventStatus[0], y, y);
    }

    void RK45Integrator::restart(double t0)
    {
        if (!model) {
            return;
        }

        if (t0 <= 0.0) {
            if (y)
            {
                model->getStateVector(y);
            }

            testRootsAtInitialTime();
        }

        model->setTime(t0);

        // copy state vector into memory
        if (y)
        {
            model->getStateVector(y);
        }
    }

    void RK45Integrator::setListener(IntegratorListenerPtr)
    {
    }

    IntegratorListenerPtr RK45Integrator::getListener()
    {
        return IntegratorListenerPtr();
    }

    std::string RK45Integrator::getName() const {
        return RK45Integrator::getRK45Name();
    }

    std::string RK45Integrator::getRK45Name() {
        return "rk45";
    }

    std::string RK45Integrator::getDescription() const {
        return RK45Integrator::getRK45Description();
    }

    std::string RK45Integrator::getRK45Description() {
        return "Runge-Kutta-Fehlberg methods are a family of algorithms for solving "
			"ODEs. This integrator is based on explicit Runge-Kutta (4,5) formula, automatically "
			"determining adaptive stepsize.";
    }

    std::string RK45Integrator::getHint() const {
        return RK45Integrator::getRK45Hint();
    }

    std::string RK45Integrator::getRK45Hint() {
        return "Internal RK45 ODE solver";
    }

    Variant RK45Integrator::getValue(std::string key)
    {
        if (key == "variable_step_size")
            return true;
        else
            return Integrator::getValue(key);
    }

    Integrator::IntegrationMethod RK45Integrator::getIntegrationMethod() const
    {
        return Integrator::Deterministic;
    }

    void RK45Integrator::resetSettings()
    {
        Solver::resetSettings();

        addSetting("variable_step_size", true, "Variable Step Size", "Perform a variable time step simulation. (bool)", "(bool) Enabling this setting will allow the integrator to adapt the size of each time step. This will result in a non-uniform time column.");
        addSetting("minimum_time_step",  1e-12, "Minimum Time Step", "Specifies the minimum absolute value of step size allowed. (double)", "(double) The minimum absolute value of step size allowed.");
        addSetting("maximum_time_step",  1.0, "Maximum Time Step", "Specifies the maximum absolute value of step size allowed. (double)", "(double) The maximum absolute value of step size allowed.");
        addSetting("epsilon",  1e-12, "Maximum error tolerance", "Specifies the maximum error tolerance allowed. (double)", "(double) The maximum error tolerance allowed.");
    }

} /* namespace rr */
