/*
 * Integrator.h
 *
 *  Created on: Sep 7, 2013
 *      Author: andy
 */

#ifndef INTEGRATOR_H_
#define INTEGRATOR_H_

#include "rrException.h"
#include "rrLogger.h"
#include "rrOSSpecifics.h"
#include "rrRoadRunnerOptions.h"
#include "Dictionary.h"
#include <stdexcept>

# define RRINT_ENABLE_COMPLEX 1
# if RRINT_ENABLE_COMPLEX
# include <complex>
# endif

#if (__cplusplus >= 201103L) || defined(_MSC_VER)
#include <memory>
#define cxx11_ns std
#else
#include <tr1/memory>
#define cxx11_ns std::tr1
#endif

namespace rr
{

class Integrator;
class ExecutableModel;

/**
 * Listen for integrator events.
 *
 * These are called after the event occured or was processed.
 *
 * The internal integrator typically iterates for many internal time
 * steps, at variable step size for each external time step. So, if
 * RoadRunner::oneStep is called with a step size of say 10 time units, the internal
 * integrator may potentially integrate for 1, 5, 10 100 or some value. The
 * onTimeStep method is called for each of these internal time steps.
 *
 * The return values are currently a place holder and are ignored.
 */
class IntegratorListener
{
public:

    /**
     * is called after the internal integrator completes each internal time step.
     */
    virtual uint onTimeStep(Integrator* integrator, ExecutableModel* model, double time) = 0;

    /**
     * whenever model event occurs and after it is procesed.
     */
    virtual uint onEvent(Integrator* integrator, ExecutableModel* model, double time) = 0;

    virtual ~IntegratorListener() {};
};

/**
 * @author JKM
 * @brief A class for setting up the integration parameters
 */
class RR_DECLSPEC TimecourseIntegrationParameters {
public:
    typedef std::size_t size_type;

    virtual ~TimecourseIntegrationParameters();

    /// Grow the vector of selected time values by adding a new element
    virtual void addTimevalue(double t);

    /// Get the number of time values
    virtual size_type getTimevalueCount() const;

    /// Get the @a ith time value
    virtual double getTimevalue(size_type i) const;

    /**
     * Get the time values as a heap array
     * @note The caller must reclaim the memory using @a free
     */
    virtual double* getTimevaluesHeapArrayDbl() const;

    /**
     * Get the time values as a heap array
     * @note The caller must reclaim the memory using @a free
     */
    virtual float* getTimevaluesHeapArrayFlt() const;

protected:
    std::vector<double> t_;
};
typedef cxx11_ns::shared_ptr<TimecourseIntegrationParameters> TimecourseIntegrationParametersPtr;

/**
 * @author JKM
 * @brief Wraps access of values of integrand variables
 * @details Can technically be real or complex (complex not used ATM)
 */
class RR_DECLSPEC VariableValue {
protected:
    enum ValueSet {
        SET_REAL,
        SET_COMPLEX
    };
public:

    /// Construct with real value
    VariableValue(double real)
      : real_(real), set_(SET_REAL) {}

# if RRINT_ENABLE_COMPLEX
    /// Construct with complex value
    VariableValue(std::complex<double> complex)
      : complex_(complex), set_(SET_COMPLEX) {}
# endif

    double getReal() const {
        if (getValueSet() == SET_REAL)
            return real_;
        else
            throw CoreException("Wrong value type");
    }

# if RRINT_ENABLE_COMPLEX
    std::complex<double> getComplex() const {
        if (getValueSet() == SET_COMPLEX)
            return complex_;
        else
            throw CoreException("Wrong value type");
    }
# endif

protected:
    ValueSet getValueSet() const {
        return set_;
    }

    ValueSet set_;
    double real_;
# if RRINT_ENABLE_COMPLEX
    std::complex<double> complex_;
# endif
};

/**
 * @author JKM
 * @brief A class to encapsulate integration results
 * @details
 * Stores the value of each variable at each time point.
 * Provides an interface for interacting with
 * the integrator that abstracts away notions such as the
 * underlying integration data type (float / double) and
 * renders the API more stable.
 */
class RR_DECLSPEC TimecourseIntegrationResults {
public:
    typedef std::size_t size_type;

    virtual size_type getTimevalueCount() const = 0;

    /// Get the real value of the variable at index i at time point index ti
    virtual VariableValue getValue(size_type ti, size_type i) const = 0;
};
typedef cxx11_ns::shared_ptr<TimecourseIntegrationResults> TimecourseIntegrationResultsPtr;

/**
 * @brief Subclass of @ref TimecourseIntegrationResults for real vector results
 */
class RR_DECLSPEC TimecourseIntegrationResultsRealVector : public TimecourseIntegrationResults {
public:
    virtual size_type getTimevalueCount() const;
    virtual void setTimevalueCount(size_type n);

    virtual size_type getVectorLength() const;
    virtual void setVectorLength(size_type n);

    virtual VariableValue getValue(size_type ti, size_type i) const ;

    virtual void setValue(size_type ti, size_type i, double val);

protected:
    void rebuild();

    /// Number of time values
    size_type ntval_;
    /// Size of vector
    size_type veclen_;
    // not the most elegant solution...
    typedef std::vector< std::vector<double> > ValMatrix;
    ValMatrix val_;
};

/**
 * listeners are shared objects, so use std smart pointers
 * to manage them.
 */
typedef cxx11_ns::shared_ptr<IntegratorListener> IntegratorListenerPtr;

/**
 * Interface to a class which advances a model forward in time.
 *
 * The Integrator is only valid if attached to a model.
 */
class RR_DECLSPEC Integrator : public Dictionary
{
public:
    typedef std::size_t size_type;

    /**
     * Set the configuration parameters the integrator uses.
     */
    virtual void setSimulateOptions(const SimulateOptions* options) = 0;

    /**
     * integrates the model from t0 to t0 + hstep
     *
     * @return the final time value. This is typically very close to t0 + hstep,
     * but may be different if variableStep is used.
     */
    virtual double integrate(double t0, double hstep) = 0;

    /**
     * @brief Performs integration on the specified time values
     * @param[in] t The array of time values
     * @param[in] n The length of the array
     */
    virtual TimecourseIntegrationResultsPtr integrate(const TimecourseIntegrationParameters& p) {
        throw CoreException("Not implemented in this integrator");
    }

    /**
     * copies the state vector out of the model and into cvode vector,
     * re-initializes cvode.
     */
    virtual void restart(double t0) = 0;

    /**
     * the integrator can hold a single listener. If clients require multicast,
     * they can create a multi-cast listener.
     */
    virtual void setListener(IntegratorListenerPtr) = 0;

    /**
     * get the integrator listener
     */
    virtual IntegratorListenerPtr getListener() = 0;

    /**
     * get a description of this object, compatable with python __str__
     */
    virtual std::string toString() const = 0;

    /**
     * get a short descriptions of this object, compatable with python __repr__.
     */
    virtual std::string toRepr() const = 0;

    /**
     * get the name of this integrator
     */
    virtual std::string getName() const = 0;

    /**
     * this is an interface, provide virtual dtor as instances are
     * returned from New which must be deleted.
     */
    virtual ~Integrator() {};

    /**
     * create a new integrator based on the settings in the
     * options class.
     *
     * The new integrator borrows a reference to an ExecutableModel object.
     */
    static Integrator* New(const SimulateOptions *o, ExecutableModel *m);
};


class IntegratorException: public std::runtime_error
{
public:
    explicit IntegratorException(const std::string& what) :
            std::runtime_error(what)
    {
        Log(rr::Logger::LOG_ERROR) << __FUNC__ << "what: " << what;
    }

    explicit IntegratorException(const std::string& what, const std::string &where) :
            std::runtime_error(what + "; In " + where)
    {
        Log(rr::Logger::LOG_ERROR) << __FUNC__ << "what: " << what << ", where: " << where;
    }
};

}

#endif /* INTEGRATOR_H_ */
