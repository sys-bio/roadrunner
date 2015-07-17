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
#include "Dictionary.h"
#include "tr1proxy/rr_memory.h"
#include <stdexcept>

# define RRINT_ENABLE_COMPLEX 1
# if RRINT_ENABLE_COMPLEX
# include <complex>
# endif


namespace rr
{

	class Integrator;
	class ExecutableModel;

	/*-------------------------------------------------------------------------------------------
		IntegratorListener listens for integrator events.
	---------------------------------------------------------------------------------------------*/
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

 * @author JKM
 * @brief A class for setting up the integration parameters
 */
class RR_DECLSPEC TimecourseIntegrationParameters {
public:
    typedef std::size_t size_type;

    virtual ~TimecourseIntegrationParameters();

    enum Precision {
        SINGLE,
        DOUBLE
    };

    /// Set the numeric precision
    virtual void setPrecision(Precision p);

    /// Set the numeric precision
    virtual Precision getPrecision() const;

    /// Empty ctor
    TimecourseIntegrationParameters()
     : prec_(SINGLE) {

    }

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
    Precision prec_;
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
	typedef cxx11_ns::shared_ptr<IntegratorListener> IntegratorListenerPtr;
	typedef std::unordered_map<std::string, std::string> HintMap;
	typedef std::unordered_map<std::string, std::string> DescriptionMap;

	/*-------------------------------------------------------------------------------------------
		Integrator is an abstract base class that provides an interface to specific integrator
		class implementations.
	---------------------------------------------------------------------------------------------*/
	class RR_DECLSPEC Integrator
	{
	public:
    typedef std::size_t size_type;
		enum IntegrationMethod
		{
			Deterministic,
			Stochastic,
			Hybrid,
			Other
		};

		virtual ~Integrator() {};

		virtual void loadConfigSettings();
		virtual void loadSBMLSettings(const std::string& filename);
		virtual std::string getIntegratorName() const = 0;
		virtual std::string getIntegratorDescription() const = 0;
		virtual std::string getIntegratorHint() const = 0;
		virtual IntegrationMethod getIntegrationMethod() const = 0;
		std::vector<std::string> getSettings();

    /**
     * @brief Performs integration on the specified time values
     * @param[in] t The array of time values
     * @param[in] n The length of the array
     */
    virtual TimecourseIntegrationResultsPtr integrate(const TimecourseIntegrationParameters& p) {
        throw CoreException("Not implemented in this integrator");
    }

		virtual Variant getValue(std::string key);
		virtual int getValueAsInt(std::string key);
		virtual unsigned int getValueAsUInt(std::string key);
		virtual long getValueAsLong(std::string key);
		virtual unsigned long getValueAsULong(std::string key);
		virtual float getValueAsFloat(std::string key);
		virtual double getValueAsDouble(std::string key);
		virtual char getValueAsChar(std::string key);
		virtual unsigned char getValueAsUChar(std::string key);
		virtual std::string getValueAsString(std::string key);
		virtual bool getValueAsBool(std::string key);
		virtual void setValue(std::string key, const Variant& value) = 0;
		const std::string& getHint(std::string key) const;
		const std::string& getDescription(std::string key) const;
		const Variant::TypeId getType(std::string key);

		virtual double integrate(double t0, double hstep) = 0;
		virtual void restart(double t0) = 0;

		/* CARRYOVER METHODS */
		virtual void setListener(IntegratorListenerPtr) = 0;
		virtual IntegratorListenerPtr getListener() = 0;
		std::string toString() const;
		/* !-- END OF CARRYOVER METHODS */

	protected:
		std::unordered_map<std::string, Variant> settings;
		HintMap hints;
		DescriptionMap descriptions;

		void AddSetting(std::string name, Variant val, std::string hint, std::string description);
	};


	class IntegratorException : public std::runtime_error
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

	/* */
	class RR_DECLSPEC IntegratorFactory
	{
	public:

		static Integrator* New(std::string name, ExecutableModel *m);

	};

}

#endif /* INTEGRATOR_H_ */
