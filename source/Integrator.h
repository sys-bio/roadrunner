// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

// == FILEDOC =================================================

/** @file Integrator.h
* @author ETS, WBC, JKM
* @date Sep 7, 2013
* @copyright Apache License, Version 2.0
* @brief Contains the base class for RoadRunner integrators
**/

# ifndef RR_INTEGRATOR_H_
# define RR_INTEGRATOR_H_

#include "rrLogger.h"
#include "rrOSSpecifics.h"
#include "Dictionary.h"
#include "rrException.h"
#include "Solver.h"

#include "tr1proxy/rr_memory.h"
#include "tr1proxy/rr_unordered_map.h"
#include "Registrable.h"
#include "RegistrationFactory.h"
#include <stdexcept>

namespace rr {

    class Integrator;

    class ExecutableModel;

    /** @class IntegratorListener
     * IntegratorListener listens for integrator events.
     */
    class IntegratorListener {
    public:

        /**
        * is called after the internal integrator completes each internal time step.
        */
        virtual uint onTimeStep(Integrator *integrator, ExecutableModel *model, double time) = 0;

        /**
        * whenever model event occurs and after it is procesed.
        */
        virtual uint onEvent(Integrator *integrator, ExecutableModel *model, double time) = 0;

        virtual ~IntegratorListener() {};
    };

    typedef cxx11_ns::shared_ptr<IntegratorListener> IntegratorListenerPtr;

    /** @class Integrator 
     * Integrator is an abstract base class that provides an interface to specific integrator
     * class implementations.
    */
    class RR_DECLSPEC Integrator : public Solver {
    public:

        using Solver::Solver;

        /**
        * Pull down the setValue from superclass.
        * We do not need to reimplement this
        * but we make it explicit.
        */
        using Solver::setValue;

        enum IntegrationMethod {
            Deterministic,
            Stochastic,
            Hybrid,
            Other
        };

        explicit Integrator(ExecutableModel* model);

        Integrator();

        virtual ~Integrator() {};

        virtual IntegrationMethod getIntegrationMethod() const = 0;

        /**
        * @author JKM
        * @brief Called whenever a new model is loaded to allow integrator
        * to reset internal state
        */
        virtual void syncWithModel(ExecutableModel *m);

        virtual void loadConfigSettings();

        virtual void loadSBMLSettings(const std::string &filename);

        virtual double integrate(double t0, double hstep) = 0;

        virtual void restart(double t0) = 0;

        /**
         * @author JKM, WBC, ETS, MTK
         * @brief Fix tolerances for SBML tests
         * @details In order to ensure that the results of the SBML test suite
         * remain valid, this method enforces a lower bound on tolerance values.
         * Sets minimum absolute and relative tolerances to
         * Config::CVODE_MIN_ABSOLUTE and Config::CVODE_MIN_RELATIVE resp.
         */
        virtual void tweakTolerances();

        /**
        * @author FY
        * @brief Set tolerance for floating species or variables that have a rate rule, will only be used in CVODEIntegrator
        */
        virtual void setIndividualTolerance(std::string sid, double value);


        /**
         * @brief Get the absolute tolerance vector for the solver.
         *
         * Most solvers do not have absolute tolerances; this retrieves the vector for the ones (like CVODE) that do.
         */
        virtual std::vector<double> getAbsoluteToleranceVector();


        /* CARRYOVER METHODS */
        virtual void setListener(IntegratorListenerPtr) = 0;

        virtual IntegratorListenerPtr getListener() = 0;

        std::string toString() const;

        /**
        * @author JKM
        * @brief Return std::string representation a la Python __repr__ method
        */
        virtual std::string toRepr() const;
        /* !-- END OF CARRYOVER METHODS */

        void setIntegrationStartTime(double time);

    protected:
        double mIntegrationStartTime;

    };

    /** @class IntegratorException
    * The exception class thrown by the Integrator.
    */
    class IntegratorException : public std::runtime_error {
    public:
        explicit IntegratorException(const std::string &what) :
                std::runtime_error(what) {
        }

        explicit IntegratorException(const std::string &what, const std::string &where) :
                std::runtime_error(what + "; In " + where) {
        }
    };

}

# endif /* RR_INTEGRATOR_H_ */
