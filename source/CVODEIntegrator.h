// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

// == FILEDOC =================================================

/** @file CVODEIntegrator.h
* @author WBC, ETS, MTK, JKM
* @copyright Apache License, Version 2.0
* @brief Contains the RoadRunner CVODE integrator interface
**/

//todo replace the include guards to match the filename
#ifndef rrCvodeInterfaceH
#define rrCvodeInterfaceH

// == INCLUDES ================================================

#include "Integrator.h"
#include "rrRoadRunnerOptions.h"

#include <string>
#include <vector>
#include <sundials/sundials_linearsolver.h>
#include <sundials/sundials_nonlinearsolver.h>

// == CODE ====================================================

/**
* CVode std::vector struct
*/
typedef struct _generic_N_Vector *N_Vector;

namespace rr
{
    using std::string;

    class ExecutableModel;
    class RoadRunner;

    /**
     * @author WBC, ETS, MTK
     * @brief A RoadRunner integrator based on CVODE; serves as RoadRunner's main integrator for ODEs
     * @details Provides a wrapper around CVODE from the SUNDIALS suite.
     * CVODE has two main solvers: an Adams-Moulton solver for non-stiff problems,
     * and a backward differentiation formula (BDF) solver for stiff problems.
     * See: https://computation.llnl.gov/casc/sundials/documentation/toms_sundials.pdf
     */
    class CVODEIntegrator : public Integrator
    {

    public:
        /**
         * @author WBC, ETS, MTK
         * @brief Constructor: takes an executable model, does not own the pointer
         */
        explicit CVODEIntegrator(ExecutableModel* oModel);

        /**
         * @author WBC, ETS, MTK
         * @brief Destructor
         */
        ~CVODEIntegrator() override;


        /**
        * @author JKM
        * @brief Called whenever a new model is loaded to allow integrator
        * to reset internal state
        */
        void syncWithModel(ExecutableModel* m) override;

        // ** Loading Settings *************************************************

        /**
         * It looks like this method only get used
         * inside resetSettings.
         */
        void loadConfigSettings() override;

        /**
         * @author WBC
         * @brief Load an SBML settings file and apply the configuration options
         * @note Can assign relative and absolute tolerances
         */
        void loadSBMLSettings(const std::string& filename) override;

        // ** Meta Info ********************************************************

        /**
         * @author WBC
         * @brief Get the name for this integrator
         * @note Delegates to @ref getName
         */
        std::string getName() const override;

        /**
         * @author JKM
         * @brief Get the name for this integrator
         */
        static std::string getCVODEIntegratorName();

        /**
         * @author WBC
         * @brief Get the description for this integrator
         * @note Delegates to @ref getDescription
         */
        std::string getDescription() const override;

        /**
         * @author JKM
         * @brief Get the description for this integrator
         */
        static std::string getCVODEIntegratorDescription();

        /**
         * @author WBC
         * @brief Get the hint for this integrator
         * @note Delegates to @ref getHint
         */
        std::string getHint() const override;

        /**
         * @author JKM
         * @brief Get the hint for this integrator
         */
        static std::string getCVODEIntegratorHint();

        // ** Getters / Setters ************************************************

        /**
         * @author WBC, ETS, MTK
         * @brief Always deterministic for CVODE
         */
        IntegrationMethod getIntegrationMethod() const override;

        /**
         * @author WBC, ETS, MTK
         * @brief Sets the value of an integrator setting (e.g. absolute_tolerance)
         */
        void setValue(const std::string& setting, Setting value) override;


		/**
		 * @author FY
		 * @brief Sets tolerance for individual species
		 */
		void setIndividualTolerance(std::string sid, double value) override;

		/**
		 * @author FY
		 * @brief Sets tolerance based on concentration of species
		 * @details First converts the concentration tolerances to amount tolerances
		 * by multiplying the compartment volume of species. Whichever is smaller
		 * will be stored as absolute_tolerance and used in the integration process.
		 */
		void setConcentrationTolerance(Setting value) override;

		/**
		 * @author FY
		 * @brief Gets tolerance based on concentration of species
		 */
		std::vector<double> getConcentrationTolerance() override;

        /**
        * @author JKM
        * @brief Reset all integrator settings to their respective default values
        */
        void resetSettings() override;


        /**
         * @author JKM, WBC, ETS, MTK
         * @brief Fix tolerances for SBML tests
         * @details In order to ensure that the results of the SBML test suite
         * remain valid, this method enforces a lower bound on tolerance values.
         * Sets minimum absolute and relative tolerances to
         * Config::CVODE_MIN_ABSOLUTE and Config::CVODE_MIN_RELATIVE resp.
         */
        void tweakTolerances() override;


        // ** Integration Routines *********************************************

        /**
         * @author WBC, ETS, MTK
         * @brief Main integration routine
         */
        double integrate(double t0, double hstep) override;

        /**
         * @author WBC, ETS, MTK
         * @brief Reset time to zero and reinitialize model
         * @details Applies events which occur before time zero.
         * Reinitializes CVODE and the executable model.
         */
        void restart(double timeStart) override;

        // ** Listeners ********************************************************

        /**
         * @author WBC, ETS
         * @brief Gets the integrator listener
         */
        IntegratorListenerPtr getListener() override;

        /**
         * @author WBC, ETS
         * @brief Sets the integrator listener
         */
        void setListener(IntegratorListenerPtr) override;

        /**
         * @author JKM
         * @brief Does a RT type check which throws if it fails, EVEN IF RTTI IS DISABLED
         */
        void checkType() const;


		/**
		* @author FY
		* @brief Does a size check which throws if it fails
		*/
		void checkVectorSize(int expected, size_t real) const;

		/**
		* @author FY
		* @brief Does a index check which throws if it is out of bound
		*/
		void checkIndex(int index, int size) const;

		/**
		* @author FY
		* @brief Converts integer to std::string for error print
		*/
		std::string ToString(int val) const;

        /**
        * @author LPS
        * @brief Converts size_t to std::string for error print
        */
        std::string ToString(size_t val) const;

        /**
         * @brief decode the cvode error code to a std::string
         */
        std::string cvodeDecodeError(int cvodeError, bool exInfo = true);


        /**
         * @brief getter for the internal state std::vector
         */
        N_Vector getStateVector() const;

        /**
         * @brief getter for the internal Sundials linear solver object
         */
        SUNNonlinearSolver getSolver() const;

        /**
         * @brief getter for the internal CVode memory buffer
         */
        void *getCvodeMemory() const;

    private:
        // defaults directly from CVODE docs
        static const int mDefaultMaxNumSteps;
        static const int mDefaultMaxAdamsOrder;
        static const int mDefaultMaxBDFOrder;

        // cvode components
        void* mCVODE_Memory;
        N_Vector mStateVector;
        SUNMatrix jac = nullptr;
        SUNNonlinearSolver nonLinSolver = nullptr;
        SUNLinearSolver  linSolver = nullptr;

        IntegratorListenerPtr listener;
        double lastEventTime;
        bool variableStepPendingEvent;
        bool variableStepTimeEndEvent;
		std::vector<double> variableStepPostEventState;
        std::vector<unsigned char> eventStatus;

        void testRootsAtInitialTime();
        bool haveVariables() const;
        void assignResultsToModel() const;
        /**
         * @author WBC, ETS, JKM
         * @brief Propagates changes in the "absolute_tolerance" and
         * "relative_tolerance" settings to the CVODE library.
         */
        void setCVODETolerances();

        /**
         * @brief Reinitialize sundials objects
         */
        void reInit(double t0);

        /**
         * @author WBC, ETS, JKM
         * @brief Propagates changes in the following settings to CVODE:\n
         * *  initial_time_step \n
         * *  minimum_time_step \n
         * *  maximum_time_step \n
         * *  maximum_num_steps \n
         * *  absolute_tolerance (via @ref setCVODETolerances) \n
         * *  relative_tolerance (via @ref setCVODETolerances) \n
         */
        void updateCVODE();

        void applyPendingEvents(double timeEnd);

        void applyEvents(double timeEnd, std::vector<unsigned char> &previousEventStatus);

        double applyVariableStepPendingEvents();

        void createCVode();

        void freeCVode();

        bool stateVectorVariables;

        unsigned long typecode_;

        friend int cvodeDyDtFcn(double t, N_Vector cv_y, N_Vector cv_ydot, void *f_data);

        friend int cvodeRootFcn(double t, N_Vector y, double *gout, void *g_data);



    };


    // ** Registration *********************************************************


    class CVODEIntegratorRegistrar : public IntegratorRegistrar {
        public:
            /**
            * @author JKM
            * @brief Gets the name associated with this integrator type
            */
            std::string getName() const override {
                return CVODEIntegrator::getCVODEIntegratorName();
            }

            /**
            * @author JKM
            * @brief Gets the description associated with this integrator type
            */
            std::string getDescription() const override {
                return CVODEIntegrator::getCVODEIntegratorDescription();
            }

            /**
            * @author JKM
            * @brief Gets the hint associated with this integrator type
            */
            std::string getHint() const override {
                return CVODEIntegrator::getCVODEIntegratorHint();
            }

            /**
            * @author JKM
            * @brief Constructs a new integrator of a given type
            */
            Integrator* construct(ExecutableModel *model) const override {
                return new CVODEIntegrator(model);
            }
    };
}

#endif
