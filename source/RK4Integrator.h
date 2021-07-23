/*
 * RK4Integrator.h
 *
 *  Created on: Jul 22, 2014
 *      Author: andy
 */

#ifndef RK4Integrator_H_
#define RK4Integrator_H_

#include <Integrator.h>
#include <rrRoadRunnerOptions.h>

namespace rr
{


    /**
    * A super basic 4'th order fixed step integrator.
    *
    * The RungeKuttaIntegrator will be the more sophisticated,
    * general purpose Runge-Kutta integrator which will support
    * different orders and adaptive time stepping.
    *
    * This object is mainly here as an example of creating a new Integrator.
    *
    */
    class RK4Integrator: public Integrator
    {
    public:

        using Integrator::Integrator;
        /**
        * Creates a new RK4Integrator.
        *
        * The IntegratorFactory is the ONLY object that creates integrators.
        *
        * Integrators are created when the IntegratorFactory::New method is called,
        * typically by the top level RoadRunner object.
        *
        * The integrator will hold onto the ExecutableModel pointer, m, and when the
        * integrate method is called, will advance the model object forward in time.
        *
        * @param m: a borrowed reference to an existing ExecutableModel object.
        * @param o: a reference to a SimulatOptions object where the configuration
        * parameters will be read from.
        */
        RK4Integrator(ExecutableModel *m);

        /**
        * clean up any mess.
        */
        ~RK4Integrator() override;

        /**
        * @author JKM
        * @brief Called whenever a new model is loaded to allow integrator
        * to reset internal state
        */
        void syncWithModel(ExecutableModel* m) override;


        /**
        * implement Integrator interface
        */
    public:

        /**
        * integrates the model from t0 to tf.
        */
        double integrate(double t0, double tf) override;

        /**
        * copies the state std::vector out of the model and into cvode std::vector,
        * re-initializes cvode.
        */
        void restart(double t0) override;

        // ** Meta Info ********************************************************

        /**
         * @author JKM
         * @brief Get the name for this integrator
         * @note Delegates to @ref getName
         */
        std::string getName() const override;

        /**
         * @author JKM
         * @brief Get the description for this integrator
         * @note Delegates to @ref getDescription
         */
        std::string getDescription() const override;

        /**
         * @author WBC
         * @brief Get the hint for this integrator
         * @note Delegates to @ref getHint
         */
        std::string getHint() const override;

        /**
         * @brief construct an instance of type RK4Integrator.
         * @details implements the Registrar interface. Used in
         * factory creation of Integrators.
         */
        Solver *construct(ExecutableModel *model) const override;

        // ** Getters / Setters ************************************************

        virtual Setting getValue(std::string key);

        /**
         * @author JKM
         * @brief Always deterministic for RK4
         */
        IntegrationMethod getIntegrationMethod() const  override;

        /**
        * @author JKM
        * @brief Reset all integrator settings to their respective default values
        */
        void resetSettings() override;

        // ** Listeners ********************************************************

        /**
        * the integrator can hold a single listener. If clients require multicast,
        * they can create a multi-cast listener.
        */
        void setListener(IntegratorListenerPtr) override;

        /**
        * get the integrator listener
        */
        IntegratorListenerPtr getListener() override;

    public:

        /**
        * set an arbitrary key
        */
//         virtual void setItem(const std::string& key, const rr::Setting& value);

        /**
        * get a value. Variants are POD.
        */
//         virtual Setting getItem(const std::string& key) const;

        /**
        * is there a key matching this name.
        */
//         virtual bool hasKey(const std::string& key) const;

        /**
        * remove a value
        */
//         virtual int deleteItem(const std::string& key);

        /**
        * list of keys in this object.
        */
//         virtual std::vector<std::string> getKeys() const;


    private:

        unsigned stateVectorSize;

        /**
        * arrays to store function eval values.
        */
        double *k1, *k2, *k3, *k4, *y, *ytmp;

        void testRootsAtInitialTime();
        void applyEvents(double timeEnd, std::vector<unsigned char> &previousEventStatus);

    };


    // ** Registration *********************************************************


//    class RK4IntegratorRegistrar : public Registrar {
//        public:
//            /**
//            * @author JKM
//            * @brief Gets the name associated with this integrator type
//            */
//            virtual std::string getName() const {
//                return RK4Integrator::getRK4Name();
//            }
//
//            /**
//            * @author JKM
//            * @brief Gets the description associated with this integrator type
//            */
//            virtual std::string getDescription() const {
//                return RK4Integrator::getRK4Description();
//            }
//
//            /**
//            * @author JKM
//            * @brief Gets the hint associated with this integrator type
//            */
//            virtual std::string getHint() const {
//                return RK4Integrator::getRK4Hint();
//            }
//
//            /**
//            * @author JKM
//            * @brief Constructs a new integrator of a given type
//            */
//            virtual Integrator* construct(ExecutableModel *model) const {
//                return new RK4Integrator(model);
//            }
//    };

} /* namespace rr */

#endif /* RK4Integrator_H_ */
