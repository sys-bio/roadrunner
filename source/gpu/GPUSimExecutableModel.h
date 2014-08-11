/*
 * GPUSimExecutableModel.h
 *
 *  Created on: Jun 3, 2013
 *
 * Author: Andy Somogyi,
 *     email decode: V1 = "."; V2 = "@"; V3 = V1;
 *     andy V1 somogyi V2 gmail V3 com
 */

#ifndef GPUSimExecutableModelH
#define GPUSimExecutableModelH

#include "rrExecutableModel.h"

// #include "EvalInitialConditionsCodeGen.h"
// #include "EvalReactionRatesCodeGen.h"
// #include "EvalRateRuleRatesCodeGen.h"
// #include "GetValuesCodeGen.h"
// #include "GetInitialValuesCodeGen.h"
// #include "GetEventValuesCodeGen.h"
// #include "EventAssignCodeGen.h"
// #include "EventTriggerCodeGen.h"
// #include "EvalVolatileStoichCodeGen.h"
// #include "EvalConversionFactorCodeGen.h"
// #include "SetValuesCodeGen.h"
// #include "SetInitialValuesCodeGen.h"
// #include "EventQueue.h"
#include "rrSelectionRecord.h"


#if (__cplusplus >= 201103L) || defined(_MSC_VER)
#include <memory>
#include <unordered_map>
#define cxx11_ns std
#else
#include <tr1/memory>
#include <tr1/unordered_map>
#define cxx11_ns std::tr1
#endif

#include <map>

namespace rrgpu
{

class ModelResources;

class RR_DECLSPEC GPUSimExecutableModel: public rr::ExecutableModel
{
public:

    /**
     * the default ctor just zeros out all our private bits, then
     * the main construction is handled by the model generator.
     */
    GPUSimExecutableModel();


    virtual ~GPUSimExecutableModel();

    /**
     * get the name of the model
     */
    virtual std::string getModelName();
    virtual void setTime(double _time);
    virtual double getTime();

    virtual bool getConservedSumChanged();

    virtual void setConservedSumChanged(bool);

    /**
     * evaluate the initial conditions specified in the sbml, this entails
     * evaluating all InitialAssigments, AssigmentRules, initial values, etc...
     *
     * The the model state is fully set.
     */
    virtual void evalInitialConditions();

    /**
     * reset the model to its original state
     */
    virtual void reset();



    virtual int getNumIndFloatingSpecies();
    virtual int getNumDepFloatingSpecies();

    virtual int getNumFloatingSpecies();
    virtual int getNumBoundarySpecies();
    virtual int getNumGlobalParameters();

    virtual int getNumCompartments();

    /**
     * get the global parameter values
     *
     * @param[in] len the length of the indx and values arrays.
     * @param[in] indx an array of length len of boundary species to return.
     * @param[out] values an array of at least length len which will store the
     *                returned boundary species amounts.
     */
    virtual int getGlobalParameterValues(int len, int const *indx,
            double *values);

    virtual int setGlobalParameterValues(int len, int const *indx,
            const double *values);

    virtual int getNumReactions();

    virtual int getReactionRates(int len, int const *indx,
                    double *values);

    /**
     * get the compartment volumes
     *
     * @param[in] len the length of the indx and values arrays.
     * @param[in] indx an array of length len of boundary species to return.
     * @param[out] values an array of at least length len which will store the
     *                returned boundary species amounts.
     */
    virtual int getCompartmentVolumes(int len, int const *indx,
            double *values);

    virtual int getNumRules();



    virtual int getNumLocalParameters(int reactionId);


    virtual void convertToAmounts();
    virtual void computeConservedTotals();


    /**
     * copy (but do not evaluate) existing rate rules values into
     * a buffer.
     */
    virtual void getRateRuleValues(double *rateRuleValues);


    virtual std::string getStateVectorId(int index);

    /**
     * copies the internal model state vector into the provided
     * buffer.
     *
     * @param[out] stateVector a buffer to copy the state vector into, if NULL,
     *         return the size required.
     *
     * @return the number of items coppied into the provided buffer, if
     *         stateVector is NULL, returns the length of the state vector.
     */
    virtual int getStateVector(double *stateVector);

    /**
     * sets the internal model state to the provided packed state vector.
     *
     * @param[in] an array which holds the packed state vector, must be
     *         at least the size returned by getStateVector.
     *
     * @return the number of items copied from the state vector, negative
     *         on failure.
     */
    virtual int setStateVector(const double *stateVector);

    const rr::SelectionRecord& getSelection(const std::string& sel);

    virtual void convertToConcentrations();
    virtual void updateDependentSpeciesValues();
    virtual void computeAllRatesOfChange();


    /**
     * where most of the juicy bits occur.
     *
     * the state vector y is the rate rule values and floating species
     * concentrations concatenated. y is of length numFloatingSpecies + numRateRules.
     *
     * The state vector is packed such that the first n raterule elements are the
     * values of the rate rules, and the last n floatingspecies are the floating
     * species values.
     *
     * @param[in] time current simulator time
     * @param[in] y state vector, must be of size returned by getStateVector
     * @param[out] dydt calculated rate of change of the state vector, if null,
     * it is ignored.
     */
    virtual void getStateVectorRate(double time, const double *y, double* dydt=0);


    virtual void testConstraints();

    virtual std::string getInfo();

    virtual int getFloatingSpeciesIndex(const std::string&);
    virtual std::string getFloatingSpeciesId(int);
    virtual int getBoundarySpeciesIndex(const std::string&);
    virtual std::string getBoundarySpeciesId(int);

    /**
     * get the floating species amounts
     *
     * @param[in] len the length of the indx and values arrays.
     * @param[in] indx an array of length len of boundary species to return.
     * @param[out] values an array of at least length len which will store the
     *                returned boundary species amounts.
     */
    virtual int getFloatingSpeciesAmounts(int len, int const *indx,
            double *values);

    virtual double getFloatingSpeciesAmountRate(int index,
            const double *reactionRates);

    virtual int getFloatingSpeciesAmountRates(int len, int const *indx,
            double *values);

    virtual int getFloatingSpeciesConcentrationRates(int len, int const *indx,
            double *values);

    /**
     * get the floating species concentrations
     *
     * @param[in] len the length of the indx and values arrays.
     * @param[in] indx an array of length len of boundary species to return.
     * @param[out] values an array of at least length len which will store the
     *                returned boundary species amounts.
     */
    virtual int getFloatingSpeciesConcentrations(int len, int const *indx,
            double *values);


    /**
     * set the floating species concentrations
     *
     * @param[in] len the length of the indx and values arrays.
     * @param[in] indx an array of length len of boundary species to return.
     * @param[in] values an array of at least length len which will store the
     *                returned boundary species amounts.
     */
    virtual int setFloatingSpeciesConcentrations(int len, int const *indx,
            double const *values);

    virtual int setFloatingSpeciesAmounts(int len, int const *indx,
            const double *values);

    /**
     * get the boundary species amounts
     *
     * @param[in] len the length of the indx and values arrays.
     * @param[in] indx an array of length len of boundary species to return.
     * @param[out] values an array of at least length len which will store the
     *                returned boundary species amounts.
     */
    virtual int getBoundarySpeciesAmounts(int len, int const *indx,
            double *values);

    /**
     * get the boundary species concentrations
     *
     * @param[in] len the length of the indx and values arrays.
     * @param[in] indx an array of length len of boundary species to return.
     * @param[out] values an array of at least length len which will store the
     *                returned boundary species amounts.
     */
    virtual int getBoundarySpeciesConcentrations(int len, int const *indx,
            double *values);

    /**
     * get the boundary species concentrations
     *
     * @param[in] len the length of the indx and values arrays.
     * @param[in] indx an array of length len of boundary species to return.
     * @param[in] values an array of at least length len which will store the
     *                returned boundary species amounts.
     */
    virtual int setBoundarySpeciesConcentrations(int len, int const *indx,
            double const *values);


    /**
      * get the boundary species amounts
      *
      * @param[in] len the length of the indx and values arrays.
      * @param[in] indx an array of length len of boundary species to return.
      * @param[in] values an array of at least length len which will store the
      *                returned boundary species amounts.
      */
     virtual int setBoundarySpeciesAmounts(int len, int const *indx,
             double const *values);


    virtual int getGlobalParameterIndex(const std::string&);
    virtual std::string getGlobalParameterId(int);
    virtual int getCompartmentIndex(const std::string&);
    virtual std::string getCompartmentId(int);
    virtual int getReactionIndex(const std::string&);
    virtual std::string getReactionId(int);

    virtual void print(std::ostream &stream);

    virtual int getNumConservedMoieties();
    virtual int getConservedMoietyIndex(const std::string& name);
    virtual std::string getConservedMoietyId(int index);
    virtual int getConservedMoietyValues(int len, int const *indx, double *values);
    virtual int setConservedMoietyValues(int len, int const *indx,
            const double *values);


    /**
     * using the current model state, evaluate and store all the reaction rates.
     */
    virtual void evalReactionRates();


    virtual int setCompartmentVolumes(int len, int const *indx,
            const double *values);


    virtual double getStoichiometry(int speciesIndex, int reactionIndex);

    /**
     * allocate a block of memory and copy the stochiometric values into it,
     * and return it.
     *
     * The caller is responsible for freeing the memory that is referenced by data.
     *
     * @param[out] rows will hold the number of rows in the matrix.
     * @param[out] cols will hold the number of columns in the matrix.
     * @param[out] data a pointer which will hold a newly allocated memory block.
     */
    virtual int getStoichiometryMatrix(int* rows, int* cols, double** data);


    /******************************* Initial Conditions Section *******************/
    #if (1) /**********************************************************************/
    /******************************************************************************/

    virtual int setFloatingSpeciesInitConcentrations(int len, const int *indx,
            double const *values);

    virtual int getFloatingSpeciesInitConcentrations(int len, const int *indx,
            double *values);

    virtual int setFloatingSpeciesInitAmounts(int len, const int *indx,
                double const *values);

    virtual int getFloatingSpeciesInitAmounts(int len, const int *indx,
                    double *values);

    virtual int setCompartmentInitVolumes(int len, const int *indx,
                double const *values);

    virtual int getCompartmentInitVolumes(int len, const int *indx,
                    double *values);

    virtual int setGlobalParameterInitValues(int len, const int *indx,
                double const *values);

    virtual int getGlobalParameterInitValues(int len, const int *indx,
                    double *values);

    /******************************* End Initial Conditions Section ***************/
    #endif /***********************************************************************/
    /******************************************************************************/


    /************************ Selection Ids Species Section ***********************/
    #if (1) /**********************************************************************/
    /******************************************************************************/

    /**
     * populates a given list with all the ids that this class can accept.
     */
    virtual void getIds(int types, std::list<std::string> &ids);

    /**
     * returns a bit field of the ids that this class supports.
     */
    virtual int getSupportedIdTypes();

    /**
     * gets the value for the given id string. The string must be a SelectionRecord
     * string that is accepted by this class.
     */
    virtual double getValue(const std::string& id);

    /**
     * sets the value coresponding to the given selection stringl
     */
    virtual void setValue(const std::string& id, double value);

    /************************ End Selection Ids Species Section *******************/
    #endif /***********************************************************************/
    /******************************************************************************/


private:

    template <typename a_type, typename b_type>
    friend void copyCachedModel(a_type* src, b_type* dst);
};

} /* namespace rrgpu */
#endif /* GPUSimExecutableModelH */
