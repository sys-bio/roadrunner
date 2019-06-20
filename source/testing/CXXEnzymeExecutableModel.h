/*
 * CXXEnzymeExecutableModel.h
 *
 *  Created on: Dec 11, 2014
 *      Author: andy
 */

#ifndef CXXENZYMEEXECUTABLEMODEL_H_
#define CXXENZYMEEXECUTABLEMODEL_H_

#include <rrExecutableModel.h>
#include "Dictionary.h"
#include "rrSparse.h"

namespace rrtesting
{

class CXXEnzymeExecutableModel: public rr::ExecutableModel
{
public:
    CXXEnzymeExecutableModel(const rr::Dictionary* options);
    virtual ~CXXEnzymeExecutableModel();


public:

    /**
     * @author JKM
     * @date 07/31/2015
     * @brief Returns a human-readable description of the code generation backend,
     * e.g. LLVM, legacy C, etc.
     */
    virtual std::string getExecutableModelDesc() const {
        return "Enzyme Executable Model";
    }

    /**
     * get the name of the model
     */
    virtual std::string getModelName();

    virtual void setTime(double _time);
    virtual double getTime();


    /**
     * Loads the initial conditions into the current model state.
     *
     * Initial conditions may have been updated at any time externally.
     */
    virtual void reset();



    /************************ Floating Species Section ****************************/
    #if (1) /**********************************************************************/
    /******************************************************************************/

    /**
     * dependent species are defined by rules and the only way to change them
     * is by changing the values on which they depend.
     */
    virtual int getNumDepFloatingSpecies();

    /**
     * total number of floating species.
     */
    virtual int getNumFloatingSpecies();

    virtual int getFloatingSpeciesIndex(const std::string& eid);
    virtual std::string getFloatingSpeciesId(int index);

    /**
     * independent species do are not defined by rules, they typically participate
     * in reactions and can have thier values set at any time.
     */
    virtual int getNumIndFloatingSpecies();

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

    virtual int setFloatingSpeciesAmounts(int len, int const *indx,
            const double *values);

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

    /**
     * Set the initial concentrations of the floating species.
     *
     * Takes the same indices as the other floating species methods.
     *
     * Note, if a floating species has an initial assignment rule,
     * than the initial conditions value can only be set by
     * updating the values on which it depends, it can not be set
     * directly.
     */
    virtual int setFloatingSpeciesInitConcentrations(int len, int const *indx,
                double const *values);

    /**
     * Get the initial concentrations of the floating species,
     * uses the same indexing as the other floating species methods.
     */
    virtual int getFloatingSpeciesInitConcentrations(int len, int const *indx,
                    double *values);

    /**
     * Set the initial amounts of the floating species.
     *
     * Takes the same indices as the other floating species methods.
     *
     * Note, if a floating species has an initial assignment rule,
     * than the initial conditions value can only be set by
     * updating the values on which it depends, it can not be set
     * directly.
     */
    virtual int setFloatingSpeciesInitAmounts(int len, int const *indx,
                double const *values);

    /**
     * Get the initial amounts of the floating species,
     * uses the same indexing as the other floating species methods.
     */
    virtual int getFloatingSpeciesInitAmounts(int len, int const *indx,
                    double *values);

    /************************ End Floating Species Section ************************/
    #endif /***********************************************************************/
    /******************************************************************************/



    /************************ Boundary Species Section ****************************/
    #if (1) /**********************************************************************/
    /******************************************************************************/


    /**
     * get the number of boundary species.
     */
    virtual int getNumBoundarySpecies();
    virtual int getBoundarySpeciesIndex(const std::string &eid);
    virtual std::string getBoundarySpeciesId(int index);

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
     * set the boundary species concentrations
     *
     * @param[in] len the length of the indx and values arrays.
     * @param[in] indx an array of length len of boundary species to return.
     * @param[in] values an array of at least length len which will store the
     *                returned boundary species amounts.
     */
    virtual int setBoundarySpeciesConcentrations(int len, int const *indx,
            double const *values);


    /************************ End Boundary Species Section ************************/
    #endif /***********************************************************************/
    /******************************************************************************/


    /************************ Global Parameters Section ***************************/
    #if (1) /**********************************************************************/
    /******************************************************************************/

    /**
     * get the number of global parameters
     */
    virtual int getNumGlobalParameters();

    /**
     * index of the global parameter id, -1 if it does not exist.
     */
    virtual int getGlobalParameterIndex(const std::string& eid);

    /**
     * id of the indexed global parameter.
     */
    virtual std::string getGlobalParameterId(int index);

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

    /**
    * Set the initial value of the global parameter.
    *
    * Takes the same indices as the other global parameter methods.
    */
    virtual int setGlobalParameterInitValues(int len, int const *indx,
        double const *values);

    /**
    * Get the initial amounts of the global parameter,
    * uses the same indexing as the other global parameter methods.
    */
    virtual int getGlobalParameterInitValues(int len, int const *indx,
        double *values);


    /************************ Global Parameters Species Section *******************/
    #endif /***********************************************************************/
    /******************************************************************************/


    /************************ Compartments Section ********************************/
    #if (1) /**********************************************************************/
    /******************************************************************************/

    virtual int getNumCompartments();
	virtual int getCompartmentIndexForFloatingSpecies(int index);
    virtual int getCompartmentIndex(const std::string& eid);
    virtual std::string getCompartmentId(int index);

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

    virtual int setCompartmentVolumes(int len, int const *indx,
                const double *values);

    /**
     * Set the initial volumes of the compartments.
     *
     * Takes the same indices as the other compartment methods.
     *
     * Note, if a compartment has an initial assignment rule,
     * than the initial conditions value can only be set by
     * updating the values on which it depends, it can not be set
     * directly.
     */
    virtual int setCompartmentInitVolumes(int len, int const *indx,
                double const *values);

    /**
     * Get the initial volume of the compartments,
     * uses the same indexing as the other compartment methods.
     */
    virtual int getCompartmentInitVolumes(int len, int const *indx,
                    double *values);


    /************************ End Compartments Species Section ********************/
    #endif /***********************************************************************/
    /******************************************************************************/


    /************************ Selection Ids Species Section ***********************/
    #if (1) /**********************************************************************/
    /******************************************************************************/

    /**
     * populates a given list with all the ids that this class can accept.
     *
     * @param ids: a list of strings that will be filled by this class.
     * @param types: the types of ids that are requested. Can be set to
     * 0xffffffff to request all the ids that this class supports.
     * This should by a bitwise OR of the filelds in SelectionRecord::SelectionType
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

    /**
     * Get the current stiochiometry value for the given species / reaction.
     *
     * If either are not valid, NaN is returned.
     */
    virtual double getStoichiometry(int speciesIndex, int reactionIndex);


    virtual int getNumConservedMoieties();
    virtual int getConservedMoietyIndex(const std::string& eid);
    virtual std::string getConservedMoietyId(int index);
    virtual int getConservedMoietyValues(int len, int const *indx, double *values);
    virtual int setConservedMoietyValues(int len, int const *indx,
            const double *values);

    virtual int getNumRateRules();

    /**
     * get the number of reactions the model has
     */
    virtual int getNumReactions();

    /**
     * get the index of a named reaction
     * @returns >= 0 on success, < 0 on failure.
     */
    virtual int getReactionIndex(const std::string& eid);

    /**
     * get the name of the specified reaction
     */
    virtual std::string getReactionId(int index);

    /**
     * get the vector of reaction rates.
     *
     * @param len: the length of the suplied buffer, must be >= reaction rates size.
     * @param indx: pointer to index array. If NULL, then it is ignored and the
     * reaction rates are copied directly into the suplied buffer.
     * @param values: pointer to user suplied buffer where rates will be stored.
     */
    virtual int getReactionRates(int len, int const *indx,
                double *values);

    /**
     * get the 'values' i.e. the what the rate rule integrates to, and
     * store it in the given array.
     *
     * The length of rateRuleValues obviously must be the number of
     * rate rules we have.
     */
    virtual void getRateRuleValues(double *rateRuleValues);

    /**
     * get the id of an element of the state vector.
     */
    virtual std::string getStateVectorId(int index);

    /**
     * The state vector is a vector of elements that are defined by
     * differential equations (rate rules) or independent floating species
     * are defined by reactions.
     *
     * To get the ids of the state vector elements, use getStateVectorId.
     *
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

    /**
     * the state vector y is the rate rule values and floating species
     * concentrations concatenated. y is of length numFloatingSpecies + numRateRules.
     *
     * The state vector is packed such that the first n raterule elements are the
     * values of the rate rules, and the last n floatingspecies are the floating
     * species values.
     *
     * @param[in] time current simulator time
     * @param[in] y state vector, must be either null, or have a size of that
     *         speciefied by getStateVector. If y is null, then the model is
     *         evaluated using its current state. If y is not null, then the
     *         y is considered the state vector.
     * @param[out] dydt calculated rate of change of the state vector, if null,
     *         it is ignored.
     */
    virtual void getStateVectorRate(double time, const double *y, double* dydt=0);

    virtual void testConstraints();

    virtual std::string getInfo();

    virtual void print(std::ostream &stream);

    /******************************* Events Section *******************************/
    #if (1) /**********************************************************************/
    /******************************************************************************/

    virtual int getNumEvents();

    /**
     * get the event status, false if the even is not triggered, true if it is.
     *
     * The reason this returns an unsigned char instead of a bool array is this
     * array is typically stuffed into an std::vector, and std::vector<bool> is
     * well, weird as it's actually implemented as a bitfield, and can not be
     * used as a C array.
     *
     * So, on every modern system I'm aware of, bool is an unsigned char, so
     * use that data type here.
     */
    virtual int getEventTriggers(int len, const int *indx, unsigned char *values);

    virtual int applyEvents(double timeEnd, const unsigned char* previousEventStatus,
                const double *initialState, double* finalState);

    virtual int applyPendingEvents(double timeEnd);

    /**
     * evaluate the event 'roots' -- when events transition form triggered - non-triggered
     * or triggered to non-triggered state.
     *
     * Simplest method is to return 1 for triggered, -1 for not-triggered, so long
     * as there is a zero crossing.
     *
     * @param time[in] current time
     * @param y[in] the state vector
     * @param gdot[out] result event roots, this is of length numEvents.
     */
    virtual void getEventRoots(double time, const double* y, double* gdot);

    virtual double getNextPendingEventTime(bool pop);

    virtual int getPendingEventSize();

    virtual void resetEvents();


    /******************************* Events Section *******************************/
     #endif /**********************************************************************/
    /******************************************************************************/


    /**
     * Gets the index for an event id.
     * If there is no event with this id, returns -1.
     */
    virtual int getEventIndex(const std::string& eid);
    virtual std::string getEventId(int index);
    virtual void setEventListener(int index, rr::EventListenerPtr eventHandler);
    virtual rr::EventListenerPtr getEventListener(int index);

    /**
     * Get the amount rate of change for the i'th floating species
     * given a reaction rates vector.
     *
     * TODO: This should be merged with getFloatingSpeciesAmountRates, but that will
     * break inteface, will do in next point release.
     *
     * TODO: If the conversion factor changes in between getting the
     * reaction rates vector via getReactionRates
     *
     * @param index: index of the desired floating speceis rate.
     * @param reactionRates: pointer to buffer of reaction rates.
     */
    virtual double getFloatingSpeciesAmountRate(int index,
            const double *reactionRates);

    /**
     * reset the model accordign to a bitfield specified by the
     * SelectionRecord::SelectionType values.
     */
    virtual void reset(int options);

    /**
     * set the seed used by the random number generator. This will by definition
     * reset the RNG.
     */
    virtual void setRandomSeed(int64_t);

    /**
     * get the seed used by the RNG.
     */
    virtual int64_t getRandomSeed();

    /**
     * Get a uniform random number between 0 and 1 created by the RNG.
     *
     * The type of RNG can be specified in the config file with the
     * RANDOM_GENERATOR_TYPE key.
     */
    virtual double getRandom();

    /**
     * Get the current set of flags
     */
    virtual uint32_t getFlags() const;

    /**
     * Set certain options that determine the state of the ExecutableModel,
     * these are listed in
     */
    virtual void setFlags(uint32_t);

    double time;

    double *reactionRates;

    double *floatingSpeciesAmountRates;

    double *floatingSpeciesAmounts;

    double *volumes;

    double evalReactionRates();

    /**
     * stoichiometry matrix
     */
    rr::csr_matrix*                     stoichiometry;                    // 13


    double Km1;
    double Km2;
    double Km3;
    double sigma;
    double Keq;
    double h;
    double k1;
    double Vm;
    double M;


    // number of source species, source -> product
    int numSource;
    int numSpecies;
    int numReactions; // same as num source
};

} /* namespace rrtesting */

#endif /* CXXENZYMEEXECUTABLEMODEL_H_ */
