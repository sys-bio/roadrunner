// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

// == FILEDOC =================================================

/** @file LLVMExecutableModel.h
* @author ETS
* @copyright Apache License, Version 2.0
* @date Jun 3, 2013
* @brief LLVM executable model interface
**/

/*
 * Author: Andy Somogyi,
 *     email decode: V1 = "."; V2 = "@"; V3 = V1;
 *     andy V1 somogyi V2 gmail V3 com
 */

#ifndef LLVMExecutableModelH
#define LLVMExecutableModelH

#include "rrExecutableModel.h"
#include "LLVMModelDataSymbols.h"
#include "LLVMIncludes.h"

#include "EvalInitialConditionsCodeGen.h"
#include "EvalReactionRatesCodeGen.h"
#include "EvalRateRuleRatesCodeGen.h"
#include "GetValuesCodeGen.h"
#include "GetInitialValuesCodeGen.h"
#include "GetEventValuesCodeGen.h"
#include "EventAssignCodeGen.h"
#include "EventTriggerCodeGen.h"
#include "EvalVolatileStoichCodeGen.h"
#include "EvalConversionFactorCodeGen.h"
#include "SetValuesCodeGen.h"
#include "SetInitialValuesCodeGen.h"
#include "EventQueue.h"
#include "rrSelectionRecord.h"

#include "tr1proxy/rr_memory.h"
#include "tr1proxy/rr_unordered_map.h"
#include <map>

namespace rrllvm
{

class ModelResources;

/**
 * @author ETS
 * @brief LLVM executable model
 * @details Interface to LLVM executable model.
 */
class RR_DECLSPEC LLVMExecutableModel: public rr::ExecutableModel
{
public:

    /**
     * @author JKM
     * @date 07/31/2015
     * @brief Returns a human-readable description of the code generation backend,
     * e.g. LLVM, legacy C, etc.
     */
    virtual std::string getExecutableModelDesc() const {
        return "LLVM Executable Model";
    }

    /**
     * the default ctor just zeros out all our private bits, then
     * the main construction is handled by the model generator.
     */
    LLVMExecutableModel();

    /**
     * takes ownership of the LLVMModelData pointer.
     */
    LLVMExecutableModel(const cxx11_ns::shared_ptr<ModelResources> &resources,
            LLVMModelData* modelData);

	/*
	* Loads a saved executable model
	*/
	LLVMExecutableModel(std::istream& in, uint modelGeneratorOpt);

    virtual ~LLVMExecutableModel();

    /**
     * get the name of the model
     */
    virtual string getModelName();
    virtual void setTime(double _time);
    virtual double getTime();

    /**
     * evaluate the initial conditions specified in the sbml, this entails
     * evaluating all InitialAssigments, AssigmentRules, initial values, etc...
     *
     * The the model state is fully set.
     */
    void evalInitialConditions(uint32_t flags = 0);

    /**
     * call reset(options) with defaut options.
     */
    virtual void reset();

    /**
     * reset the model accordign to a bitfield specified by the
     * SelectionRecord::SelectionType values.
     */
    virtual void reset(int options);



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

    virtual int getNumRateRules();

    /**
     * @author JKM
     * @date 07/31/2015
     * @brief Gets the symbols defined by rate rules, i.e.
     * returns all x such that x' = f(x) is a rule which defines parameter x.
     */
    virtual std::vector<std::string> getRateRuleSymbols() const;

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

    virtual string getInfo();

    virtual int getFloatingSpeciesIndex(const string&);
    virtual string getFloatingSpeciesId(int);
    virtual int getBoundarySpeciesIndex(const string&);
    virtual string getBoundarySpeciesId(int);

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


    virtual int getGlobalParameterIndex(const string&);
    virtual string getGlobalParameterId(int);
    virtual int getCompartmentIndex(const string&);
    virtual string getCompartmentId(int);
    virtual int getReactionIndex(const string&);
    virtual string getReactionId(int);

    virtual void print(std::ostream &stream);

    virtual int getNumConservedMoieties();
    virtual int getConservedMoietyIndex(const string& name);
    virtual string getConservedMoietyId(int index);
    virtual int getConservedMoietyValues(int len, int const *indx, double *values);
    virtual int setConservedMoietyValues(int len, int const *indx,
            const double *values);


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

    /******************************* Random Section *******************************/
    #if (1) /**********************************************************************/
    /******************************************************************************/

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

    /******************************* End Random Section ***************************/
    #endif  /**********************************************************************/
    /******************************************************************************/


    /******************************* Events Section *******************************/
    #if (1) /**********************************************************************/
    /******************************************************************************/
public:
    virtual int getNumEvents();

    /**
     * @brief Get status of event triggers
     * @author ETS
     * @details When len <= 0, returns number of events.
     * Otherwise, sets bits in @ref eventState to the corresponding trigger.
     * @param[in] len The length of @ref indx
     * @param[in] indx An index into @ref eventState
     * @param[out] eventState The state of the event triggers
     */
    virtual int getEventTriggers(int len, const int *indx, unsigned char *eventState);

    /**
     * erases expired events, and applies ripe events as long as we have ripe
     * events.
     */
    virtual int applyEvents(double timeEnd, const unsigned char* previousEventState,
            const double *initialState, double* finalState);

    virtual void getEventRoots(double time, const double* y, double* gdot);

    virtual double getNextPendingEventTime(bool pop);

    virtual int getPendingEventSize();

    virtual void resetEvents();

    inline double getEventDelay(uint event)
    {
        return getEventDelayPtr(modelData, event);
    }

    inline double getEventPriority(uint event)
    {
        return getEventPriorityPtr(modelData, event);
    }

    inline bool getEventTrigger(uint event)
    {
        assert(event < symbols->getEventAttributes().size()
                        && "event out of bounds");
        if (modelData->time >= 0.0)
        {
            return getEventTriggerPtr(modelData, event);
        }
        else
        {
            return symbols->getEventAttributes()[event] & EventInitialValue
                    ? true : false;
        }
    }

    inline bool getEventUseValuesFromTriggerTime(uint event)
    {
        assert(event < symbols->getEventAttributes().size()
            && "event out of bounds");
        return symbols->getEventAttributes()[event]
                & EventUseValuesFromTriggerTime;
    }

    inline bool getEventInitialValue(uint event)
    {
        assert(event < symbols->getEventAttributes().size()
            && "event out of bounds");
        return symbols->getEventAttributes()[event] & EventInitialValue;
    }

    inline bool getEventPersistent(uint event)
    {
        assert(event < symbols->getEventAttributes().size()
            && "event out of bounds");
        return symbols->getEventAttributes()[event] & EventPersistent;
    }

    inline uint getEventBufferSize(uint event)
    {
        return symbols->getEventBufferSize(event);
    }

    inline void getEventData(uint eventId, double* data)
    {
        eventTriggerPtr(modelData, eventId, data);
    }

    /**
     * assign or apply the event using the given data.
     */
    inline void assignEvent(uint eventId, double* data)
    {
        // apply the sbml JITed event assignments
        eventAssignPtr(modelData, eventId, data);

        const rr::EventListenerPtr &handler = eventListeners[eventId];
        if(handler)
        {
            uint result = handler->onAssignment(this, eventId, symbols->getEventId(eventId));

            if(result & rr::EventListener::HALT_SIMULATION) {
                throw rr::EventListenerException(result);
            }
        }
    }

    bool getEventTieBreak(uint eventA, uint eventB);

    virtual int getEventIndex(const std::string& eid);
    virtual std::string getEventId(int index);
    virtual void setEventListener(int index, rr::EventListenerPtr eventHandler);
    virtual rr::EventListenerPtr getEventListener(int index);


    virtual double getFloatingSpeciesAmountRate(int index,
            const double *reactionRates);

    /**
     * calculate rate rule values.
     * TODO redo this function, not very effecient.
     */
    int getRateRueRates(int len, int const *indx, double *values);


    /**
     * Get the current set of flags
     */
    virtual uint32_t getFlags() const { return flags; }

    /**
     * Set certain options that determine the state of the ExecutableModel,
     * these are listed in
     */
    virtual void setFlags(uint32_t val) { flags = val; }
	
	void saveState(std::ostream & out);


private:

    /**
     * get a selection record for a given stirng. if the string is valid,
     * the SelectionRecord is created and cached.
     *
     * if the string is invalid, and exception is thrown.
     */
    const rr::SelectionRecord& getSelection(const std::string& sel);

    /**
     * previous state
     * get current state
     * current state becomes previous state for next itteration
     * evaluate first pending event
     */
    bool applyEvents(unsigned char* prevEventState,
            unsigned char* currEventState);


    rrllvm::EventQueue pendingEvents;

    /**
     * the time delayed events were triggered.
     */
    std::vector<double> eventAssignTimes;

    typedef unsigned long long TieBreakKey;
    typedef std::map<TieBreakKey, bool> TieBreakMap;
    TieBreakMap tieBreakMap;

    /******************************* Events Section *******************************/
    #endif /***********************************************************************/
    /******************************************************************************/
private:
    /**
     * the model generator maintians a cached of generated models.
     */
    LLVMModelData *modelData;
    cxx11_ns::shared_ptr<ModelResources> resources;
    const LLVMModelDataSymbols *symbols;


    EvalInitialConditionsCodeGen::FunctionPtr evalInitialConditionsPtr;
    EvalReactionRatesCodeGen::FunctionPtr evalReactionRatesPtr;
    GetBoundarySpeciesAmountCodeGen::FunctionPtr getBoundarySpeciesAmountPtr;
    GetFloatingSpeciesAmountCodeGen::FunctionPtr getFloatingSpeciesAmountPtr;
    GetBoundarySpeciesConcentrationCodeGen::FunctionPtr getBoundarySpeciesConcentrationPtr;
    GetFloatingSpeciesConcentrationCodeGen::FunctionPtr getFloatingSpeciesConcentrationPtr;
    GetCompartmentVolumeCodeGen::FunctionPtr getCompartmentVolumePtr;
    GetGlobalParameterCodeGen::FunctionPtr getGlobalParameterPtr;
    EvalRateRuleRatesCodeGen::FunctionPtr evalRateRuleRatesPtr;
    GetEventTriggerCodeGen::FunctionPtr getEventTriggerPtr;
    GetEventPriorityCodeGen::FunctionPtr getEventPriorityPtr;
    GetEventDelayCodeGen::FunctionPtr getEventDelayPtr;
    EventTriggerCodeGen::FunctionPtr eventTriggerPtr;
    EventAssignCodeGen::FunctionPtr eventAssignPtr;
    EvalVolatileStoichCodeGen::FunctionPtr evalVolatileStoichPtr;
    EvalConversionFactorCodeGen::FunctionPtr evalConversionFactorPtr;

    // set model values externally.
    SetBoundarySpeciesAmountCodeGen::FunctionPtr setBoundarySpeciesAmountPtr;
    SetFloatingSpeciesAmountCodeGen::FunctionPtr setFloatingSpeciesAmountPtr;
    SetBoundarySpeciesConcentrationCodeGen::FunctionPtr setBoundarySpeciesConcentrationPtr;
    SetFloatingSpeciesConcentrationCodeGen::FunctionPtr setFloatingSpeciesConcentrationPtr;
    SetCompartmentVolumeCodeGen::FunctionPtr setCompartmentVolumePtr;
    SetGlobalParameterCodeGen::FunctionPtr setGlobalParameterPtr;


    // init value accessors
    SetFloatingSpeciesInitConcentrationCodeGen::FunctionPtr setFloatingSpeciesInitConcentrationsPtr;
    GetFloatingSpeciesInitConcentrationCodeGen::FunctionPtr getFloatingSpeciesInitConcentrationsPtr;
    SetFloatingSpeciesInitAmountCodeGen::FunctionPtr setFloatingSpeciesInitAmountsPtr;
    GetFloatingSpeciesInitAmountCodeGen::FunctionPtr getFloatingSpeciesInitAmountsPtr;
    SetCompartmentInitVolumeCodeGen::FunctionPtr setCompartmentInitVolumesPtr;
    GetCompartmentInitVolumeCodeGen::FunctionPtr getCompartmentInitVolumesPtr;
    GetGlobalParameterInitValueCodeGen::FunctionPtr getGlobalParameterInitValuePtr;
    SetGlobalParameterInitValueCodeGen::FunctionPtr setGlobalParameterInitValuePtr;


    typedef string (LLVMExecutableModel::*GetNameFuncPtr)(int);

    /**
     * cache the selection records
     */
    typedef cxx11_ns::unordered_map<std::string, rr::SelectionRecord> SelectionMap;
    SelectionMap selectionRecordCache;

    /**
     * event handlers, we don't own these, just borrow them
     *
     * array of modelData.numEvents length.
     */
    std::vector<rr::EventListenerPtr> eventListeners;

    /**
     * get the values from the model struct and populate the given values array.
     */
    int getValues(double (*funcPtr)(LLVMModelData*, int), int len,
            const int *indx, double *values);

    /**
     * set the model struct values from the given array.
     */
    int setValues(bool (*funcPtr)(LLVMModelData*, int, double), GetNameFuncPtr, int len,
            const int *indx, const double *values);

    static LLVMExecutableModel* dummy();

    friend class LLVMModelGenerator;

    template <typename a_type, typename b_type>
    friend void copyCachedModel(a_type* src, b_type* dst);

    /**
     * the sbml conversion factor.
     *
     * TODO: this has issues in that its possible for the conversion factor to change.
     * This needs to be moved into an LLVM generated function instead of a
     * class variable.
     */
    double conversionFactor;


    /**
     * what items are dirty
     */
    uint32_t dirty;

    enum
    {
        // init cond for species have changed
        DIRTY_INIT_SPECIES            = (0x1 << 0),  // => 0x00000001

        // conserved moieties have changes.
        DIRTY_CONSERVED_MOIETIES      = (0x1 << 1),  // => 0x00000010

        // reaction rates need to be re-calculated.
        DIRTY_REACTION_RATES          = (0x1 << 2)   //
    };


    uint32_t flags;
};

} /* namespace rr */
#endif /* LLVMExecutableModelH */
