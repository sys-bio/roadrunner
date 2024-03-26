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
#include "GetPiecewiseTriggerCodeGen.h"
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
class Jit;

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
    LLVMExecutableModel(const std::shared_ptr<ModelResources> &resources, LLVMModelData* modelData);

	/*
	* Loads a saved executable model
	*/
	LLVMExecutableModel(std::istream& in, uint modelGeneratorOpt);

    virtual ~LLVMExecutableModel();

    /**
     * get the name of the model
     */
    virtual std::string getModelName();
    virtual void setTime(double _time);
    virtual double getTime();

    /**
     * evaluate the initial conditions specified in the sbml, this entails
     * evaluating all InitialAssigments, AssigmentRules, initial values, etc...
     *
     * Then the model state is fully set.
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
    virtual int getCompartmentIndexForFloatingSpecies(size_t index);
    virtual int getCompartmentIndexForBoundarySpecies(size_t index);

    /**
     * get the global parameter values
     *
     * @param[in] len the length of the indx and values arrays.
     * @param[in] indx an array of length len of boundary species to return.
     * @param[out] values an array of at least length len which will store the
     *                returned boundary species amounts.
     */
    virtual int getGlobalParameterValues(size_t len, int const *indx,
            double *values);

    virtual int setGlobalParameterValues(size_t len, int const *indx,
            const double *values);

    virtual int setGlobalParameterValues(size_t len, int const* indx,
        const double* values, bool strict);

    virtual int getNumReactions();

    virtual int getReactionRates(size_t len, int const *indx,
                    double *values);

    /**
     * get the compartment volumes
     *
     * @param[in] len the length of the indx and values arrays.
     * @param[in] indx an array of length len of boundary species to return.
     * @param[out] values an array of at least length len which will store the
     *                returned boundary species amounts.
     */
    virtual int getCompartmentVolumes(size_t len, int const *indx,
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


    virtual std::string getStateVectorId(size_t index);

    /**
     * copies the internal model state std::vector into the provided
     * buffer.
     *
     * @param[out] stateVector a buffer to copy the state std::vector into, if NULL,
     *         return the size required.
     *
     * @return the number of items coppied into the provided buffer, if
     *         stateVector is NULL, returns the length of the state std::vector.
     */
    virtual int getStateVector(double *stateVector);

    /**
     * sets the internal model state to the provided packed state std::vector.
     *
     * @param[in] stateVector an array which holds the packed state std::vector, must be
     *         at least the size returned by getStateVector.
     *
     * @return the number of items copied from the state std::vector, negative
     *         on failure.
     */
    virtual int setStateVector(const double *stateVector);

    /**
     * where most of the juicy bits occur.
     *
     * the state std::vector y is the rate rule values and floating species
     * concentrations concatenated. y is of length numFloatingSpecies + numRateRules.
     *
     * The state std::vector is packed such that the first n raterule elements are the
     * values of the rate rules, and the last n floatingspecies are the floating
     * species values.
     *
     * @param[in] time current simulator time
     * @param[in] y state std::vector, must be of size returned by getStateVector
     * @param[out] dydt calculated rate of change of the state std::vector, if null,
     * it is ignored.
     */
    virtual void getStateVectorRate(double time, const double *y, double* dydt=0);


    virtual void testConstraints();

    virtual std::string getInfo();

    virtual int getFloatingSpeciesIndex(const std::string&);
    virtual std::string getFloatingSpeciesId(size_t index);
    virtual int getBoundarySpeciesIndex(const std::string&);
    virtual std::string getBoundarySpeciesId(size_t index);

    /**
     * get the floating species amounts
     *
     * @param[in] len the length of the indx and values arrays.
     * @param[in] indx an array of length len of boundary species to return.
     * @param[out] values an array of at least length len which will store the
     *                returned boundary species amounts.
     */
    virtual int getFloatingSpeciesAmounts(size_t len, int const *indx,
            double *values);

    virtual int getFloatingSpeciesAmountRates(size_t len, int const *indx,
            double *values);

    virtual int getFloatingSpeciesConcentrationRates(size_t len, int const *indx,
            double *values);

    /**
     * get the floating species concentrations
     *
     * @param[in] len the length of the indx and values arrays.
     * @param[in] indx an array of length len of boundary species to return.
     * @param[out] values an array of at least length len which will store the
     *                returned boundary species amounts.
     */
    virtual int getFloatingSpeciesConcentrations(size_t len, int const *indx,
            double *values);


    /**
     * set the floating species concentrations
     *
     * @param[in] len the length of the indx and values arrays.
     * @param[in] indx an array of length len of boundary species to return.
     * @param[in] values an array of at least length len which will store the
     *                returned boundary species amounts.
     */
    virtual int setFloatingSpeciesConcentrations(size_t len, int const *indx,
            double const *values);

    virtual int setFloatingSpeciesAmounts(size_t len, int const *indx,
            const double *values);

    virtual int setFloatingSpeciesAmounts(size_t len, int const* indx,
        const double* values, bool strict);

    /**
     * get the boundary species amounts
     *
     * @param[in] len the length of the indx and values arrays.
     * @param[in] indx an array of length len of boundary species to return.
     * @param[out] values an array of at least length len which will store the
     *                returned boundary species amounts.
     */
    virtual int getBoundarySpeciesAmounts(size_t len, int const *indx,
            double *values);

    /**
     * get the boundary species concentrations
     *
     * @param[in] len the length of the indx and values arrays.
     * @param[in] indx an array of length len of boundary species to return.
     * @param[out] values an array of at least length len which will store the
     *                returned boundary species amounts.
     */
    virtual int getBoundarySpeciesConcentrations(size_t len, int const *indx,
            double *values);

    /**
     * get the boundary species concentrations
     *
     * @param[in] len the length of the indx and values arrays.
     * @param[in] indx an array of length len of boundary species to return.
     * @param[in] values an array of at least length len which will store the
     *                returned boundary species amounts.
     */
    virtual int setBoundarySpeciesConcentrations(size_t len, int const *indx,
            double const *values);


    /**
      * get the boundary species amounts
      *
      * @param[in] len the length of the indx and values arrays.
      * @param[in] indx an array of length len of boundary species to return.
      * @param[in] values an array of at least length len which will store the
      *                returned boundary species amounts.
      */
     virtual int setBoundarySpeciesAmounts(size_t len, int const *indx,
             double const *values);

     virtual int setBoundarySpeciesAmounts(size_t len, int const* indx,
         double const* values, bool strict);


    virtual int getGlobalParameterIndex(const std::string&);
    virtual std::string getGlobalParameterId(size_t);
    virtual int getCompartmentIndex(const std::string&);
    virtual std::string getCompartmentId(size_t);
    virtual int getReactionIndex(const std::string&);
    virtual std::string getReactionId(size_t);
    virtual int getStoichiometryIndex(const std::string&);
    virtual int getStoichiometryIndex(const std::string& speciesId, const std::string& reactionId);
    virtual std::string getStoichiometryId(size_t);

    virtual void print(std::ostream &stream);

    virtual int getNumConservedMoieties();
    virtual int getConservedMoietyIndex(const std::string& name);
    virtual std::string getConservedMoietyId(size_t index);
    virtual int getConservedMoietyValues(size_t len, int const *indx, double *values);
    virtual int setConservedMoietyValues(size_t len, int const *indx,
            const double *values);

    virtual int setCompartmentVolumes(size_t len, int const* indx,
        const double* values);

    virtual int setCompartmentVolumes(size_t len, int const* indx,
        const double* values, bool strict);

    virtual int setStoichiometries(size_t len, int const* indx,
                                      const double* values);

    virtual int setStoichiometries(size_t len, int const* indx,
                                      const double* values, bool strict);

    virtual int setStoichiometry(int index, double value);

    virtual int setStoichiometry(int speciesIndex, int reactionIndex, double value);

    virtual double getStoichiometry(int index);

    virtual double getStoichiometry(int speciesIndex, int reactionIndex);

    /******************************* Initial Conditions Section *******************/
    #if (1) /**********************************************************************/
    /******************************************************************************/

    virtual int setFloatingSpeciesInitConcentrations(size_t len, const int *indx,
            double const *values);

    virtual int setBoundarySpeciesInitConcentrations(size_t len, const int* indx,
        double const* values);

    virtual int getFloatingSpeciesInitConcentrations(size_t len, const int *indx,
            double *values);

    virtual int getBoundarySpeciesInitConcentrations(size_t len, const int* indx,
        double* values);

    virtual int setFloatingSpeciesInitAmounts(size_t len, const int *indx,
                double const *values);

    virtual int setBoundarySpeciesInitAmounts(size_t len, const int* indx,
        double const* values);

    virtual int getFloatingSpeciesInitAmounts(size_t size_t, const int *indx,
                    double *values);

    virtual int getBoundarySpeciesInitAmounts(size_t size_t, const int* indx,
        double* values);

    virtual int setCompartmentInitVolumes(size_t len, const int *indx,
                double const *values);

    virtual int getCompartmentInitVolumes(size_t len, const int *indx,
                    double *values);

    virtual int setGlobalParameterInitValues(size_t len, const int *indx,
                double const *values);

    virtual int getGlobalParameterInitValues(size_t len, const int *indx,
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
     * gets the value for the given id std::string. The std::string must be a SelectionRecord
     * std::string that is accepted by this class.
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
    virtual int getEventTriggers(size_t len, const int *indx, unsigned char *eventState);

    /**
     * erases expired events, and applies ripe events as long as we have ripe
     * events.
     */
    virtual int applyEvents(double timeEnd, const unsigned char* previousEventState,
            const double *initialState, double* finalState);

    virtual void getEventRoots(double time, const double* y, double* gdot);

    virtual void getPiecewiseTriggerRoots(double time, const double* y, double* gdot);

    virtual double getNextPendingEventTime(bool pop);

    virtual int getPendingEventSize();

    virtual void resetEvents();

    inline double getEventDelay(size_t event)
    {
        return getEventDelayPtr(modelData, event);
    }

    inline double getEventPriority(size_t event)
    {
        return getEventPriorityPtr(modelData, event);
    }

    bool getEventTrigger(size_t event);

    inline bool getEventUseValuesFromTriggerTime(size_t event)
    {
        assert(event < symbols->getEventAttributes().size()
            && "event out of bounds");
        return symbols->getEventAttributes()[event]
                & EventUseValuesFromTriggerTime;
    }

    inline bool getEventInitialValue(size_t event)
    {
        assert(event < symbols->getEventAttributes().size()
            && "event out of bounds");
        return symbols->getEventAttributes()[event] & EventInitialValue;
    }

    inline bool getEventPersistent(size_t event)
    {
        assert(event < symbols->getEventAttributes().size()
            && "event out of bounds");
        return symbols->getEventAttributes()[event] & EventPersistent;
    }

    inline size_t getEventBufferSize(size_t event)
    {
        return symbols->getEventBufferSize(event);
    }

    inline void getEventData(size_t eventId, double* data)
    {
        eventTriggerPtr(modelData, eventId, data);
    }

    /**
     * assign or apply the event using the given data.
     */
    inline void assignEvent(size_t eventId, double* data)
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

    bool getEventTieBreak(size_t eventA, size_t eventB);

    virtual int getEventIndex(const std::string& eid);
    virtual std::string getEventId(size_t index);
    virtual void getEventIds(std::list<std::string>& out);

    virtual int getNumPiecewiseTriggers();

    virtual void getAssignmentRuleIds(std::list<std::string>& out);
    virtual void getRateRuleIds(std::list<std::string>& out);
    virtual void getInitialAssignmentIds(std::list<std::string>& out);
    virtual void setEventListener(size_t index, rr::EventListenerPtr eventHandler);
    virtual rr::EventListenerPtr getEventListener(size_t index);


    virtual double getFloatingSpeciesAmountRate(size_t index,
            const double *reactionRates);

    /**
     * calculate rate rule values.
     * TODO redo this function, not very effecient.
     */
    virtual int getRateRuleRates(size_t len, int const *indx, double *values);


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
     * get a selection record for a given stirng. if the std::string is valid,
     * the SelectionRecord is created and cached.
     *
     * if the std::string is invalid, and exception is thrown.
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

    //Used by 'reset' to reset one type of element, while keeping track of which versions were reset that have initial assignments.
    void resetOneType(int& opt, int thistype, int independents, int total, int(LLVMExecutableModel::*getInit)(size_t, const int*, double*), int(LLVMExecutableModel::*setCurrent)(size_t, const int*, const double*), string(LLVMModelDataSymbols::* getTypeId)(size_t) const, double* buffer, std::map<std::string, int>& inits, std::map<std::string, double>& initvals);


    /******************************* Events Section *******************************/
    #endif /***********************************************************************/
    /******************************************************************************/
private:
    /**
     * the model generator maintians a cache of generated models.
     */

    LLVMModelData *modelData;
    cxx11_ns::shared_ptr<ModelResources> resources;
    const LLVMModelDataSymbols *symbols;


    EvalReactionRatesCodeGen::FunctionPtr evalReactionRatesPtr;
    EvalInitialConditionsCodeGen::FunctionPtr evalInitialConditionsPtr;
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
    GetPiecewiseTriggerCodeGen::FunctionPtr getPiecewiseTriggerPtr;
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
    SetBoundarySpeciesInitConcentrationCodeGen::FunctionPtr setBoundarySpeciesInitConcentrationsPtr;
    GetFloatingSpeciesInitConcentrationCodeGen::FunctionPtr getFloatingSpeciesInitConcentrationsPtr;
    GetBoundarySpeciesInitConcentrationCodeGen::FunctionPtr getBoundarySpeciesInitConcentrationsPtr;
    SetFloatingSpeciesInitAmountCodeGen::FunctionPtr setFloatingSpeciesInitAmountsPtr;
    GetFloatingSpeciesInitAmountCodeGen::FunctionPtr getFloatingSpeciesInitAmountsPtr;
    SetBoundarySpeciesInitAmountCodeGen::FunctionPtr setBoundarySpeciesInitAmountsPtr;
    GetBoundarySpeciesInitAmountCodeGen::FunctionPtr getBoundarySpeciesInitAmountsPtr;
    SetCompartmentInitVolumeCodeGen::FunctionPtr setCompartmentInitVolumesPtr;
    GetCompartmentInitVolumeCodeGen::FunctionPtr getCompartmentInitVolumesPtr;
    GetGlobalParameterInitValueCodeGen::FunctionPtr getGlobalParameterInitValuePtr;
    SetGlobalParameterInitValueCodeGen::FunctionPtr setGlobalParameterInitValuePtr;


    typedef std::string (LLVMExecutableModel::*GetNameFuncPtr)(size_t);

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
    int getValues(double (*funcPtr)(LLVMModelData*, size_t), size_t len,
            const int *indx, double *values);

    /**
     * set the model struct values from the given array.
     */
    int setValues(bool (*funcPtr)(LLVMModelData*, int, double), GetNameFuncPtr, size_t len,
            const int *indx, const double *values, bool strict=true);

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
