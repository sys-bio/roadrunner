/*
 * LLVMModelDataSymbols.h
 *
 * Created on: Jun 28, 2013
 *
 * Author: Andy Somogyi,
 *     email decode: V1 = "."; V2 = "@"; V3 = V1;
 *     andy V1 somogyi V2 gmail V3 com
 */

#ifndef LLVMModelDataSymbolsH
#define LLVMModelDataSymbolsH

#include "LLVMModelData.h"
#include "rrExecutableModel.h"

#include "tr1proxy/rr_memory.h"
#include "tr1proxy/rr_unordered_map.h"


#include <map>
#include <set>
#include <list>

namespace libsbml
{
    class Model;
    class SimpleSpeciesReference;
    class ASTNode;
}

namespace rrllvm
{

enum ModelDataFields {
    Size = 0,                                 // 0
    Flags,                                    // 1
    Time,                                     // 2
    NumIndCompartments,                       // 3
    NumIndFloatingSpecies,                    // 4
    NumIndBoundarySpecies,                    // 5
    NumIndGlobalParameters,                   // 6
    NumRateRules,                             // 7
    NumReactions,                             // 8

    NumInitCompartments,                      // 9
    NumInitFloatingSpecies,                   // 10
    NumInitBoundarySpecies,                   // 11
    NumInitGlobalParameters,                  // 12

    Stoichiometry,                            // 13
    RandomPtr,                                // 14
    NumEvents,                                // 15
    NumPiecewiseTriggers,                     // 16
    StateVectorSize,                          // 17
    StateVector,                              // 18
    StateVectorRate,                          // 19
    RateRuleRates,                            // 20
    FloatingSpeciesAmountRates,               // 21

    CompartmentVolumesAlias,                  // 22
    InitCompartmentVolumesAlias,              // 23
    InitFloatingSpeciesAmountsAlias,          // 24
    BoundarySpeciesAmountsAlias,              // 25
    InitBoundarySpeciesAmountsAlias,          // 26
    GlobalParametersAlias,                    // 27
    InitGlobalParametersAlias,                // 28
    ReactionRatesAlias,                       // 29

    RateRuleValuesAlias,                      // 30
    FloatingSpeciesAmountsAlias,              // 31

    CompartmentVolumes,                       // 32
    InitCompartmentVolumes,                   // 33
    InitFloatingSpeciesAmounts,               // 34
    BoundarySpeciesAmounts,                   // 35
    InitBoundarySpeciesAmounts,               // 36
    GlobalParameters,                         // 37
    InitGlobalParameters,                     // 38
    ReactionRates,                            // 39
    NotSafe_RateRuleValues,                   // 40
    NotSafe_FloatingSpeciesAmounts,           // 41
};

enum EventAtributes
{
    EventUseValuesFromTriggerTime = (0x1 << 0), // => 0x00000001
    EventInitialValue             = (0x1 << 1), // => 0x00000010
    EventPersistent               = (0x1 << 2)  // => 0x00000100
};



/**
 * stores the names of all the symbols in the sbml model and thier
 * indexes in the ModelData arrays. This class essentially maps symbol
 * names into ModelData indexes.
 *
 * The logic behind splitting up LLVMModelDataSymbols and LLVMModelSymbols
 * is that LLVMModelSymbols contains all the ASTs from the model, and this
 * contains only ModelData indices -- once the all the model functions
 * are generated, there is no need to keep AST information around, but we
 * still need to know what symbol is at what index for the various
 * Model accessor functions.
 *
 * * data storage:
 * There can exist rules (assigment, rate, and eventually algebraic) rules
 * that determine the value of a symbol.
 *
 * * All items for which a rate rule exists are stored in the
 * modelData::rateRules array.
 *
 * * No space is allocated for items determined by rate rules.
 *
 * * All elements get an index value, even the dependent ones, this allows
 * us uniquely identify them in the generated accessor functions.
 *
 * * Most of the indexes used in this class are indexes into ModelData
 * arrays, therefore we use unsigned integers -- these are less error
 * prone and its easier to check if they are valid i.e. only check that
 * they are less than the array size and we do not have to check that
 * it is positive.
 *
 * * All symbols from the sbml are reordered such that the independent
 * ones are first, followed by the dependent values.
 */
class LLVMModelDataSymbols
{
public:

    typedef std::map<std::string, uint> StringUIntMap;
    typedef std::map<std::string, std::vector<uint> > StringUIntVectorMap;
    typedef std::pair<std::string, uint> StringUIntPair;
    typedef std::unordered_map<uint, uint> UIntUIntMap;

    enum SpeciesReferenceType
    {
        Reactant, Product, Modifier, MultiReactantProduct
    };

    /**
     * The model frequently looks for indices to sbml symbols.
     *
     * This is a list of the types of sbml symbols that this
     * class keeps track of.
     */
    enum SymbolIndexType
    {
        FLOATING_SPECIES,
        BOUNDARY_SPECIES,
        COMPARTMENT,
        GLOBAL_PARAMETER,
        REACTION,
        EVENT,
        STOICHIOMETRY,
        INVALID_SYMBOL
    };

    /**
     * info about an entry in the stoich matrix.
     *
     * there is exactly one of these for each entry.
     */
    struct SpeciesReferenceInfo
    {
        uint row;
        uint column;
        SpeciesReferenceType type;
        std::string id;
    };

    LLVMModelDataSymbols();

    LLVMModelDataSymbols(libsbml::Model const* model, unsigned options);

	LLVMModelDataSymbols(std::istream& in);

    virtual ~LLVMModelDataSymbols();

    const std::string& getModelName() const;

    int getCompartmentIndex(std::string const&) const;

    /**
     * checks to see if we have the given symbol in the sbml.
     *
     * @param name the name of the sbml symbol to look up.
     * @param result if successfull, this is set to the index
     * @returns the type of symbol, if no symbol is found, then
     * INVALID_SYMBOL is returned.
     */
    SymbolIndexType getSymbolIndex(const std::string& name, int& result) const;

    /**
     * Get the index of a floating species.
     *
     * @param id the id of the species to retrieve.
     * @param requireIndependent if true, the float species must be independent --
     * not defined by a rule.
     */
    int getFloatingSpeciesIndex(std::string const& id, bool requireIndependent = true) const;

    /**
     * Get the index of a boundary species.
     *
     * @param id the id of the species to retrieve.
     */
    int getBoundarySpeciesIndex(std::string const& id) const;

    /**
     * number of boundary species not defined by rules.
     */
    size_t getIndependentBoundarySpeciesSize() const;

    /**
     * number of fully indenpendent species, these are species that
     * have thier dynamics fully determined by the reaction rates /
     * stoichiometric matrix -- these DO NOT have any rules determining
     * their dynamics.
     *
     * This is size of the allocated array in the ModelData struct.
     */
    size_t getIndependentFloatingSpeciesSize() const;

    /**
     * index of a global param given its name.
     * @throw exception if invalid name.
     */
    int getGlobalParameterIndex(std::string const&) const;

    int getRateRuleIndex(std::string const&) const;

    size_t getRateRuleSize() const;

    /**
     * get the symbolic id of the i'th rate rule
     */
    std::string getRateRuleId(size_t indx) const;

    /**
     * number of global parameters which are not determined by rules.
     */
    size_t getIndependentGlobalParameterSize() const;

    int getReactionIndex(std::string const&) const;
    std::vector<std::string> getReactionIds() const;
    size_t getReactionSize() const;

    int getStoichiometryIndex(std::string const&) const;
    int getStoichiometryIndex(const std::string&, const std::string&) const;
    std::vector<std::string> getStoichiometryIds() const;
    size_t getStoichiometrySize() const;


    std::vector<std::string> getGlobalParameterIds() const;

    /**
     * is the global parameter index defined by a rate rule.
     */
    bool isRateRuleGlobalParameter(uint gid) const;

    /**
     * the list that is returned by ExecutableModel, so order must
     * remain constant.
     */
    std::vector<std::string> getFloatingSpeciesIds() const;


    /**
     * get the symbolic id of the i'th floating species.
     */
    std::string getFloatingSpeciesId(size_t indx) const;


    /**
     * get the symbolic id of the i'th boundary species.
     */
    std::string getBoundarySpeciesId(size_t indx) const;


    /**
     * get the symbolic id of the i'th compartment.
     */
    std::string getCompartmentId(size_t indx) const;


    /**
     * find the id of the given global parameter index.
     */
    std::string getGlobalParameterId(size_t indx) const;

    /**
     * total size of all floating species.
     */
    size_t getFloatingSpeciesSize() const;

    size_t getBoundarySpeciesSize() const;

    size_t getCompartmentsSize() const;

    size_t getGlobalParametersSize() const;

    std::vector<std::string> getCompartmentIds() const;

    /**
     * number of compartments which are not determined by rules.
     */
    size_t getIndependentCompartmentSize() const;


    std::vector<std::string> getBoundarySpeciesIds() const;

    /**
     * get a list of all the non-zero entries in the
     * stoichiometry matrix.
     *
     * in the list of pairs, first is the row (species) index,
     * and second is the column (reaction) index.
     */
    std::list<SpeciesReferenceInfo> getStoichiometryList() const;

    /**
     * initialize and allocate the buffers (including the stoich matrix)
     * for an existing ModelData struct.
     */
    //void initAllocModelDataBuffers(LLVMModelData& m) const;

    void print() const;

    /**
     * if there are no rules for an element, then they are considered
     * independent.
     *
     * These are only for elements, not init values, will return false
     * for all init symbols.
     *
     * Only valid after class has been constructed.
     */
    bool isIndependentElement(const std::string& id) const;

    bool hasRateRule(const std::string& id) const;

    bool hasAssignmentRule(const std::string& id) const;

    bool hasInitialAssignmentRule(const std::string& id) const;

    bool isIndependentFloatingSpecies(const std::string& id) const;

    bool isIndependentBoundarySpecies(const std::string& id) const;

    bool isBoundarySpecies(const std::string& id) const;

    bool isIndependentGlobalParameter(const std::string& id) const;

    bool isIndependentCompartment(const std::string& id) const;

    bool isNamedSpeciesReference(const std::string& id) const;

    const SpeciesReferenceInfo& getNamedSpeciesReferenceInfo(
            const std::string& id);


    /******* Conserved Moiety Section ********************************************/

    /**
     * checks if the given symbol is a init value for a conserved species.
     *
     * Global parameters or floating species can be conservied moieties,
     *
     * a global parameter is a CM if it is defined by a inital assignment rules
     * as a linear of one CM species and a set of indepdent
     * floating species.
     *
     * A floating species may be a CM if it is defined by a rule
     * as a linear combination of independent species.
     *
     * If a floating species is a CM, then it must by definition
     * be a dependent floating species.
     *
     * It can however have independent initial conditions defined
     * either by intial values or initial assignment rules.
     */
    bool isConservedMoietySpecies(const std::string& symbol) const;

    /**
     * checks if the given floating species id corresponds to
     * to a conserved moiety species.
     *
     * @param id index of floating species
     * @param [out] result the index of the conserved moiety if this
     * is a conserved moiety species.
     * @returns true or false if this is a cm species or not.
     */
    bool isConservedMoietySpecies(uint id, uint &result) const;

    /**
     * check if the global parameter with the given id is a
     * conserved moiety.
     */
    bool isConservedMoietyParameter(uint id) const;

    /**
     * get the number of conserved moieties.
     */
    size_t getConservedMoietySize() const;

    /**
     * get the dependent species for a given conserved moiety id
     */
    int getDepSpeciesIndexForConservedMoietyId(std::string id) const;

    /**
     * get all the independent species for a given conserved moiety id
     */
    const std::vector<uint>& getIndSpeciesIndexForConservedMoietyId(std::string id) const;

    /**
     * get the index of a global parameter given a conserved moiety index.
     */
    int getConservedMoietyGlobalParameterIndex(uint cmIndex) const;

    /**
     * find the id of the given conserved moiety index.
     */
    std::string getConservedMoietyId(size_t indx) const;

    /**
     * get the id of a conserved moiety given its name.
     */
    int getConservedMoietyIndex(const std::string& name) const;

    /**
     * check if the conserved moiety is turned on for this model
     */
    bool isConservedMoietyAnalysis() const;

private:

    /**
     * initialized in initFloatingSpecies.
     */
    std::set<std::string> conservedMoietySpeciesSet;

    /**
     * global parameter id conserved moiety status.
     */
    std::vector<bool> conservedMoietyGlobalParameter;

    /**
     * the index of a global parameter from a conserved moeity
     * index.
     *
     * This array will be the size of the number of CM, each value
     * of this array will be a global parameter index.
     */
    std::vector<uint> conservedMoietyGlobalParameterIndex;

    /**
     * std::map of floating species ids to conserved moiety ids.
     *
     * assume conserved moeity speces are stored in the sbml
     * in the same order as the conserved moieties.
     */
    UIntUIntMap floatingSpeciesToConservedMoietyIdMap;

    // holds the conserved moiety id for each dependent species
    StringUIntMap conservedMoietyDepSpecies;

    // holds the conserved moiety id for each independent species
    StringUIntVectorMap conservedMoietyIndSpecies;


    /* End Conserved Moiety Section ******************************************/
public:

    const std::vector<unsigned char>& getEventAttributes() const;

    /**
     * get the textual form of the field names.
     */
    static const char* getFieldName(ModelDataFields field);

    /**
     * get the size (in number of doubles) of the buffer
     * that events need to save the event data to.
     */
    size_t getEventBufferSize(size_t eventId) const;

    /**
     * the row indices of non-zero stoichiometry values
     */
    const std::vector<uint>& getStoichRowIndx() const;

    /**
     * the column indices of non-zero stoichiometry values
     */
    const std::vector<uint>& getStoichColIndx() const;


/************************ Initial Conditions Section *************************/

    /**
     * checks if the given symbol is an init value for an independent
     * floating species.
     *
     * Conserved Moiety species are considered to have independent
     * initial condtions as in this case, the assignment rule only applies
     * at time t > 0.
     */
    bool isIndependentInitFloatingSpecies(const std::string& symbol) const;

    /**
     * checks if the given symbol is an init value for an independent
     * boundary species.
     *
     * Conserved Moiety species are considered to have independent
     * initial condtions as in this case, the assignment rule only applies
     * at time t > 0.
     */
    bool isIndependentInitBoundarySpecies(const std::string& symbol) const;

    /**
     * Is this sbml element an independent initial value.
     *
     * True if this value does NOT have an assignment or initial
     * assignment rule.
     *
     * Independent initial values do not have assignment or
     * initial assigment rules, but may have rate rules.
     */
    bool isIndependentInitElement(const std::string& symbol) const;

    bool isIndependentInitCompartment(const std::string& symbol) const;

    /**
     * has this std::string been found to be an independent init global param--
     * is a global param and not having an assignment or init assignment rule.
     */
    bool isIndependentInitGlobalParameter(const std::string& symbol) const;


    /**
     * is this global parameter id an independent init param --
     * a global param without an init assignment rule.
     */
    bool isIndependentInitGlobalParameter(size_t id) const;

    /**
     * get the index of a floating species initial value.
     *
     * has the same index as the run time floating species.
     */
    int getFloatingSpeciesInitIndex(const std::string& symbol) const;

    /**
     * get the index of a boundary species initial value.
     *
     * has the same index as the run time boundary species.
     */
    int getBoundarySpeciesInitIndex(const std::string& symbol) const;

    /**
     * get the index of a compartment initial value
     *
     * has the same index as the run time compartment.
     */
    int getCompartmentInitIndex(const std::string& symbol) const;

    /**
     * get the index of a compartment for a float species.
     */
    int getCompartmentIndexForFloatingSpecies(size_t floatIndex) const;

    /**
     * get the index of a compartment for a boundary species.
     */
    int getCompartmentIndexForBoundarySpecies(size_t floatIndex) const;

    /**
     * get the index of a global parameter initial value
     *
     * has the same index as the run time global parameter.
     */
    int getGlobalParameterInitIndex(const std::string& symbol) const;

    size_t getInitCompartmentSize() const;
    size_t getInitFloatingSpeciesSize() const;
    size_t getInitBoundarySpeciesSize() const;
    size_t getInitGlobalParameterSize() const;

    std::vector<std::string> getEventIds() const;

    std::vector<std::string> getAssignmentRuleIds() const;

    std::vector<std::string> getRateRuleIds() const;

    std::vector<std::string> getInitialAssignmentIds() const;

    std::string getEventId(size_t indx) const;

    int getEventIndex(const std::string& id) const;

	void saveState(std::ostream&) const;

	void loadState(std::istream&);

private:

    std::set<std::string> initAssignmentRules;

    /**
     * std::map of floating species init value symbols to thier
     * index in the array.
     *
     * All floating species are stored in this std::map, however
     * they are reordered and only the ones with index <
     * independentInitFloatingSpeciesSize are allocated storage.
     */
    StringUIntMap initFloatingSpeciesMap;
    StringUIntMap initBoundarySpeciesMap;
    StringUIntMap initCompartmentsMap;
    StringUIntMap initGlobalParametersMap;

    /**
     * Elements that do NOT have assignment rules are considered
     * independent initial conditions.
     *
     * Ind init conditions CAN however defined by rules.
     *
     * ind init conditions are all elements that do not have
     * assignment or initial assignment rules.
     */

    /**
     * float species init array has floating species NOT
     * defined by init assignment rules, assignment rules,
     * but rate rules are OK. This does not contain the init values
     * for conserved species.
     */
    uint independentInitFloatingSpeciesSize;

    /**
     * boundary species not defined by any rule
     * including init assignment rules.
     */
    uint independentInitBoundarySpeciesSize;

    /**
     * global params not defined by any rule
     */
    uint independentInitGlobalParameterSize;

    /**
     * compartments not defined by any rules.
     */
    uint independentInitCompartmentSize;

    /**
     * index of compartments for each float species.
     */
    std::vector<uint> floatingSpeciesCompartmentIndices;

    /**
     * index of compartments for each boundary species.
     */
    std::vector<uint> boundarySpeciesCompartmentIndices;

    /************************ End Initial Conditions Section *********************/


private:

    std::string modelName;
    StringUIntMap floatingSpeciesMap;
    StringUIntMap boundarySpeciesMap;
    StringUIntMap compartmentsMap;
    StringUIntMap globalParametersMap;


    /**
     * std::map of all identified species reference (species references with ids)
     * to their indices in the stoichiometric matrix.
     */
    typedef std::map<std::string, SpeciesReferenceInfo> StringRefInfoMap;
    StringRefInfoMap namedSpeciesReferenceInfo;

    /**
     * all reactions are named.
     */
    StringUIntMap reactionsMap;

    /**
     * the stochiometry matrix is # reactions rows by # species columns.
     *
     * the species indices go here.
     */
    std::vector<uint> stoichColIndx;

    /**
     * and the reaction indices go here.
     */
    std::vector<uint> stoichRowIndx;

    /**
     * most species references are un-named, this is used so that if named ones
     * have a rate rule, we can set the proper initial conditions.
     */
    std::vector<std::string> stoichIds;

    std::vector<SpeciesReferenceType> stoichTypes;

    /**
     * the set of rule, these contain the variable name of the rule so that
     * we can quickly see if a symbol has an associated rule.
     */
    std::set<std::string> assignmentRules;

    /**
     * rate rules, index by variable name.
     */
    StringUIntMap rateRules;

    /**
     * are global params defined by rate rules,
     * set in initGlobalParam
     */
    std::vector<bool> globalParameterRateRules;

    size_t  independentFloatingSpeciesSize;
    size_t  independentBoundarySpeciesSize;
    size_t  independentGlobalParameterSize;
    size_t  independentCompartmentSize;

    /**
     * the number of assignments each event has
     */
    std::vector<size_t> eventAssignmentsSize;

    std::vector<unsigned char> eventAttributes;

    /**
     * std::map of event ids
     */
    StringUIntMap eventIds;

    void initBoundarySpecies(const libsbml::Model *);

    /**
     * init the floating species symbols.
     *
     * @param model the model to check.
     * @param conservedMoieties: are conserved moieties enabled?
     */
    void initFloatingSpecies(const libsbml::Model *model,
            bool conservedMoieties);

    /**
     *
     */
    void initCompartments(const libsbml::Model *);

    /**
     * get the global parameters, need to reorder them to set the independent
     * ones first
     *
     * @param model the model to check.
     * @param conservedMoieties: are conserved moieties enabled?
     */
    void initGlobalParameters(const libsbml::Model *model,
            bool conservedMoieties);

    void initReactions(const libsbml::Model *model);

    void displayCompartmentInfo();

    void initEvents(const libsbml::Model *model);

    /**
     * determine is this species can be used as a species reference,
     * in the sense that it will add a column to the stochiometry
     * matrix.
     *
     * Boundary species are not consumed or produced in reactions.
     * If this is invalid float species, such as its defined by a rule
     * this will log the reason.
     */
    bool isValidFloatingSpeciesReference(const libsbml::SimpleSpeciesReference*,
            const std::string& reacOrProd);

    /**
     * set the type of all species references.
     * row is species, column is reaction
     */
    void setNamedSpeciesReferenceInfo(uint row, uint column,
            SpeciesReferenceType type);

	void saveStringRefInfoMap(std::ostream&, const StringRefInfoMap&) const;

	void loadStringRefInfoMap(std::istream&, StringRefInfoMap&);

	void saveBinarySpeciesReferenceInfo(std::ostream&, SpeciesReferenceInfo& sri) const;

	void loadBinarySpeciesReferenceInfo(std::istream&, SpeciesReferenceInfo&);
};

} /* namespace rr */
#endif /* RRLLVMMODELDATASYMBOLS_H_ */
