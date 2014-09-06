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
    NumConservedSpecies,                      // 5
    NumIndBoundarySpecies,                    // 6
    NumIndGlobalParameters,                   // 7
    NumRateRules,                             // 8
    NumReactions,                             // 9

    NumInitCompartments,                      // 10
    NumInitFloatingSpecies,                   // 11
    NumInitBoundarySpecies,                   // 12
    NumInitGlobalParameters,                  // 13

    Stoichiometry,                            // 14
    NumEvents,                                // 15
    StateVectorSize,                          // 16
    StateVector,                              // 17
    StateVectorRate,                          // 18
    RateRuleRates,                            // 19
    FloatingSpeciesAmountRates,               // 20

    CompartmentVolumesAlias,                  // 21
    InitCompartmentVolumesAlias,              // 22
    InitFloatingSpeciesAmountsAlias,          // 23
    InitConservedSpeciesAmountsAlias,         // 24
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
    InitConservedSpeciesAmounts,              // 35
    BoundarySpeciesAmounts,                   // 36
    InitBoundarySpeciesAmounts,               // 37
    GlobalParameters,                         // 38
    InitGlobalParameters,                     // 39
    ReactionRates,                            // 40
    NotSafe_RateRuleValues,                   // 41
    NotSafe_FloatingSpeciesAmounts,           // 42
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
 * All items for which a rate rule exists are stored in the
 * modelData::rateRules array.
 *
 * No space is allocated for items determined by rate rules.
 *
 * * All elements get an index value, even the dependent ones, this allows
 * us uniquely indentify them in the generated accessor functions.
 *
 * * Most of the indexes used in this class are indexes into ModelData
 * arrays, therefore we use unsigned integers -- these are less error
 * prone and its easier to check if they are valid i.e. only check that
 * they are less than the the array size and we do not have to check that
 * it is positive.
 *
 * * All symbols from the sbml are reordered such that the independent
 * ones are first, followed by the dependent values.
 */
class LLVMModelDataSymbols
{
public:

    typedef std::map<std::string, uint> StringUIntMap;
    typedef std::pair<std::string, uint> StringUIntPair;
    typedef cxx11_ns::unordered_map<uint, uint> UIntUIntMap;

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
        FLOATING_SPECIES, BOUNDARY_SPECIES, COMPARTMENT, GLOBAL_PARAMETER, REACTION, EVENT, INVALID_SYMBOL
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

    virtual ~LLVMModelDataSymbols();

    const std::string& getModelName() const;

    uint getCompartmentIndex(std::string const&) const;

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
     * get the index of a floating species.
     *
     * @param requireIndependent if true, the float species must be independent --
     * not defined by a rule.
     */
    uint getFloatingSpeciesIndex(std::string const&, bool requireIndependent = true) const;

    uint getBoundarySpeciesIndex(std::string const&) const;

    /**
     * number of boundary species not defined by rules.
     */
    uint getIndependentBoundarySpeciesSize() const;

    /**
     * number of fully indenpendent species, these are species that
     * have thier dynamics fully determined by the reaction rates /
     * stoichiometric matrix -- these DO NOT have any rules determining
     * their dynamics.
     *
     * This is size of the allocated array in the ModelData struct.
     */
    uint getIndependentFloatingSpeciesSize() const;

    /**
     * index of a global param given its name.
     * @throw exception if invalid name.
     */
    uint getGlobalParameterIndex(std::string const&) const;

    uint getRateRuleIndex(std::string const&) const;

    uint getRateRuleSize() const;

    /**
     * get the symbolic id of the i'th rate rule
     */
    std::string getRateRuleId(uint indx) const;

    /**
     * number of global parameters which are not determined by rules.
     */
    uint getIndependentGlobalParameterSize() const;

    uint getReactionIndex(std::string const&) const;
    std::vector<std::string> getReactionIds() const;
    uint getReactionSize() const;

    std::vector<std::string> getGlobalParameterIds() const;

    /**
     * is the global parameter index defined by a rate rule.
     */
    bool isRateRuleGlobalParameter(uint gid) const;

    /**
     * the list that is returned by ExecutableModel, so order must
     * remain constant.
     *
     * All floating species ids, independent first.
     */
    std::vector<std::string> getFloatingSpeciesIds() const;


    /**
     * get the symbolic id of the i'th floating species.
     */
    std::string getFloatingSpeciesId(uint indx) const;


    /**
     * find the id of the given global parameter index.
     */
    std::string getGlobalParameterId(uint indx) const;

    /**
     * total size of all floating species.
     */
    uint getFloatingSpeciesSize() const;

    uint getBoundarySpeciesSize() const;

    uint getCompartmentsSize() const;

    uint getGlobalParametersSize() const;

    std::vector<std::string> getCompartmentIds() const;

    /**
     * number of compartments which are not determined by rules.
     */
    uint getIndependentCompartmentSize() const;



    /**
     * all the boundary species ids, independent first.
     */
    std::vector<std::string> getBoundarySpeciesIds() const;

    /**
     * get a list of all the non-zero entries in the
     * stoichiometry matrix.
     *
     * in the list of pairs, first is the row (species) index,
     * and second is the column (reaction) index.
     */
    std::list<SpeciesReferenceInfo> getStoichiometryIndx() const;

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
            const std::string& id) const;


/******* Conserved Moiety Section ********************************************/
#if (1) /*********************************************************************/
/*****************************************************************************/




    /**
     * checks if the given symbol is a init value for a conserved species.
     *
     * Global parameters or floating species can be conservied moieties,
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
     * The number of conserved species. Thes are species which are defined
     * by assignment rules.
     */
    uint getConservedSpeciesSize() const;

    /**
     * get the number of conserved moieties.
     */
    uint getConservedMoietySize() const;

    /**
     * get the index of a global parameter given a conserved moiety index.
     */
    uint getConservedMoietyGlobalParameterIndex(uint cmIndex) const;

    /**
     * find the id of the given conserved moiety index.
     */
    std::string getConservedMoietyId(uint indx) const;

    /**
     * get the id of a conserved moiety given its name.
     */
    uint getConservedMoietyIndex(const std::string& name) const;

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
     * map of floating species ids to conserved moiety ids.
     *
     * assume conserved moeity speces are stored in the sbml
     * in the same order as the conserved moieties.
     */
    UIntUIntMap floatingSpeciesToConservedMoietyIdMap;


/*****************************************************************************/
#endif /* Conserved Moiety Section ******************************************/
/*****************************************************************************/
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
    uint getEventBufferSize(uint eventId) const;

    /**
     * the the row indices of non-zero stoichiometry values
     */
    const std::vector<uint>& getStoichRowIndx() const;

    /**
     * the the column indices of non-zero stoichiometry values
     */
    const std::vector<uint>& getStoichColIndx() const;


/************************ Initial Conditions Section *************************/
#if (1) /*********************************************************************/
/*****************************************************************************/

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
     * has this string been found to be an independent init global param--
     * is a global param and not having an assignment or init assignment rule.
     */
    bool isIndependentInitGlobalParameter(const std::string& symbol) const;

    /**
     * get the index of a floating species initial value.
     *
     * has the same index as the run time floating species.
     */
    uint getFloatingSpeciesInitIndex(const std::string& symbol) const;

    /**
     * get the index of a compartment initial value
     *
     * has the same index as the run time compartment.
     */
    uint getCompartmentInitIndex(const std::string& symbol) const;

    /**
     * get the index of a compartment for a float species.
     */
    uint getCompartmentIndexForFloatingSpecies(uint floatIndex) const;

    /**
     * get the index of a global parameter initial value
     *
     * has the same index as the run time global parameter.
     */
    uint getGlobalParameterInitIndex(const std::string& symbol) const;

    uint getInitCompartmentSize() const;
    uint getInitFloatingSpeciesSize() const;
    uint getInitBoundarySpeciesSize() const;
    uint getInitGlobalParameterSize() const;

    std::vector<std::string> getEventIds() const;

    std::string getEventId(uint indx) const;

    uint getEventIndex(const std::string& id) const;

private:

    std::set<std::string> initAssignmentRules;

    /**
     * map of floating species init value symbols to thier
     * index in the array.
     *
     * All floating species are stored in this map, however
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

/************************ End Initial Conditions Section *********************/
#endif /**********************************************************************/
/*****************************************************************************/


private:

    std::string modelName;
    StringUIntMap floatingSpeciesMap;
    StringUIntMap boundarySpeciesMap;
    StringUIntMap compartmentsMap;
    StringUIntMap globalParametersMap;



    /**
     * map of all identified species reference (species references with ids)
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
    std::set<std::string> assigmentRules;

    /**
     * rate rules, index by variable name.
     */
    StringUIntMap rateRules;

    /**
     * are global params defined by rate rules,
     * set in initGlobalParam
     */
    std::vector<bool> globalParameterRateRules;

    uint independentFloatingSpeciesSize;
    uint independentBoundarySpeciesSize;
    uint independentGlobalParameterSize;
    uint independentCompartmentSize;

    /**
     * the number of assignments each event has
     */
    std::vector<uint> eventAssignmentsSize;

    std::vector<unsigned char> eventAttributes;

    /**
     * map of event ids
     */
    StringUIntMap eventIds;

    void initBoundarySpecies(const libsbml::Model *);

    /**
     * init the floating species symbols.
     *
     * @param conservedMoieties: are conserved moieties enabled?
     */
    void initFloatingSpecies(const libsbml::Model *,
            bool conservedMoieties);

    /**
     *
     */
    void initCompartments(const libsbml::Model *);

    /**
     * get the global parameters, need to reorder them to set the independent
     * ones first
     *
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
};

} /* namespace rr */
#endif /* RRLLVMMODELDATASYMBOLS_H_ */
