// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

// == FILEDOC =================================================

/** @file GPUSimBiochemModel.h
  * @author JKM
  * @date 08/14/2014
  * @copyright Apache License, Version 2.0
  * @brief Classes and data structures for representing the model
**/

#ifndef rrGPUSimBiochemModelH
#define rrGPUSimBiochemModelH

// == MACROS ==================================================

# define GPUSIM_MODEL_USE_SBML 1

// TODO: make this a .hpp

// == INCLUDES ================================================

#include "GPUSimModel.hpp"
#include "conservation/ConservedMoietyConverter.h"
#include "conservation/ConservationExtension.h"
#include "patterns/AccessPtrIterator.hpp"
#include "patterns/Range.hpp"

#include <memory>
#include <set>

// == CODE ====================================================

namespace rr
{

namespace rrgpu
{

class RR_DECLSPEC Species : public ModelElement
{
public:
    Species(const std::string& id)
      : id_(id) {}

    void settId(const std::string& id) { id_ = id; }
    std::string getId() const { return id_; }
    /// Return true if @ref id matches this object's id
    bool matchId(const std::string& id) const { return id_ == id; }

    void setIsConservedMoiety(bool val) { consrvMoity_ = val; }
    bool getIsConservedMoiety() const { return consrvMoity_; }

    void setInitialConcentration(Real val) { init_conc_ = val; }
    Real getInitialConcentration() const { return init_conc_; }

    virtual bool matchesType(int typebits) const = 0;
protected:
    std::string id_;
    bool consrvMoity_ = false;
    Real init_conc_ = 0.;
};

class RR_DECLSPEC BoundarySpecies : public Species
{
public:
    BoundarySpecies(const std::string& id)
      : Species(id) {}

    virtual bool matchesType(int typebits) const {
        return false;
    }
protected:
};

class RR_DECLSPEC FloatingSpecies : public Species
{
public:
    FloatingSpecies(const std::string& id)
      : Species(id) {}

    // deprecate?
    void setIsIndepInitFltSpecies(bool val) { indInitFltSpc_ = val; }
    bool getIsIndepInitFltSpecies() const { return indInitFltSpc_; }

    void setIsIndependent(bool val) { isIndependent_ = val; }
    bool getIsIndependent() const { return isIndependent_; }

    void setIndex(int index) { index_ = index; }
    int getIndex() const { return index_; }

    virtual bool matchesType(int typebits) const {
        return ((typebits & SelectionRecord::FLOATING) &&
                (((typebits & SelectionRecord::INDEPENDENT) && getIsIndependent()) || ((typebits & SelectionRecord::DEPENDENT) && !getIsIndependent())) //&&
//                 (typebits & SelectionRecord::INITIAL) == getIsIndepInitFltSpecies()
        );
    }
protected:
    bool indInitFltSpc_ = false;
    bool isIndependent_ = true;
    int index_ = 0;
    Real init_conc_ = 0.;
};

/**
 * @brief Kinetic law of a process (e.g. a reaction)
 */
class RR_DECLSPEC KineticLaw
{
public:
    ModelAlgebra& getAlgebra() { return alg_; }
    const ModelAlgebra& getAlgebra() const { return alg_; }

protected:
    ModelAlgebra alg_;
};

/**
 * @brief Kinetic law of a reaction
 */
class RR_DECLSPEC ReactionKineticLaw : public KineticLaw
{
public:
protected:
};

class RR_DECLSPEC ReactionParticipant
{
public:
    enum class ReactionSide {
        Reactant,
        Product,
        Both
    };

    ReactionParticipant(const Species* spec, ReactionSide side)
      : spec_(spec), side_(side) {
        checkSide();
    }

    void checkSide() {
        if (side_ == ReactionSide::Both)
            throw_gpusim_exception("Side 'both' is not supported");
    }

    const Species* getSpecies() const { return spec_; }

    bool isFloatingSpecies() const {
        if (dynamic_cast<const FloatingSpecies*>(spec_))
            return true;
        else
            return false;
    }

    /**
     * @note Throws if not a floating species
     */
    const FloatingSpecies* getFloatingSpecies() const {
        auto p = dynamic_cast<const FloatingSpecies*>(spec_);
        if (p)
            return p;
        else
            throw_gpusim_exception("Not a floating species");
    }

    /**
     * @note Throws if not a boundary species
     */
    const BoundarySpecies* getBoundarySpecies() const {
        auto p = dynamic_cast<const BoundarySpecies*>(spec_);
        if (p)
            return p;
        else
            throw_gpusim_exception("Not a boundary species");
    }

    ReactionSide getSide() const { return side_; }

protected:
    const Species* spec_ = nullptr;
    ReactionSide side_;
};

/**
 * @brief Represents a reaction in the model
 */
class RR_DECLSPEC Reaction : public ModelElement
{
public:
    typedef std::string String;
    typedef ReactionParticipant::ReactionSide ReactionSide;

    Reaction(const String& id)
      : id_(id) {}

    void settId(const String& id) { id_ = id; }
    String getId() const { return id_; }
    /// Return true if @ref id matches this object's id
    bool matchId(const String& id) { return id_ == id; }

    bool isReversible() const { return reversible_; }
    void setIsReversible(bool val) { reversible_ = val; }

    /// Add a participant (reactant or product)
    void addParticipant(const Species* spec, ReactionSide side) {
        if (isParticipant(spec))
            throw_gpusim_exception("Floating species \"" + spec->getId() + "\" is already a participant in the reaction");
        addParticipantForce(spec, side);
    }

    /// Does not check if the species is already present in the reaction
    void addParticipantForce(const Species* spec, ReactionSide side) {
        part_.emplace_back(new ReactionParticipant(spec, side));
    }

    void addReactant(const Species* spec) {
        addParticipant(spec, ReactionSide::Reactant);
    }

    void addProduct(const Species* spec) {
        addParticipant(spec, ReactionSide::Product);
    }

    bool isParticipant(const Species* spec) const {
        for (auto const & p : part_)
            if (p->getSpecies() == spec)
                return true;
        return false;
    }

    bool isParticipant(const String& id) const {
        for (auto const & p : part_)
            if (p->getSpecies()->getId() == id)
                return true;
        return false;
    }

    /// Get the species from an id if it participates in the reaction
    const Species* getSpecies(const String& id) const {
        for (auto const & p : part_)
            if (p->getSpecies()->getId() == id)
                return p->getSpecies();
        throw_gpusim_exception("No such participant \"" + id +"\"");
    }

    /**
     * @note Throws if not a floating species
     */
    const FloatingSpecies* getFloatingSpecies(const String& id) const {
        for (auto const & p : part_)
            if (p->isFloatingSpecies() && p->getFloatingSpecies()->getId() == id)
                return p->getFloatingSpecies();
        throw_gpusim_exception("No such participant \"" + id +"\"");
    }

    ReactionSide getSide(const Species* spec) const {
        for (auto const & p : part_)
            if (p->getSpecies() == spec)
                return p->getSide();
        throw_gpusim_exception("Floating species \"" + spec->getId() + "\" is not a participant in the reaction");
    }

    bool isParameter(const String& p) const;

    double getParameterVal(const String& p) const;

    ReactionKineticLaw& getKineticLaw() { return klaw_; }

    const ReactionKineticLaw& getKineticLaw() const { return klaw_; }

# if GPUSIM_MODEL_USE_SBML
    const libsbml::Reaction* getSBMLReaction() const {
        return sbmlrxn_;
    }

    void setSBMLReaction(const libsbml::Reaction* r) {
        sbmlrxn_ = r;
    }

    void setSBMLModel(const libsbml::Model* m) {
        sbmlmodel_ = m;
    }

    const libsbml::ASTNode* getSBMLMath() const {
        return getSBMLReaction()->getKineticLaw()->getMath();
    }
# endif

protected:
    std::string id_;
    bool reversible_ = false;
    typedef std::unique_ptr<ReactionParticipant> ReactionParticipantPtr;
    typedef std::vector<ReactionParticipantPtr> Participants;
    Participants part_;
    ReactionKineticLaw klaw_;
# if GPUSIM_MODEL_USE_SBML
    const libsbml::Reaction* sbmlrxn_ = nullptr;
    const libsbml::Model* sbmlmodel_ = nullptr;
# endif
};

class BiochemModelASTNodeVisitor;

/**
 * @brief Reference to floating species
 */
class RR_DECLSPEC FloatingSpeciesRefASTNode : public ModelASTNode
{
public:
    FloatingSpeciesRefASTNode(const FloatingSpecies* spec)
      : spec_(spec) {}

    /// Get the associated floating species
    const FloatingSpecies* getFloatingSpecies() const { return spec_; }

protected:
    /// Non-owning
    const FloatingSpecies* spec_;
};

/**
 * @brief Reference to floating species
 */
class RR_DECLSPEC ParameterRefASTNode : public ModelASTNode
{
public:
    typedef std::string String;

    ParameterRefASTNode(const Reaction* r, const String& param)
      : r_(r), param_(param) {}

    /// Get the associated parameter value
    double getParameterVal() const { return r_->getParameterVal(param_); }

protected:
    /// Non-owning
    const Reaction* r_;
    String param_;
};

class ModelASTNodeVisitor {
public:
    virtual void visit(const ProductASTNode* node) = 0;
};

class RR_DECLSPEC ModelRule : public ModelElement
{
public:

    virtual bool contains(const FloatingSpecies* s) const {
        return false;
    }

    virtual bool isAssignmentRule() const {
        return false;
    }

    virtual bool isInitialAssignmentRule() const {
        return false;
    }
};

class RR_DECLSPEC AssignmentRule : public ModelRule
{
public:
    std::string getId() const { return ""; }

    AssignmentRule(const std::string& variable) {}

    bool isAssignmentRule() const {
        return true;
    }
};

class RR_DECLSPEC RateRule : public ModelRule
{
public:
    std::string getId() const { return ""; }

    RateRule(const std::string& id) {}
};

class RR_DECLSPEC AlgebraicRule : public ModelRule
{
public:
    std::string getId() const { return ""; }

    AlgebraicRule(const std::string& formula) {}
};

class RR_DECLSPEC InitialAssignmentRule : public ModelRule
{
public:
    std::string getId() const { return symbol_; }

    InitialAssignmentRule(const std::string& symbol)
      : symbol_(symbol) {}

    bool isInitialAssignmentRule() const {
        return true;
    }

protected:
    std::string symbol_;
};

class RR_DECLSPEC ModelRules
{
protected:
    typedef std::unique_ptr<ModelRule> ModelRulePtr;
    typedef std::vector<ModelRulePtr> Rules;
public:
    typedef AccessPtrIterator<Rules::iterator> iterator;
    typedef AccessPtrIterator<Rules::const_iterator> const_iterator;

    iterator begin() { return iterator(rules_.begin()); }
    iterator end() { return iterator(rules_.end()); }

    const_iterator begin() const { return const_iterator(rules_.begin()); }
    const_iterator end() const { return const_iterator(rules_.begin()); }

    /// Return true if there is a rule that references @ref s
    bool contains(const FloatingSpecies* s) const {
        for(const ModelRule* r : *this)
            if(r->contains(s))
                return true;
        return false;
    }

    void addRule(ModelRulePtr&& r) {
        rules_.emplace_back(std::move(r));
    }

protected:
    Rules rules_;
};

/**
 * @author JKM
 * @brief Class that holds model data, symbols, and contextual information
 */
class RR_DECLSPEC GPUSimModel
{
protected:
    /// Owning model rule pointer
    typedef std::unique_ptr<ModelRule> ModelRulePtr;
    /// Owning floating species pointer
    typedef std::unique_ptr<FloatingSpecies> FloatingSpeciesPtr;
    /// Owning boundary species pointer
    typedef std::unique_ptr<BoundarySpecies> BoundarySpeciesPtr;
    /// Collection of all floating species
    typedef std::vector<FloatingSpeciesPtr> FloatingSpeciesCollection;
    /// Collection of all boundary species
    typedef std::vector<BoundarySpeciesPtr> BoundarySpeciesCollection;
    /// Owning pointer to reaction
    typedef std::unique_ptr<Reaction> ReactionPtr;
    /// Collection of all reactions
    typedef std::vector<ReactionPtr> ReactionCollection;
public:
    typedef std::size_t size_type;

    /// Iterator for all model elements
//     typedef MultiIteratorT<MultiIteratorRegPol<std::vector<int>>, ModelElement, FloatingSpeciesCollection, ModelRules> ModelElementsIterator;
    typedef std::vector<ModelElement*> ModelElements;
    typedef std::vector<const ModelElement*> ConstModelElements;

    typedef ReactionParticipant::ReactionSide ReactionSide;

    // SBML string ctor
    GPUSimModel(std::string const &sbml, unsigned loadSBMLOptions);

    // -- accessor/iterator section --

    // * FloatingSpecies *

    /// Iterator for floating species access pointer
    typedef AccessPtrIterator<FloatingSpeciesCollection::iterator> FloatingSpeciesIterator;
    /// Const iterator for floating species access pointer
    typedef AccessPtrIterator<FloatingSpeciesCollection::const_iterator> FloatingSpeciesConstIterator;

    typedef Range<FloatingSpeciesIterator> FloatingSpeciesRange;
    typedef Range<FloatingSpeciesConstIterator> FloatingSpeciesConstRange;

    /// Get all floating species
    FloatingSpeciesRange getFloatingSpecies() {
        return FloatingSpeciesRange(floatingSpecies_);
    }

    /// Get all floating species (const)
    FloatingSpeciesConstRange getFloatingSpecies() const {
        return FloatingSpeciesConstRange(floatingSpecies_);
    }

    // * BoundarySpecies *

    /// Iterator for floating species access pointer
    typedef AccessPtrIterator<BoundarySpeciesCollection::iterator> BoundarySpeciesIterator;
    /// Const iterator for floating species access pointer
    typedef AccessPtrIterator<BoundarySpeciesCollection::const_iterator> BoundarySpeciesConstIterator;

    typedef Range<BoundarySpeciesIterator> BoundarySpeciesRange;
    typedef Range<BoundarySpeciesConstIterator> BoundarySpeciesConstRange;

    /// Get all floating species
    BoundarySpeciesRange getBoundarySpecies() {
        return BoundarySpeciesRange(boundarySpecies_);
    }

    /// Get all floating species (const)
    BoundarySpeciesConstRange getBoundarySpecies() const {
        return BoundarySpeciesConstRange(boundarySpecies_);
    }

    size_type getNumIndepFloatingSpecies() const {
        size_type result = 0;
        for (const FloatingSpecies* f : getFloatingSpecies())
            if (f->getIsIndependent())
                result++;
        return result;
    }

    /// Get all model elements
    ModelElements getElements() {
        ModelElements elts;
        for(ModelElement* e : getFloatingSpecies())
            elts.push_back(e);
        for(ModelElement* e : getRules())
            elts.push_back(e);
        return elts;
    }

    ConstModelElements getElements() const {
        ConstModelElements elts;
        for(const ModelElement* e : getFloatingSpecies())
            elts.push_back(e);
        for(const ModelElement* e : getRules())
            elts.push_back(e);
        return elts;
    }

    virtual void getIds(int types, std::list<std::string> &ids);

    // -- end accessor/iterator section --

    /// Get the rules included in this model
    const ModelRules& getRules() const { return rules_; }

    // -- Species --

    /**
     * @brief Find the species with the given id
     * @note Throws if nonexistent
     */
    Species* getSpeciesById(const std::string& id);

    /**
     * @brief Find the species with the given id
     * @note Throws if nonexistent
     */
    const Species* getSpeciesById(const std::string& id) const;

    /**
     * @brief Find the floating species with the given id
     * @note Throws if nonexistent
     */
    FloatingSpecies* getFloatingSpeciesById(const std::string& id);

    /**
     * @brief Find the floating species with the given id
     * @note Throws if nonexistent
     */
    const FloatingSpecies* getFloatingSpeciesById(const std::string& id) const;

    bool hasInitialAssignmentRule(const FloatingSpecies* s);

    bool hasAssignmentRule(const FloatingSpecies* s) const;

    /**
     * @brief Gets the size of the state vector
     */
    size_type getStateVecSize() const;

    /**
     * @brief Gets the component of the state vector
     * which represents the conc. of @ref s
     */
    int getStateVecComponent(const FloatingSpecies* s) const;

    /**
     * @brief Inverse of @ref getStateVecComponent
     */
    const FloatingSpecies* getFloatingSpeciesFromSVComponent(int i) const;

    /**
     * @brief Dumps state vector assignments to log
     */
    void dumpStateVecAssignments() const;

    /**
     * @details Given a component of the state vector,
     * determine which side (if any) of the reaction
     * it is on
     */
    ReactionSide getReactionSide(const Reaction* r, const FloatingSpecies* s) const;

    /**
     * @brief Return 0, 1, or -1 depending on the side of the species
     * @return 1 if @ref species_id is a product, -1 if @ref species_id
     * is a reactant, and 0 if @ref species_id does not participate in the
     * reaction
     * @details Given a component of the state vector,
     * determine which side (if any) of the reaction
     * it is on
     */
    int getReactionSideFac(const Reaction* r, const FloatingSpecies* s) const {
        if (!r->isParticipant(s))
            return 0;
        switch (getReactionSide(r, s)) {
            case ReactionSide::Reactant:
                return -1;
            case ReactionSide::Product:
                return 1;
            default:
                assert(0 && "Should not happen");
        }
    }

    // -- Reactions --

    /**
     * @brief Find the floating species with the given id
     * @note Throws if nonexistent
     */
    const Reaction* getReactionById(const std::string& id) const;

    /// Iterator for floating species access pointer
    typedef AccessPtrIterator<ReactionCollection::iterator> ReactionIterator;
    /// Const iterator for floating species access pointer
    typedef AccessPtrIterator<ReactionCollection::const_iterator> ReactionConstIterator;

    typedef Range<ReactionIterator> ReactionRange;
    typedef Range<ReactionConstIterator> ReactionConstRange;

    /// Get all floating species
    ReactionRange getReactions() {
        return ReactionRange(reactions_);
    }

    /// Get all floating species (const)
    ReactionConstRange getReactions() const {
        return ReactionConstRange(reactions_);
    }

    size_type reactionCount() const {
        return reactions_.size();
    }

# if GPUSIM_MODEL_USE_SBML
    /// Returns the document access pointer
    libsbml::SBMLDocument* getDocument();

    /// Returns the document access pointer
    const libsbml::SBMLDocument* getDocument() const;

    const libsbml::Model* getModel();
# endif

protected:

//     bool isIndependentElement(const std::string& id) const;

    /**
     * checks if the given symbol is an init value for an independent
     * floating species.
     *
     * Conserved Moiety species are considered to have independent
     * initial condtions as in this case, the assignment rule only applies
     * at time t > 0.
     */
//     bool isIndependentInitFloatingSpecies(const std::string& symbol) const;

    /**
     * Is this sbml element an independent initial value.
     *
     * True if this value does NOT have an assignment or initial
     * assignment rule.
     *
     * Independent initial values do not have assignment or
     * initial assigment rules, but may have rate rules.
     */
//     bool isIndependentInitElement(const std::string& symbol) const;

//     bool isIndependentInitCompartment(const std::string& symbol) const;

//     void initCompartments();

    /// Forward new rule to @ref ModelRules
    void addRule(ModelRulePtr&& r) {
        rules_.addRule(std::move(r));
    }

    void addSpecies(BoundarySpeciesPtr&& s) {
        boundarySpecies_.emplace_back(std::move(s));
    }

    void addSpecies(FloatingSpeciesPtr&& s) {
        floatingSpecies_.emplace_back(std::move(s));
    }

    /// Add a reaction
    Reaction* addReaction(ReactionPtr&& r) {
        reactions_.emplace_back(std::move(r));
        return reactions_.back().get();
    }

    typedef std::unique_ptr<conservation::ConservedMoietyConverter> ConservedMoietyConverterPtr;
    /**
     * the moiety converter, for the time being owns the
     * converted document.
     */
    ConservedMoietyConverterPtr moietyConverter;

# if GPUSIM_MODEL_USE_SBML
    typedef std::unique_ptr<libsbml::SBMLDocument> SBMLDocumentPtr;
    SBMLDocumentPtr ownedDoc;
# endif

    ModelRules rules_;

    FloatingSpeciesCollection floatingSpecies_;
    BoundarySpeciesCollection boundarySpecies_;
    ReactionCollection reactions_;
};

} // namespace rrgpu

} // namespace rr

#endif /* GPUSimModelH */
