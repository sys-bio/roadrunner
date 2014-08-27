// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

/*
 * GPUSimModel.h
 *
 *  Created on: Aug 14, 2014
 *
 * Author: JKM
 */

#ifndef rrGPUSimModelH
#define rrGPUSimModelH

// == INCLUDES ================================================

#include "rrSelectionRecord.h"
#include "conservation/ConservedMoietyConverter.h"
#include "conservation/ConservationExtension.h"
#include "patterns/AccessPtrIterator.hpp"
#include "patterns/Range.hpp"
// #include "patterns/MultiIterator.hpp"

#include <memory>
#include <set>

// == CODE ====================================================

namespace rr
{

namespace rrgpu
{

class RR_DECLSPEC ModelElement
{
public:
    virtual ~ModelElement();

    virtual bool matchesType(int typebits) const { return false; }

    virtual std::string getId() const = 0;
};

class RR_DECLSPEC FloatingSpecies : public ModelElement
{
public:
    FloatingSpecies(const std::string& id)
      : id_(id) {}

    void settId(const std::string& id) { id_ = id; }
    std::string getId() const { return id_; }
    /// Return true if @ref id matches this object's id
    bool matchId(const std::string& id) { return id_ == id; }

    void setIsConservedMoiety(bool val) { consrvMoity_ = val; }
    bool getIsConservedMoiety() const { return consrvMoity_; }

    // depreacte? 
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
    std::string id_;
    bool consrvMoity_ = false;
    bool indInitFltSpc_ = false;
    bool isIndependent_ = true;
    int index_ = 0;
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
    /// Collection of all floating species
    typedef std::vector<FloatingSpeciesPtr> FloatingSpeciesCollection;
public:
    typedef std::size_t size_type;

    /// Iterator for all model elements
//     typedef MultiIteratorT<MultiIteratorRegPol<std::vector<int>>, ModelElement, FloatingSpeciesCollection, ModelRules> ModelElementsIterator;
    typedef std::vector<ModelElement*> ModelElements;
    typedef std::vector<const ModelElement*> ConstModelElements;

    // SBML string ctor
    GPUSimModel(std::string const &sbml, unsigned loadSBMLOptions);

    // -- accessor/iterator section --

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

    /// Find the floating species with the given id; throw if nonexistent
    FloatingSpecies* findFloatingSpeciesById(const std::string& id);

    bool hasInitialAssignmentRule(const FloatingSpecies* s);

    bool hasAssignmentRule(const FloatingSpecies* s);

protected:
    /// Returns the document access pointer
    libsbml::SBMLDocument* getDocument();

    /// Returns the document access pointer
    const libsbml::SBMLDocument* getDocument() const;

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

    void addSpecies(FloatingSpeciesPtr&& s) {
        floatingSpecies_.emplace_back(std::move(s));
    }

    typedef std::unique_ptr<conservation::ConservedMoietyConverter> ConservedMoietyConverterPtr;
    /**
     * the moiety converter, for the time being owns the
     * converted document.
     */
    ConservedMoietyConverterPtr moietyConverter;

    typedef std::unique_ptr<libsbml::SBMLDocument> SBMLDocumentPtr;
    SBMLDocumentPtr ownedDoc;

    ModelRules rules_;

    FloatingSpeciesCollection floatingSpecies_;
};

} // namespace rrgpu

} // namespace rr

#endif /* GPUSimModelH */
