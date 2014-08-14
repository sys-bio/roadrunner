// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

/*
 * GPUSimModel.h
 *
 *  Created on: Aug 14, 2014
 *
 * Author: JKM
 */

#ifndef GPUSimModelH
#define GPUSimModelH

// == INCLUDES ================================================

#include "rrSelectionRecord.h"
#include "conservation/ConservedMoietyConverter.h"
#include "conservation/ConservationExtension.h"
#include "AccessPtrIterator.hpp"

#include <memory>
#include <set>

// == CODE ====================================================

namespace rr
{

namespace rrgpu
{


class RR_DECLSPEC FloatingSpecies
{
public:
    FloatingSpecies(const std::string& id)
      : id_(id) {}

    void setIsConservedMoiety(bool val) { consrvMoity_ = val; }
    bool getIsConservedMoiety() const { return consrvMoity_; }

    void setIsIndepInitFltSpecies(bool val) { indInitFltSpc_ = val; }
    bool getIsIndepInitFltSpecies() const { return indInitFltSpc_; }

    void setIsIndependent(bool val) { isIndependent_ = val; }
    bool getIsIndependent() const { return isIndependent_; }
protected:
    std::string id_;
    bool consrvMoity_ = false;
    bool indInitFltSpc_ = false;
    bool isIndependent_ = false;
};

class RR_DECLSPEC ModelRule
{
public:
    virtual bool contains(const FloatingSpecies* s) {
        return false;
    }
};

class RR_DECLSPEC AssignmentRule : public ModelRule
{

};

class RR_DECLSPEC InitialAssignmentRule : public ModelRule
{
public:
    InitialAssignmentRule(const std::string& symbol)
      : symbol_(symbol) {}

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
    typedef AccessPtrIterator<Rules::iterator> const_iterator;

    iterator begin() { return iterator(rules_.begin()); }
    iterator end() { return iterator(rules_.end()); }

    const_iterator begin() const { return const_iterator(rules_.begin()); }
    const_iterator end() const { return const_iterator(rules_.begin()); }

    /// Return true if there is a rule that references @ref s
    bool contains(const FloatingSpecies* s) {
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
    typedef std::vector<FloatingSpeciesPtr> FloatingSpeciesCollection;
public:

    // SBML string ctor
    GPUSimModel(std::string const &sbml, unsigned loadSBMLOptions);

    // -- iterator section --

    typedef AccessPtrIterator<FloatingSpeciesCollection::iterator> FloatingSpeciesIterator;
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

    // -- end iterator section --

    /// Get the rules included in this model
    const ModelRules& getRules() const { return rules_; }

    /// Find the floating species with the given id; throw if nonexistent
    FloatingSpecies* findFloatingSpeciesById(const std::string& id);

protected:
    typedef std::unique_ptr<ModelRule> ModelRulePtr;
    typedef std::unique_ptr<FloatingSpecies> FloatingSpeciesPtr;

    /// Returns the document access pointer
    libsbml::SBMLDocument* getDocument();

    /// Returns the document access pointer
    const libsbml::SBMLDocument* getDocument() const;

    bool isIndependentElement(const std::string& id) const;

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

    void initCompartments();

    /// Forward to @ref ModelRules
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

    /**
     * the set of rule, these contain the variable name of the rule so that
     * we can quickly see if a symbol has an associated rule.
     */
//     std::set<std::string> assigmentRules;

    FloatingSpeciesCollection floatingSpecies_;
};

} // namespace rrgpu

} // namespace rr

#endif /* GPUSimModelH */
