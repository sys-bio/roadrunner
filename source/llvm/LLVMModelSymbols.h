/*
 * LLVMModelSymbols.h
 *
 *  Created on: Jul 16, 2013
 *      Author: andy
 */

#ifndef LLVMLLVMModelSymbolsH
#define LLVMLLVMModelSymbolsH

#include "SymbolForest.h"
#include "ASTNodeFactory.h"
#include "ModelDataIRBuilder.h"
#include <sbml/Model.h>
#include <sbml/SBMLVisitor.h>

namespace rrllvm
{

/**
 * Hold all the un-evaluated symbolic inforamtion in the model.
 *
 * TODO: some real docs...
 */
class LLVMModelSymbols: private libsbml::SBMLVisitor
{
    using libsbml::SBMLVisitor::visit;

public:
    LLVMModelSymbols(libsbml::Model const *m, LLVMModelDataSymbols const &sym);

    ~LLVMModelSymbols();

    /**
     * create an ASTNode for the species id / reaction id pair.
     *
     * This assembles the mess of items stored in the reactions array.
     */
    libsbml::ASTNode *createStoichiometryNode(int row, int col) const;

    /**
     * assignment rules are always active
     */
    const SymbolForest& getAssigmentRules() const;

    /**
     * only valid before the model is started.
     *
     * There cannot be both an InitialAssignment and an AssignmentRule for the
     * same symbol in a model, because both kinds of constructs apply prior to
     * and at the start of simulated timeÑallowing both to exist for a given
     * symbol would result in indeterminism.
     *
     *
     * \conservation: In the case of conserved moieties, these are species that
     * are defined by assignment rules, they however can have initial values
     * defined by initial value or assignment rules.
     *
     * So, at run time, the assignment rule is active, however before run time,
     * the initial value or initial assignment rule is active.
     */
    const SymbolForest& getInitialAssignmentRules() const;

    /**
     * contains the intial symbols along with the intial assignments which
     * override the initial values.
     */
    const SymbolForest& getInitialValues() const;

    const SymbolForest& getRateRules() const;

protected:

    enum SpeciesReferenceType {
        Reactant, Product
    };

    virtual bool visit(const libsbml::Compartment &x);
    virtual bool visit(const libsbml::Species &x);
    virtual bool visit(const libsbml::AssignmentRule  &x);

    /**
     * InitialAssignments override the initial value specified in the element
     * definition.
     *
     * The actions of all InitialAssignment objects are in general terms the same,
     * but differ in the precise details depending on the type of variable being set:
     *
     * * In the case of a species, an InitialAssignment sets the referenced species
     * initial quantity (concentration or amount) to the value determined by the
     * formula in math. The unit associated with the value produced by the math
     * formula should be equal to the unit associated with the species quantity.
     *
     * * In the case of a species reference, an InitialAssignment sets the initial
     * stoichiometry of the reactant or product referenced by the SpeciesReference
     * object to the value determined by the formula in math. The unit associated
     * with the value produced by the math formula should be consistent with the
     * unit dimensionless, because reactant and product stoichiometries in reactions
     * are dimensionless quantities.
     *
     * * In the case of a compartment, an InitialAssignment sets the referenced
     * compartment's initial size to the size determined by the formula in math.
     * The unit associated with the value produced by the math formula should be
     * the same as that specified for the compartment's size.
     *
     * * In the case of a parameter, an InitialAssignment sets the parameter's
     * initial value to the value of the formula in math. The unit associated
     * with the value produced by the math formula should be the same as parameter's
     * units attribute value.
     */
    virtual bool visit(const libsbml::InitialAssignment &x);

    /**
     * * In the case of a species, a RateRule sets the rate of change of the
     * speciesÕ quantity (concentration or amount) to the value determined
     * by the formula in math. The unit associated with the ruleÕs math
     * element should be equal to the unit of the speciesÕ quantity
     * (Section 4.6.5) divided by the model-wide unit of time (Section 4.2.4),
     * or in other words, {unit of species quantity}/{unit of time}.
     *
     * * Restrictions: There must not be both a RateRule variable attribute
     * and a SpeciesReference species attribute having the same value, unless
     *  that species has its boundaryCondition attribute is set to ÒtrueÓ. This
     * means a rate rule cannot be defined for a species that is created or
     * destroyed in a reaction, unless that species is defined as a boundary
     * condition in the model.
     *
     * * In the case of a species reference, a RateRule sets the rate of change
     * of the stoichiometry of the referenced reactant or product to the value
     * determined by the formula in math. The unit associated with the value
     * produced by the formula should be consistent with {unit derived from
     * dimensionless}/{unit of time}.
     *
     * * In the case of a compartment, a RateRule sets the rate of change of
     * the compartmentÕs size to the value determined by the formula in math.
     * The unit of the ruleÕs math element should be identical to the compartmentÕs
     * units attribute divided by the model-wide unit of time. (In other words,
     * {unit of compartment size}/{unit of time}.)
     *
     * * In the case of a parameter, a RateRule sets the rate of change of the
     * parameterÕs value to that determined by the formula in math. The unit
     * associated with the ruleÕs math element should be equal to the parameterÕs
     * units attribute value divided by the model-wide unit of time. (In other
     * words, {parameter units}/{unit of time}.)
     */
    virtual bool visit(const libsbml::RateRule &rule);

    /**
     * visit the reactions so we can get all the SpeciesReferences and stuff
     * them in the initialConditions map.
     */
    virtual bool visit(const libsbml::Reaction  &x);

    /**
     * tell the acceptor to process all rules, even the ones
     * we don't handle so the iteration continues over all rules.
     *
     * The left-hand side (the variable attribute) of an assignment rule can
     * refer to the identifier of a Species, SpeciesReference, Compartment,
     * or global Parameter object in the model (but not a reaction)
     */
    virtual bool visit(const libsbml::Rule &x);

    /**
     * fatal error if we find these, event not supported yet
     */
    virtual bool visit (const libsbml::Event &event);

    /**
     * The only differences in how initialAssigments and assignmentRules
     * are handled is whether they are stuffed in the initialAssigment
     * or assigmentRules maps.
     *
     * This figures out what they refer to, and stuffs the AST in the
     * appropriate map.
     */
    void processElement(SymbolForest &currentSymbols,
            const libsbml::SBase *element, const libsbml::ASTNode *math);

    /**
     * specialized logic to write both amounts and concentrations here.
     */
    void processSpecies(SymbolForest &currentSymbols,
            const libsbml::Species *element, const libsbml::ASTNode *math);

    /**
     * get the MathML element for a SpeciesReference if it is set, otherwise,
     * create a ASTNode from its stoichiometry field.
     */
    const libsbml::ASTNode *getSpeciesReferenceStoichMath(const libsbml::SpeciesReference *reference);

    SymbolForest initialValues;

	// Have to change the forest to include arrays Vin
    SymbolForest assigmentRules;

    SymbolForest initialAssignmentRules;

    SymbolForest rateRules;

    /**
     * all ASTNodes we create are stored here.
     */
    ASTNodeFactory nodes;

    /**
     * a species can appear more than once in a reaction, there can be
     * several stoichiometries for a species both as a product and as
     * a reactant, for example, we could have
     *
     * A + 2A + B -> A + C
     *
     * In this case, we calculate the stochiometry for each species as
     * S(A) = -1 -2 + 1 = -1
     * S(B) = -1 + 0 = -1
     * S(C) = 0 + 1 = 1
     *
     * Stochiometries can however change, so we have to delay evaluation
     * of them -- if we have a named species reference, we add an ASTNode
     * with a reference to it: this way, when it get evaluated, we can look
     * up to see if we have any assigment rules or intitial assigments
     * overriding the original value.
     */

    /**
     * each species has a list of stoichiometry nodes, this way it can appear
     * more than once.
     */
    typedef std::list<const libsbml::ASTNode*> ASTNodeList;

    /**
     * we reference the reactants and products by the species index
     */
    typedef std::map<int, ASTNodeList> IntASTNodeListMap;

    /**
     * hold the symbols that belong to a reactions.
     */
    struct ReactionSymbols {
        IntASTNodeListMap reactants;
        IntASTNodeListMap products;
    };

    /**
     * these are indexed by reaction index.
     *
     * the stoichiometry matrix is a N species x N reaction matrix.
     */
    std::vector<ReactionSymbols> reactions;

private:
    libsbml::Model const
    *model;
    const LLVMModelDataSymbols &symbols;
};

} /* namespace rr */
#endif /* LLVMLLVMModelSymbolsH */
