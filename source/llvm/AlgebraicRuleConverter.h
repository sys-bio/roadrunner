/*
 * AlgebraicRuleConverter.h
 *
 * Rewrites linear SBML algebraic rules as assignment rules, so that the rest of
 * the LLVM back end only ever sees rules it already knows how to compile.
 */

#ifndef AlgebraicRuleConverter_h
#define AlgebraicRuleConverter_h

#include "rrExporter.h"

#include <sbml/SBMLDocument.h>

namespace rrllvm {

    /**
     * @brief Returns a copy of @p doc with its algebraic rules replaced.
     *
     * Each rule '0 = c * x + r' becomes the assignment rule 'x = -r / c', where
     * x is a symbol that nothing else in the model determines and c is constant.
     * Rules with no math are dropped, as the SBML test suite requires.
     *
     * When a rule could determine several symbols, the one chosen is, in order:
     * a symbol with no initial value, then one that is not a boundary species.
     * Symbols changed by an event are never chosen, because an assignment rule
     * would immediately overwrite the event's result. Nor are symbols that rateOf
     * is applied to anywhere in the model, as SBML Level 3 Version 2 requires.
     * SBML itself only requires that some valid choice exists, so the order
     * above follows the expected results in the SBML Test Suite.
     *
     * Rules that share their unknowns are solved together by Gaussian elimination.
     * The result is still a set of assignment rules, written as formulas so that
     * changing a constant after loading is honoured. The value of each constant
     * at load only decides the elimination order and whether a solution exists.
     *
     * @param doc the document to read. It is not changed.
     * @return a new document owned by the caller, or nullptr if @p doc has no
     * algebraic rules and can be used as it is.
     * @throws rrllvm::LLVMException if a rule is nonlinear, has a coefficient
     * that can change during a simulation or isn't known at load, can't determine
     * any symbol, or is part of a set of rules with no unique solution.
     */
    RR_DECLSPEC libsbml::SBMLDocument* convertAlgebraicRules(const libsbml::SBMLDocument* doc);

}

#endif /* AlgebraicRuleConverter_h */
