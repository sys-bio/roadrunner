/*
 * AlgebraicRuleConverter.cpp
 */
#pragma hdrstop

#include "AlgebraicRuleConverter.h"
#include "LLVMException.h"

#include <sbml/SBMLTransforms.h>
#include <sbml/SBMLTypes.h>
#include <sbml/math/ASTNode.h>
#include <sbml/math/L3FormulaFormatter.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace libsbml;

namespace rrllvm {

    namespace {

        using ASTPtr = std::unique_ptr<ASTNode>;
        using SymbolSet = std::set<std::string>;

        /** a rule that has not been solved yet */
        struct PendingRule {
            ASTPtr math;
            SymbolSet candidates;
        };

        /** node = coefficient * x + rest. A null pointer stands for zero. */
        struct LinearSplit {
            ASTPtr coefficient;
            ASTPtr rest;
        };

        struct SolvedRule {
            std::string variable;
            ASTPtr solution;
        };

        std::string formulaOf(const ASTNode* math) {
            char* formula = SBML_formulaToL3String(math);
            std::string result = formula ? formula : "";
            free(formula);
            return result;
        }

        [[noreturn]] void throwUnsupported(const ASTNode* math, const std::string& reason) {
            std::stringstream err;
            err << "Unable to support algebraic rules.  The formula '0 = " << formulaOf(math)
                << "' is not supported: " << reason;
            throw_llvm_exception(err.str());
        }

        ASTPtr clone(const ASTNode* node) {
            return ASTPtr(node->deepCopy());
        }

        ASTPtr makeNumber(double value) {
            ASTPtr node(new ASTNode(AST_REAL));
            node->setValue(value);
            return node;
        }

        ASTPtr makeBinary(ASTNodeType_t type, ASTPtr left, ASTPtr right) {
            ASTPtr node(new ASTNode(type));
            node->addChild(left.release());
            node->addChild(right.release());
            return node;
        }

        ASTPtr makeNegative(ASTPtr operand) {
            ASTPtr node(new ASTNode(AST_MINUS));
            node->addChild(operand.release());
            return node;
        }

        void collectNames(const ASTNode* node, SymbolSet& names) {
            if (node->getType() == AST_NAME) {
                names.insert(node->getName());
            }
            for (unsigned i = 0; i < node->getNumChildren(); ++i) {
                collectNames(node->getChild(i), names);
            }
        }

        SymbolSet namesIn(const ASTNode* node) {
            SymbolSet names;
            collectNames(node, names);
            return names;
        }

        bool mentions(const ASTNode* node, const std::string& symbol) {
            return namesIn(node).count(symbol) > 0;
        }

        /**
         * Replaces each symbol in @p replacements inside @p node. The argument of
         * rateOf is left alone, because rateOf only accepts a plain symbol.
         * libsbml's replaceArgument isn't used as it skips the root of the tree.
         */
        void substitute(ASTPtr& node, const std::map<std::string, const ASTNode*>& replacements) {
            if (node->getType() == AST_NAME) {
                auto found = replacements.find(node->getName());
                if (found != replacements.end()) {
                    node = clone(found->second);
                }
                return;
            }
            if (node->getType() == AST_FUNCTION_RATE_OF) {
                return;
            }
            for (unsigned i = 0; i < node->getNumChildren(); ++i) {
                ASTPtr child = clone(node->getChild(i));
                substitute(child, replacements);
                node->replaceChild(i, child.release(), true);
            }
        }

        ASTPtr addTerms(ASTPtr left, ASTPtr right) {
            if (!left) return right;
            if (!right) return left;
            return makeBinary(AST_PLUS, std::move(left), std::move(right));
        }

        ASTPtr subtractTerms(ASTPtr left, ASTPtr right) {
            if (!right) return left;
            if (!left) return makeNegative(std::move(right));
            return makeBinary(AST_MINUS, std::move(left), std::move(right));
        }

        /**
         * Splits @p node into coefficient * @p x + rest. The two parts are built
         * symbolically instead of by evaluating the formula at two points, which
         * would lose the coefficient next to large terms.
         *
         * @return false if the formula isn't linear in x.
         */
        bool splitLinear(const ASTNode* node, const std::string& x, LinearSplit& out) {
            if (!mentions(node, x)) {
                out.rest = clone(node);
                return true;
            }

            const unsigned count = node->getNumChildren();
            switch (node->getType()) {
            case AST_NAME:
                out.coefficient = makeNumber(1);
                return true;

            case AST_PLUS:
                for (unsigned i = 0; i < count; ++i) {
                    LinearSplit part;
                    if (!splitLinear(node->getChild(i), x, part)) {
                        return false;
                    }
                    out.coefficient = addTerms(std::move(out.coefficient), std::move(part.coefficient));
                    out.rest = addTerms(std::move(out.rest), std::move(part.rest));
                }
                return true;

            case AST_MINUS: {
                LinearSplit left;
                if (!splitLinear(node->getChild(0), x, left)) {
                    return false;
                }
                if (count == 1) {
                    out.coefficient = subtractTerms(nullptr, std::move(left.coefficient));
                    out.rest = subtractTerms(nullptr, std::move(left.rest));
                    return true;
                }
                LinearSplit right;
                if (!splitLinear(node->getChild(1), x, right)) {
                    return false;
                }
                out.coefficient = subtractTerms(std::move(left.coefficient), std::move(right.coefficient));
                out.rest = subtractTerms(std::move(left.rest), std::move(right.rest));
                return true;
            }

            case AST_TIMES: {
                // x may only appear in one factor, the others scale that factor.
                LinearSplit factor;
                ASTPtr scale;
                bool found = false;
                for (unsigned i = 0; i < count; ++i) {
                    const ASTNode* child = node->getChild(i);
                    if (mentions(child, x)) {
                        if (found || !splitLinear(child, x, factor)) {
                            return false;
                        }
                        found = true;
                    }
                    else {
                        scale = scale ? makeBinary(AST_TIMES, std::move(scale), clone(child)) : clone(child);
                    }
                }
                auto scaled = [&scale](ASTPtr term) -> ASTPtr {
                    if (!term || !scale) return term;
                    return makeBinary(AST_TIMES, clone(scale.get()), std::move(term));
                };
                out.coefficient = scaled(std::move(factor.coefficient));
                out.rest = scaled(std::move(factor.rest));
                return true;
            }

            case AST_DIVIDE: {
                const ASTNode* denominator = node->getChild(1);
                LinearSplit top;
                if (mentions(denominator, x) || !splitLinear(node->getChild(0), x, top)) {
                    return false;
                }
                auto divided = [denominator](ASTPtr term) -> ASTPtr {
                    if (!term) return term;
                    return makeBinary(AST_DIVIDE, std::move(term), clone(denominator));
                };
                out.coefficient = divided(std::move(top.coefficient));
                out.rest = divided(std::move(top.rest));
                return true;
            }

            default:
                return false;
            }
        }

        /** true if @p node can't change during a simulation. */
        bool isConstantExpression(const ASTNode* node, const SymbolSet& constants) {
            switch (node->getType()) {
            case AST_NAME:
                if (constants.count(node->getName()) == 0) {
                    return false;
                }
                break;
            case AST_NAME_TIME:
            case AST_FUNCTION_DELAY:
            case AST_FUNCTION_RATE_OF:
                return false;
            default:
                break;
            }
            for (unsigned i = 0; i < node->getNumChildren(); ++i) {
                if (!isConstantExpression(node->getChild(i), constants)) {
                    return false;
                }
            }
            return true;
        }

        /** true for a formula such as '1 - 1' where x has cancelled out. */
        bool isLiteralZero(const ASTNode* node) {
            return namesIn(node).empty() && SBMLTransforms::evaluateASTNode(node) == 0.0;
        }

        /** Everything in a model that decides how its symbols may be treated. */
        struct ModelSymbols {
            SymbolSet constants;    // never change
            SymbolSet determined;   // already set by a rule or a reaction
            SymbolSet undetermined; // free for an algebraic rule to set
            SymbolSet eventTargets;
            SymbolSet rateOfTargets;
            SymbolSet boundarySpecies;
            SymbolSet withoutValue;
            std::map<std::string, double> constantValues; // when known at load
        };

        /** Adds every symbol that rateOf is applied to inside @p node. */
        void collectRateOfTargets(const ASTNode* node, SymbolSet& targets) {
            if (node->getType() == AST_FUNCTION_RATE_OF) {
                collectNames(node, targets);
                return;
            }
            for (unsigned i = 0; i < node->getNumChildren(); ++i) {
                collectRateOfTargets(node->getChild(i), targets);
            }
        }

        /** The value of a constant expression, or NaN if it can't be worked out. */
        double valueAtLoad(const ASTNode* node, const std::map<std::string, double>& values) {
            for (const std::string& name : namesIn(node)) {
                if (values.count(name) == 0) {
                    return std::numeric_limits<double>::quiet_NaN();
                }
            }
            return SBMLTransforms::evaluateASTNode(node, values);
        }

        ModelSymbols readSymbols(const Model* model, const SBMLDocument* doc) {
            ModelSymbols s;
            SymbolSet changeable;
            auto add = [&](const std::string& id, bool constant, bool hasValue, double value) {
                (constant ? s.constants : changeable).insert(id);
                if (!hasValue) {
                    s.withoutValue.insert(id);
                }
                else if (constant) {
                    s.constantValues[id] = value;
                }
            };

            // Level 1 has no 'constant' attribute, and libsbml reports parameters
            // and compartments as constant there even though rules can change them.
            const bool hasConstantAttribute = doc->getLevel() > 1;

            for (unsigned i = 0; i < model->getNumParameters(); ++i) {
                const Parameter* p = model->getParameter(i);
                add(p->getId(), hasConstantAttribute && p->getConstant(), p->isSetValue(), p->getValue());
            }
            for (unsigned i = 0; i < model->getNumCompartments(); ++i) {
                const Compartment* c = model->getCompartment(i);
                add(c->getId(), hasConstantAttribute && c->getConstant(), c->isSetSize() || c->isSetVolume(), c->getSize());
            }
            for (unsigned i = 0; i < model->getNumSpecies(); ++i) {
                const Species* sp = model->getSpecies(i);
                // Only used to tell whether a coefficient is zero, which an amount and
                // a concentration agree on.
                add(sp->getId(), sp->getConstant(), sp->isSetInitialAmount() || sp->isSetInitialConcentration(),
                    sp->isSetInitialConcentration() ? sp->getInitialConcentration() : sp->getInitialAmount());
                if (sp->getBoundaryCondition()) {
                    s.boundarySpecies.insert(sp->getId());
                }
            }

            for (unsigned i = 0; i < model->getNumRules(); ++i) {
                const Rule* rule = model->getRule(i);
                if (!rule->isAlgebraic()) {
                    s.determined.insert(rule->getVariable());
                }
            }

            for (unsigned i = 0; i < model->getNumReactions(); ++i) {
                const Reaction* reaction = model->getReaction(i);
                s.determined.insert(reaction->getId());
                for (const ListOfSpeciesReferences* list : { reaction->getListOfReactants(), reaction->getListOfProducts() }) {
                    for (unsigned j = 0; j < list->size(); ++j) {
                        const SpeciesReference* ref = static_cast<const SpeciesReference*>(list->get(j));
                        if (s.boundarySpecies.count(ref->getSpecies()) == 0) {
                            s.determined.insert(ref->getSpecies());
                        }
                        if (ref->isSetId()) {
                            add(ref->getId(), doc->getLevel() < 3 || ref->getConstant(), ref->isSetStoichiometry(), ref->getStoichiometry());
                        }
                    }
                }
            }

            for (unsigned i = 0; i < model->getNumEvents(); ++i) {
                const Event* event = model->getEvent(i);
                for (unsigned j = 0; j < event->getNumEventAssignments(); ++j) {
                    s.eventTargets.insert(event->getEventAssignment(j)->getVariable());
                }
            }

            // SBML Level 3 Version 2 doesn't let an algebraic rule determine a
            // symbol that rateOf is applied to anywhere in the model.
            auto addRateOfTargets = [&s](const ASTNode* math) {
                if (math) {
                    collectRateOfTargets(math, s.rateOfTargets);
                }
            };
            for (unsigned i = 0; i < model->getNumRules(); ++i) {
                addRateOfTargets(model->getRule(i)->getMath());
            }
            for (unsigned i = 0; i < model->getNumInitialAssignments(); ++i) {
                addRateOfTargets(model->getInitialAssignment(i)->getMath());
            }
            for (unsigned i = 0; i < model->getNumConstraints(); ++i) {
                addRateOfTargets(model->getConstraint(i)->getMath());
            }
            for (unsigned i = 0; i < model->getNumReactions(); ++i) {
                const Reaction* reaction = model->getReaction(i);
                if (reaction->isSetKineticLaw()) {
                    addRateOfTargets(reaction->getKineticLaw()->getMath());
                }
            }
            for (unsigned i = 0; i < model->getNumEvents(); ++i) {
                const Event* event = model->getEvent(i);
                if (event->isSetTrigger()) {
                    addRateOfTargets(event->getTrigger()->getMath());
                }
                if (event->isSetDelay()) {
                    addRateOfTargets(event->getDelay()->getMath());
                }
                if (event->isSetPriority()) {
                    addRateOfTargets(event->getPriority()->getMath());
                }
                for (unsigned j = 0; j < event->getNumEventAssignments(); ++j) {
                    addRateOfTargets(event->getEventAssignment(j)->getMath());
                }
            }

            std::set_difference(changeable.begin(), changeable.end(),
                s.determined.begin(), s.determined.end(),
                std::inserter(s.undetermined, s.undetermined.end()));

            // An initial assignment can depend on another one, so repeat until
            // every chain has been followed.
            for (unsigned pass = 0; pass < model->getNumInitialAssignments(); ++pass) {
                for (unsigned i = 0; i < model->getNumInitialAssignments(); ++i) {
                    const InitialAssignment* assignment = model->getInitialAssignment(i);
                    if (s.constants.count(assignment->getSymbol()) == 0 || !assignment->isSetMath()) {
                        continue;
                    }
                    ASTPtr math = clone(assignment->getMath());
                    SBMLTransforms::replaceFD(math.get(), model->getListOfFunctionDefinitions());
                    const double value = valueAtLoad(math.get(), s.constantValues);
                    if (!std::isnan(value)) {
                        s.constantValues[assignment->getSymbol()] = value;
                    }
                }
            }
            return s;
        }

        /** Lower sorts first: the order in which symbols should be solved for. */
        int preference(const std::string& symbol, const ModelSymbols& symbols) {
            int rank = symbols.withoutValue.count(symbol) ? 0 : 2;
            return rank + (symbols.boundarySpecies.count(symbol) ? 1 : 0);
        }

        /** The symbols that @p math can be solved for. */
        SymbolSet solvableSymbols(const ASTNode* math, const ModelSymbols& symbols) {
            SymbolSet result;
            for (const std::string& name : namesIn(math)) {
                if (symbols.undetermined.count(name) == 0 || symbols.eventTargets.count(name) > 0
                    || symbols.rateOfTargets.count(name) > 0) {
                    continue;
                }
                LinearSplit split;
                if (splitLinear(math, name, split) && split.coefficient
                    && isConstantExpression(split.coefficient.get(), symbols.constants)
                    && !isLiteralZero(split.coefficient.get())) {
                    result.insert(name);
                }
            }
            return result;
        }

        SolvedRule solveRule(const ASTNode* math, const std::string& x) {
            LinearSplit split;
            splitLinear(math, x, split);
            ASTPtr negated = split.rest ? makeNegative(std::move(split.rest)) : makeNumber(0);
            return { x, makeBinary(AST_DIVIDE, std::move(negated), std::move(split.coefficient)) };
        }

        /**
         * One coefficient of the system. The formula is what ends up in the model,
         * so a constant changed after loading is still honoured. The value is only
         * used to choose pivots. A null formula stands for zero.
         */
        struct Coefficient {
            ASTPtr math;
            double value = 0;
        };

        Coefficient numberCoefficient(double value) {
            Coefficient result;
            result.value = value;
            if (value != 0) {
                result.math = makeNumber(value);
            }
            return result;
        }

        /** Plain numbers are folded, which stops formulas growing at every step. */
        Coefficient combined(ASTPtr math, double value) {
            if (namesIn(math.get()).empty()) {
                return numberCoefficient(value);
            }
            Coefficient result;
            result.math = std::move(math);
            result.value = value;
            return result;
        }

        /** @return a - (top / bottom) * b */
        Coefficient eliminated(const Coefficient& a, const Coefficient& top, const Coefficient& bottom, const Coefficient& b) {
            if (!b.math) {
                Coefficient copy;
                copy.value = a.value;
                copy.math = a.math ? clone(a.math.get()) : nullptr;
                return copy;
            }
            ASTPtr factor = makeBinary(AST_DIVIDE, clone(top.math.get()), clone(bottom.math.get()));
            ASTPtr product = makeBinary(AST_TIMES, std::move(factor), clone(b.math.get()));
            ASTPtr left = a.math ? clone(a.math.get()) : nullptr;
            return combined(subtractTerms(std::move(left), std::move(product)), a.value - top.value / bottom.value * b.value);
        }

        /** The symbols that every rule mentioning them can be solved for. */
        std::vector<std::string> sharedUnknowns(const std::vector<PendingRule>& rules, const SymbolSet& available) {
            SymbolSet unknowns;
            for (const PendingRule& rule : rules) {
                for (const std::string& name : rule.candidates) {
                    if (available.count(name)) {
                        unknowns.insert(name);
                    }
                }
            }
            for (const PendingRule& rule : rules) {
                for (const std::string& name : namesIn(rule.math.get())) {
                    if (rule.candidates.count(name) == 0) {
                        unknowns.erase(name);
                    }
                }
            }
            return { unknowns.begin(), unknowns.end() };
        }

        /**
         * Solves rules that share their unknowns, with Gaussian elimination on
         *
         *     0 = c1 * x1 + c2 * x2 + ... + 1 * rest
         *
         * Each row also has one column per rule for its 'rest', so that each
         * 'rest' is written only once however many steps the elimination takes.
         * After elimination, every solved symbol depends only on the symbols
         * solved after it, so the results are ordinary chained assignment rules.
         * Unknowns left without a pivot stay free and are treated like parameters.
         */
        void solveSimultaneously(const std::vector<PendingRule>& rules, const SymbolSet& available,
            const ModelSymbols& symbols, std::vector<SolvedRule>& solved) {
            const std::vector<std::string> unknowns = sharedUnknowns(rules, available);
            const size_t columns = unknowns.size() + rules.size();

            std::vector<ASTPtr> rests;
            std::vector<std::vector<Coefficient>> rows(rules.size());
            std::vector<const ASTNode*> rowRule;
            double largest = 0;

            for (size_t r = 0; r < rules.size(); ++r) {
                rows[r].resize(columns);
                rowRule.push_back(rules[r].math.get());
                ASTPtr rest = clone(rules[r].math.get());
                for (size_t c = 0; c < unknowns.size() && rest; ++c) {
                    if (!mentions(rest.get(), unknowns[c])) {
                        continue;
                    }
                    LinearSplit split;
                    splitLinear(rest.get(), unknowns[c], split);
                    const double value = valueAtLoad(split.coefficient.get(), symbols.constantValues);
                    if (std::isnan(value)) {
                        throwUnsupported(rules[r].math.get(), "the value of a coefficient is not known when the model loads.");
                    }
                    rows[r][c] = combined(std::move(split.coefficient), value);
                    largest = std::max(largest, std::fabs(value));
                    rest = std::move(split.rest);
                }
                rests.push_back(std::move(rest));
                rows[r][unknowns.size() + r] = numberCoefficient(1);
            }

            std::vector<size_t> order(unknowns.size());
            for (size_t c = 0; c < order.size(); ++c) {
                order[c] = c;
            }
            std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                return preference(unknowns[a], symbols) < preference(unknowns[b], symbols);
            });

            const double tolerance = largest * 1e-12;
            std::vector<size_t> pivotColumn;
            for (size_t k = 0; k < rows.size(); ++k) {
                size_t pivotRow = rows.size();
                size_t chosen = 0;
                double best = tolerance;
                for (size_t c : order) {
                    if (std::find(pivotColumn.begin(), pivotColumn.end(), c) != pivotColumn.end()) {
                        continue;
                    }
                    for (size_t r = k; r < rows.size(); ++r) {
                        if (std::fabs(rows[r][c].value) > best) {
                            best = std::fabs(rows[r][c].value);
                            pivotRow = r;
                            chosen = c;
                        }
                    }
                    if (pivotRow != rows.size()) {
                        break;
                    }
                }
                if (pivotRow == rows.size()) {
                    throwUnsupported(rowRule[k],
                        "the algebraic rules it must be solved simultaneously with have no unique linear solution.");
                }

                std::swap(rows[k], rows[pivotRow]);
                std::swap(rowRule[k], rowRule[pivotRow]);
                pivotColumn.push_back(chosen);

                for (size_t r = k + 1; r < rows.size(); ++r) {
                    if (!rows[r][chosen].math) {
                        continue;
                    }
                    std::vector<Coefficient> reduced(columns);
                    for (size_t c = 0; c < columns; ++c) {
                        if (c != chosen) {
                            reduced[c] = eliminated(rows[r][c], rows[r][chosen], rows[k][chosen], rows[k][c]);
                        }
                    }
                    rows[r] = std::move(reduced);
                }
            }

            for (size_t k = 0; k < rows.size(); ++k) {
                ASTPtr sum;
                for (size_t c = 0; c < columns; ++c) {
                    const Coefficient& cell = rows[k][c];
                    if (!cell.math || c == pivotColumn[k]) {
                        continue;
                    }
                    ASTPtr symbol;
                    if (c < unknowns.size()) {
                        symbol.reset(new ASTNode(AST_NAME));
                        symbol->setName(unknowns[c].c_str());
                    }
                    else if (rests[c - unknowns.size()]) {
                        symbol = clone(rests[c - unknowns.size()].get());
                    }
                    if (symbol) {
                        sum = addTerms(std::move(sum), makeBinary(AST_TIMES, clone(cell.math.get()), std::move(symbol)));
                    }
                }
                ASTPtr negated = sum ? makeNegative(std::move(sum)) : makeNumber(0);
                solved.push_back({ unknowns[pivotColumn[k]],
                    makeBinary(AST_DIVIDE, std::move(negated), clone(rows[k][pivotColumn[k]].math.get())) });
            }
        }

        /**
         * Pairs every rule with a symbol. A rule with a single option goes first.
         * After that, a symbol used by just one remaining rule can be taken without
         * affecting the others. If neither applies, the remaining rules depend on
         * each other and are solved simultaneously.
         */
        std::vector<SolvedRule> solveRules(std::vector<PendingRule>& rules, const ModelSymbols& symbols) {
            std::vector<SolvedRule> solved;
            SymbolSet available;
            for (const PendingRule& rule : rules) {
                available.insert(rule.candidates.begin(), rule.candidates.end());
            }

            auto optionsOf = [&available](const PendingRule& rule) {
                SymbolSet options;
                for (const std::string& name : rule.candidates) {
                    if (available.count(name)) {
                        options.insert(name);
                    }
                }
                return options;
            };
            auto takeRule = [&](size_t index, const std::string& symbol) {
                solved.push_back(solveRule(rules[index].math.get(), symbol));
                available.erase(symbol);
                rules.erase(rules.begin() + index);
            };

            while (!rules.empty()) {
                bool progress = false;
                for (size_t i = 0; i < rules.size() && !progress; ++i) {
                    const SymbolSet options = optionsOf(rules[i]);
                    if (options.size() == 1) {
                        takeRule(i, *options.begin());
                        progress = true;
                    }
                }

                for (size_t i = 0; i < rules.size() && !progress; ++i) {
                    std::vector<std::string> unshared;
                    for (const std::string& option : optionsOf(rules[i])) {
                        const bool shared = std::any_of(rules.begin(), rules.end(), [&](const PendingRule& other) {
                            return &other != &rules[i] && mentions(other.math.get(), option);
                        });
                        if (!shared) {
                            unshared.push_back(option);
                        }
                    }
                    if (!unshared.empty()) {
                        std::stable_sort(unshared.begin(), unshared.end(), [&](const std::string& a, const std::string& b) {
                            return preference(a, symbols) < preference(b, symbols);
                        });
                        takeRule(i, unshared.front());
                        progress = true;
                    }
                }

                if (!progress) {
                    for (const PendingRule& rule : rules) {
                        if (optionsOf(rule).empty()) {
                            throwUnsupported(rule.math.get(),
                                "it is not linear with a constant coefficient in any symbol it could determine.");
                        }
                    }
                    solveSimultaneously(rules, available, symbols, solved);
                    rules.clear();
                }
            }
            return solved;
        }

        /** Throws if solutions refer to each other, directly or indirectly. */
        void rejectCycles(const std::vector<SolvedRule>& solved) {
            std::map<std::string, const SolvedRule*> byVariable;
            for (const SolvedRule& rule : solved) {
                byVariable[rule.variable] = &rule;
            }

            enum class Visit { Active, Done };
            std::map<std::string, Visit> visits;

            struct Walker {
                std::map<std::string, const SolvedRule*>& byVariable;
                std::map<std::string, Visit>& visits;

                void visit(const SolvedRule& rule) {
                    visits[rule.variable] = Visit::Active;
                    for (const std::string& name : namesIn(rule.solution.get())) {
                        auto next = byVariable.find(name);
                        if (next == byVariable.end()) {
                            continue;
                        }
                        auto seen = visits.find(name);
                        if (seen == visits.end()) {
                            visit(*next->second);
                        }
                        else if (seen->second == Visit::Active) {
                            throwUnsupported(next->second->solution.get(),
                                "it depends on another algebraic rule in a way that is not linear.");
                        }
                    }
                    visits[rule.variable] = Visit::Done;
                }
            };

            Walker walker{ byVariable, visits };
            for (const SolvedRule& rule : solved) {
                if (visits.count(rule.variable) == 0) {
                    walker.visit(rule);
                }
            }
        }

        /** Math of a rule, with function calls and assignment rules written out. */
        ASTPtr expandedMath(const ASTNode* math, const Model* model) {
            std::map<std::string, const ASTNode*> assignments;
            for (unsigned i = 0; i < model->getNumRules(); ++i) {
                const Rule* rule = model->getRule(i);
                if (rule->isAssignment() && rule->isSetMath()) {
                    assignments[rule->getVariable()] = rule->getMath();
                }
            }

            ASTPtr expanded = clone(math);
            // Assignment rules can't form loops, so this ends within one pass per rule.
            for (size_t pass = 0; pass <= assignments.size(); ++pass) {
                substitute(expanded, assignments);
                const SymbolSet names = namesIn(expanded.get());
                const bool done = std::none_of(names.begin(), names.end(), [&](const std::string& name) {
                    return assignments.count(name) > 0;
                });
                if (done) {
                    break;
                }
            }
            SBMLTransforms::replaceFD(expanded.get(), model->getListOfFunctionDefinitions());
            return expanded;
        }

        bool hasAlgebraicRule(const Model* model) {
            for (unsigned i = 0; i < model->getNumRules(); ++i) {
                if (model->getRule(i)->isAlgebraic()) {
                    return true;
                }
            }
            return false;
        }
    }

    SBMLDocument* convertAlgebraicRules(const SBMLDocument* doc) {
        const Model* source = doc->getModel();
        if (!source || !hasAlgebraicRule(source)) {
            return nullptr;
        }

        std::unique_ptr<SBMLDocument> converted(doc->clone());
        Model* model = converted->getModel();

        const ModelSymbols symbols = readSymbols(model, converted.get());

        std::vector<PendingRule> pending;
        for (unsigned i = model->getNumRules(); i-- > 0;) {
            const Rule* rule = model->getRule(i);
            if (rule->isAlgebraic() && rule->isSetMath()) {
                PendingRule item;
                item.math = expandedMath(rule->getMath(), model);
                item.candidates = solvableSymbols(item.math.get(), symbols);
                pending.push_back(std::move(item));
            }
        }
        std::reverse(pending.begin(), pending.end());

        const std::vector<SolvedRule> solved = solveRules(pending, symbols);
        rejectCycles(solved);

        for (unsigned i = model->getNumRules(); i-- > 0;) {
            if (model->getRule(i)->isAlgebraic()) {
                delete model->removeRule(i);
            }
        }
        for (const SolvedRule& rule : solved) {
            AssignmentRule* assignment = model->createAssignmentRule();
            assignment->setVariable(rule.variable);
            assignment->setMath(rule.solution.get());
        }
        return converted.release();
    }
}
