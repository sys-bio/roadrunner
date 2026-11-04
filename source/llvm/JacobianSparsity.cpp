#include "JacobianSparsity.h"

#include "LLVMModelDataSymbols.h"

#include <sbml/SBMLTypes.h>

#include <algorithm>
#include <numeric>

namespace rrllvm {

    JacobianSparsity::JacobianSparsity(const libsbml::Model &model, const LLVMModelDataSymbols &symbols)
            : model(model), symbols(symbols),
              nRateRules((int) symbols.getRateRuleSize()),
              nStates((int) (symbols.getRateRuleSize() + symbols.getIndependentFloatingSpeciesSize())) {

        // reaction rows
        int nReactions = (int) symbols.getReactionSize();
        std::vector<std::set<int>> reactionRows(nReactions);
        for (unsigned i = 0; i < model.getNumReactions(); ++i) {
            const libsbml::Reaction *r = model.getReaction(i);
            int row = symbols.getReactionIndex(r->getId());
            if (row < 0 || !r->isSetKineticLaw() || !r->getKineticLaw()->isSetMath()) {
                continue;
            }
            // local parameters shadow the global symbols and are constant
            Env env;
            const libsbml::KineticLaw *kl = r->getKineticLaw();
            for (unsigned k = 0; k < kl->getNumParameters(); ++k) {
                env[kl->getParameter(k)->getId()] = Deps();
            }
            for (unsigned k = 0; k < kl->getNumLocalParameters(); ++k) {
                env[kl->getLocalParameter(k)->getId()] = Deps();
            }
            astDeps(kl->getMath(), env, reactionRows[row]);
        }

        // rate rule rows
        std::vector<std::set<int>> rateRuleRows(nRateRules);
        for (int i = 0; i < nRateRules; ++i) {
            const std::string &id = symbols.getRateRuleId(i);
            const libsbml::Rule *rule = model.getRateRuleByVariable(id);
            if (rule != nullptr && rule->isSetMath()) {
                astDeps(rule->getMath(), Env(), rateRuleRows[i]);
            }
            // roadrunner integrates the amount of a species whose rate rule is in concentration:
            // dn/dt = c * dS/dt, plus S * dc/dt when the compartment has a rate rule itself
            // (EvalRateRuleRatesCodeGen)
            const libsbml::Species *species = model.getSpecies(id);
            if (species != nullptr && !species->getHasOnlySubstanceUnits()) {
                Deps c = symbolDeps(species->getCompartment());
                rateRuleRows[i].insert(c.begin(), c.end());
                const libsbml::Rule *compartmentRule = model.getRateRuleByVariable(species->getCompartment());
                if (compartmentRule != nullptr && compartmentRule->isSetMath()) {
                    Deps s = symbolDeps(id);
                    rateRuleRows[i].insert(s.begin(), s.end());
                    astDeps(compartmentRule->getMath(), Env(), rateRuleRows[i]);
                }
            }
        }

        // Jacobian rows: rate rules, then species as N * dv/dy, plus the diagonal
        std::vector<std::set<int>> jacRows(nStates);
        for (int i = 0; i < nRateRules; ++i) {
            jacRows[i] = rateRuleRows[i];
        }
        int nIndFloating = nStates - nRateRules;
        for (const auto &ref: symbols.getStoichiometryList()) {
            if (ref.type == LLVMModelDataSymbols::Modifier || (int) ref.row >= nIndFloating ||
                (int) ref.column >= nReactions) {
                continue;
            }
            const std::set<int> &v = reactionRows[ref.column];
            jacRows[nRateRules + ref.row].insert(v.begin(), v.end());
        }
        for (int i = 0; i < nStates; ++i) {
            jacRows[i].insert(i);
        }

        dvdy = CscPattern::fromRows(nStates, reactionRows);
        drdy = CscPattern::fromRows(nStates, rateRuleRows);
        jac = CscPattern::fromRows(nStates, jacRows);
        color();
    }

    bool JacobianSparsity::dependsOnState(const std::string &id) {
        return !symbolDeps(id).empty();
    }

    namespace {
        /** Why J = cf * N * dv/dy does not hold for the model, or an empty string. */
        std::string unsupported(const libsbml::Model &model, const LLVMModelDataSymbols &symbols) {
            for (unsigned i = 0; i < model.getNumReactions(); ++i) {
                const libsbml::Reaction *r = model.getReaction(i);
                for (const libsbml::ListOfSpeciesReferences *refs: {r->getListOfReactants(), r->getListOfProducts()}) {
                    for (unsigned k = 0; k < refs->size(); ++k) {
                        auto *ref = static_cast<const libsbml::SpeciesReference *>(refs->get(k));
                        bool ruled = ref->isSetId() && !ref->getId().empty() &&
                                     (symbols.hasAssignmentRule(ref->getId()) || symbols.hasRateRule(ref->getId()));
                        if (ruled || ref->isSetStoichiometryMath()) {
                            return "the stoichiometry of reaction '" + r->getId() +
                                   "' is set by a rule (volatile stoichiometry)";
                        }
                    }
                }
            }
            for (unsigned i = 0; i < model.getNumSpecies(); ++i) {
                const libsbml::Species *s = model.getSpecies(i);
                if (s->isSetConversionFactor()) {
                    return "species '" + s->getId() + "' has a conversion factor";
                }
            }
            return "";
        }
    }

    JacobianStructure JacobianSparsity::analyze(const libsbml::Model &model, const LLVMModelDataSymbols &symbols,
                                                bool conservedMoieties) {
        JacobianStructure res;
        if (conservedMoieties) {
            res.unavailableReason = "conserved moiety analysis is switched on";
            return res;
        }
        std::string reason = unsupported(model, symbols);
        if (!reason.empty()) {
            res.unavailableReason = reason;
            return res;
        }
        JacobianSparsity sparsity(model, symbols);
        if (model.isSetConversionFactor() && sparsity.dependsOnState(model.getConversionFactor())) {
            res.unavailableReason = "the conversion factor '" + model.getConversionFactor() + "' depends on the state";
            return res;
        }
        res.reactionRates = sparsity.reactionRates();
        res.rateRules = sparsity.rateRules();
        res.jacobian = sparsity.jacobian();
        res.colors = sparsity.colors();
        res.numColors = sparsity.numColors();
        res.unavailableReason.clear();
        return res;
    }

    JacobianSparsity::Deps JacobianSparsity::allColumns() const {
        Deps d;
        for (int j = 0; j < nStates; ++j) {
            d.insert(j);
        }
        return d;
    }

    JacobianSparsity::Deps JacobianSparsity::symbolDeps(const std::string &id) {
        auto cached = symbolCache.find(id);
        if (cached != symbolCache.end()) {
            return cached->second;
        }
        if (inProgress.count(id)) {
            // a cycle of rules cannot be resolved here; stay conservative
            return allColumns();
        }
        inProgress.insert(id);

        Deps d;
        int rateRule = symbols.getRateRuleIndex(id);
        if (rateRule >= 0) {
            d.insert(rateRule);
        } else if (const libsbml::Rule *rule = model.getAssignmentRuleByVariable(id)) {
            if (rule->isSetMath()) {
                astDeps(rule->getMath(), Env(), d);
            }
        } else if (symbols.isIndependentFloatingSpecies(id)) {
            d.insert(nRateRules + symbols.getFloatingSpeciesIndex(id));
        } else if (const libsbml::Reaction *r = model.getReaction(id)) {
            // a reaction id in math is the rate of the reaction
            if (r->isSetKineticLaw() && r->getKineticLaw()->isSetMath()) {
                Env env;
                const libsbml::KineticLaw *kl = r->getKineticLaw();
                for (unsigned k = 0; k < kl->getNumParameters(); ++k) {
                    env[kl->getParameter(k)->getId()] = Deps();
                }
                for (unsigned k = 0; k < kl->getNumLocalParameters(); ++k) {
                    env[kl->getLocalParameter(k)->getId()] = Deps();
                }
                astDeps(kl->getMath(), env, d);
            }
        }
        // a species read as a concentration also depends on the volume of its compartment
        if (const libsbml::Species *s = model.getSpecies(id)) {
            Deps c = symbolDeps(s->getCompartment());
            d.insert(c.begin(), c.end());
        }

        inProgress.erase(id);
        symbolCache[id] = d;
        return d;
    }

    void JacobianSparsity::astDeps(const libsbml::ASTNode *node, const Env &env, Deps &out) {
        if (node == nullptr) {
            return;
        }
        switch (node->getType()) {
            case libsbml::AST_NAME: {
                auto bound = env.find(node->getName());
                if (bound != env.end()) {
                    out.insert(bound->second.begin(), bound->second.end());
                } else {
                    Deps d = symbolDeps(node->getName());
                    out.insert(d.begin(), d.end());
                }
                return;
            }
            case libsbml::AST_FUNCTION_DELAY:
            case libsbml::AST_FUNCTION_RATE_OF: {
                // the derivative of a delayed value or of a rate is not read from the math here
                Deps d = allColumns();
                out.insert(d.begin(), d.end());
                return;
            }
            case libsbml::AST_FUNCTION: {
                const libsbml::FunctionDefinition *fd = model.getFunctionDefinition(node->getName());
                if (fd != nullptr && fd->getBody() != nullptr) {
                    Env inner;
                    for (unsigned i = 0; i < fd->getNumArguments() && i < node->getNumChildren(); ++i) {
                        Deps arg;
                        astDeps(node->getChild(i), env, arg);
                        inner[fd->getArgument(i)->getName()] = arg;
                    }
                    astDeps(fd->getBody(), inner, out);
                    return;
                }
                break;  // an unknown function: the union of its arguments
            }
            default:
                break;
        }
        for (unsigned i = 0; i < node->getNumChildren(); ++i) {
            astDeps(node->getChild(i), env, out);
        }
    }

    void JacobianSparsity::color() {
        // rows of the stacked [dv/dy; d(rate rule rates)/dy] and the columns of each
        std::vector<std::vector<int>> rowsOfColumn(nStates);
        int nRows = dvdy.nRows + drdy.nRows;
        std::vector<std::vector<int>> columnsOfRow(nRows);
        for (int j = 0; j < nStates; ++j) {
            for (int k = dvdy.colPtr[j]; k < dvdy.colPtr[j + 1]; ++k) {
                rowsOfColumn[j].push_back(dvdy.rowIdx[k]);
                columnsOfRow[dvdy.rowIdx[k]].push_back(j);
            }
            for (int k = drdy.colPtr[j]; k < drdy.colPtr[j + 1]; ++k) {
                rowsOfColumn[j].push_back(dvdy.nRows + drdy.rowIdx[k]);
                columnsOfRow[dvdy.nRows + drdy.rowIdx[k]].push_back(j);
            }
        }

        // greedy distance-1 coloring of the column intersection graph, densest columns first
        std::vector<int> order(nStates);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            return rowsOfColumn[a].size() > rowsOfColumn[b].size();
        });
        columnColors.assign(nStates, -1);
        nColors = 0;
        std::vector<int> forbidden(nStates + 1, -1);
        for (int j: order) {
            for (int row: rowsOfColumn[j]) {
                for (int other: columnsOfRow[row]) {
                    if (columnColors[other] >= 0) {
                        forbidden[columnColors[other]] = j;
                    }
                }
            }
            int c = 0;
            while (forbidden[c] == j) {
                ++c;
            }
            columnColors[j] = c;
            nColors = std::max(nColors, c + 1);
        }
    }

}
