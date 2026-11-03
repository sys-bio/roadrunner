#include "JacobianSparsity.h"

#include "LLVMModelDataSymbols.h"

#include <sbml/SBMLTypes.h>

#include <algorithm>
#include <numeric>

namespace rrllvm {

    CscPattern CscPattern::fromRows(int nCols, const std::vector<std::set<int>> &rows) {
        CscPattern p;
        p.nRows = (int) rows.size();
        p.nCols = nCols;
        std::vector<std::vector<int>> columns(nCols);
        for (int i = 0; i < p.nRows; ++i) {
            for (int j: rows[i]) {
                columns[j].push_back(i);
            }
        }
        p.colPtr.assign(nCols + 1, 0);
        for (int j = 0; j < nCols; ++j) {
            p.colPtr[j + 1] = p.colPtr[j] + (int) columns[j].size();
            p.rowIdx.insert(p.rowIdx.end(), columns[j].begin(), columns[j].end());
        }
        return p;
    }

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
            const libsbml::Rule *rule = model.getRateRuleByVariable(symbols.getRateRuleId(i));
            if (rule != nullptr && rule->isSetMath()) {
                astDeps(rule->getMath(), Env(), rateRuleRows[i]);
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
