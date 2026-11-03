#ifndef RR_LLVM_JACOBIAN_SPARSITY_H
#define RR_LLVM_JACOBIAN_SPARSITY_H

#include "rrSparsityPattern.h"

#include <map>
#include <set>
#include <string>
#include <vector>

namespace libsbml {
    class ASTNode;
    class Model;
}

namespace rrllvm {

    class LLVMModelDataSymbols;

    using rr::CscPattern;

    /**
     * @brief Conservative sparsity pattern of the Jacobian of a model, read from its SBML math,
     * and a column coloring for compressed forward mode differentiation.
     *
     * The state columns are ordered as the state vector CVODE integrates:
     * [rate rule values; independent floating species amounts]. A reaction rate or a rate rule
     * depends on a state if the state appears in its math, after expanding assignment rules and
     * function definitions, in any branch of any piecewise. A species also brings in its
     * compartment, since its concentration is its amount divided by the volume. The pattern is
     * therefore a superset of the nonzeros of every Jacobian the model can produce.
     */
    class JacobianSparsity {
    public:
        JacobianSparsity(const libsbml::Model &model, const LLVMModelDataSymbols &symbols);

        /** Pattern of the derivative of the reaction rates, reactions x states. */
        const CscPattern &reactionRates() const { return dvdy; }

        /** Pattern of the derivative of the rate rule rates, rate rules x states. */
        const CscPattern &rateRules() const { return drdy; }

        /**
         * Pattern of the Jacobian, states x states: the rate rule rows, the species rows of
         * N * dv/dy, and the full diagonal (CVODE forms I - gamma J).
         */
        const CscPattern &jacobian() const { return jac; }

        /**
         * Color of every state column. Columns of the same color share no row of the stacked
         * [dv/dy; d(rate rule rates)/dy], so one seed direction per color recovers both exactly.
         */
        const std::vector<int> &colors() const { return columnColors; }

        int numColors() const { return nColors; }

        /** True if the value of the symbol id depends on a state of the model. */
        bool dependsOnState(const std::string &id);

    private:
        typedef std::set<int> Deps;
        typedef std::map<std::string, Deps> Env;

        Deps symbolDeps(const std::string &id);
        void astDeps(const libsbml::ASTNode *node, const Env &env, Deps &out);
        Deps allColumns() const;
        void color();

        const libsbml::Model &model;
        const LLVMModelDataSymbols &symbols;
        int nRateRules;
        int nStates;

        std::map<std::string, Deps> symbolCache;
        std::set<std::string> inProgress;

        CscPattern dvdy;
        CscPattern drdy;
        CscPattern jac;
        std::vector<int> columnColors;
        int nColors = 0;
    };

}

#endif // RR_LLVM_JACOBIAN_SPARSITY_H
