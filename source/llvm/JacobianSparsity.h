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
     * @brief The structure of the Jacobian of the state vector rate of a model: the
     * sparsity patterns and the column coloring, or why the model has none.
     * @details The Jacobian is J = cf * N * dv/dy plus the rows of the rate rules, with
     * cf the global conversion factor, N the stoichiometry matrix and v the reaction rates.
     * That holds unless the stoichiometry or a species conversion factor depends on the
     * state, or the global conversion factor does; such a model has no structure.
     */
    struct JacobianStructure {
        /** Why the model has no structure; empty if it has one. */
        std::string unavailableReason = "the structure of the Jacobian was not computed";

        /** Patterns of dv/dy, d(rate rule rates)/dy and the Jacobian (with the full diagonal). */
        CscPattern reactionRates;
        CscPattern rateRules;
        CscPattern jacobian;

        /** Color of every state column, see JacobianSparsity::colors. */
        std::vector<int> colors;
        int numColors = 0;

        bool available() const { return unavailableReason.empty(); }
    };

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

        /**
         * The structure of the Jacobian of the model, or why it has none: conserved moiety
         * analysis (@param conservedMoieties), a stoichiometry set by a rule, a species
         * conversion factor or a global conversion factor that depends on the state.
         */
        static JacobianStructure analyze(const libsbml::Model &model, const LLVMModelDataSymbols &symbols,
                                         bool conservedMoieties);

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
