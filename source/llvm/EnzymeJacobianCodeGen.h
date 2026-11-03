#ifndef RR_LLVM_ENZYME_JACOBIAN_CODEGEN_H
#define RR_LLVM_ENZYME_JACOBIAN_CODEGEN_H

#include "LLVMModelData.h"
#include "rrSparsityPattern.h"
#include "rrExecutableModel.h"

#include <string>
#include <vector>

namespace llvm {
    class Module;
}

namespace rrllvm {

    class ModelGeneratorContext;

    /**
     * Signature of a generated tangents function: evaluates the primal function at md and
     * writes the tangents of its outputs into the W shadows shadows[0..W-1], whose state
     * aliases hold the seed directions.
     */
    typedef void (*EnzymeTangents_FunctionPtr)(LLVMModelData *md, LLVMModelData **shadows);

    /**
     * @brief Everything a model needs to evaluate its Enzyme Jacobian, shared by every model
     * made from the same ModelResources.
     */
    struct EnzymeJacobianResources {
        /** Why the Enzyme Jacobian is not available; empty if it is. */
        std::string unavailableReason = rr::ExecutableModel::StateVectorJacobianNotRequested;

        /** Patterns of dv/dy, d(rate rule rates)/dy and the Jacobian (with full diagonal). */
        rr::CscPattern reactionRates;
        rr::CscPattern rateRules;
        rr::CscPattern jacobian;

        /** Color of every state column, number of colors, and vector width of the tangents functions. */
        std::vector<int> colors;
        int numColors = 0;
        int width = 0;

        EnzymeTangents_FunctionPtr reactionRatesTangents = nullptr;
        EnzymeTangents_FunctionPtr rateRuleRatesTangents = nullptr;

        bool available() const { return unavailableReason.empty(); }
    };

    /**
     * @brief Generates the forward mode derivatives of the generated model functions with Enzyme.
     *
     * For a primal function `ret f(LLVMModelData*)` already in the module, emitTangents adds
     *
     *     void name(LLVMModelData* md, LLVMModelData** shadows)
     *
     * which calls `__enzyme_fwddiff(f, enzyme_width, W, enzyme_dup, md, shadows[0], ..., shadows[W-1])`.
     * runEnzymePass then replaces every such call by the derivative code.
     */
    class EnzymeJacobianCodeGen {
    public:
        static const char *ReactionRatesTangentsName;
        static const char *RateRuleRatesTangentsName;

        /** Largest vector width of a generated tangents function. */
        static const int MaxWidth = 16;

        /**
         * @brief Prepare the Enzyme Jacobian of the model of the context: check the cases it
         * does not cover, compute the sparsity pattern and coloring, emit the tangents functions
         * and run Enzyme on the module. On any failure the module is left as it was and the
         * reason is stored in res.unavailableReason; it never throws. The function pointers are
         * filled after the module was compiled, see lookup.
         */
        static void prepare(const ModelGeneratorContext &context, unsigned options, EnzymeJacobianResources &res);

        /** @brief Fill the function pointers of res from the compiled module, if res is available. */
        static void lookup(const ModelGeneratorContext &context, EnzymeJacobianResources &res);

        /**
         * @brief Emit the tangents wrapper `name` of the primal function `primal` with vector width `width`.
         */
        static void emitTangents(llvm::Module &module, const std::string &primal, const std::string &name,
                                 unsigned width);

        /**
         * @brief Give the C++ helpers with a simple derivative an IR body in the module, so
         * that Enzyme differentiates them like any other code: rr_max, rr_min, rr_logd,
         * rr_rootd, arccot and rr_arccot_negzero. Every other helper stays an external call,
         * which the JIT resolves to the C++ function and Enzyme cannot differentiate. The
         * bodies compute exactly what the C++ helpers in SBMLSupportFunctions.cpp compute.
         */
        static void defineDifferentiableHelpers(llvm::Module &module);

        /**
         * @brief Run the Enzyme pass on the module. Errors Enzyme reports through the
         * diagnostics of the LLVM context are collected and thrown as std::runtime_error
         * instead of terminating the process.
         */
        static void runEnzymePass(llvm::Module &module);
    };

}

#endif // RR_LLVM_ENZYME_JACOBIAN_CODEGEN_H
