#ifndef RR_LLVM_ENZYME_JACOBIAN_CODEGEN_H
#define RR_LLVM_ENZYME_JACOBIAN_CODEGEN_H

#include "LLVMModelData.h"
#include "rrSparsityPattern.h"
#include "rrExecutableModel.h"

#include <string>
#include <vector>

namespace llvm {
    class Function;
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

        /** Number of blocks the reaction rates and the rate rule rates are differentiated in. */
        int reactionRateBlocks = 0;
        int rateRuleBlocks = 0;

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

        /**
         * Default vector width of the generated tangents functions: one seed direction per pass.
         * Measured on the benchmark models (prototype/enzyme_jacobian/RESULTS.md), a wider vector
         * hardly changes the time of a Jacobian but multiplies the compilation time (Chen_MSB2009,
         * 7 colors, MCJit, before the blocks: 11 s to load at width 1, 59 s at width 7). The environment variable
         * RR_ENZYME_MAX_WIDTH sets a larger width.
         */
        static const int DefaultWidth = 1;

        /**
         * Default number of rates (stores of the primal function) per block. Enzyme's activity
         * analysis grows faster than linearly with the size of the differentiated function, so
         * the primal is differentiated in blocks, each a copy reduced to the rates of the block
         * (see splitByStores). On Chen_MSB2009 (827 reactions, prototype/enzyme_jacobian/blocks.csv)
         * 64 loads fastest with both JITs: the Enzyme pass takes 2.6 s instead of 8.6 s with LLJit;
         * smaller blocks cost more in copying and per-block overhead than they save. The time of a
         * Jacobian does not change. The environment variable RR_ENZYME_BLOCK_SIZE sets another size.
         */
        static const int DefaultBlockSize = 64;

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
         * @brief Emit the tangents wrapper `name` of the primal function `primal` with vector width
         * `width`, differentiating the primal in blocks of at most `blockSize` rates (see
         * splitByStores). Returns the number of blocks.
         */
        static int emitTangents(llvm::Module &module, const std::string &primal, const std::string &name,
                                unsigned width, unsigned blockSize);

        /**
         * @brief Split the primal function f into internal copies that each write only a block of
         * at most blockSize of the rates f writes, with the code no longer needed removed. The
         * copies together write what f writes. This needs the shape of the generated
         * evalReactionRates, which is checked: every store writes the reaction rates of the model
         * data (the first argument), no load reads them (a reaction id in the math is computed
         * again from its kinetic law, see LoadSymbolResolverBase::loadReactionRate), every load
         * reads a field of the model data or through one of its alias pointers, and every call is
         * to a pure function (an intrinsic, the math library or a helper with an IR body).
         * Returns {f} if f has another shape or at most blockSize stores.
         */
        static std::vector<llvm::Function *> splitByStores(llvm::Function *f, unsigned blockSize);

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
