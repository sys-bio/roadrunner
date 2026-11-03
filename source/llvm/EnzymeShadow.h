#ifndef RR_LLVM_ENZYME_SHADOW_H
#define RR_LLVM_ENZYME_SHADOW_H

#include "LLVMModelData.h"

namespace rrllvm {

    /**
     * @brief Shadow copies of an LLVMModelData for the forward mode derivatives of Enzyme.
     *
     * Enzyme differentiates a generated function f(LLVMModelData* md) with md passed as
     * enzyme_dup: every load through md has a twin load through the shadow, which holds the
     * tangent of that value. A shadow therefore has the layout of md: the same header and a
     * data block of the same size, with every alias pointer moved to the same offset in the
     * shadow's own data block. The data block starts zeroed, so every quantity of the model
     * has a zero tangent until a caller seeds one.
     */
    struct EnzymeShadow {
        /**
         * @brief Allocate a shadow of md: a copy of the header, a zeroed data block, and every
         * double* field that points into md->data rebased to the shadow; double* fields that
         * point elsewhere are set to null. Free it with EnzymeShadow::free.
         */
        static LLVMModelData *create(const LLVMModelData *md);

        /** @brief Free a shadow made by create. */
        static void free(LLVMModelData *shadow);

        /**
         * @brief The pointer in shadow at the offset p has in md->data, or null if p does not
         * point into md->data.
         */
        static double *rebase(const LLVMModelData *md, LLVMModelData *shadow, const double *p);

        /** @brief Number of doubles in the data block of md. */
        static size_t dataSize(const LLVMModelData *md);
    };

}

#endif // RR_LLVM_ENZYME_SHADOW_H
