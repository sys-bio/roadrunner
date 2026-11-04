#include "EnzymeShadow.h"

#include <cstdlib>
#include <cstring>

namespace rrllvm {

    size_t EnzymeShadow::dataSize(const LLVMModelData *md) {
        return (md->size - sizeof(LLVMModelData)) / sizeof(double);
    }

    double *EnzymeShadow::rebase(const LLVMModelData *md, LLVMModelData *shadow, const double *p) {
        if (p == nullptr) {
            return nullptr;
        }
        const double *begin = md->data;
        const double *end = md->data + dataSize(md);
        if (p < begin || p >= end) {
            return nullptr;
        }
        return shadow->data + (p - begin);
    }

    LLVMModelData *EnzymeShadow::create(const LLVMModelData *md) {
        auto *shadow = static_cast<LLVMModelData *>(std::calloc(md->size, 1));
        std::memcpy(shadow, md, sizeof(LLVMModelData));

        double **fields[] = {
                &shadow->stateVector,
                &shadow->stateVectorRate,
                &shadow->rateRuleRates,
                &shadow->floatingSpeciesAmountRates,
                &shadow->compartmentVolumesAlias,
                &shadow->initCompartmentVolumesAlias,
                &shadow->initFloatingSpeciesAmountsAlias,
                &shadow->boundarySpeciesAmountsAlias,
                &shadow->initBoundarySpeciesAmountsAlias,
                &shadow->globalParametersAlias,
                &shadow->initGlobalParametersAlias,
                &shadow->reactionRatesAlias,
                &shadow->rateRuleValuesAlias,
                &shadow->floatingSpeciesAmountsAlias,
                &shadow->multiSpeciesReferencesAlias,
                &shadow->multiSpeciesReferencesInitAlias,
        };
        for (double **field: fields) {
            *field = rebase(md, shadow, *field);
        }
        return shadow;
    }

    void EnzymeShadow::free(LLVMModelData *shadow) {
        // the stoichiometry and random pointers belong to the primal model data
        std::free(shadow);
    }

}
