//
// Section memory for objects loaded by LLJit (sys-bio/roadrunner issue 1368).
//
// The code generated for a model holds 32-bit offsets between the code section and the
// read-only data section (jump tables, PC-relative references). The default allocator can
// place the sections of one object farther apart than that, and the model then crashes
// (observed on Linux and arm64 macOS; the sections end up too far apart on every platform).
// The object layer made here allocates all sections of an object from one contiguous range.
//
// Written for LLVM 13. Later LLVM versions may keep an object's sections together
// themselves; if so, this file, its use in LLJit.cpp and its entries in CMakeLists.txt can
// go, and test/llvm_tests/LLJitSectionRange.cpp can be rewritten to measure the
// section distance through whatever the new memory manager offers.
//

#ifndef ROADRUNNER_LLJITSECTIONMEMORY_H
#define ROADRUNNER_LLJITSECTIONMEMORY_H

#include <atomic>
#include <cstdint>
#include <memory>

#ifdef _MSC_VER
#pragma warning(disable: 4146)
#pragma warning(disable: 4141)
#pragma warning(disable: 4267)
#pragma warning(disable: 4624)
#endif

#include "llvm/TargetParser/Triple.h"
#include "llvm/ExecutionEngine/Orc/Layer.h"
#include "llvm/Support/Error.h"

#ifdef _MSC_VER
#pragma warning(default: 4146)
#pragma warning(default: 4141)
#pragma warning(default: 4267)
#pragma warning(default: 4624)
#endif

namespace rrllvm {

    /**
     * @brief Creates the object layer for LLJit: RuntimeDyld, with every object's sections
     * allocated from one contiguous range.
     * @details Matches the signature of llvm::orc::LLJITBuilder::setObjectLinkingLayerCreator.
     */
    llvm::Expected<std::unique_ptr<llvm::orc::ObjectLayer>>
    createRangeObjectLayer(llvm::orc::ExecutionSession &ES, const llvm::Triple &triple);

    /**
     * @brief Hooks for tests. Nothing else reads or writes them.
     */
    struct LLJitSectionMemoryTesting {

        /**
         * @brief If not 0, sections that do not come from the contiguous range are placed about
         * this many GiB from the code section, to make the crash of issue 1368 happen on demand.
         * @details Read when an LLJit is created. Has no effect while the contiguous range works.
         */
        static std::atomic<unsigned> farDataGiB;

        /**
         * @brief Largest distance in bytes between the lowest and highest section of any object
         * loaded in this process. Tests reset it to 0 to measure one load.
         */
        static std::atomic<std::uint64_t> largestSectionSpan;
    };
}

#endif //ROADRUNNER_LLJITSECTIONMEMORY_H
