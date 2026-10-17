//
// See LLJitSectionMemory.h.
//

#include "LLJitSectionMemory.h"
#include "rrLogger.h"

#include <algorithm>
#include <system_error>
#include <vector>

#ifdef _MSC_VER
#pragma warning(disable: 4146)
#pragma warning(disable: 4141)
#pragma warning(disable: 4267)
#pragma warning(disable: 4624)
#endif

#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/Process.h"

#ifdef _MSC_VER
#pragma warning(default: 4146)
#pragma warning(default: 4141)
#pragma warning(default: 4267)
#pragma warning(default: 4624)
#endif

using namespace rr;

namespace rrllvm {

    std::atomic<unsigned> LLJitSectionMemoryTesting::farDataGiB{0};
    std::atomic<std::uint64_t> LLJitSectionMemoryTesting::largestSectionSpan{0};

    namespace {

        // Hands out section memory from one contiguous range. Falls back to ordinary
        // allocation if the range cannot be reserved or is used up.
        class ObjectRangeMapper : public llvm::SectionMemoryManager::MemoryMapper {
        public:
            ObjectRangeMapper() = default;

            ~ObjectRangeMapper() override {
                if (range.base() != nullptr) {
                    llvm::sys::Memory::releaseMappedMemory(range);
                }
            }

            bool hasRange() const {
                return range.base() != nullptr;
            }

            // Reserves the range that later section allocations are carved from.
            void reserve(uint64_t numBytes) {
                if (range.base() != nullptr || rangeFailed || numBytes == 0) {
                    return;
                }
                std::error_code ec;
                range = llvm::sys::Memory::allocateMappedMemory(
                        numBytes, nullptr, llvm::sys::Memory::MF_READ | llvm::sys::Memory::MF_WRITE, ec);
                if (ec || range.base() == nullptr) {
                    range = llvm::sys::MemoryBlock();
                    rangeFailed = true;
                    rrLogWarn << "LLJit: could not reserve " << numBytes
                              << " bytes for the sections of a model; using separate allocations";
                }
            }

            llvm::sys::MemoryBlock
            allocateMappedMemory(llvm::SectionMemoryManager::AllocationPurpose, size_t numBytes,
                                 const llvm::sys::MemoryBlock *const nearBlock, unsigned flags,
                                 std::error_code &ec) override {
                ec = std::error_code();
                if (range.base() != nullptr) {
                    const size_t pageSize = llvm::sys::Process::getPageSizeEstimate();
                    const size_t chunk = (numBytes + pageSize - 1) / pageSize * pageSize;
                    if (used + chunk <= range.allocatedSize()) {
                        uint8_t *addr = static_cast<uint8_t *>(range.base()) + used;
                        used += chunk;
                        return llvm::sys::MemoryBlock(addr, chunk);
                    }
                    if (!warnedExhausted) {
                        warnedExhausted = true;
                        rrLogWarn << "LLJit: the sections of a model outgrew their reserved range; "
                                  << "using separate allocations";
                    }
                }
                return llvm::sys::Memory::allocateMappedMemory(numBytes, nearBlock, flags, ec);
            }

            std::error_code protectMappedMemory(const llvm::sys::MemoryBlock &block, unsigned flags) override {
                return llvm::sys::Memory::protectMappedMemory(block, flags);
            }

            std::error_code releaseMappedMemory(llvm::sys::MemoryBlock &block) override {
                const uint8_t *p = static_cast<const uint8_t *>(block.base());
                const uint8_t *b = static_cast<const uint8_t *>(range.base());
                if (b != nullptr && p >= b && p < b + range.allocatedSize()) {
                    block = llvm::sys::MemoryBlock();
                    return std::error_code();
                }
                return llvm::sys::Memory::releaseMappedMemory(block);
            }

        private:
            llvm::sys::MemoryBlock range;
            size_t used = 0;
            bool rangeFailed = false;
            bool warnedExhausted = false;
        };

        // Constructed before the memory manager that points at it, destroyed after it.
        struct RangeMapperHolder {
            ObjectRangeMapper mapper;
        };

        // Memory manager for JIT'd objects: all sections of an object come from one range.
        class RangeSectionMemoryManager : private RangeMapperHolder, public llvm::SectionMemoryManager {
        public:
            // farGiB and farAllData are test hooks (LLJitSectionMemoryTesting::farDataGiB).
            // farAllData: RuntimeDyld classifies no Mach-O section as read-only, so every data
            // section is moved, not only the read-only ones.
            RangeSectionMemoryManager(unsigned farGiB, bool farAllData)
                    : RangeMapperHolder(), llvm::SectionMemoryManager(&mapper),
                      farGiB(farGiB), farAllData(farAllData) {}

            ~RangeSectionMemoryManager() override {
                for (llvm::sys::MemoryBlock &block: farBlocks) {
                    llvm::sys::Memory::releaseMappedMemory(block);
                }
            }

            bool needsToReserveAllocationSpace() override {
                return false;
            }

            void reserveAllocationSpace(uintptr_t codeSize, uint32_t codeAlign, uintptr_t roDataSize,
                                        uint32_t roDataAlign, uintptr_t rwDataSize, uint32_t rwDataAlign) override {
                // Each kind of section gets its size, its alignment padding and some slack for the
                // page rounding of individual sections and for sections the totals do not cover (GOT).
                constexpr uint64_t slack = uint64_t(1) << 20;
                auto needed = [](uint64_t size, uint32_t align) -> uint64_t {
                    if (size == 0) {
                        return 0;
                    }
                    align = std::max<uint32_t>(align, 16);
                    return llvm::alignTo(size, align) + align + slack;
                };
                mapper.reserve(needed(codeSize, codeAlign) + needed(roDataSize, roDataAlign) +
                               needed(rwDataSize, rwDataAlign));
            }

            uint8_t *allocateCodeSection(uintptr_t size, unsigned alignment, unsigned sectionID,
                                         llvm::StringRef sectionName) override {
                uint8_t *addr = SectionMemoryManager::allocateCodeSection(size, alignment, sectionID, sectionName);
                if (codeBase == 0) {
                    codeBase = reinterpret_cast<uintptr_t>(addr);
                }
                trackSection(addr, size);
                return addr;
            }

            uint8_t *allocateDataSection(uintptr_t size, unsigned alignment, unsigned sectionID,
                                         llvm::StringRef sectionName, bool isReadOnly) override {
                uint8_t *addr = nullptr;
                if (farGiB != 0 && !mapper.hasRange() && codeBase != 0 && (isReadOnly || farAllData)) {
                    addr = allocateFar(size, alignment);
                }
                if (addr == nullptr) {
                    addr = SectionMemoryManager::allocateDataSection(size, alignment, sectionID, sectionName,
                                                                     isReadOnly);
                }
                trackSection(addr, size);
                return addr;
            }

            bool finalizeMemory(std::string *errMsg = nullptr) override {
                const bool failed = SectionMemoryManager::finalizeMemory(errMsg);
                if (spanHigh > spanLow) {
                    const uint64_t span = spanHigh - spanLow;
                    std::atomic<std::uint64_t> &largest = LLJitSectionMemoryTesting::largestSectionSpan;
                    uint64_t seen = largest.load();
                    while (seen < span && !largest.compare_exchange_weak(seen, span)) {
                    }
                }
                return failed;
            }

        private:
            // Records the extent of all sections of this object.
            void trackSection(const uint8_t *addr, uintptr_t size) {
                if (addr == nullptr || size == 0) {
                    return;
                }
                const uintptr_t low = reinterpret_cast<uintptr_t>(addr);
                spanLow = spanLow == 0 ? low : std::min(spanLow, low);
                spanHigh = std::max(spanHigh, low + size);
            }

            // Carves a section out of a pool mapped about farGiB GiB from the code section.
            uint8_t *allocateFar(uintptr_t size, unsigned alignment) {
                if (alignment == 0) {
                    alignment = 1;
                }
                uintptr_t offset = (poolUsed + alignment - 1) / alignment * alignment;
                if (pool.base() != nullptr && offset + size <= pool.allocatedSize()) {
                    poolUsed = offset + size;
                    return static_cast<uint8_t *>(pool.base()) + offset;
                }
                if (!allocatePool(size + alignment)) {
                    return nullptr;
                }
                poolUsed = size;
                return static_cast<uint8_t *>(pool.base());
            }

            // Maps a new pool at least farGiB GiB from the code section; later pools follow the first.
            bool allocatePool(uintptr_t minSize) {
                const uintptr_t gib = uintptr_t(1) << 30;
                const uintptr_t minDistance = gib * farGiB - gib / 2;
                const uintptr_t poolSize = minSize > (uintptr_t(16) << 20) ? minSize : (uintptr_t(16) << 20);
                const auto protection = llvm::sys::Memory::MF_READ | llvm::sys::Memory::MF_WRITE;
                std::vector<uintptr_t> hints;
                if (pool.base() != nullptr) {
                    hints.push_back(reinterpret_cast<uintptr_t>(pool.base()) + pool.allocatedSize());
                } else {
                    for (uintptr_t distance: {farGiB * gib, (farGiB + 1) * gib, (farGiB + 3) * gib}) {
                        hints.push_back(codeBase + distance);
                        hints.push_back(codeBase - distance);
                    }
                }
                for (uintptr_t hint: hints) {
                    llvm::sys::MemoryBlock nearBlock(reinterpret_cast<void *>(hint), 0);
                    std::error_code ec;
                    llvm::sys::MemoryBlock block = llvm::sys::Memory::allocateMappedMemory(poolSize, &nearBlock,
                                                                                           protection, ec);
                    if (ec || block.base() == nullptr) {
                        continue;
                    }
                    uintptr_t base = reinterpret_cast<uintptr_t>(block.base());
                    uintptr_t apart = base > codeBase ? base - codeBase : codeBase - base;
                    if (apart >= minDistance) {
                        farBlocks.push_back(block);
                        pool = block;
                        return true;
                    }
                    llvm::sys::Memory::releaseMappedMemory(block);
                }
                return false;
            }

            unsigned farGiB;
            bool farAllData;
            uintptr_t codeBase = 0;
            uintptr_t spanLow = 0;
            uintptr_t spanHigh = 0;
            llvm::sys::MemoryBlock pool;
            uintptr_t poolUsed = 0;
            std::vector<llvm::sys::MemoryBlock> farBlocks;
        };
    }

    llvm::Expected<std::unique_ptr<llvm::orc::ObjectLayer>>
    createRangeObjectLayer(llvm::orc::ExecutionSession &ES, const llvm::Triple &triple) {
        const unsigned farGiB = LLJitSectionMemoryTesting::farDataGiB.load();
        const bool machO = triple.isOSBinFormatMachO();
        auto layer = std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(
                ES, [farGiB, machO]() {
                    return std::make_unique<RangeSectionMemoryManager>(farGiB, machO);
                });
        // LLJIT's default layer does the same for COFF
        if (triple.isOSBinFormatCOFF()) {
            layer->setOverrideObjectFlagsWithResponsibilityFlags(true);
            layer->setAutoClaimResponsibilityForObjectSymbols(true);
        }
        return std::unique_ptr<llvm::orc::ObjectLayer>(std::move(layer));
    }
}
