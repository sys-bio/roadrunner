//
// Created by Ciaran on 25/10/2021.
//


#include "LLJit.h"
#include "rrConfig.h"
#include "rrLogger.h"
#include "SBMLSupportFunctions.h"
#include "ModelDataIRBuilder.h"
#include "SBMLModelObjectCache.h"
#include "rrRoadRunnerOptions.h"
#include "rrSparse.h"
#include <thread>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <string>

#ifdef _MSC_VER
#pragma warning(disable: 4146)
#pragma warning(disable: 4141)
#pragma warning(disable: 4267)
#pragma warning(disable: 4624)
#endif

#include "llvm/IR/Function.h"
#include "llvm/ExecutionEngine/Orc/EPCDynamicLibrarySearchGenerator.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Process.h"

#ifdef _MSC_VER
#pragma warning(default: 4146)
#pragma warning(default: 4141)
#pragma warning(default: 4267)
#pragma warning(default: 4624)
#endif

#if LIBSBML_HAS_PACKAGE_DISTRIB

#   include "Random.h"

#endif

using namespace llvm;
using namespace rr;
using namespace sbmlsupport;

namespace rrllvm {

    std::atomic<std::uint64_t> LLJit::largestSectionSpan{0};

#define strToFnPtrMapElement(funcName, fnPtr)    {llJit->mangleAndIntern(funcName), JITEvaluatedSymbol::fromPointer(fnPtr)}

    /**
     * Debug switches read from the environment; all are off by default.
     *   RR_LLJIT_RELOC=static          static relocation model instead of PIC
     *   RR_LLJIT_NO_COMPILE_THREAD=1   compile on the calling thread
     *   RR_LLJIT_NO_EH_FRAMES=1        do not register JIT'd .eh_frame with the unwinder
     *   RR_LLJIT_FAR_RODATA=1          place read-only data sections about 3 GiB from the code; on Mach-O,
     *                                  where RuntimeDyld treats no section as read-only, all data sections
     *                                  (ignored unless RR_LLJIT_RESERVE=0, the range keeps sections together)
     *   RR_LLJIT_FAR_RODATA_GIB=n      distance in GiB for RR_LLJIT_FAR_RODATA (default 3)
     *   RR_LLJIT_RESERVE=0             do not allocate the sections of an object from one contiguous range
     */
    static bool envEquals(const char *name, const char *value) {
        const char *v = std::getenv(name);
        return v != nullptr && std::string(v) == value;
    }

    // Hands out section memory from one contiguous range, so the code, read-only and
    // read-write sections of an object stay within a bounded distance of each other. Jump
    // tables and PC-relative relocations in the generated code hold 32-bit offsets between
    // sections, and the default allocator can place the sections farther apart than that.
    // Falls back to ordinary allocation if the range cannot be reserved or is used up.
    class ObjectRangeMapper : public llvm::SectionMemoryManager::MemoryMapper {
    public:
        ObjectRangeMapper() = default;

        ~ObjectRangeMapper() override {
            if (range.base() != nullptr) {
                llvm::sys::Memory::releaseMappedMemory(range);
            }
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

    // Memory manager for JIT'd objects. Allocates all sections of an object from one contiguous
    // range, and, for the debug switches, can skip EH frame registration or place read-only data
    // sections far from the code section.
    class JitSectionMemoryManager : private RangeMapperHolder, public llvm::SectionMemoryManager {
    public:
        JitSectionMemoryManager(bool noEhFrames, bool farReadOnlyData, unsigned farGiB, bool reserveRange,
                                bool farAllData)
                : RangeMapperHolder(), llvm::SectionMemoryManager(reserveRange ? &mapper : nullptr),
                  noEhFrames(noEhFrames), farReadOnlyData(farReadOnlyData && !reserveRange), farGiB(farGiB),
                  reserveRange(reserveRange), farAllData(farAllData) {}

        ~JitSectionMemoryManager() override {
            for (llvm::sys::MemoryBlock &block: farBlocks) {
                llvm::sys::Memory::releaseMappedMemory(block);
            }
        }

        bool needsToReserveAllocationSpace() override {
            return reserveRange;
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
            if (farReadOnlyData && (isReadOnly || farAllData)) {
                addr = codeBase != 0 ? allocateFar(size, alignment) : nullptr;
                if (addr == nullptr && !warnedFallback) {
                    warnedFallback = true;
                    std::fprintf(stderr, "LLJit: could not place read-only section %s far from the code (%s)\n",
                                 sectionName.str().c_str(),
                                 codeBase == 0 ? "no code section allocated yet" : "no free address range");
                }
            }
            if (addr == nullptr) {
                addr = SectionMemoryManager::allocateDataSection(size, alignment, sectionID, sectionName, isReadOnly);
            }
            trackSection(addr, size);
            return addr;
        }

        bool finalizeMemory(std::string *errMsg = nullptr) override {
            const bool failed = SectionMemoryManager::finalizeMemory(errMsg);
            if (spanHigh > spanLow) {
                const uint64_t span = spanHigh - spanLow;
                uint64_t largest = LLJit::largestSectionSpan.load();
                while (largest < span && !LLJit::largestSectionSpan.compare_exchange_weak(largest, span)) {
                }
                if (span > (uint64_t(1) << 31)) {
                    rrLogWarn << "LLJit: the sections of a model span " << span
                              << " bytes; 32-bit offsets between them may overflow";
                }
            }
            return failed;
        }

        void registerEHFrames(uint8_t *addr, uint64_t loadAddr, size_t size) override {
            if (!noEhFrames) {
                SectionMemoryManager::registerEHFrames(addr, loadAddr, size);
            }
        }

        void deregisterEHFrames() override {
            if (!noEhFrames) {
                SectionMemoryManager::deregisterEHFrames();
            }
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

        // Carves a read-only section out of a pool mapped about farGiB GiB from the code section.
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
                    if (farBlocks.empty()) {
                        std::fprintf(stderr, "LLJit: read-only data placed %.2f GiB from the code section\n",
                                     static_cast<double>(apart) / static_cast<double>(gib));
                    }
                    farBlocks.push_back(block);
                    pool = block;
                    return true;
                }
                llvm::sys::Memory::releaseMappedMemory(block);
            }
            return false;
        }

        bool noEhFrames;
        bool farReadOnlyData;
        unsigned farGiB;
        bool reserveRange;
        bool farAllData;
        uintptr_t codeBase = 0;
        uintptr_t spanLow = 0;
        uintptr_t spanHigh = 0;
        bool warnedFallback = false;
        llvm::sys::MemoryBlock pool;
        uintptr_t poolUsed = 0;
        std::vector<llvm::sys::MemoryBlock> farBlocks;
    };

    static std::mutex LLJitMtx;

    llvm::CodeGenOpt::Level convertRRCodeGenOptLevelToLLVM(std::uint32_t options) {
        // these should be mutually exclusive
        if (options & rr::LoadSBMLOptions::NONE)
            return llvm::CodeGenOpt::None;
        else if (options & rr::LoadSBMLOptions::LESS)
            return llvm::CodeGenOpt::Less;
        else if (options & rr::LoadSBMLOptions::DEFAULT)
            return llvm::CodeGenOpt::Default;
        else if (options & rr::LoadSBMLOptions::AGGRESSIVE)
            return llvm::CodeGenOpt::Aggressive;
        else {
            throw std::invalid_argument("None of the LLJIT_OPTIMIZATION_LEVELS are set to true");
        }
    }

    std::string LLJit::mangleName(const std::string &unmangledName) const {
        std::string mangledName;
        llvm::raw_string_ostream mangledNameStream(mangledName);
        llvm::Mangler::getNameWithPrefix(mangledNameStream, unmangledName, getDataLayout());
        return mangledNameStream.str();
    }

    LLJit::LLJit()
        : LLJit(LoadSBMLOptions().modelGeneratorOpt)
    {
    }

    LLJit::LLJit(std::uint32_t options)
        : Jit(options)
        , llJit()
    {

        // todo, can we cross compile providing a different host arch?

        /**
         * Create a JTMB.
         *
         * This is moved into the LLJitBuilder and becomes harder to access, so pull out
         * things which we need later on and store them as member variables.
         */
        llvm::Expected<llvm::orc::JITTargetMachineBuilder> expectedJTMB = llvm::orc::JITTargetMachineBuilder::detectHost();
        if (!expectedJTMB) {
            llvm::logAllUnhandledErrors(std::move(expectedJTMB.takeError()), llvm::errs(),
                                        "Cannot create JitTargetMachineBuilder");
        }
        auto JTMB = *expectedJTMB;

        /**
         * None, Less, Default or Aggressive are the options.
         */
        JTMB.setCodeGenOptLevel(convertRRCodeGenOptLevelToLLVM(options));

        // position independent code unless RR_LLJIT_RELOC=static
        const bool staticReloc = envEquals("RR_LLJIT_RELOC", "static");
        if (staticReloc) {
            rrLogWarn << "LLJit: using the static relocation model (RR_LLJIT_RELOC)";
        }
        JTMB.setRelocationModel(staticReloc ? llvm::Reloc::Static : llvm::Reloc::PIC_);

        // augment the default options
        llvm::TargetOptions targetOptions = JTMB.getOptions();


        targetOptions.ThreadModel = llvm::ThreadModel::POSIX;

        targetOptions.AllowFPOpFusion = FPOpFusion::Fast;
        JTMB.setOptions(targetOptions);

        // query the target machine builder for its data layout.
        auto DL = JTMB.getDefaultDataLayoutForTarget();
        if (!DL) {
            std::string err = "Unable to get default data layout";
            rrLogErr << err;
            llvm::logAllUnhandledErrors(std::move(DL.takeError()), llvm::errs(), err);
            throw_llvm_exception(err);
        }

        llvm::Expected<std::unique_ptr<llvm::TargetMachine>> expectedTargetMachine = JTMB.createTargetMachine();
        if (!expectedTargetMachine) {
            std::string err = "Could not create target machine";
            rrLogErr << err;
            llvm::logAllUnhandledErrors(std::move(expectedTargetMachine.takeError()), llvm::errs(), err);
            throw_llvm_exception(err);
        }

        // tell the LLJit instance to look in the current process
        // for symbols before complaining that they can't be found,
        auto DLSG = llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(DL->getGlobalPrefix());
        if (!DLSG) {
            llvm::logAllUnhandledErrors(
                    std::move(DLSG.takeError()),
                    llvm::errs(),
                    "DynamicLibrarySearchGenerator not built successfully"
            );
        }

        /**
         * todo expose as option to user
         */
        int numCompileThreads = rr::Config::getValue(rr::Config::LLJIT_NUM_THREADS).getAs<int>();

        // in case c++ fails to detect num cores from hardware
        // when using default options.
        if (envEquals("RR_LLJIT_NO_COMPILE_THREAD", "1")) {
            rrLogWarn << "LLJit: compiling on the calling thread (RR_LLJIT_NO_COMPILE_THREAD)";
            numCompileThreads = 0;
        }
        else if (numCompileThreads == 0) {
            numCompileThreads = 1;
        }

        rrLogDebug << "Compiling model with LLJit, num threads: " << numCompileThreads;

        // Create the LLJitBuilder
        llvm::orc::LLJITBuilder llJitBuilder;
        llJitBuilder
                .setNumCompileThreads(numCompileThreads)
                .setDataLayout(std::move(*DL))
                .setJITTargetMachineBuilder(std::move(JTMB))

                        /**
                         * Set a custom compile function that 1) caches objects
                         * and 2) stores a pointer to the targetMachine for access later.
                         * The targetMachine pointer is owned by the JITTargetMachineBuilder.
                         */
                .setCompileFunctionCreator([&](llvm::orc::JITTargetMachineBuilder JTMB)
                                                   -> Expected<std::unique_ptr<llvm::orc::IRCompileLayer::IRCompiler>> {
                    // Create the target machine.
                    auto TM = JTMB.createTargetMachine();
                    if (!TM)
                        return TM.takeError();

                    // store a pointer to it for later use
                    targetMachineNonOwning = (*TM).get();
                    if (numCompileThreads > 0)
                        return std::make_unique<llvm::orc::ConcurrentIRCompiler>(
                                std::move(JTMB),
                                &SBMLModelObjectCache::getObjectCache());

                    return std::make_unique<llvm::orc::TMOwningSimpleCompiler>(
                            std::move(*TM),
                            &SBMLModelObjectCache::getObjectCache());
                });


        const bool noEhFrames = envEquals("RR_LLJIT_NO_EH_FRAMES", "1");
        const bool farReadOnlyData = envEquals("RR_LLJIT_FAR_RODATA", "1");
        const bool reserveRange = !envEquals("RR_LLJIT_RESERVE", "0");
        unsigned farGiB = 3;
        if (const char *gibText = std::getenv("RR_LLJIT_FAR_RODATA_GIB")) {
            if (std::atoi(gibText) > 0) {
                farGiB = static_cast<unsigned>(std::atoi(gibText));
            }
        }
        if (noEhFrames) {
            rrLogWarn << "LLJit: not registering EH frames (RR_LLJIT_NO_EH_FRAMES)";
        }
        if (farReadOnlyData) {
            rrLogWarn << "LLJit: placing read-only data about " << farGiB << " GiB from the code (RR_LLJIT_FAR_RODATA)";
        }
        if (!reserveRange) {
            rrLogWarn << "LLJit: not allocating the sections of a model from one range";
        }
        if (noEhFrames || farReadOnlyData || reserveRange) {
            llJitBuilder.setObjectLinkingLayerCreator(
                    [noEhFrames, farReadOnlyData, farGiB, reserveRange](llvm::orc::ExecutionSession &ES,
                                                                    const llvm::Triple &triple)
                            -> Expected<std::unique_ptr<llvm::orc::ObjectLayer>> {
                        // RuntimeDyld classifies no Mach-O section as read-only
                        const bool machO = triple.isOSBinFormatMachO();
                        auto layer = std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(
                                ES, [noEhFrames, farReadOnlyData, farGiB, reserveRange, machO]() {
                                    return std::make_unique<JitSectionMemoryManager>(noEhFrames, farReadOnlyData,
                                                                                     farGiB, reserveRange, machO);
                                });
                        // LLJIT's default layer does the same for COFF
                        if (triple.isOSBinFormatCOFF()) {
                            layer->setOverrideObjectFlagsWithResponsibilityFlags(true);
                            layer->setAutoClaimResponsibilityForObjectSymbols(true);
                        }
                        return std::unique_ptr<llvm::orc::ObjectLayer>(std::move(layer));
                    });
        }

        auto llJitExpected = llJitBuilder.create();
        if (!llJitExpected) {
            std::string err = "failed to create an LLJit instance";
            rrLogErr << err;
            llvm::logAllUnhandledErrors(
                    std::move(llJitExpected.takeError()), llvm::errs(), "[LLJit creation failure] ");
        }
        // if success, grab the LLJit obj from the llvm::Expected object.
        llJit = std::move(*llJitExpected);

        // adds the generater created above
        llJit->getMainJITDylib().addGenerator(std::move(*DLSG));

        LLJit::mapFunctionsToJitSymbols();

    }


    void LLJit::mapFunctionsToJitSymbols() {
        llvm::DenseMap<llvm::orc::SymbolStringPtr, llvm::JITEvaluatedSymbol> map{};
        for (auto[fnName, fnTy_addr_pair]: externalFunctionSignatures()) {
            auto[fnTy, addr] = fnTy_addr_pair;
            /**
             * A note on name mangling. LLVM docs, tutorials and examples all use name mangling
             * for function names. We can do this here by using the mangleName functor and then pass
             * this as input to Function::Create (3rd arg). However, if we want to do this, we
             * need to mangle the names at the time these functions are needed (ASTNodeCodeGen),
             * or we wont be able to locate them. This is easily doable but currently I don't see
             * much point, as we're not likely to have many name clashes.
             */
            // std::string mangled = mangleName(fnName);
            Function::Create(fnTy, Function::ExternalLinkage, fnName, getModuleNonOwning());
            map.insert({
                               llJit->mangleAndIntern(fnName), JITEvaluatedSymbol::fromPointer(addr)
                       });
        }
        // these call out to functions that create the necessary function if
        // not already been created before.
        ModelDataIRBuilder::getCSRMatrixSetNZDecl(getModuleNonOwning());
        ModelDataIRBuilder::getCSRMatrixGetNZDecl(getModuleNonOwning());

        map.insert({llJit->mangleAndIntern(std::string(ModelDataIRBuilder::csr_matrix_set_nzName)),
                    JITEvaluatedSymbol::fromPointer(&rr::csr_matrix_set_nz)}
        );
        map.insert({llJit->mangleAndIntern(std::string(ModelDataIRBuilder::csr_matrix_get_nzName)),
                    JITEvaluatedSymbol::fromPointer(&rr::csr_matrix_get_nz)}
        );

        llvm::Error err = llJit->getMainJITDylib().define(llvm::orc::absoluteSymbols(map));
        if (err) {
            llvm::logAllUnhandledErrors(std::move(err), llvm::errs(), "[ error mapping functions to symbols ]");
        }

    }


    std::uint64_t LLJit::lookupFunctionAddress(const std::string &name) {
        // automatically looks up manged name
        llvm::Expected<llvm::JITEvaluatedSymbol> expectedSymbol = llJit->lookup(name);
        if (!expectedSymbol) {
            std::string err = "Could not find symbol " + mangleName(name);
            rrLogErr << err;
            llvm::logAllUnhandledErrors(
                    std::move(expectedSymbol.takeError()),
                    llvm::errs(),
                    "[symbol lookup error] ");
            throw std::logic_error(err);
        }
        llvm::JITEvaluatedSymbol symbol = *expectedSymbol;
        rrLogInfo << "LLJit has loaded jit'd function called " << name;
        return symbol.getAddress();
    }


    void LLJit::addObjectFile(rrOwningBinary owningObject) {
        llvm::Error err = llJit->addObjectFile(std::move(owningObject.takeBinary().second));
        if (err) {
            std::string errMsg = "Could not add object to LLJit";
            rrLogErr << errMsg;
            llvm::logAllUnhandledErrors(std::move(err), llvm::errs(), errMsg);
        }
    }

    const llvm::DataLayout &LLJit::getDataLayout() const {
        return llJit->getDataLayout();
    }

    void LLJit::addModule(llvm::orc::ThreadSafeModule tsm) {
        std::cout << "full module: " << std::endl;
        std::cout << emitToString() << std::endl;
        if (llvm::Error err = llJit->addIRModule(std::move(tsm))) {
            std::string errMsg = "Unable to add ThreadSafeModule to LLJit";
            rrLogErr << errMsg;
            llvm::logAllUnhandledErrors(std::move(err), llvm::errs(), errMsg);
        }
    }

    void LLJit::addModule(std::unique_ptr<llvm::Module> M, std::unique_ptr<llvm::LLVMContext> ctx) {
        llvm::Error err = llJit->addIRModule(
                llvm::orc::ThreadSafeModule(std::move(M), std::move(ctx))
        );
        if (err) {
            llvm::logAllUnhandledErrors(std::move(err), llvm::errs(), "error adding module");
        }
    }


    void LLJit::addModule(llvm::Module *m) {

    }

    void LLJit::addObjectFile(std::unique_ptr<llvm::MemoryBuffer> obj) {
        llvm::Error err = llJit->addObjectFile(std::move(obj));
        if (err) {
            std::string s = "Unable to add object file to LLJit";
            rrLogErr << s;
            llvm::logAllUnhandledErrors(std::move(err), llvm::errs(), s);
        }
    }

    void LLJit::addObjectFile(std::unique_ptr<llvm::object::ObjectFile> objectFile) {
        // Only needed for MCJit, which will be removed in the distant future.
    }


    void LLJit::addModule() {
        llvm::orc::ThreadSafeModule TSM(std::move(module), std::move(context));
        llvm::Error err = llJit->addIRModule(std::move(TSM));
        if (err) {
            std::string errMsg = "Could not add main JITDylib to LLJit";
            llvm::logAllUnhandledErrors(std::move(err), llvm::errs(), errMsg);
            rrLogErr << errMsg;
        }
    }

    std::unique_ptr<llvm::MemoryBuffer> LLJit::getCompiledModelFromCache(const std::string &sbmlMD5) {
        SBMLModelObjectCache &cache = SBMLModelObjectCache::getObjectCache();
        auto v = cache.inspect();
        LLVMContext ctx;
        std::unique_ptr<llvm::Module> m = std::make_unique<llvm::Module>(sbmlMD5, ctx);
        std::unique_ptr<llvm::MemoryBuffer> objBuf = cache.getObject(m.get());
        return std::move(objBuf);
    }

    llvm::orc::LLJIT *LLJit::getLLJitNonOwning() {
        return llJit.get();
    }

    std::string LLJit::dump() {
        std::string s;
        llvm::raw_string_ostream os(s);
        llJit->getExecutionSession().dump(os);
        return s;
    }

    std::ostream &operator<<(std::ostream &os, LLJit *llJit) {
        os << llJit->dump();
        return os;
    }


    void LLJit::addIRModule() {
        llvm::orc::ThreadSafeModule tsm(std::move(module), std::move(context));
        llvm::Error err = llJit->addIRModule(std::move(tsm));
        if (err) {
            std::string errMsg = "Could not add module to LLJit";
            rrLogErr << errMsg;
            llvm::logAllUnhandledErrors(std::move(err), llvm::errs(), errMsg);
        }
    }

    std::string LLJit::getModuleAsString(std::string sbmlMD5) {
        std::unique_ptr<llvm::MemoryBuffer> memBuf = getCompiledModelFromCache(sbmlMD5);
        MemoryBufferRef memBufRef = memBuf->getMemBufferRef();
        std::string s = memBufRef.getBuffer().str();
        if (s.empty()) {
            std::string err = "Unable to convert module to string. Have you made a call to addModule or addObject yet?";
            rrLogErr << err;
            throw_llvm_exception(err);
        }
        return s;
    };


}