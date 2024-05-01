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

#define strToFnPtrMapElement(funcName, fnPtr)    {llJit->mangleAndIntern(funcName), JITEvaluatedSymbol::fromPointer(fnPtr)}

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

        // enable position independent code
        JTMB.setRelocationModel(llvm::Reloc::PIC_);

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
        if (numCompileThreads == 0) {
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