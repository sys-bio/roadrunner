//
// Created by Ciaran on 25/10/2021.
//

#include "MCJit.h"
#include "ModelDataIRBuilder.h"
#include "SBMLSupportFunctions.h"
#include "rrRoadRunnerOptions.h"
#include "rrLogger.h"
#include "ModelResources.h"

#ifdef _MSC_VER
#pragma warning(disable: 4146)
#pragma warning(disable: 4141)
#pragma warning(disable: 4267)
#pragma warning(disable: 4624)
#endif

#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/Mangler.h"

#ifdef _MSC_VER
#pragma warning(default: 4146)
#pragma warning(default: 4141)
#pragma warning(default: 4267)
#pragma warning(default: 4624)
#endif

#ifdef LIBSBML_HAS_PACKAGE_DISTRIB

#   include "Random.h"

#endif

using namespace rr;

namespace rrllvm {
    /** @cond PRIVATE */

    /**
     * @brief cross platform mechanism for getting the target machine
     * file type.
     */
#if LLVM_VERSION_MAJOR == 6
    llvm::LLVMTargetMachine::CodeGenFileType getCodeGenFileType()
    {
        return llvm::TargetMachine::CGFT_ObjectFile;
    }
#elif LLVM_VERSION_MAJOR >= 12

    llvm::CodeGenFileType getCodeGenFileType() {
        return llvm::CGFT_ObjectFile;
    }

#endif


    MCJit::MCJit(std::uint32_t opt)
        : Jit(opt)
        , engineBuilder(EngineBuilder(std::move(module)))
        , executionEngine()
        , functionPassManager()
        , errString()
    {
        compiledModuleBinaryStream = std::make_unique<llvm::raw_svector_ostream>(moduleBuffer);

        engineBuilder
            .setErrorStr(errString.get())
            .setMCJITMemoryManager(std::make_unique<SectionMemoryManager>());
        executionEngine.reset(engineBuilder.create());

        MCJit::mapFunctionsToJitSymbols();
        MCJit::initFunctionPassManager();
    }

    MCJit::MCJit()
        : MCJit(LoadSBMLOptions().modelGeneratorOpt)
    {
    }

    ExecutionEngine* MCJit::getExecutionEngineNonOwning() const {
        return executionEngine.get();
    }

    std::string MCJit::mangleName(const std::string& unmangledName) const {
        std::string mangledName;
        llvm::raw_string_ostream mangledNameStream(mangledName);
        llvm::Mangler::getNameWithPrefix(mangledNameStream, unmangledName, getDataLayout());
        return mangledNameStream.str();
    }

    void MCJit::mapFunctionsToJitSymbols() {
        llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr); // for symbols in current process


        for (auto [fnName, fnTy_addr_pair] : externalFunctionSignatures()) {
            auto [fnTy, addr] = fnTy_addr_pair;
            rrLogDebug << "Creating function \"" << fnName << "\"; fn type: " << toStringRef(fnTy).str() << "; at addr: " << addr;
            Function::Create(fnTy, Function::ExternalLinkage, fnName, getModuleNonOwning());
            llvm::sys::DynamicLibrary::AddSymbol(fnName, addr);
        }

        // these call out to functions that create the necessary function if
        // not already been created before (see getCSRMatrixSetNZDecl)
        ModelDataIRBuilder::getCSRMatrixSetNZDecl(getModuleNonOwning());
        ModelDataIRBuilder::getCSRMatrixGetNZDecl(getModuleNonOwning());
        // Add the symbol to the library
        llvm::sys::DynamicLibrary::AddSymbol(ModelDataIRBuilder::csr_matrix_set_nzName, (void*)rr::csr_matrix_set_nz);
        llvm::sys::DynamicLibrary::AddSymbol(ModelDataIRBuilder::csr_matrix_get_nzName, (void*)rr::csr_matrix_get_nz);
    }

    void MCJit::transferObjectsToResources(std::shared_ptr<rrllvm::ModelResources> modelResources) {
        Jit::transferObjectsToResources(modelResources);
        // todo check whether we break stuff by not giving the execution engine to modelResources
        modelResources->executionEngine = std::move(executionEngine);
        modelResources->executionEngine = nullptr;
        modelResources->errStr = std::move(errString);
        errString = nullptr;

    }

    std::uint64_t MCJit::lookupFunctionAddress(const std::string& name) {
        void* v = executionEngine->getPointerToNamedFunction(mangleName(name));
        return (std::uint64_t)v;
    }

    llvm::TargetMachine* MCJit::getTargetMachine() {
        return executionEngine->getTargetMachine();
    }

    void MCJit::addObjectFile(rrOwningBinary owningObject) {
        getExecutionEngineNonOwning()->addObjectFile(std::move(owningObject));
        // loadState does not work without a call to finalizeObject()
        getExecutionEngineNonOwning()->finalizeObject();
    }

    void MCJit::addObjectFile(std::unique_ptr<llvm::object::ObjectFile> objectFile) {
        getExecutionEngineNonOwning()->addObjectFile(std::move(objectFile));
        // loadState does not work without a call to finalizeObject()

        getExecutionEngineNonOwning()->finalizeObject();
    }

    void MCJit::addObjectFile(std::unique_ptr<llvm::MemoryBuffer> obj) {


        llvm::Expected<std::unique_ptr<llvm::object::ObjectFile> > objectFileExpected =
            llvm::object::ObjectFile::createObjectFile(obj->getMemBufferRef());
        if (!objectFileExpected) {
            throw std::invalid_argument("Failed to load object data");
        }
        std::unique_ptr<llvm::object::ObjectFile> objectFile(std::move(objectFileExpected.get()));
        getExecutionEngineNonOwning()->addObjectFile(std::move(objectFile));
    }


    const llvm::DataLayout& MCJit::getDataLayout() const {
        return getExecutionEngineNonOwning()->getDataLayout();
    }

    void MCJit::addModule(llvm::Module* M) {

    }

    void MCJit::addModule(std::unique_ptr<llvm::Module> M, std::unique_ptr<llvm::LLVMContext> ctx) {
        executionEngine->addModule(std::move(M));
        writeObjectToBinaryStream();
    }

    void MCJit::addModule() {
        addModuleViaObjectFile();
    }

    void MCJit::addModuleViaObjectFile() {
        writeObjectToBinaryStream();

        if (compiledModuleBinaryStream->str().empty()) {
            std::string err = "Attempt to add module before its been written to binary. Make a call to "
                "MCJit::writeObjectToBinaryStream() before addModule()";
            rrLogErr << err;
            throw_llvm_exception(err);
        }

        auto memBuffer(llvm::MemoryBuffer::getMemBuffer(getModuleBinaryStreamAsString()));

        llvm::Expected<std::unique_ptr<llvm::object::ObjectFile> > objectFileExpected =
            llvm::object::ObjectFile::createObjectFile(
                llvm::MemoryBufferRef(compiledModuleBinaryStream->str(), "id"));

        if (!objectFileExpected) {
            //LS DEBUG:  find a way to get the text out of the error.
            auto err = objectFileExpected.takeError();
            std::string s = "LLVM object supposed to be file, but is not.";
            rrLog(Logger::LOG_FATAL) << s;
            throw_llvm_exception(s);
        }

        std::unique_ptr<llvm::object::ObjectFile> objectFile(std::move(objectFileExpected.get()));
        llvm::object::OwningBinary<llvm::object::ObjectFile> owningObject(std::move(objectFile), std::move(memBuffer));

        addObjectFile(std::move(owningObject));

        getExecutionEngineNonOwning()->finalizeObject();
    }

    std::unique_ptr<llvm::MemoryBuffer> MCJit::getCompiledModelFromCache(const std::string& sbmlMD5) {
        return nullptr;
    }


    void MCJit::writeObjectToBinaryStream() {
        //Currently we save jitted functions in object file format
        //in save state. Compiling the functions into this format in the first place
        //makes saveState significantly faster than creating the object file when it is called
        //We then load the object file into the jit engine to avoid compiling the functions twice

        auto TargetMachine = getTargetMachine();

        llvm::InitializeNativeTarget(); // todo may have already been called in RoadRunner constructor

        //Write the object file to modBufferOut
        std::error_code EC;
        // llvm::SmallVector<char, 10> modBufferOut;
        // postOptimizedModuleStream(modBufferOut);

        llvm::legacy::PassManager pass;
        auto FileType = getCodeGenFileType();


#if LLVM_VERSION_MAJOR == 6
        if (TargetMachine->addPassesToEmitFile(pass, *compiledModuleBinaryStream, FileType))
#elif LLVM_VERSION_MAJOR >= 12
        if (TargetMachine->addPassesToEmitFile(pass, *compiledModuleBinaryStream, nullptr, FileType))
#endif
        {
            throw std::logic_error("TargetMachine can't emit a file of type CGFT_ObjectFile");
        }

        pass.run(*getModuleNonOwning());
    }


    void MCJit::initFunctionPassManager() {
        if (options & LoadSBMLOptions::OPTIMIZE) {
            functionPassManager = std::make_unique<legacy::FunctionPassManager>(getModuleNonOwning());

            // Set up the optimizer pipeline.  Start with registering info about how the
            // target lays out data structures.

            /**
             * Note to developers - passes are stored in llvm/Transforms/Scalar.h.
             */

             // we only support LLVM >= 3.1
#if (LLVM_VERSION_MAJOR == 3) && (LLVM_VERSION_MINOR == 1)
            //#if (LLVM_VERSION_MAJOR == 6)
            functionPassManager->add(new TargetData(*executionEngine->getTargetData()));
#elif (LLVM_VERSION_MAJOR == 3) && (LLVM_VERSION_MINOR <= 4)
            functionPassManager->add(new DataLayout(*executionEngine->getDataLayout()));
#elif (LLVM_VERSION_MINOR > 4)
            // Needed for LLVM 3.5 regardless of architecture
            // also, should use DataLayoutPass(module) per Renato (http://reviews.llvm.org/D4607)
            functionPassManager->add(new DataLayoutPass(module));
#elif (LLVM_VERSION_MAJOR >= 6)
            // Do nothing, because I'm not sure what the non-legacy equivalent of DataLayoutPass is
#endif

            // Provide basic AliasAnalysis support for GVN.
            // FIXME: This was deprecated somewhere along the line
            //functionPassManager->add(createBasicAliasAnalysisPass());


            if (options & LoadSBMLOptions::OPTIMIZE_INSTRUCTION_SIMPLIFIER) {
                rrLog(Logger::LOG_INFORMATION) << "using OPTIMIZE_INSTRUCTION_SIMPLIFIER";
#if LLVM_VERSION_MAJOR == 6
                functionPassManager->add(createInstructionSimplifierPass());
#elif LLVM_VERSION_MAJOR >= 12
                functionPassManager->add(createInstSimplifyLegacyPass());
#else
                rrLogWarn << "Not using llvm optimization \"OPTIMIZE_INSTRUCTION_SIMPLIFIER\" "
                    "because llvm version is " << LLVM_VERSION_MAJOR;
#endif
            }

            if (options & LoadSBMLOptions::OPTIMIZE_INSTRUCTION_COMBINING) {
                rrLog(Logger::LOG_INFORMATION) << "using OPTIMIZE_INSTRUCTION_COMBINING";
                functionPassManager->add(createInstructionCombiningPass());
            }

            if (options & LoadSBMLOptions::OPTIMIZE_GVN) {
                rrLog(Logger::LOG_INFORMATION) << "using GVN optimization";
                functionPassManager->add(createNewGVNPass());

            }

            if (options & LoadSBMLOptions::OPTIMIZE_CFG_SIMPLIFICATION) {
                rrLog(Logger::LOG_INFORMATION) << "using OPTIMIZE_CFG_SIMPLIFICATION";
                functionPassManager->add(createCFGSimplificationPass());
            }

            if (options & LoadSBMLOptions::OPTIMIZE_DEAD_INST_ELIMINATION) {
                rrLog(Logger::LOG_INFORMATION) << "using OPTIMIZE_DEAD_INST_ELIMINATION";
#if LLVM_VERSION_MAJOR == 6
                functionPassManager->add(createDeadInstEliminationPass());
#elif LLVM_VERSION_MAJOR >= 12
                // do nothing since this seems to have been removed
                // or replaced with createDeadCodeEliminationPass, which we add below anyway
#else
                rrLogWarn << "Not using OPTIMIZE_DEAD_INST_ELIMINATION because you are using"
                    "LLVM version " << LLVM_VERSION_MAJOR;
#endif
            }

            if (options & LoadSBMLOptions::OPTIMIZE_DEAD_CODE_ELIMINATION) {
                rrLog(Logger::LOG_INFORMATION) << "using OPTIMIZE_DEAD_CODE_ELIMINATION";
                functionPassManager->add(createDeadCodeEliminationPass());
            }

            functionPassManager->doInitialization();
        }
    }

    llvm::legacy::FunctionPassManager* MCJit::getFunctionPassManager() const {
        return functionPassManager.get();
    }

    llvm::raw_svector_ostream& MCJit::getCompiledModuleStream() {
        return *compiledModuleBinaryStream;
    }

    std::string MCJit::getModuleAsString(std::string sbmlMD5) {
        std::string s = getCompiledModuleStream().str().str();
        if (s.empty()) {
            std::string err = "Unable to convert module to string. Have you made a call to addModule or addObject yet?";
            rrLogErr << err;
            throw_llvm_exception(err);
        }
        return s;
    };

    /** @endcond PRIVATE */
}
