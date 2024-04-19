//
// Created by Ciaran on 25/10/2021.
//

#ifndef ROADRUNNER_LLJIT_H
#define ROADRUNNER_LLJIT_H

#define NOMINMAX

#include "SBMLModelObjectCache.h"

#ifdef _MSC_VER
#pragma warning(disable: 4146)
#pragma warning(disable: 4141)
#pragma warning(disable: 4267)
#pragma warning(disable: 4624)
#endif

#include <llvm/IRReader/IRReader.h>
#include "llvm/Jit.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/Support/SourceMgr.h"

#ifdef _MSC_VER
#pragma warning(default: 4146)
#pragma warning(default: 4141)
#pragma warning(default: 4267)
#pragma warning(default: 4624)
#endif

using namespace rr;

namespace rrllvm {

    class SBMLModelObjectCache;

    /**
     * @brief Thin layer around the llvm::orc::LLJit.
     *
     * @details The interface for LLJit is used as a framework
     */
    class LLJit : public Jit {
    public:

        LLJit();

        ~LLJit() override = default;

        std::string mangleName(const std::string &unmangledName) const override;

        explicit LLJit(std::uint32_t options);

        void mapFunctionsToJitSymbols() override;

        std::uint64_t lookupFunctionAddress(const std::string &name) override;

        void addObjectFile(rrOwningBinary owningObject) override;

        const llvm::DataLayout &getDataLayout() const override;

        void addModule(llvm::Module *M) override;

        void addModule() override;

        void addObjectFile(std::unique_ptr<llvm::MemoryBuffer> obj) override;

        void addObjectFile(std::unique_ptr<llvm::object::ObjectFile> objectFile) override;

        void addModule(llvm::orc::ThreadSafeModule tsm);

        void addModule(std::unique_ptr<llvm::Module> M, std::unique_ptr<llvm::LLVMContext> ctx) override;

        /**
         * @brief lookup a model in the roadrunner object cache and return a memory
         * buffer to it.
         * @details When compiled (with LLJit), object files are stored in the SBMLModelObjectCache.
         * This is basically a string to MemoryBuffer map, where the string is the sbml's md5. In the case
         * where the moiety conservation is turned on, the sbmlMD5 will be identical. Therefore it is appended
         * with the string "_conserved".
         */
        std::unique_ptr<llvm::MemoryBuffer> getCompiledModelFromCache(const std::string& sbmlMD5) override;

        llvm::orc::LLJIT *getLLJitNonOwning();

        void addIRModule();

        /**
         * @brief prints the symbol tables currently in the jit.
         */
        std::string dump();

        friend std::ostream& operator<<(std::ostream& os, LLJit* llJit);

        std::string getModuleAsString(std::string sbmlMD5) override;


    private:

//        void mapFunctionToAbsoluteSymbol(const std::string &funcName, std::uint64_t funcAddress);

        std::unique_ptr<llvm::orc::LLJIT> llJit;
        llvm::TargetMachine* targetMachineNonOwning = nullptr;
    };

}

#endif //ROADRUNNER_LLJIT_H
