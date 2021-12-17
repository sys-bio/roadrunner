//
// Created by Ciaran on 25/10/2021.
//

#ifndef ROADRUNNER_MCJIT_H
#define ROADRUNNER_MCJIT_H

#include <rrRoadRunnerOptions.h>
#include "Jit.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/IR/LegacyPassManager.h"


using namespace llvm;
using namespace rr;

namespace rrllvm {

    class MCJit : public Jit {
    public:

        explicit MCJit(std::uint32_t options);

        void mapFunctionsToJitSymbols() override;

        void transferObjectsToResources(std::shared_ptr<rrllvm::ModelResources> modelResources) override;

        std::uint64_t lookupFunctionAddress(const std::string& name) override;

        void addObjectFile(rrOwningBinary owningObject) override;

        void addObjectFile(std::unique_ptr<llvm::object::ObjectFile> objectFile) override;

        void addObjectFile(std::unique_ptr<llvm::MemoryBuffer> obj) override;

        const llvm::DataLayout& getDataLayout() const override;

        void addModule() override;

        void addModule(llvm::Module* M) override;

        void addModule(std::unique_ptr<llvm::Module> M, std::unique_ptr<llvm::LLVMContext> ctx) override;

        /**
         * @brief Adds the member variable context and associated module
         * to current jit engine. The module is first converted into
         * an in memory object file and then stored as a string
         * in the member cariable compiledModuleBinaryStream.
         */
        void addModuleViaObjectFile();

        std::unique_ptr<llvm::MemoryBuffer> getCompiledModelFromCache(const std::string &sbmlMD5) override;

        ExecutionEngine* getExecutionEngineNonOwning() const;

        llvm::TargetMachine *getTargetMachine() ;

        llvm::legacy::FunctionPassManager *getFunctionPassManager() const ;

        llvm::raw_svector_ostream &getCompiledModuleStream();

        std::string getModuleAsString(std::string sbmlMD5) override;

    private:
        /**
         * @brief make the calls to PassManager to
         * optimize the LLVM IR module.
         * @details This is different between MCJit and LLJit
         * in that the optimization module is already present in
         * the LLJit layer stack, so optimization occurs automatically when
         * a module is added. In contrast, in MCJit we need to call
         * this ourselves.
         */
        void writeObjectToBinaryStream();

        void initFunctionPassManager();

        /**
         * Docs say to stack allocate the EngineBuilder
         */
        EngineBuilder engineBuilder;
        std::unique_ptr<ExecutionEngine> executionEngine;
        std::unique_ptr<llvm::legacy::FunctionPassManager> functionPassManager;
        std::unique_ptr<std::string> errString;



    };

}

#endif //ROADRUNNER_MCJIT_H