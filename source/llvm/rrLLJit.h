//
// Created by Ciaran on 25/10/2021.
//

#ifndef ROADRUNNER_RRLLJIT_H
#define ROADRUNNER_RRLLJIT_H

#include "llvm/Jit.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"

using namespace rr;

namespace rrllvm {

    class rrLLJit : public Jit{
    public:

        explicit rrLLJit(std::uint32_t options);

        void mapFunctionsToJitSymbols() override;

        std::uint64_t getFunctionAddress(const std::string &name) override;

        llvm::TargetMachine *getTargetMachine() override;

        void addObjectFile(llvm::object::OwningBinary<llvm::object::ObjectFile> owningObject) override;

        void finalizeObject() override;

        const llvm::DataLayout &getDataLayout() override;

        void addModule(llvm::Module *M) override;

        llvm::orc::LLJIT* getLLJitNonOwning() ;

        void addIRModule();

    private:
        void mapFunctionToJitAbsoluteSymbol(const std::string& funcName, std::uint64_t funcAddress);

        std::unique_ptr<llvm::orc::LLJIT> llJit;

    };

}

#endif //ROADRUNNER_RRLLJIT_H