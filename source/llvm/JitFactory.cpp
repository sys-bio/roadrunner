//
// Created by Ciaran on 15/11/2021.
//

#include "JitFactory.h"

namespace rrllvm {

    Jit* JitFactory::makeJitEngine(std::uint32_t opt) {
        rrLog(Logger::LOG_DEBUG) << __FUNC__;
        Jit* jit = NULL;
        if (opt & LoadSBMLOptions::MCJIT) {
            rrLog(Logger::LOG_DEBUG) << "Creating an MCJit object.";
            jit = new rrllvm::MCJit(opt);
        }

        else if (opt & LoadSBMLOptions::LLJIT) {
            jit = new rrllvm::LLJit(opt);
        }
        
        else {
            throw std::invalid_argument("Cannot create JIT object; need to say whether it's MCJit or LLJit in the options.");
        }

        rrLog(Logger::LOG_DEBUG) << "Done creating a Jit object.";
        return jit;
    }

    Jit* JitFactory::makeJitEngine() {
        LoadSBMLOptions opt;
        return JitFactory::makeJitEngine(opt.modelGeneratorOpt);
    }
}
