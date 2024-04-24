//
// Created by Ciaran on 15/11/2021.
//

#ifndef ROADRUNNER_JITFACTORY_H
#define ROADRUNNER_JITFACTORY_H

#include "rrConfig.h"
#include "MCJit.h"
#include "LLJit.h"
#include "Jit.h"

namespace rrllvm {

    /** @class JitFactory
     * @brief Use the options in LoadSBMLOptions.ModelGeneratorOpt
     * or fall back on the global Config settings to create a
     * Jit.
     *
     * @example
     *  @code
     *   LoadSBMLOptions opt;
     *   opt.setLLVMBackend(LoadSBMLOptions::MCJIT); // or LoadSBMLOptions::LLJit;
     *   std::unique_ptr<Jit> j = JitFactory::makeJitEngine(opt.modelGeneratorOpt);
     *   // OR
     *   Config::setValue(LLVM_BACKEND, Config::LLVM_BACKEND_VALUES::MCJIT); // default
     *   // OR
     *   Config::setValue(LLVM_BACKEND, Config::LLVM_BACKEND_VALUES::LLJIT); // alternative
     *   std::unique_ptr<Jit> j = makeJitEngine();
     *  @endcode
     */
    class JitFactory {
    public:

        JitFactory() = default;

        /**
         * @brief Create a Jit engine using local options provided by the user
         */
        static Jit* makeJitEngine(std::uint32_t opt);

        /**
         * @brief Create a Jit engine using the global options in Config.
         * @details LoadSBMLOptions is populated based on the global Config values.
         * This function instantiates the LoadSBMLOptions and provides the default
         * modelGeneratorOpt to JitFactory::makeJitEngine(opt);
         */
        static Jit* makeJitEngine();

    };

}

#endif //ROADRUNNER_JITFACTORY_H
