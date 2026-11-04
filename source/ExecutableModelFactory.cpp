/*
 * ExecutableModelFactory.cpp
 *
 *  Created on: Dec 11, 2014
 *      Author: andy
 */

#include <ExecutableModelFactory.h>
#include "rrRoadRunnerOptions.h"
#include <iostream>

#if defined(BUILD_LLVM)
#ifdef _MSC_VER
#pragma warning(disable: 4146)
#pragma warning(disable: 4141)
#pragma warning(disable: 4267)
#pragma warning(disable: 4624)
#endif
#include "llvm/LLVMModelGenerator.h"
#include "llvm/LLVMCompiler.h"
#include "llvm/LLVMExecutableModel.h"
#ifdef _MSC_VER
#pragma warning(default: 4146)
#pragma warning(default: 4141)
#pragma warning(default: 4267)
#pragma warning(default: 4624)
#endif
#endif

#if defined(BUILD_LEGACY_C)
#include "c/rrCModelGenerator.h"
#endif

#include "rrLogger.h"
#include <string>
#include <algorithm>

namespace rr {

    /*
    static ModelGenerator* createModelGenerator(const std::string& compiler, const std::string& tempFolder,
                const std::string& supportCodeFolder);
    */

    /**
     * implement the couple Compiler methods, this will go, here for source compatiblity.
     */

    std::string Compiler::getDefaultCompiler() {
#if defined(BUILD_LLVM)
        return "LLVM";
#else
#if defined(_WIN32)
        return joinPath("..", "compilers", "tcc", "tcc.exe");
#else
        // the default compiler on Unix systems is 'cc', the standard enviornment
        // for the default compiler is 'CC'.
        return getenv("CC") ? getenv("CC") : "gcc";
#endif
#endif
    }

    Compiler* Compiler::New() {
        return new rrllvm::LLVMCompiler();
    }

    ExecutableModel* rr::ExecutableModelFactory::createModel(
        const libsbml::SBMLDocument* sbml, const std::string& md5, const Dictionary* dict) {
        // note: the load options, e.g. conserved moieties, are taken into account in rr::getSBMLMD5
        LoadSBMLOptions opt(dict);
        return rrllvm::LLVMModelGenerator::createModel(sbml, md5, opt.modelGeneratorOpt);
    }

    ExecutableModel* rr::ExecutableModelFactory::createModel(std::istream& in, uint modelGeneratorOpt) {
        return new rrllvm::LLVMExecutableModel(in, modelGeneratorOpt);
    }

    ExecutableModel*
        ExecutableModelFactory::regenerateModel(ExecutableModel* oldModel, libsbml::SBMLDocument* doc, uint options) {
        return rrllvm::LLVMModelGenerator::regenerateModel(oldModel, doc, options);
    }


} /* namespace rr */

