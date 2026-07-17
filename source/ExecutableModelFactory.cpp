/*
 * ExecutableModelFactory.cpp
 *
 *  Created on: Dec 11, 2014
 *      Author: andy
 */

#include <ExecutableModelFactory.h>
#include "rrRoadRunnerOptions.h"
#include "rrUtils.h"
#include "rrException.h"
#include <iostream>
#include <filesystem>

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
#include "c/rrCCompiler.h"
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
        return (std::filesystem::path("..") / "compilers" / "tcc" / "tcc.exe").string();
#else
        // the default compiler on Unix systems is 'cc', the standard enviornment
        // for the default compiler is 'CC'.
        return getenv("CC") ? getenv("CC") : "gcc";
#endif
#endif
    }

    Compiler* Compiler::New() {
#if defined(BUILD_LLVM)
        return new rrllvm::LLVMCompiler();
#elif defined(BUILD_LEGACY_C)
        // no code generation / compilation is needed to select the legacy
        // C back end, but a Compiler instance is still handed out for
        // source compatibility with code that expects one.
        return new CCompiler("", Compiler::getDefaultCompiler());
#else
        throw Exception("No model generation back end (BUILD_LLVM or BUILD_LEGACY_C) was enabled at build time");
#endif
    }

    ExecutableModel* rr::ExecutableModelFactory::createModel(
        const libsbml::SBMLDocument* sbml, const std::string& md5, const Dictionary* dict) {
        // note: conserved moieties are now taken into account in rrUtils::getSBMLMD5
        LoadSBMLOptions opt(dict);
#if defined(BUILD_LLVM)
        return rrllvm::LLVMModelGenerator::createModel(sbml, md5, opt.modelGeneratorOpt);
#elif defined(BUILD_LEGACY_C)
        char* cstr = libsbml::writeSBMLToString(const_cast<libsbml::SBMLDocument*>(sbml));
        std::string sbmlStr(cstr ? cstr : "");
        CModelGenerator gen(getTempDir(), "", Compiler::getDefaultCompiler());
        return gen.createModel(sbmlStr, opt.modelGeneratorOpt);
#else
        throw Exception("No model generation back end (BUILD_LLVM or BUILD_LEGACY_C) was enabled at build time");
#endif
    }

    ExecutableModel* rr::ExecutableModelFactory::createModel(std::istream& in, uint modelGeneratorOpt) {
#if defined(BUILD_LLVM)
        return new rrllvm::LLVMExecutableModel(in, modelGeneratorOpt);
#else
        throw Exception("Restoring a saved model from a stream is not supported by the legacy C back end");
#endif
    }

    ExecutableModel*
        ExecutableModelFactory::regenerateModel(ExecutableModel* oldModel, libsbml::SBMLDocument* doc, uint options) {
#if defined(BUILD_LLVM)
        return rrllvm::LLVMModelGenerator::regenerateModel(oldModel, doc, options);
#else
        throw Exception("Regenerating a model in place is not supported by the legacy C back end");
#endif
    }


} /* namespace rr */

