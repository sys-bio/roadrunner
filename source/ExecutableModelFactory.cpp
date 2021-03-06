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

#include "testing/CXXExecutableModel.h"
#include "testing/CXXEnzymeExecutableModel.h"


namespace rr {

/*
static ModelGenerator* createModelGenerator(const std::string& compiler, const std::string& tempFolder,
            const std::string& supportCodeFolder);
*/

/**
 * implement the couple Compiler methods, this will go, here for source compatiblity.
 */

std::string Compiler::getDefaultCompiler()
{
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
        const std::string& sbml, const Dictionary* dict)
{
    LoadSBMLOptions opt(dict);

    if(opt.hasKey("cxxEnzymeTest")) {
        return new rrtesting::CXXEnzymeExecutableModel(dict);
    }
    return rrllvm::LLVMModelGenerator::createModel(sbml, opt.modelGeneratorOpt);
}

ExecutableModel *rr::ExecutableModelFactory::createModel(std::istream& in, uint modelGeneratorOpt)
{
	return new rrllvm::LLVMExecutableModel(in, modelGeneratorOpt);
}

ExecutableModel* ExecutableModelFactory::regenerateModel(ExecutableModel* oldModel, libsbml::SBMLDocument* doc, uint options)
{
	return  rrllvm::LLVMModelGenerator::regenerateModel(oldModel, doc, options);
}

/*
ModelGenerator* createModelGenerator(const std::string& compiler, const std::string& tempFolder,
            const std::string& supportCodeFolder)
{
#if defined(BUILD_LLVM) && !defined(BUILD_LEGACY_C)

    rrLog(Logger::LOG_INFORMATION) << "Creating LLVM based model generator.";
    return new rrllvm::LLVMModelGenerator();

#endif

#if defined(BUILD_LLVM) && defined(BUILD_LEGACY_C)
    std::string ucomp = compiler;
    std::transform(ucomp.begin(), ucomp.end(),ucomp.begin(), ::toupper);

    if (ucomp == "LLVM")
    {
        rrLog(Logger::LOG_INFORMATION) << "Creating LLVM based model generator.";
        return new rrllvm::LLVMModelGenerator();
    }

    rrLog(Logger::LOG_INFORMATION) << "Creating C based model generator using " << compiler << " compiler.";

    return new CModelGenerator(tempFolder, supportCodeFolder, compiler);
#endif

#if !defined(BUILD_LLVM) && defined(BUILD_LEGACY_C)

    rrLog(Logger::LOG_INFORMATION) << "Creating C based model generator using " << compiler << " compiler.";

    // default (for now...), the old C code generating model generator.
    return new CModelGenerator(tempFolder, supportCodeFolder, compiler);
#endif

#if !defined(BUILD_LLVM) && !defined(BUILD_LEGACY_C)
#error Must built at least one ModelGenerator backend, either BUILD_LLVM or BUILD_LEGACY_C
#endif
}
*/


} /* namespace rr */

