/*
 * ModelGenerator.cpp
 *
 *  Created on: May 20, 2013
 *      Author: andy
 */
#pragma hdrstop
#include "ModelGenerator.h"

#if defined(BUILD_GPUSIM)
#include "gpu/GPUSimModelGenerator.h"
#endif

#if defined(BUILD_LLVM)
#include "llvm/LLVMModelGenerator.h"
#endif

#if defined(BUILD_LEGACY_C)
#include "c/rrCModelGenerator.h"
#endif

#include "rrLogger.h"
#include <string>
#include <algorithm>

using namespace std;
namespace rr {

ModelGenerator* ModelGenerator::New(const string& compiler, const string& tempFolder,
            const string& supportCodeFolder)
{
    Log(Logger::LOG_INFORMATION) << "createing model generator, compiler: \"" << compiler << "\"";

#if defined(BUILD_GPUSIM)
    {
	string ucomp = compiler;
	std::transform(ucomp.begin(), ucomp.end(),ucomp.begin(), ::toupper);
	
	if (ucomp == "GPUSIM")
	{
	    Log(Logger::LOG_INFORMATION) << "Creating GPU based model generator.";
	    return new rrgpu::GPUSimModelGenerator(compiler);
	}
    }
#endif
    
#if defined(BUILD_LLVM) && !defined(BUILD_LEGACY_C)

    Log(Logger::LOG_INFORMATION) << "Creating LLVM based model generator.";
    return new rrllvm::LLVMModelGenerator(compiler);

#endif

#if defined(BUILD_LLVM) && defined(BUILD_LEGACY_C)
    string ucomp = compiler;
    std::transform(ucomp.begin(), ucomp.end(),ucomp.begin(), ::toupper);

    if (ucomp == "LLVM")
    {
        Log(Logger::LOG_INFORMATION) << "Creating LLVM based model generator.";
        return new rrllvm::LLVMModelGenerator(compiler);
    }

    Log(Logger::LOG_INFORMATION) << "Creating C based model generator using " << compiler << " compiler.";

    return new CModelGenerator(tempFolder, supportCodeFolder, compiler);
#endif

#if !defined(BUILD_LLVM) && defined(BUILD_LEGACY_C)

    Log(Logger::LOG_INFORMATION) << "Creating C based model generator using " << compiler << " compiler.";

    // default (for now...), the old C code generating model generator.
    return new CModelGenerator(tempFolder, supportCodeFolder, compiler);
#endif

#if !defined(BUILD_LLVM) && !defined(BUILD_LEGACY_C) && !defined(BUILD_GPUSIM)
#error Must built at least one ModelGenerator backend, either BUILD_LLVM or BUILD_LEGACY_C or BUILD_GPUSIM
#endif


}

} /* namespace rr */
