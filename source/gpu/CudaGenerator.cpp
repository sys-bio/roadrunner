// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

/*
 * CudaGenerator.cpp
 *
 *  Created on: Aug 21, 2014
 *
 *  Author: JKM
 */

// == INCLUDES ================================================

# include "CudaGenerator.hpp"

#include "Poco/SharedLibrary.h"

#include <fstream>
#include <sstream>

// == CODE ====================================================


namespace rr
{

namespace rrgpu
{

namespace dom
{

void CudaGenerator::generate(const GPUSimModel& model) {
    CudaModule mod;
    std::string entryName = "cuEntryPoint";

    // construct the DOM

    CudaKernelPtr kernel(new CudaKernel("GPUIntMEBlockedRK4", BaseTypes::getTp(BaseTypes::VOID)));

    // printf
    kernel->addStatement(ExpressionPtr(new FunctionCallExpression(mod.getPrintf(), ExpressionPtr(new StringLiteralExpression("in kernel\\n")))));

    CudaModule::CudaFunctionPtr entry(new CudaFunction(entryName, BaseTypes::getTp(BaseTypes::VOID)));
//     entry->setHasCLinkage(true);

    // printf
    entry->addStatement(ExpressionPtr(new FunctionCallExpression(mod.getPrintf(), ExpressionPtr(new StringLiteralExpression("in cuda\\n")))));

    // call the kernel
    ExpressionPtr calltokern(new CudaKernelCallExpression(1, 1, 1, kernel.get()));
    entry->addStatement(StatementPtr(new ExpressionStatement(std::move(calltokern))));

    // cudaDeviceSynchronize
    entry->addStatement(ExpressionPtr(new FunctionCallExpression(mod.getCudaDeviceSynchronize())));

    mod.addFunction(std::move(kernel));
    mod.addFunction(std::move(entry));

    // serialize the module to a document

    {
        Serializer s("/tmp/rr_cuda_model.cu");
        mod.serialize(s);
    }

    // compile the module

    FILE* pp = popen("nvcc -D__CUDACC__ -ccbin gcc -m32 -I/home/jkm/devel/src/roadrunner/source --ptxas-options=-v --compiler-options '-fPIC' -Drr_cuda_model_EXPORTS -Xcompiler ,\"-fPIC\",\"-fPIC\",\"-g\" -DNVCC --shared -o /tmp/rr_cuda_model.so /tmp/rr_cuda_model.cu 2>&1 >/dev/null", "r");

#define SBUFLEN 512
    char sbuf[SBUFLEN];
    fgets(sbuf, SBUFLEN, pp);

    int code = pclose(pp);
    pp = NULL;
    std::cerr << "Return code: " << code << "\n";

    if(code/256) {
        std::stringstream ss;
        ss << "Compiler errors:\n" << sbuf << "\n";
        throw_gpusim_exception(ss.str());
    }

    // get the mangled name of the entry point

    pp = popen("nm -g /tmp/rr_cuda_model.so | grep -ohe '_.*cuEntryPoint[^\\s]*$'", "r");

    fgets(sbuf, SBUFLEN, pp);
    std::string entryMangled{sbuf};
    while(entryMangled.back() == '\n' && entryMangled.size())
        entryMangled = std::string(entryMangled,0,entryMangled.size()-1);

    pclose(pp);

    // load the module

    Poco::SharedLibrary so;
    try {
        so.load("/tmp/rr_cuda_model.so");
    } catch(Poco::LibraryLoadException e) {
        throw_gpusim_exception("Cannot load lib: " + e.message());
    }

//     std::cerr << "Loading symbol " + entryMangled + "\n";
    if(!so.hasSymbol(entryMangled))
        throw_gpusim_exception("Lib has no symbol \"" + entryMangled + "\"");

    typedef void (*EntryPointSig)();
    EntryPointSig entryPoint;
    entryPoint = (EntryPointSig)so.getSymbol(entryMangled);

    entryPoint();

}

} // namespace dom

} // namespace rrgpu

} // namespace rr
