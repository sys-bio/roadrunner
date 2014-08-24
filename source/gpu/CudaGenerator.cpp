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
    CudaKernelPtr kernel(new CudaKernel("GPUIntMEBlockedRK4", BaseTypes::getTp(BaseTypes::VOID)));
    CudaModule::CudaFunctionPtr entry(new CudaFunction("entryPoint", BaseTypes::getTp(BaseTypes::VOID)));

    // call the kernel
    ExpressionPtr calltokern(new CudaKernelCallExpression(1, 1, 1, kernel.get()));
    entry->addStatement(StatementPtr(new ExpressionStatement(std::move(calltokern))));

    mod.addFunction(std::move(kernel));
    mod.addFunction(std::move(entry));

    {
        Serializer s("/tmp/rr_cuda_model.cu");
        mod.serialize(s);
    }

    FILE* pp = popen("nvcc -M -D__CUDACC__ -ccbin gcc -m32 -I/home/jkm/devel/src/roadrunner/source --ptxas-options=-v --compiler-options '-fPIC' -o /tmp/rr_cuda_model.so --shared /tmp/rr_cuda_model.cu 2>&1 >/dev/null", "r");

    char err[512];
    fgets(err, 512, pp);

    int code = pclose(pp);
    std::cerr << "Return code: " << code << "\n";

    if(code/256) {
        std::stringstream ss;
        ss << "Compiler errors:\n" << err << "\n";
        throw_gpusim_exception(ss.str());
    }
}

} // namespace dom

} // namespace rrgpu

} // namespace rr
