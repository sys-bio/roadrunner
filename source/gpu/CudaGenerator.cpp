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
    ExpressionPtr calltokern(new FunctionCallExpression(kernel.get()));
    entry->addStatement(StatementPtr(new ExpressionStatement(std::move(calltokern))));

    mod.addFunction(std::move(kernel));
    mod.addFunction(std::move(entry));

    Serializer s("/tmp/rr_cuda_model.cu");
    mod.serialize(s);
}

} // namespace dom

} // namespace rrgpu

} // namespace rr
