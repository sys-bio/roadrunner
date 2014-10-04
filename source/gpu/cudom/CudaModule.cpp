// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

/*
 * CudaModule.cpp
 *
 *  Created on: Aug 21, 2014
 *
 *  Author: JKM
 */

// == INCLUDES ================================================

# include "CudaModule.hpp"
# include "gpu/GPUSimException.h"

// == CODE ====================================================


namespace rr
{

namespace rrgpu
{

namespace dom
{

void CudaVariableDeclarationExpression::serialize(Serializer& s) const {
    if (isShared())
        s << "extern __shared__ ";
    VariableDeclarationExpression::serialize(s);
}

void CudaFunction::serialize(Serializer& s) const {
    if (getIsDeviceFun())
        s << "__device__" << " ";
    Function::serialize(s);
}

void CudaKernel::serialize(Serializer& s) const {
    s << "__global__" << " ";
    CudaFunction::serialize(s);
}

CudaKernelCallExpression::CudaKernelCallExpression(int nblocks, int nthreads, int shared_mem_size, Function* func) : FunctionCallExpression(func) {
    CudaKernel* k = dynamic_cast<CudaKernel*>(func);
    if (!k)
        throw_gpusim_exception("Not a CUDA kernel");

    nblocks_.reset(new LiteralIntExpression(nblocks));
    nthreads_.reset(new LiteralIntExpression(nthreads));
    shared_mem_size_.reset(new LiteralIntExpression(shared_mem_size));
}

void CudaKernelCallExpression::serialize(Serializer& s) const {
    s << func_->getName();
    s << "<<<";
    s << *nblocks_;
    s << ", ";
    s << *nthreads_;
    s << ", ";
    s << *shared_mem_size_;
    s << ">>>";
    s << "(";
    serializeArgs(s);
    s << ")";
}

// -- CudaExponentiationExpressionSP --

void CudaExponentiationExpressionSP::serialize(Serializer& s) const {
    // NOTE: approximation
    s << "__powf(" << *getLHS() << ", " << *getRHS() << ")";
}

// -- CudaExponentiationExpressionDP --

void CudaExponentiationExpressionDP::serialize(Serializer& s) const {
    // NOTE: approximation
    s << "pow(" << *getLHS() << ", " << *getRHS() << ")";
}

// -- CudaModule --

ExponentiationExpressionPtr CudaModule::powSP(ExpressionPtr&& x, ExpressionPtr&& y) const {
    return ExponentiationExpressionPtr(new CudaExponentiationExpressionSP(std::move(x), std::move(y)));
}

ExponentiationExpressionPtr CudaModule::pow(ExpressionPtr&& x, ExpressionPtr&& y) const {
    return ExponentiationExpressionPtr(new CudaExponentiationExpressionDP(std::move(x), std::move(y)));
}

void CudaModule::serialize(Serializer& s) const {
    s << "#include <stdlib.h>\n";
    s << "#include <stdio.h>\n";
    s << "\n";

    serializeMacros(s);
    s << "\n";

    serializeStatements(s);
    s <<  "\n";

    for (Function* f : getFunctions()) {
        f->serialize(s);
        s << "\n";
    }
}

} // namespace dom

} // namespace rrgpu

} // namespace rr
