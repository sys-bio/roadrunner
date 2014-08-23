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

void CudaFunction::serialize(Serializer& s) const {
    Function::serialize(s);
}

void CudaKernel::serialize(Serializer& s) const {
    s << "__global__" << " ";
    CudaFunction::serialize(s);
}

void CudaModule::serialize(Serializer& s) const {
    for (Function* f : getFunctions()) {
        f->serialize(s);
        s << "\n";
    }
}

} // namespace dom

} // namespace rrgpu

} // namespace rr
