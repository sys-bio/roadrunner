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

void CudaFunction::serialize(std::ostream& os) const {
    Function::serialize(os);
}

void CudaKernel::serialize(std::ostream& os) const {
    CudaFunction::serialize(os);
}

void CudaModule::serialize(std::ostream& os) const {
    for (Function* f : getFunctions())
        f->serialize(os);
}

} // namespace dom

} // namespace rrgpu

} // namespace rr
