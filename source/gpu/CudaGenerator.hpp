// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

// == FILEDOC =================================================

/** @file CudaGenerator.h
  * @author JKM
  * @date 08/21/2014
  * @copyright Apache License, Version 2.0
  * @brief Generates the CUDA documents
**/

#ifndef rrGPU_CudaGenerator_H
#define rrGPU_CudaGenerator_H

// == MACROS ==================================================

#if !defined(__cplusplus)
#   error "You are including a .hpp in non-C++ code"
#endif

#if (__cplusplus < 201103L) && !defined(_MSC_VER)
#   error "This file requires C++11 support in order to compile"
#endif

// == INCLUDES ================================================

# include "gpu/cudom/CudaModule.hpp"
# include "GPUSimExecutableModel.h"

// == CODE ====================================================


namespace rr
{

namespace rrgpu
{

namespace dom
{

class CudaGeneratorImpl;

/**
  * @author JKM
  * @brief CUDA generator
  * @note Uses PIMPL
  */
class CudaGenerator {
public:
    typedef GPUSimExecutableModel::EntryPointSig EntryPointSig;

    CudaGenerator();

    ~CudaGenerator();

    void generate(GPUSimExecutableModel& model);

    EntryPointSig getEntryPoint();

protected:
    std::unique_ptr<CudaGeneratorImpl> impl_;
};

} // namespace dom

} // namespace rrgpu

} // namespace rr

#endif // header guard
