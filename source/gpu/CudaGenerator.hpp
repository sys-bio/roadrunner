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
# include "CudaCodeCompiler.hpp"

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
    typedef CudaExecutableModule::Precision Precision;

    /// Empty ctor
    CudaGenerator();

    ~CudaGenerator();

    /// Set numerical precision
    void setPrecision(Precision p);

    /// Get numerical precision
    Precision getPrecision() const;

    /// Generate the executable module
    void generate(GPUSimExecutableModel& model);

    /// Get the callable entry point after the module has been generated
    GPUEntryPoint getEntryPoint();

protected:
    std::unique_ptr<CudaGeneratorImpl> impl_;

    Precision p_ = Precision::Single;
};

} // namespace dom

} // namespace rrgpu

} // namespace rr

#endif // header guard
