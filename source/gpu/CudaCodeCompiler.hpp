// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

// == FILEDOC =================================================

/** @file CudaCodeCompiler.hpp
  * @author JKM
  * @date 09/16/2014
  * @copyright Apache License, Version 2.0
  * @brief Compiles CUDA code
**/

#ifndef rrGPUSimCudaCodeCompilerH
#define rrGPUSimCudaCodeCompilerH

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

class CudaExecutableModuleImpl;

/**
 * @brief Compiles CUDA code
 */
struct RR_DECLSPEC CudaExecutableModule
{
public:
    typedef GPUSimExecutableModel::EntryPointSigSP EntryPointSigSP;
    typedef GPUSimExecutableModel::EntryPointSigDP EntryPointSigDP;

    /// Empty ctor
    CudaExecutableModule();

    /// Dtor
    ~CudaExecutableModule();

    /// Copy ctor
    CudaExecutableModule(const CudaExecutableModule&);

    /// Get the callable entry point
    EntryPointSigSP getEntrySP() const;
    EntryPointSigDP getEntryDP() const;

protected:
    friend class CudaCodeCompiler;

    std::unique_ptr<CudaExecutableModuleImpl> impl_;
};

/**
 * @brief Compiles CUDA code
 */
class RR_DECLSPEC CudaCodeCompiler
{
public:

    CudaExecutableModule generate(dom::CudaModule& mod, const std::string hashedid, const std::string& entryName);
};

} // namespace rrgpu

} // namespace rr

#endif /* GPUSimModelH */
