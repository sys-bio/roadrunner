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
    typedef GPUEntryPoint::Precision Precision;

    /// Empty ctor
    CudaExecutableModule();

    /// Move ctor
    CudaExecutableModule(CudaExecutableModule&& other);

    /// Dtor
    ~CudaExecutableModule();

    /// Copy ctor
    CudaExecutableModule(const CudaExecutableModule&);

    /// Move assn
    CudaExecutableModule& operator=(CudaExecutableModule&& other);

    /// Get the callable entry point
    GPUEntryPoint getEntry() const;

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
    struct GenerateParams {
        GenerateParams(CudaExecutableModule::Precision precision_, const std::string& hashedid_, const std::string& entryName_)
          : precision(precision_), hashedid(hashedid_), entryName(entryName_) {}

        CudaExecutableModule::Precision precision;
        std::string hashedid;
        std::string entryName;
    };

    CudaExecutableModule generate(dom::CudaModule& mod, const GenerateParams& params);
};

} // namespace rrgpu

} // namespace rr

#endif /* GPUSimModelH */
