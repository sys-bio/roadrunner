// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README
//   -- USE AT YOUR OWN RISK --

// == FILEDOC =================================================

/** @file CudaModule.h
  * @author JKM
  * @date 08/21/2014
  * @copyright Apache License, Version 2.0
  * @brief DOM for single CUDA file
**/

#ifndef rrGPU_CUDADOM_CudaModule_H
#define rrGPU_CUDADOM_CudaModule_H

// == MACROS ==================================================

#if !defined(__cplusplus)
#   error "You are including a .hpp in non-C++ code"
#endif

#if (__cplusplus < 201103L) && !defined(_MSC_VER)
#   error "This file requires C++11 support in order to compile"
#endif

// == INCLUDES ================================================

# include <gpu/bdom/Structures.hpp>

// == CODE ====================================================


namespace rr
{

namespace rrgpu
{

namespace dom
{

/**
 * @author JKM
 * @brief Variable initialization expressions
 */
class CudaVariableDeclarationExpression : public VariableDeclarationExpression {
public:
    /// Ctor
    CudaVariableDeclarationExpression(Variable* var, bool shared=false)
      : VariableDeclarationExpression(var), cuda_shared_(shared) {}

    /// Is this a shared memory declaration?
    bool isShared() const { return cuda_shared_; }
    /// Set whether this is a shared memory declaration?
    void setIsShared(bool val) { cuda_shared_ = val; }

    virtual void serialize(Serializer& s) const;


    // TODO: replace with LLVM-style casting
    static CudaVariableDeclarationExpression* downcast(Expression* s) {
        auto result = dynamic_cast<CudaVariableDeclarationExpression*>(s);
        if (!result)
            throw_gpusim_exception("Downcast failed: incorrect type");
        return result;
    }

protected:
    bool cuda_shared_ = false;
};

/**
 * @author JKM
 * @brief CUDA function
 */
class CudaFunction : public Function {
public:
    using Function::Function;
    virtual void serialize(Serializer& s) const;
};

/**
 * @author JKM
 * @brief CUDA function
 */
class CudaKernel : public CudaFunction {
public:
    using CudaFunction::CudaFunction;

    ~CudaKernel() {}

    virtual bool requiresSpecialCallingConvention() const { return true; }

    virtual void serialize(Serializer& s) const;
};
typedef std::unique_ptr<CudaKernel> CudaKernelPtr;

/**
 * @author JKM
 * @brief A class to encapsulate expressions
 */
class CudaKernelCallExpression : public FunctionCallExpression {
public:
    CudaKernelCallExpression(int nblocks, int nthreads, int shared_mem_size, Function* func);

    /// N-arg ctor
    template <class... Args>
    CudaKernelCallExpression(int nblocks, int nthreads, int shared_mem_size, Function* func, Args&&... args)
      : FunctionCallExpression(func, std::forward<Args>(args)...) {
        nblocks_.reset(new LiteralIntExpression(nblocks));
        nthreads_.reset(new LiteralIntExpression(nthreads));
        shared_mem_size_.reset(new LiteralIntExpression(shared_mem_size));
    }

    virtual void serialize(Serializer& s) const;

protected:
    ExpressionPtr nblocks_;
    ExpressionPtr nthreads_;
    ExpressionPtr shared_mem_size_;
};

/**
 * @author JKM
 * @brief CUDA module
 */
class CudaModule : public Module {
public:
    typedef std::unique_ptr<CudaFunction> CudaFunctionPtr;

    CudaModule()
      : printf_(new Function("printf",
                             BaseTypes::getTp(BaseTypes::INT),
                             FunctionParameterPtr(new FunctionParameter(BaseTypes::getTp(BaseTypes::CSTR), "format_str")))),
        cudaDeviceSynchronize_(new Function("cudaDeviceSynchronize",
                                            BaseTypes::getTp(BaseTypes::VOID)))
        {
        printf_->setIsVarargs(true);
    }

    /// Serialize to a .cu source file
    virtual void serialize(Serializer& s) const;

    void addFunction(CudaFunction&& f) {
        func_.emplace_back(new CudaFunction(std::move(f)));
    }

    void addFunction(CudaFunctionPtr&& f) {
        func_.emplace_back(std::move(f));
    }

    const Function* getPrintf() {
        return printf_.get();
    }

    const Function* getCudaDeviceSynchronize() {
        return cudaDeviceSynchronize_.get();
    }

protected:
    FunctionPtr printf_;
    FunctionPtr cudaDeviceSynchronize_;
};

} // namespace dom

} // namespace rrgpu

} // namespace rr

#endif // header guard
