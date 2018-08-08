/*
 * LLVMIncludes.h
 *
 *  Created on: Jun 29, 2013
 *      Author: andy
 */

#ifndef RRLLVMINCLUDES_H_
#define RRLLVMINCLUDES_H_

#if !defined(LLVM_VERSION_MINOR)

// this is likely a very common macro, so save it
#pragma push_macro("CONFIG_H")
#include <llvm/Config/llvm-config.h>
#pragma pop_macro("CONFIG_H")
#endif

#pragma push_macro("min")
#pragma push_macro("max")

#undef min
#undef max

#ifdef _MSC_VER
#pragma warning( push )
#pragma warning( disable : 4355 )
#pragma warning( disable : 4244 )
#endif

#if (LLVM_VERSION_MAJOR == 6)
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Verifier.h>
#elif (LLVM_VERSION_MAJOR == 3) && (LLVM_VERSION_MINOR >= 5)
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Verifier.h>
#elif (LLVM_VERSION_MAJOR == 3) && ((LLVM_VERSION_MINOR == 3) || (LLVM_VERSION_MINOR == 4))
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/Analysis/Verifier.h>
#elif (LLVM_VERSION_MAJOR == 3) && (LLVM_VERSION_MINOR == 2)
#include <llvm/DataLayout.h>
#include <llvm/DerivedTypes.h>
#include <llvm/IRBuilder.h>
#include <llvm/LLVMContext.h>
#include <llvm/Module.h>
#include <llvm/Intrinsics.h>
#include <llvm/Analysis/Verifier.h>
#elif (LLVM_VERSION_MAJOR == 3) && (LLVM_VERSION_MINOR == 1)
#include <llvm/DerivedTypes.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/JIT.h>
#include <llvm/LLVMContext.h>
#include <llvm/Module.h>
#include <llvm/PassManager.h>
#include <llvm/Target/TargetData.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Support/IRBuilder.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Analysis/Verifier.h>
#else
#error "Unsupported LLVM Version, require LLVM >= 3.1"
#endif

#include <llvm/Analysis/Passes.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/Host.h>

#ifdef _MSC_VER
#pragma warning( pop )
#endif

#pragma pop_macro("min")
#pragma pop_macro("max")

namespace rrllvm {
/**
 * C++ 11 style to_string for LLVM types
 */
std::string to_string(const llvm::Value *);
}




#endif /* RRLLVMINCLUDES_H_ */
