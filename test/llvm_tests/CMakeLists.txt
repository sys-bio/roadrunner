add_test_executable(test_llvm_LLVMModelDataSymbolsTests test_targets LLVMModelDataSymbolsTests.cpp)
add_test_executable(test_llvm_MCJitTests test_targets MCJitTests.cpp JitTests.h)
add_test_executable(test_llvm_LLJitTests test_targets LLJitTests.cpp JitTests.h)
add_test_executable(test_llvm_MCJitMapFunctionsToJitSymbolsTests test_targets FunctionMapping/MCJitMapFunctionsToJitSymbolsTests.cpp)
add_test_executable(test_llvm_LLJitMapFunctionsToJitSymbolsTests test_targets FunctionMapping/LLJitMapFunctionsToJitSymbolsTests.cpp)
add_test_executable(test_llvm_CsrMatrixJitAliasSignatureTests test_targets FunctionMapping/CsrMatrixJitAliasSignatureTests.cpp)
add_test_executable(test_llvm_RandomTests test_targets RandomTests.cpp)
add_test_executable(
        test_llvm_EvalInitialConditionsTests test_targets
        EvalInitialConditionsTests.cpp
        LLVMGeneratedFunctionTests.h
        LLVMGeneratedFunctionTests.cpp

)
add_test_executable(test_llvm_JitFactoryTests test_targets JitFactoryTests.cpp)

#add_executable(BuildAModel BuildAModel.cpp)
#target_link_libraries(BuildAModel PRIVATE roadrunner-static gtest gtest_main)
#target_include_directories(BuildAModel PRIVATE roadrunner-static)

set(test_targets ${test_targets} PARENT_SCOPE)
