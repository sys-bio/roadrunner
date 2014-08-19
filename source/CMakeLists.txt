CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(target roadrunner)

configure_file(rrConfigure.h.in rrConfigure.h)
include_directories(${RR_GENERATED_HEADER_PATH}) #we have to do this to get the generated header

OPTION(USE_POCO_NET "Use poco net library" ON)

# pre-process the GitInfo.h.in file and insert the response from git about 
# last commit and current branch.
# may not have git, set default replacement strings then. 
find_program(GIT git)

if(GIT)
    message("found git: ${GIT}")
    execute_process(
        COMMAND ${GIT} symbolic-ref -q --short HEAD
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        OUTPUT_VARIABLE ROADRUNNER_GIT_BRANCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )


    execute_process(
        COMMAND ${GIT} log -1 --format=%H
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        OUTPUT_VARIABLE ROADRUNNER_GIT_LAST_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )
else()
    # could not find GIT
    message("could not find git")

	set(ROADRUNNER_GIT_BRANCH "git not found, unknown git branch")
    set(ROADRUNNER_GIT_LAST_COMMIT "git not found, unknown git commit")
endif(GIT)

# writes the results of the git commands to GitInfo.h
message("ROADRUNNER_GIT_BRANCH ${ROADRUNNER_GIT_BRANCH}")
message("ROADRUNNER_GIT_LAST_COMMIT: ${ROADRUNNER_GIT_LAST_COMMIT}")
configure_file(GitInfo.h.in GitInfo.h @ONLY)

include_directories(
    .
    ${THIRD_PARTY_INSTALL_FOLDER}/include
    ${THIRD_PARTY_INSTALL_FOLDER}/include/rr-libstruct
    ${THIRD_PARTY_INSTALL_FOLDER}/include/sbml
    ${THIRD_PARTY_INSTALL_FOLDER}/include/cvode
    )

set(rrCoreSources
    rrConfig
    rrSteadyStateSolver
    Configurable
    rrConstants
    rrException
    rrGetOptions
    rrLogger
    rrExecutableModel
    rrRoadRunner
    rrRoadRunnerOptions
    rrStringUtils
    rrUtils
    Integrator
    CVODEIntegrator
    GillespieIntegrator
    RK4Integrator
    rrNLEQInterface
    rrTestSuiteModelSimulation
    rrIniKey
    rrIniSection
    rrIniFile
    rrFileName
    rrRoadRunnerData
    rrSelectionRecord
    ModelGenerator
    rrVersionInfo
    rrSparse
    rrSBMLModelSimulation
    rrSBMLReader
    Variant
    conservation/ConservationExtension
    conservation/ConservationDocumentPlugin
    conservation/ConservedMoietyPlugin
    conservation/ConservedMoietyConverter
    )

# initial sources (core)
set(rrSources ${rrCoreSources})

#     set_source_files_properties(platform/Backtrace.cpp PROPERTIES COMPILE_FLAGS "-Wno-format")

# Sources for the LLVM component
if(BUILD_LLVM)
    set(rrLLVMSources
        llvm/AssignmentRuleEvaluator
        llvm/ASTNodeCodeGen
        llvm/ASTNodeFactory
        llvm/ModelResources
        llvm/CodeGenBase
        llvm/LLVMCompiler
        llvm/EvalConversionFactorCodeGen
        llvm/EvalInitialConditionsCodeGen
        llvm/EvalRateRuleRatesCodeGen
        llvm/EvalReactionRatesCodeGen
        llvm/EventAssignCodeGen
        llvm/EventTriggerCodeGen
        llvm/EventQueue
        llvm/FunctionResolver
        llvm/LLVMExecutableModel
        llvm/GetRateRuleValuesCodeGen
        llvm/GetValuesCodeGen
        llvm/LoadSymbolResolverBase
        llvm/GetInitialValuesCodeGen
        llvm/GetEventValuesCodeGen
        llvm/KineticLawParameterResolver
        llvm/LLVMModelData
        llvm/ModelDataIRBuilder
        llvm/ModelDataSymbolResolver
        llvm/ModelInitialValueSymbolResolver
        llvm/LLVMModelDataSymbols
        llvm/LLVMModelGenerator
        llvm/ModelGeneratorContext
        llvm/LLVMModelSymbols
        llvm/SetRateRuleValuesCodeGen
        llvm/SetValuesCodeGen
        llvm/SetInitialValuesCodeGen
        llvm/SBMLSupportFunctions
        llvm/EvalVolatileStoichCodeGen
        llvm/SBMLInitialValueSymbolResolver
        )

    # set up appropriate flags for using LLVM
    set_source_files_properties(${rrLLVMSources} PROPERTIES COMPILE_FLAGS ${LLVM_FLAGS})
    # add LLVM-based model sources to list
    set(rrSources ${rrSources} ${rrLLVMSources})
endif(BUILD_LLVM)

# Sources for the GPUSim component
if(BUILD_GPUSIM)
    set(rrGPUSIMCoreSources
        gpu/GPUSimCompiler.cpp
        gpu/GPUSimExecutableModel.cpp
        gpu/GPUSimIntegrator.cpp
        gpu/GPUSimModel.cpp
        gpu/GPUSimModelGenerator.cpp
        gpu/ModelResources.cpp
        platform/Backtrace.cpp
        )

    # a system without C++11 will not have the hardware to run the GPU sim anyway
    set_source_files_properties(${rrGPUSIMCoreSources} PROPERTIES COMPILE_FLAGS "-std=c++11")
    # TODO: eliminate these when GPU model generator is not dependent on LLVM
    set_source_files_properties(${rrGPUSIMCoreSources} PROPERTIES COMPILE_FLAGS ${LLVM_FLAGS})

    # add GPU sim sources to list
    set(rrSources ${rrSources} ${rrGPUSIMCoreSources})

    if (WITH_CUDA)
        cuda_add_library(CudaRKInt SHARED gpu/CudaRKInt.cu)
        set_source_files_properties(gpu/CudaRKInt.cu PROPERTIES COMPILE_FLAGS --compiler-bindir=gcc)
#         set_target_properties(CudaRKInt PROPERTIES COMPILE_FLAGS --compiler-bindir=gcc)
        install(TARGETS CudaRKInt DESTINATION lib)
    endif()

    set(GPUSIM_LIBS CudaRKInt)
endif(BUILD_GPUSIM)

# Sources for the C code-generating component
if(BUILD_LEGACY_C)
    set(rrSources
        ${rrSources}
        c/rrStringList
        c/mtrand
        c/rrStringBuilder
        c/rrStringListContainer
        c/rrSBMLSymbolDependencies
        c/rrSBMLSymbol
        c/rrHashTable
        c/rrRule
        c/rrModelSymbols
        c/rrSymbol
        c/rrSymbolList
        c/rrCompiledModelState
        c/rrNOMSupport
        c/rrPendingAssignment
        c/rrCompiledModelGenerator
        c/rrCModelGenerator
        c/rrCodeBuilder
        c/rrCCompiler
        c/rrScanner
        c/rrToken
        c/rrEvent
        c/rrModelSharedLibrary
        c/rrCModelDataUtil
        c/rrCompiledExecutableModel
        c/rrRandom
        )
endif(BUILD_LEGACY_C)

# Build shared roadrunner lib if enabled
if(RR_BUILD_SHARED_CORE)
    add_library(${target} SHARED ${rrSources})

    set_property(TARGET ${target}
        PROPERTY  COMPILE_DEFINITIONS
        EXPORT_RR
        LIBSBML_STATIC
        LIBLAX_STATIC
        STATIC_LIBSTRUCT
        STATIC_PUGI
        STATIC_NLEQ
        POCO_STATIC
        POCO_NO_AUTOMATIC_LIBS
        )

    #libraries to link with our shared lib
    if(UNIX)
        set(staticLibExt ".a")
    else()
        set(staticLibExt "")
    endif(UNIX)

    target_link_libraries (${target}
        sundials_cvode
        sundials_nvecserial
        blas
        lapack
        nleq-static
        rr-libstruct-static
        PocoNet
        PocoFoundation
        )


    # link with libxml, aperently borland can't link with a
    # conventional C library, need to be specially converted for borland

    if(UNIX)
    target_link_libraries(${target} xml2)
    elseif(${MSVC})
    target_link_libraries (${target} libxml2)
    elseif(${BORLAND})
        target_link_libraries (${target} libxml2_xe)
    endif()

    if(UNIX)
        target_link_libraries (${target}
            pthread
            sbml
            dl
            f2c${staticLibExt}
            ${ZLIB_LIBRARIES} # LLVM implcit dep
            )
    else()
        target_link_libraries (${target}
            libf2c
            libsbml
        ws2_32
        iphlpapi
            )
    endif(UNIX)

    #Libxml is not built. Use a VS converted import file for Borland
    if(${BORLAND})
        target_link_libraries (${target}
            libxml2_xe
            )
    endif()


    if(${INSTALL_CXX_API})
        install(TARGETS ${target}
            RUNTIME DESTINATION bin COMPONENT rr_core
            LIBRARY DESTINATION lib COMPONENT rr_core
            ARCHIVE DESTINATION lib COMPONENT rr_core
            )
    endif()


    if(BUILD_LLVM)
        target_link_libraries(${target}
            ${LLVM_LIBRARIES}
            )
    endif(BUILD_LLVM)

    # Link with GPU code if built
    if(BUILD_GPUSIM AND WITH_CUDA)
        target_link_libraries(${target}
            ${GPUSIM_LIBS}
            )
    endif()

endif(RR_BUILD_SHARED_CORE)

# Build the static roadruner library if enabled
if(RR_BUILD_STATIC)

    add_library(${target}-static STATIC ${rrSources})

    set_property(TARGET ${target}-static
        PROPERTY COMPILE_DEFINITIONS
        STATIC_RR
        LIBSBML_USE_CPP_NAMESPACE
        LIBSBML_STATIC
        LIBLAX_STATIC
        STATIC_LIBSTRUCT
        STATIC_PUGI
        STATIC_NLEQ
        POCO_STATIC
        POCO_NO_AUTOMATIC_LIBS
        )

    #libraries to link with static lib

    if(BUILD_LLVM)
        target_link_libraries(${target}-static
            ${LLVM_LIBRARIES}
            )
    endif(BUILD_LLVM)

    # Link with GPU code if built
    if(BUILD_GPUSIM AND WITH_CUDA)
        target_link_libraries(${target}
            ${GPUSIM_LIBS}
            )
    endif()


    #======== We need to split up this one to satisfy linking in both windows and posix.
    #=== windows need libsbml-static and libf2c and posix uses sbml-static and f2c
    target_link_libraries (${target}-static
        sundials_cvode
        sundials_nvecserial
        blas
        lapack
        nleq-static
        rr-libstruct-static
        PocoNet
        PocoFoundation
        )

    if(WIN32)
        target_link_libraries (${target}-static
            libsbml-static
            libf2c
        ws2_32
        iphlpapi
            )
    endif(WIN32)


    if(UNIX)
        target_link_libraries (${target}-static
            sbml-static
            f2c
            xml2
            dl
            pthread
            ${ZLIB_LIBRARIES} # LLVM implcit dep
            )
    endif(UNIX)

    #Libxml we can't build right now
    if(${BORLAND})
        target_link_libraries (${target}-static libxml2_xe)
    endif()

    if(${MSVC})
        target_link_libraries(${target}-static libxml2)
    endif()

    #if(INSTALL_STATIC_LIB)
    install(TARGETS ${target}-static
        LIBRARY DESTINATION lib COMPONENT rr_core
        ARCHIVE DESTINATION lib COMPONENT rr_core
        )
    #endif()
endif()

if(BUILD_TEST_TOOLS)
    if(BUILD_LLVM)
        add_subdirectory(llvm_testing)
    endif(BUILD_LLVM)
    add_subdirectory(testing)
endif()

# Install header files
if(${INSTALL_CXX_API})

    #=== HEADERS
    set(subfolders
        .
        #./mtrand
        )

    foreach(subfolder ${subfolders})
        FILE (GLOB headers ${subfolder}/*.h)
        install (   FILES ${headers}
            DESTINATION include/rr/${subfolder}
            COMPONENT rr_core)
    endforeach(subfolder)
endif() #Install CXX API

