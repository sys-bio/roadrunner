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

set(rrSources
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

# Add the LLVM sources to compilation
if(BUILD_LLVM)
    set(rrSources
        ${rrSources}
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
endif(BUILD_LLVM)

# Add the GPUSim sources to compilation
if(BUILD_GPUSIM)
    set(rrSources
        ${rrSources}
        gpu/GPUSimCompiler
        gpu/GPUSimExecutableModel
        gpu/GPUSimModelGenerator
        )
endif(BUILD_GPUSIM)

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


# set up linking libs
if(RR_BUILD_SHARED_CORE)
    #Link with static libs
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

endif(RR_BUILD_SHARED_CORE)

# static allways gets build

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
#======== We need to split up this one to satisfy linking in both windows and posix.
#=== windows need libsbml-static and libf2c and posix uses sbml-static and f2c
if(BUILD_LLVM)
    target_link_libraries(${target}-static
        ${LLVM_LIBRARIES}
        )
endif(BUILD_LLVM)


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

if(BUILD_TEST_TOOLS)
    if(BUILD_LLVM)
        add_subdirectory(llvm_testing)
    endif(BUILD_LLVM)
    add_subdirectory(testing)
endif()

#--- INSTALL STUFF
if(${INSTALL_CXX_API})

    #if(INSTALL_STATIC_LIB)
    install(TARGETS ${target}-static
        LIBRARY DESTINATION lib COMPONENT rr_core
        ARCHIVE DESTINATION lib COMPONENT rr_core
        )
    #endif()

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

