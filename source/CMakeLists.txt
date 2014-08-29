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
    rrConfig.cpp
    rrSteadyStateSolver.cpp
    Configurable.cpp
    rrConstants.cpp
    rrException.cpp
    rrGetOptions.cpp
    rrLogger.cpp
    rrExecutableModel.cpp
    rrRoadRunner.cpp
    rrRoadRunnerOptions.cpp
    rrStringUtils.cpp
    rrUtils.cpp
    Integrator.cpp
    CVODEIntegrator.cpp
    GillespieIntegrator.cpp
    RK4Integrator.cpp
    rrNLEQInterface.cpp
    rrTestSuiteModelSimulation.cpp
    rrIniKey.cpp
    rrIniSection.cpp
    rrIniFile.cpp
    rrFileName.cpp
    rrRoadRunnerData.cpp
    rrSelectionRecord.cpp
    ModelGenerator.cpp
    rrVersionInfo.cpp
    rrSparse.cpp
    rrSBMLModelSimulation.cpp
    rrSBMLReader.cpp
    SBMLValidator.cpp
    Variant.cpp
    conservation/ConservationExtension.cpp
    conservation/ConservationDocumentPlugin.cpp
    conservation/ConservedMoietyPlugin.cpp
    conservation/ConservedMoietyConverter.cpp
    rrConfig.h
    rrSteadyStateSolver.h
    Configurable.h
    rrConstants.h
    rrException.h
    rrGetOptions.h
    rrLogger.h
    rrExecutableModel.h
    rrRoadRunner.h
    rrRoadRunnerOptions.h
    rrStringUtils.h
    rrUtils.h
    Integrator.h
    CVODEIntegrator.h
    GillespieIntegrator.h
    RK4Integrator.h
    rrNLEQInterface.h
    rrTestSuiteModelSimulation.h
    rrIniKey.h
    rrIniSection.h
    rrIniFile.h
    rrFileName.h
    rrRoadRunnerData.h
    rrSelectionRecord.h
    ModelGenerator.h
    rrVersionInfo.h
    rrSparse.h
    rrSBMLModelSimulation.h
    rrSBMLReader.h
    SBMLValidator.h
    Variant.h
    conservation/ConservationExtension.h
    conservation/ConservationDocumentPlugin.h
    conservation/ConservedMoietyPlugin.h
    conservation/ConservedMoietyConverter.h
    )

# Add the LLVM sources to compilation
if(BUILD_LLVM)
    set(rrSources
        ${rrSources}
        llvm/AssignmentRuleEvaluator.cpp
        llvm/ASTNodeCodeGen.cpp
        llvm/ASTNodeFactory.cpp
        llvm/ModelResources.cpp
        llvm/CodeGenBase.cpp
        llvm/LLVMCompiler.cpp
        llvm/EvalConversionFactorCodeGen.cpp
        llvm/EvalInitialConditionsCodeGen.cpp
        llvm/EvalRateRuleRatesCodeGen.cpp
        llvm/EvalReactionRatesCodeGen.cpp
        llvm/EventAssignCodeGen.cpp
        llvm/EventTriggerCodeGen.cpp
        llvm/EventQueue.cpp
        llvm/FunctionResolver.cpp
        llvm/LLVMExecutableModel.cpp
        llvm/GetRateRuleValuesCodeGen.cpp
        llvm/GetValuesCodeGen.cpp
        llvm/LoadSymbolResolverBase.cpp
        llvm/GetInitialValuesCodeGen.cpp
        llvm/GetEventValuesCodeGen.cpp
        llvm/KineticLawParameterResolver.cpp
        llvm/LLVMModelData.cpp
        llvm/ModelDataIRBuilder.cpp
        llvm/ModelDataSymbolResolver.cpp
        llvm/ModelInitialValueSymbolResolver.cpp
        llvm/LLVMModelDataSymbols.cpp
        llvm/LLVMModelGenerator.cpp
        llvm/ModelGeneratorContext.cpp
        llvm/LLVMModelSymbols.cpp
        llvm/SetRateRuleValuesCodeGen.cpp
        llvm/SetValuesCodeGen.cpp
        llvm/SetInitialValuesCodeGen.cpp
        llvm/SBMLSupportFunctions.cpp
        llvm/EvalVolatileStoichCodeGen.cpp
        llvm/SBMLInitialValueSymbolResolver.cpp
        llvm/AssignmentRuleEvaluator.h
        llvm/ASTNodeCodeGen.h
        llvm/ASTNodeFactory.h
        llvm/ModelResources.h
        llvm/CodeGenBase.h
        llvm/LLVMCompiler.h
        llvm/EvalConversionFactorCodeGen.h
        llvm/EvalInitialConditionsCodeGen.h
        llvm/EvalRateRuleRatesCodeGen.h
        llvm/EvalReactionRatesCodeGen.h
        llvm/EventAssignCodeGen.h
        llvm/EventTriggerCodeGen.h
        llvm/EventQueue.h
        llvm/FunctionResolver.h
        llvm/LLVMExecutableModel.h
        llvm/GetRateRuleValuesCodeGen.h
        llvm/GetValuesCodeGen.h
        llvm/LoadSymbolResolverBase.h
        llvm/GetInitialValuesCodeGen.h
        llvm/GetEventValuesCodeGen.h
        llvm/KineticLawParameterResolver.h
        llvm/LLVMModelData.h
        llvm/ModelDataIRBuilder.h
        llvm/ModelDataSymbolResolver.h
        llvm/ModelInitialValueSymbolResolver.h
        llvm/LLVMModelDataSymbols.h
        llvm/LLVMModelGenerator.h
        llvm/ModelGeneratorContext.h
        llvm/LLVMModelSymbols.h
        llvm/SetRateRuleValuesCodeGen.h
        llvm/SetValuesCodeGen.h
        llvm/SetInitialValuesCodeGen.h
        llvm/SBMLSupportFunctions.h
        llvm/EvalVolatileStoichCodeGen.h
        llvm/SBMLInitialValueSymbolResolver.h
        )
endif(BUILD_LLVM)

if(BUILD_LEGACY_C)
    set(rrSources
        ${rrSources}
        c/rrStringList.cpp
        c/mtrand.cpp
        c/rrStringBuilder.cpp
        c/rrStringListContainer.cpp
        c/rrSBMLSymbolDependencies.cpp
        c/rrSBMLSymbol.cpp
        c/rrHashTable.cpp
        c/rrRule.cpp
        c/rrModelSymbols.cpp
        c/rrSymbol.cpp
        c/rrSymbolList.cpp
        c/rrCompiledModelState.cpp
        c/rrNOMSupport.cpp
        c/rrPendingAssignment.cpp
        c/rrCompiledModelGenerator.cpp
        c/rrCModelGenerator.cpp
        c/rrCodeBuilder.cpp
        c/rrCCompiler.cpp
        c/rrScanner.cpp
        c/rrToken.cpp
        c/rrEvent.cpp
        c/rrModelSharedLibrary.cpp
        c/rrCModelDataUtil.cpp
        c/rrCompiledExecutableModel.cpp
        c/rrRandom.cpp
        c/rrStringList.h
        c/mtrand.h
        c/rrStringBuilder.h
        c/rrStringListContainer.h
        c/rrSBMLSymbolDependencies.h
        c/rrSBMLSymbol.h
        c/rrHashTable.h
        c/rrRule.h
        c/rrModelSymbols.h
        c/rrSymbol.h
        c/rrSymbolList.h
        c/rrCompiledModelState.h
        c/rrNOMSupport.h
        c/rrPendingAssignment.h
        c/rrCompiledModelGenerator.h
        c/rrCModelGenerator.h
        c/rrCodeBuilder.h
        c/rrCCompiler.h
        c/rrScanner.h
        c/rrToken.h
        c/rrEvent.h
        c/rrModelSharedLibrary.h
        c/rrCModelDataUtil.h
        c/rrCompiledExecutableModel.h
        c/rrRandom.h
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

