#####################################
#
include(GNUInstallDirs)

set(target roadrunner)

# Configures

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/rrConfigure.h.in ${CMAKE_CURRENT_BINARY_DIR}/rrConfigure.h @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/rrVersionInfo.h.in ${CMAKE_CURRENT_BINARY_DIR}/rrVersionInfo.h @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/GitInfo.h.in ${CMAKE_CURRENT_BINARY_DIR}/GitInfo.h @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/roadrunner.pc.in ${CMAKE_CURRENT_BINARY_DIR}/roadrunner.pc @ONLY)
set(
        GENERATED_HEADERS
        ${CMAKE_CURRENT_BINARY_DIR}/rrConfigure.h
        ${CMAKE_CURRENT_BINARY_DIR}/rrVersionInfo.h
        ${CMAKE_CURRENT_BINARY_DIR}/GitInfo.h
        ${CMAKE_CURRENT_BINARY_DIR}/roadrunner.pc
)
set_source_files_properties(${GENERATED_HEADERS} PROPERTIES GENERATED TRUE)

# Create the .pc file
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/roadrunner.pc DESTINATION lib/pkgconfig)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/roadrunner.pc DESTINATION share/pkgconfig)
install(FILES ${GENERATED_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rr)


############################################################
# Get a list of source file to build
#

set(rrSources)
set(rrHeaders)

# add core sources. These are always needed
set(
        rrCoreSources
        rrConfig.cpp
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
        Dictionary.cpp
        GillespieIntegrator.cpp
        RK4Integrator.cpp
        RK45Integrator.cpp
        NLEQ1Solver.cpp
        NLEQ2Solver.cpp
        NLEQSolver.cpp
        rrNLEQ1Interface.cpp
        rrNLEQ2Interface.cpp
        rrTestSuiteModelSimulation.cpp
        rrIniKey.cpp
        rrIniSection.cpp
        rrIniFile.cpp
        rrFileName.cpp
        rrRoadRunnerData.cpp
        rrSelectionRecord.cpp
        ExecutableModelFactory.cpp
        rrVersionInfo.cpp
        rrSparse.cpp
        rrSBMLModelSimulation.cpp
        rrSBMLReader.cpp
        SBMLValidator.cpp
        SteadyStateSolver.cpp
        Solver.cpp
        #        Variant.cpp
        Setting.cpp
        ForwardSensitivitySolver.cpp

        ForwardSensitivitySolver.cpp

        RegistrationFactory.cpp
        SensitivitySolverFactory.cpp
        SteadyStateSolverFactory.cpp
        IntegratorFactory.cpp

        KinsolSteadyStateSolver.cpp
        FixedPointIteration.cpp
        NewtonIteration.cpp
        BasicNewtonIteration.cpp
        LinesearchNewtonIteration.cpp
        KinsolErrHandler.cpp
        SVD.cpp
        SteadyStateSolverDecorator.cpp
        PresimulationDecorator.cpp
        PresimulationProgramDecorator.cpp
        ApproxSteadyStateDecorator.cpp

        conservation/ConservationExtension.cpp
        conservation/ConservationDocumentPlugin.cpp
        conservation/ConservedMoietyPlugin.cpp
        conservation/ConservedMoietyConverter.cpp
        #todo pull test files out of main library sources.
        rrRoadRunnerMap.cpp
        thread-pool/thread_pool.hpp
)

set(
        rrCoreHeaders
        rrExporter.h
        rrOSSpecifics.h
        rrConfig.h
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
        Dictionary.h
        GillespieIntegrator.h
        RK4Integrator.h
        RK45Integrator.h
        NLEQ1Solver.h
        NLEQ2Solver.h
        NLEQSolver.h
        rrNLEQ1Interface.h
        rrNLEQ2Interface.h
        rrTestSuiteModelSimulation.h
        rrIniKey.h
        rrIniSection.h
        rrIniFile.h
        rrFileName.h
        rrRoadRunnerData.h
        rrSelectionRecord.h
        ExecutableModelFactory.h
        EulerIntegrator.h
        ${CMAKE_CURRENT_BINARY_DIR}/rrVersionInfo.h
        rrSparse.h
        rrSBMLModelSimulation.h
        rrSBMLReader.h
        SBMLValidator.h
        SteadyStateSolver.h
        Solver.h
        #        Variant.h
        Setting.h
        setting_t.h
        RegistrationFactory.h
        ForwardSensitivitySolver.h
        ForwardSensitivitySolver.h
        SensitivitySolver.h
        Registrable.h
        SensitivitySolverFactory.h
        SteadyStateSolverFactory.h
        IntegratorFactory.h

        Matrix.h
        Matrix3D.h

        KinsolErrHandler.h
        KinsolSteadyStateSolver.h
        FixedPointIteration.h
        NewtonIteration.h
        BasicNewtonIteration.h
        LinesearchNewtonIteration.h
        SVD.h
        SteadyStateSolverDecorator.h
        PresimulationDecorator.h
        PresimulationProgramDecorator.h
        ApproxSteadyStateDecorator.h

        _Variant.h

        conservation/ConservationExtension.h
        conservation/ConservationDocumentPlugin.h
        conservation/ConservedMoietyPlugin.h
        conservation/ConservedMoietyConverter.h

        rrRoadRunnerMap.h
)
install(FILES ${rrCoreHeaders} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rr)


set(
        rrCoreConservationHeaders
        conservation/ConservationExtension.h
        conservation/ConservationDocumentPlugin.h
        conservation/ConservedMoietyPlugin.h
        conservation/ConservedMoietyConverter.h

)
install(FILES ${rrCoreConservationHeaders} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rr/conservation)
set(
        rrCoreTr1ProxyHeaders
        tr1proxy/cxx11_ns.h
        tr1proxy/rr_memory.h
        tr1proxy/rr_random.h
        tr1proxy/rr_unordered_map.h

)
install(FILES ${rrCoreTr1ProxyHeaders} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rr/tr1proxy)

set(rrSources ${rrSources} ${rrCoreSources})
set(rrHeaders ${rrHeaders} ${rrCoreHeaders} ${rrCoreConservationHeaders} ${rrCoreTr1ProxyHeaders})


#
# Add the LLVM sources to compilation
if (BUILD_LLVM)
    set(rrLLVMSources
            llvm/ASTNodeCodeGen.cpp
            llvm/ASTNodeFactory.cpp
            llvm/AssignmentRuleEvaluator.cpp
            llvm/CodeGenBase.cpp
            llvm/EvalConversionFactorCodeGen.cpp
            llvm/EvalInitialConditionsCodeGen.cpp
            llvm/EvalRateRuleRatesCodeGen.cpp
            llvm/EvalReactionRatesCodeGen.cpp
            llvm/EvalVolatileStoichCodeGen.cpp
            llvm/EventAssignCodeGen.cpp
            llvm/EventQueue.cpp
            llvm/EventTriggerCodeGen.cpp
            llvm/FunctionResolver.cpp
            llvm/GetEventValuesCodeGen.cpp
            llvm/GetInitialValuesCodeGen.cpp
            llvm/GetPiecewiseTriggerCodeGen.cpp
            llvm/GetValuesCodeGen.cpp
            llvm/Jit.cpp
            llvm/JitFactory.cpp
            llvm/KineticLawParameterResolver.cpp
            llvm/LLJit.cpp
            llvm/LLVMCompiler.cpp
            llvm/LLVMExecutableModel.cpp
            llvm/LLVMModelData.cpp
            llvm/LLVMModelDataSymbols.cpp
            llvm/LLVMModelGenerator.cpp
            llvm/LLVMModelSymbols.cpp
            llvm/LoadSymbolResolverBase.cpp
            llvm/MCJit.cpp
            llvm/ModelDataIRBuilder.cpp
            llvm/ModelDataSymbolResolver.cpp
            llvm/ModelGeneratorContext.cpp
            llvm/ModelInitialValueSymbolResolver.cpp
            llvm/ModelResources.cpp
            llvm/Random.cpp
            llvm/SBMLInitialValueSymbolResolver.cpp
            llvm/SBMLModelObjectCache.cpp
            llvm/SBMLSupportFunctions.cpp
            llvm/SetInitialValuesCodeGen.cpp
            llvm/SetValuesCodeGen.cpp
            )

    set(rrLLVMHeaders
            llvm/ASTNodeCodeGen.h
            llvm/ASTNodeFactory.h
            llvm/AssignmentRuleEvaluator.h
            llvm/CodeGen.h
            llvm/CodeGenBase.h
            llvm/EvalConversionFactorCodeGen.h
            llvm/EvalInitialConditionsCodeGen.h
            llvm/EvalRateRuleRatesCodeGen.h
            llvm/EvalReactionRatesCodeGen.h
            llvm/EvalVolatileStoichCodeGen.h
            llvm/EventAssignCodeGen.h
            llvm/EventCodeGenBase.h
            llvm/EventQueue.h
            llvm/EventTriggerCodeGen.h
            llvm/FunctionResolver.h
            llvm/GetEventValuesCodeGen.h
            llvm/GetInitialValueCodeGenBase.h
            llvm/GetInitialValuesCodeGen.h
            llvm/GetPiecewiseTriggerCodeGen.h
            llvm/GetValueCodeGenBase.h
            llvm/GetValuesCodeGen.h
            llvm/Jit.h
            llvm/JitFactory.h
            llvm/KineticLawParameterResolver.h
            llvm/LLJit.h
            llvm/LLVMCompiler.h
            llvm/LLVMExecutableModel.h
            llvm/LLVMException.h
            llvm/LLVMModelData.h
            llvm/LLVMModelDataSymbols.h
            llvm/LLVMModelGenerator.h
            llvm/LLVMModelSymbols.h
            llvm/LoadSymbolResolverBase.h
            llvm/MCJit.h
            llvm/ModelDataIRBuilder.h
            llvm/ModelDataSymbolResolver.h
            llvm/ModelGeneratorContext.h
            llvm/ModelInitialValueSymbolResolver.h
            llvm/ModelResources.h
            llvm/Random.h
            llvm/SBMLInitialValueSymbolResolver.h
            llvm/SBMLModelObjectCache.h
            llvm/SBMLSupportFunctions.h
            llvm/SetInitialValueCodeGenBase.h
            llvm/SetInitialValuesCodeGen.h
            llvm/SetValueCodeGenBase.h
            llvm/SetValuesCodeGen.h
            )
    install(FILES ${rrLLVMHeaders} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rr/llvm)


    set(rrSources ${rrSources} ${rrLLVMSources})
    set(rrHeaders ${rrHeaders} ${rrLLVMHeaders})

    message(STATUS "LLVM_FLAGS ${LLVM_FLAGS}")

    if (NOT LLVM_FLAGS)
        message(FATAL_ERROR "No LLVM_FLAGS variable found")
    endif ()
    # Don't use add_definitions for LLVM flags per http://stackoverflow.com/questions/13638408/cmake-override-compile-flags-for-single-files
    # Line below kept commented out on purpose - we don't want this.
    #     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LLVM_FLAGS}") # Set per-source
    # set up appropriate flags for using LLVM
    # TODO remove the code below and check for errors. Are these actually needed?
    # Done: on g++, commenting this line out does not break the build. In fact, it fixes
    #   this build since llvm adds extra flags which override roadrunner flags (-std=c++ for example)
    #    set_source_files_properties(${rrLLVMSources} PROPERTIES COMPILE_FLAGS ${LLVM_FLAGS})
    # add LLVM-based model sources to list
    set(rrSources ${rrSources} ${rrLLVMSources} ${rrLLVMHeaders})
endif (BUILD_LLVM)

# Add C legacy code to sources if needed
if (${BUILD_LEGACY_C})
    set(
            rrCSources
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
    )
    set(
            rrCHeaders
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

    install(FILES ${rrCHeaders} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rr/c)

    set(rrSources ${rrSources} ${rrCSources})
    set(rrHeaders ${rrHeaders} ${rrCHeaders})

    # install the rr_support if building legacy c
    add_subdirectory(rr_support)
endif (${BUILD_LEGACY_C})

##############################################################
#   Build the static library
#

add_library(${target}-static STATIC ${rrSources} ${rrHeaders})


# we add compiler flag to announce the static library.
# make it public so consumers of the library do not need to manually
# specify STATIC_RR
target_compile_definitions(${target}-static PUBLIC STATIC_RR)

set(INCLUDE_DIRS "$<BUILD_INTERFACE:${LLVM_INCLUDE_DIRS}>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/llvm>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/thread-pool>"
        "$<BUILD_INTERFACE:${LLVM_INCLUDE_DIRS}>"
        "$<BUILD_INTERFACE:${DEPENDENCY_INCLUDE_DIR}>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/parallel-hashmap-1.33/parallel_hashmap>"
        # though roadrunner does not link libxml2 directly, it does
        # so in the plugins. This next item enables targets that
        # link to roadrunner-static to use
        # #include "libxml/tree.h", for instance in the plugin code.
        "$<BUILD_INTERFACE:${DEPENDENCY_INCLUDE_DIR}/libexpat>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>")
#if (NOT WIN32)
#    set(INCLUDE_DIRS ${INCLUDE_DIRS} "$<BUILD_INTERFACE:${CURSES_INCLUDE_DIRS}>")
#endif ()

target_include_directories(
        ${target}-static PUBLIC ${INCLUDE_DIRS}
)

# Extra platform specific libraries
# These were not needed to actually build roadrunner binaries
# but linking the tests fails without them.
set(PLATFORM_SPECIFIC_LIBS)
if (UNIX)
    #    icui18n icuuc (can be used instead of iconv)
    set(PLATFORM_SPECIFIC_LIBS ${PLATFORM_SPECIFIC_LIBS} pthread dl m)
elseif (WIN32)
    set(PLATFORM_SPECIFIC_LIBS ${PLATFORM_SPECIFIC_LIBS} Ws2_32)

elseif (APPLE AND NOT UNIX)
    set(PLATFORM_SPECIFIC_LIBS ${PLATFORM_SPECIFIC_LIBS})

endif ()

message(STATUS " LLVM_LIBRARIES ${LLVM_LIBRARIES}}")
    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")


# link the libraries -- remember that order here
#   is important on linux builds so don't change
target_link_libraries(
        ${target}-static
        PUBLIC
        ${PLATFORM_SPECIFIC_LIBS}

        ${LLVM_LIBRARIES}
        rr-libstruct::rr-libstruct-static
        ${SBML_TARGET_NAME}
        #These libraries libraries seem to be already linked in a different way.
        #libz2
        #LibXml2::LibXml2
        zlib::zlibstatic
        nleq2::nleq2-static
        nleq1::nleq1-static
        clapack::lapack clapack::blas clapack::f2c

        SUNDIALS::generic_static
        SUNDIALS::nvecserial_static
        SUNDIALS::nvecmanyvector_static
        SUNDIALS::sunmatrixband_static
        SUNDIALS::sunmatrixdense_static
        SUNDIALS::sunmatrixsparse_static
        SUNDIALS::sunlinsolband_static
        SUNDIALS::sunlinsoldense_static
        SUNDIALS::sunlinsolpcg_static
        SUNDIALS::sunlinsolspbcgs_static
        SUNDIALS::sunlinsolspfgmr_static
        SUNDIALS::sunlinsolspgmr_static
        SUNDIALS::sunlinsolsptfqmr_static
        SUNDIALS::sunnonlinsolnewton_static
        SUNDIALS::sunnonlinsolfixedpoint_static
        # cvodes is a superset of cvode.
        # linking the both can cause a duplicate symbol problem
        # SUNDIALS::cvode_static
        SUNDIALS::cvodes_static
        SUNDIALS::kinsol_static


        Poco::Foundation
        Poco::Net
        Poco::XML
        #iconv::libcharset iconv::libiconv-static
        expat::expat
)

#get_target_property(v ${target}-static LINK_FLAGS)
#MESSAGE(STATUS "vty ${v}")

add_library(${target} SHARED ${rrSources} ${rrHeaders})
target_compile_definitions(${target} PRIVATE EXPORT_RR)
target_link_libraries(${target} PUBLIC $<TARGET_PROPERTY:${target}-static,LINK_LIBRARIES>)
target_include_directories(${target} PUBLIC $<TARGET_PROPERTY:${target}-static,INCLUDE_DIRECTORIES>)


install(
        TARGETS ${target}
        EXPORT ${target}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT ${target}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT ${target}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT ${target}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rr COMPONENT ${target}
)

# install export info so roadrunner can be used with find_package in CONFIG mode
install(
        EXPORT ${target}
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake
        NAMESPACE ${target}::
        FILE ${target}-config.cmake
        COMPONENT ${target}
        EXPORT_LINK_INTERFACE_LIBRARIES
)

#################################################################3
#   install code for COMPONENT RR
#

install(
        TARGETS ${target}-static
        EXPORT ${target}-static
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT ${target}-static
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT ${target}-static
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT ${target}-static
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rr COMPONENT ${target}-static
)

# install export info so roadrunner can be used with find_package in CONFIG mode
install(
        EXPORT ${target}-static
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake
        NAMESPACE ${target}-static::
        FILE ${target}-static-config.cmake
        COMPONENT ${target}-static
        EXPORT_LINK_INTERFACE_LIBRARIES
)


#Libxml we can't build right now
if (${BORLAND})
    target_link_libraries(${target}-static libxml2_xe)
endif ()

