# A simple configuration summary for roadrunner.
macro(ConfigurationSummary)
    message(STATUS "
########################################################
# Build Summary for libroadrunner ${ROADRUNNER_VERSION}
########################################################

For up-to-date news and releases visit libroadrunner.org

Configured on host $ENV{COMPUTERNAME} ${HOSTNAME}

-------------------------------------------------------------------------------
Options
-------------------------------------------------------------------------------

    Host OS                             ${CMAKE_SYSTEM_NAME}
    Host architecture                   ${CMAKE_SYSTEM_PROCESSOR}
    With LLVM?                          ${BUILD_LLVM}
    With legacy C? (DEP)                ${BUILD_LEGACY_C}
    llvm-6.x install prefix             ${LLVM_INSTALL_PREFIX}
    Build test tools?                   ${BUILD_TEST_TOOLS}
    Build tests?                        ${BUILD_TESTS}
    Build Python bindings               ${BUILD_PYTHON}
    Site-packages directory             ${PYTHON_PACKAGE_SITE_DIR}
    Python roadrunner install prefix    ${PYTHON_PACKAGE_INSTALL_PREFIX}
    Build Python ctypes api             ${BUILD_PYTHON_CTYPES_API}
    Build conda recipe                  ${WITH_CONDA_BUILDER}
    Build roadrunner C API              ${BUILD_RR_CAPI}
    Build apps                          ${BUILD_APPS}
    Build examples                      ${BUILD_EXAMPLES}
    Install sbml models?                ${INSTALL_SBML_MODELS}
    Build Java bindings?                ${BUILD_JAVA}
    Package roadrunner?                 ${BUILD_PACKAGING}
    Build with address sanitizer?       ${WITH_ADDRESS_SANITIZER}

-------------------------------------------------------------------------------
rrplugins
-------------------------------------------------------------------------------

    Features:
        Build rrplugins package?           ${BUILD_RR_PLUGINS}
        Build rrplugins as shared library? ${RR_PLUGINS_BUILD_SHARED_LIB}
        Build rrplugins as static library? ${RR_PLUGINS_BUILD_STATIC_LIB}
        Where to install rrplugins?        ${RR_PLUGINS_INSTALL_PREFIX}

    Languages

        Build the C API?                   ${RR_PLUGINS_BUILD_C_API}
        Build the Python API?              ${RR_PLUGINS_BUILD_PYTHON_API}

    Plugins available:
      Test Model                           ${RR_PLUGINS_BUILD_TEST_MODEL_PLUGIN}
      Add Noise                            ${RR_PLUGINS_BUILD_ADD_NOISE_PLUGIN}
      Chi-squared                          ${RR_PLUGINS_BUILD_CHISQUARE_PLUGIN}
      Bifurcation Auto2k                   ${RR_PLUGINS_BUILD_AUTO2000_PLUGIN}
      Demo Plugin                          ${RR_PLUGINS_BUILD_HELLO_PLUGIN}


-------------------------------------------------------------------------------
CMake variables
-------------------------------------------------------------------------------
    CMAKE_C_COMPILER                   ${CMAKE_C_COMPILER}
    CMAKE_CXX_COMPILER                 ${CMAKE_CXX_COMPILER}
    BUILD_DEFINITIONS                  ${BUILD_DEFINITIONS}
    CMAKE_SYSTEM_VERSION               ${CMAKE_SYSTEM_VERSION}
    ROADRUNNER_VERSION                 ${ROADRUNNER_VERSION}
    CMAKE_CXX_COMPILER_ID              ${CMAKE_CXX_COMPILER_ID}
    CMAKE_CXX_COMPILER_VERSION         ${CMAKE_CXX_COMPILER_VERSION}
    CMAKE_CXX_STANDARD                 ${CMAKE_CXX_STANDARD}
    CMAKE_POSITION_INDEPENDENT_CODE    ${CMAKE_POSITION_INDEPENDENT_CODE}
    CMAKE_INSTALL_RPATH                ${CMAKE_INSTALL_RPATH}
    CMAKE_CXX_FLAGS                    ${CMAKE_CXX_FLAGS}
    CMAKE_C_FLAGS                      ${CMAKE_C_FLAGS}
    CMAKE_CXX_FLAGS_RELEASE            ${CMAKE_CXX_FLAGS_RELEASE}
    CMAKE_C_FLAGS_RELEASE              ${CMAKE_C_FLAGS_RELEASE}
    CMAKE_CXX_FLAGS_DEBUG              ${CMAKE_CXX_FLAGS_DEBUG}
    CMAKE_C_FLAGS_DEBUG                ${CMAKE_C_FLAGS_DEBUG}
    CMAKE_INSTALL_PREFIX               ${CMAKE_INSTALL_PREFIX}
    CMAKE_VERBOSE_MAKEFILE             ${CMAKE_VERBOSE_MAKEFILE}
    CMAKE_BUILD_TYPE                   ${CMAKE_BUILD_TYPE}
    CMAKE_RUNTIME_OUTPUT_DIRECTORY     ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    CMAKE_LIBRARY_OUTPUT_DIRECTORY     ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    CMAKE_ARCHIVE_OUTPUT_DIRECTORY     ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}


End Summary
########################################################################################
")
endmacro()
