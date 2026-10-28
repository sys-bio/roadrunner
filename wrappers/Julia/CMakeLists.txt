###############################################################################
#
# Description  : CMake build script for the roadrunner Julia bindings
#
# The bindings are a pure-Julia package that calls the roadrunner C API
# (roadrunner_c_api), and includes the Antimony Julia bindings so that Antimony
# strings can be loaded. Nothing is compiled here: the package is assembled in
# the build tree for testing, and installed to bindings/julia/RoadRunner.
#
###############################################################################

if (NOT BUILD_RR_CAPI)
    message(FATAL_ERROR "BUILD_JULIA requires BUILD_RR_CAPI, because the Julia bindings call the C API.")
endif ()

###############################################################################
# The Antimony Julia package: a self-contained folder (Project.toml, src and
# lib) that is published as the antimony-julia-<platform> asset of an Antimony
# release. Set ANTIMONY_JULIA_PACKAGE_DIR to use a copy you already have, for
# example the bindings/julia/Antimony folder of an Antimony install built with
# -DWITH_JULIA=ON.
###############################################################################

set(ANTIMONY_JULIA_VERSION "latest" CACHE STRING "Antimony release to take the Julia package from: 'latest', or a version such as 3.3.0")
set(ANTIMONY_JULIA_PACKAGE_DIR "" CACHE PATH "Use this Antimony Julia package folder instead of downloading one")

if (ANTIMONY_JULIA_PACKAGE_DIR)
    set(antimony_package "${ANTIMONY_JULIA_PACKAGE_DIR}")
else ()
    if (WIN32)
        set(antimony_platform "windows-latest")
    elseif (APPLE)
        if (CMAKE_OSX_ARCHITECTURES)
            set(antimony_arch "${CMAKE_OSX_ARCHITECTURES}")
        else ()
            set(antimony_arch "${CMAKE_SYSTEM_PROCESSOR}")
        endif ()
        if (antimony_arch MATCHES "arm64|aarch64")
            set(antimony_platform "macos-14")
        else ()
            set(antimony_platform "macos-15-intel")
        endif ()
    else ()
        if (NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
            message(FATAL_ERROR "There is no prebuilt Antimony Julia package for ${CMAKE_SYSTEM_PROCESSOR}. Set ANTIMONY_JULIA_PACKAGE_DIR to one you built.")
        endif ()
        set(antimony_platform "manylinux_2_28")
    endif ()

    set(antimony_asset "antimony-julia-${antimony_platform}-release.zip")
    if (ANTIMONY_JULIA_VERSION STREQUAL "latest")
        set(antimony_url "https://github.com/sys-bio/antimony/releases/latest/download/${antimony_asset}")
    else ()
        set(antimony_url "https://github.com/sys-bio/antimony/releases/download/v${ANTIMONY_JULIA_VERSION}/${antimony_asset}")
    endif ()
    set(antimony_download_dir "${CMAKE_CURRENT_BINARY_DIR}/antimony-julia-download")
    set(antimony_extract_dir "${antimony_download_dir}/${ANTIMONY_JULIA_VERSION}-${antimony_platform}")

    # The download is kept in the build tree. To pick up a newer release, delete the
    # antimony-julia-download folder (or the whole build tree) and configure again.
    if (NOT EXISTS "${antimony_extract_dir}/Antimony/Project.toml")
        message(STATUS "Downloading ${antimony_url}")
        set(CMAKE_TLS_VERIFY TRUE)
        file(DOWNLOAD "${antimony_url}" "${antimony_download_dir}/${antimony_asset}"
                STATUS antimony_status
                SHOW_PROGRESS)
        list(GET antimony_status 0 antimony_status_code)
        if (NOT antimony_status_code EQUAL 0)
            list(GET antimony_status 1 antimony_status_message)
            message(FATAL_ERROR "Could not download ${antimony_url}: ${antimony_status_message}")
        endif ()
        file(REMOVE_RECURSE "${antimony_extract_dir}")
        file(MAKE_DIRECTORY "${antimony_extract_dir}")
        execute_process(
                COMMAND ${CMAKE_COMMAND} -E tar xf "${antimony_download_dir}/${antimony_asset}"
                WORKING_DIRECTORY "${antimony_extract_dir}"
                RESULT_VARIABLE antimony_extract_result)
        if (NOT antimony_extract_result EQUAL 0)
            message(FATAL_ERROR "Could not extract ${antimony_download_dir}/${antimony_asset}")
        endif ()
    endif ()
    set(antimony_package "${antimony_extract_dir}/Antimony")
endif ()

if (NOT EXISTS "${antimony_package}/Project.toml")
    message(FATAL_ERROR "${antimony_package} is not an Antimony Julia package folder (no Project.toml).")
endif ()
message(STATUS "  Using the Antimony Julia package         = ${antimony_package}")

###############################################################################
# Assemble the package in the build tree
###############################################################################

set(JULIA_PACKAGE_BUILD_DIR "${CMAKE_BINARY_DIR}/bindings/julia/RoadRunner")
set(JULIA_PACKAGE_INSTALL_DIR "bindings/julia/RoadRunner")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/Project.toml.in" "${JULIA_PACKAGE_BUILD_DIR}/Project.toml" @ONLY)

set(julia_source_files RoadRunner.jl rrc_types.jl rrc_utilities_binding.jl)
foreach (f ${julia_source_files})
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/src/${f}" "${JULIA_PACKAGE_BUILD_DIR}/src/${f}" COPYONLY)
endforeach ()

file(GLOB julia_test_files "${CMAKE_CURRENT_SOURCE_DIR}/test/*")
foreach (f ${julia_test_files})
    get_filename_component(julia_test_file_name "${f}" NAME)
    configure_file("${f}" "${JULIA_PACKAGE_BUILD_DIR}/test/${julia_test_file_name}" COPYONLY)
endforeach ()

file(COPY "${antimony_package}/" DESTINATION "${JULIA_PACKAGE_BUILD_DIR}/Antimony")

# The C API library goes inside the package, so the package is self-contained.
add_custom_target(roadrunner_julia_package ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory "${JULIA_PACKAGE_BUILD_DIR}/lib"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:roadrunner_c_api>" "${JULIA_PACKAGE_BUILD_DIR}/lib/"
        COMMENT "Copying the roadrunner C API library into the Julia package")
add_dependencies(roadrunner_julia_package roadrunner_c_api)

###############################################################################
# Install
###############################################################################

install(FILES "${JULIA_PACKAGE_BUILD_DIR}/Project.toml" DESTINATION ${JULIA_PACKAGE_INSTALL_DIR})
foreach (f ${julia_source_files})
    install(FILES "${JULIA_PACKAGE_BUILD_DIR}/src/${f}" DESTINATION ${JULIA_PACKAGE_INSTALL_DIR}/src)
endforeach ()
foreach (f ${julia_test_files})
    get_filename_component(julia_test_file_name "${f}" NAME)
    install(FILES "${JULIA_PACKAGE_BUILD_DIR}/test/${julia_test_file_name}" DESTINATION ${JULIA_PACKAGE_INSTALL_DIR}/test)
endforeach ()
install(PROGRAMS "$<TARGET_FILE:roadrunner_c_api>" DESTINATION ${JULIA_PACKAGE_INSTALL_DIR}/lib)
install(DIRECTORY "${antimony_package}/" DESTINATION ${JULIA_PACKAGE_INSTALL_DIR}/Antimony USE_SOURCE_PERMISSIONS)
install(FILES "${CMAKE_SOURCE_DIR}/NOTICE.txt" DESTINATION ${JULIA_PACKAGE_INSTALL_DIR})

###############################################################################
# Tests
###############################################################################

find_program(JULIA_EXECUTABLE julia)
if (BUILD_TESTS AND JULIA_EXECUTABLE)
    message(STATUS "  Julia binding tests will run with        = ${JULIA_EXECUTABLE}")
    add_test(NAME julia_roadrunner_bindings
            COMMAND ${CMAKE_COMMAND} -E env "ROADRUNNER_C_API_LIBRARY=$<TARGET_FILE:roadrunner_c_api>"
            "${JULIA_EXECUTABLE}" "--project=${JULIA_PACKAGE_BUILD_DIR}" "${JULIA_PACKAGE_BUILD_DIR}/test/runtests.jl"
            WORKING_DIRECTORY "${JULIA_PACKAGE_BUILD_DIR}/test")
    set_tests_properties(julia_roadrunner_bindings PROPERTIES LABELS "julia")
else ()
    message(STATUS "  The Julia binding tests will not be registered (needs BUILD_TESTS and julia on the path)")
endif ()
