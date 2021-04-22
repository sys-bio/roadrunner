
add_subdirectory(CVODEIntegratorTests)
add_subdirectory(SundialsSteadyStateTests)

if (BUILD_PYTHON AND BUILD_TESTS)
    message(STATUS "Building TestModelFactory")
    # temporary: See if this works before cleaning it up
    find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)
    find_package(SWIG 4.0.2 REQUIRED
            COMPONENTS python
            )
    include(UseSWIG)

    set_source_files_properties(TestModelFactory.i
            PROPERTIES
            CPLUSPLUS ON
            SWIG_FLAGS
                # turns on automatic annotation of python arg and return types
                # BUT does so in c++ types. Therefore I'm not sure how much value this adds.
                "-py3"
            )

    swig_add_library(TestModelFactory
            TYPE MODULE
            LANGUAGE python
            SOURCES TestModelFactory.i TestModelFactory
            "../../wrappers/Python/roadrunner/PyUtils"
            )

    #    set(CMAKE_SWIG_FLAGS "-py3 -castmode -keyword")
    set_target_properties(TestModelFactory PROPERTIES
            SWIG_USE_TARGET_INCLUDE_DIRECTORIES TRUE
            )

    message(STATUS "Python_INCLUDE_DIRS ${Python_INCLUDE_DIRS}")
    message(STATUS "Python_LIBRARIES ${Python_LIBRARIES}")

    target_include_directories(TestModelFactory PUBLIC
            ${Python_INCLUDE_DIRS}
            ${CMAKE_CURRENT_SOURCE_DIR}
            $<TARGET_PROPERTY:roadrunner-static,INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:roadrunner_python,INCLUDE_DIRECTORIES>
            )
    target_link_libraries(TestModelFactory PUBLIC
            roadrunner-static
            ${Python_LIBRARIES}
            )
    # copy Python files to build tree post build
    add_custom_command(
            TARGET TestModelFactory POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:TestModelFactory> "${RR_PYTHON_TESTING_DIR}"
    )

    add_custom_command(
            TARGET TestModelFactory POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/TestModelFactory.py" "${RR_PYTHON_TESTING_DIR}"
    )

    configure_file(testTestModelFactory.py "${RR_PYTHON_TESTING_DIR}" COPYONLY)

    # install python test files to install tree
    install(TARGETS TestModelFactory DESTINATION site-packages/roadrunner/testing)
    install(
            FILES
                "${CMAKE_CURRENT_BINARY_DIR}/TestModelFactory.py"
                "${CMAKE_CURRENT_SOURCE_DIR}/testTestModelFactory.py"
            DESTINATION
            site-packages/roadrunner/testing
    )

endif ()




























