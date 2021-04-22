include(UseSWIG)

if (BUILD_PYTHON AND BUILD_TESTS)

    #####################################################################
    #   Minimal swig bindings to a minimal C++ object.
    #   This is proof of principle of how to pickle a PySWIGObject
    #   and does not involve roadrunner itself.
    #

    #todo write a cmake function for building swig targets
    set_source_files_properties(picklable_swig.i CPPObjectForPickling.cpp CPPObjectForPickling.h
            PROPERTIES
            CPLUSPLUS ON
            #            SWIG_FLAGS
            )
    swig_add_library(picklable_swig
            TYPE
            MODULE
            LANGUAGE
            python
            SOURCES
            picklable_swig.i
            CPPObjectForPickling
            )

    set_target_properties(picklable_swig PROPERTIES
            SWIG_USE_TARGET_INCLUDE_DIRECTORIES TRUE
            )

    message(STATUS "Python_INCLUDE_DIRS ${Python_INCLUDE_DIRS}")
    message(STATUS "Python_LIBRARIES ${Python_LIBRARIES}")

    target_include_directories(picklable_swig PUBLIC
            ${Python_INCLUDE_DIRS}
            ${CMAKE_CURRENT_SOURCE_DIR}
            #            $<TARGET_PROPERTY:roadrunner-static,INCLUDE_DIRECTORIES>
            #            $<TARGET_PROPERTY:roadrunner_python,INCLUDE_DIRECTORIES>
            )

    target_link_libraries(picklable_swig PUBLIC
            #            roadrunner-static
            ${Python_LIBRARIES}
            )


    # copy Python files to build tree post build
    # *.pyd file
    add_custom_command(
            TARGET picklable_swig POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:picklable_swig> "${RR_PYTHON_TESTING_DIR}"
    )
    message(STATUS "\"$<TARGET_FILE:picklable_swig>\" copied to \"${RR_PYTHON_TESTING_DIR}\"")


    # *.py file
    add_custom_command(
            TARGET picklable_swig POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/picklable_swig.py" "${RR_PYTHON_TESTING_DIR}"
    )
    message(STATUS "\"${CMAKE_CURRENT_BINARY_DIR}/picklable_swig.py\" copied to \"${RR_PYTHON_TESTING_DIR}\"")


endif ()








