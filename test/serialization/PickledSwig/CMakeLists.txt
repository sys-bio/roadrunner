
include(UseSWIG)


if (BUILD_PYTHON AND BUILD_TESTS)

    #####################################################################
    #   Minimal swig bindings to a minimal C++ object.
    #   This is proof of principle of how to serialization a PySWIGObject
    #   and does not involve roadrunner itself.
    #

    #todo write a cmake function for building swig targets
    set_source_files_properties(picklable_swig.i CPPObjectForPickling.cpp CPPObjectForPickling.h
            PROPERTIES
            CPLUSPLUS ON
            #            SWIG_FLAGS
            )
    add_rr_test_swig_library(
            picklable_swig
            picklable_swig.i
            CPPObjectForPickling.cpp
            CPPObjectForPickling.h
    )

endif ()



# make test_targets list global to all tests
set(test_targets ${test_targets} PARENT_SCOPE)




