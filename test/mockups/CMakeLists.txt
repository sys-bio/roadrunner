##########################################################
#   Library for containing mock objects only
#

# Note: mocking requires a code base which adheres to the principles of coding to
# interfaces. Only some of roadrunner does this (Executable Model, Integrator and Solver)
# and so mocking is of limtied value here

add_library(
        rr-mockups OBJECT
        MockExecutableModel.h
        MockSolver.h
        MockSBase.h
)
set(MOCKUPS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(
        rr-mockups PUBLIC
        ${MOCKUPS_DIRECTORY}
)

target_link_libraries(rr-mockups PUBLIC
        roadrunner-static gmock gmock_main gtest gtest_main
        )
































