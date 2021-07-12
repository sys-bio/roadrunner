

set(SharedTestFiles "${SharedTestFiles}" "${CMAKE_CURRENT_SOURCE_DIR}/SundialsSteadyStateSolverUnitTest")
## FixedPointIteration not finished due to difficulties getting it to work.
#add_test_executable(
#        FixedPointIterationTests test_targets
#        FixedPointIterationUnitTests.cpp SundialsSteadyStateSolverUnitTest ${SharedTestFiles}
#)

add_test_executable(
        test_steady_state_unit_approx test_targets
        ApproxSteadyStateDecoratorTests.cpp ${SharedTestFiles}
)

add_test_executable(
        test_steady_state_unit_automatic_moiety_conservation test_targets
        AutomaticMoietyConservationTests.cpp
        SundialsSteadyStateSolverUnitTest
        ${SharedTestFiles}
)


add_test_executable(
        test_steady_state_unit_basic_newton test_targets
        BasicNewtonIterationUnitTests.cpp
        SundialsSteadyStateSolverUnitTest
        ${SharedTestFiles}
)

add_test_executable(
        test_steady_state_unit_presimulation test_targets
        PresimulationDecoratorTests.cpp ${SharedTestFiles}
)

add_test_executable(
        test_unit_svd test_targets SVDTests.cpp ${SharedTestFiles}
)

add_test_executable(
        test_steady_state_integration test_targets
        SteadyStateIntegrationTests.h
        BasicNewtonIntegrationTests.cpp
        DecoratorIntegrationTests.cpp
        LinesearchNewtonIntegrationTests.cpp
        NLEQ1IntegrationTests.cpp
        NLEQ2IntegrationTests.cpp
        ${SharedTestFiles}
)


# copy Python tests to binary tree site packages
configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/SteadyStateSolverTests.py"
        "${CMAKE_BINARY_DIR}/lib/site-packages/roadrunner/testing/SteadyStateSolverTests.py"
        COPYONLY
)
configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/AutomaticMoietyConservationAnalysisTests.py"
        "${CMAKE_BINARY_DIR}/lib/site-packages/roadrunner/testing/"
        COPYONLY
)

# install python tests to install tree
install(
        FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/SteadyStateSolverTests.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/AutomaticMoietyConservationAnalysisTests.py"
        DESTINATION
            site-packages/roadrunner/testing
        )






#add_test_executable(
#        SteadyStateIntegrationTests test_targets
#        SteadyStateIntegrationTests.h
##        BasicNewtonIntegrationTests.cpp
##        DecoratorIntegrationTests.cpp
##        LinesearchNewtonIntegrationTests.cpp
##        NLEQ1IntegrationTests.cpp
##        NLEQ2IntegrationTests.cpp
#        ${SharedTestFiles}
#)




# make test_targets list global to all tests
set(test_targets ${test_targets} PARENT_SCOPE)














