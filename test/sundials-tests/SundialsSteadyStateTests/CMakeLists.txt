

set(SharedTestFiles "${SharedTestFiles}" "${CMAKE_CURRENT_SOURCE_DIR}/SundialsSteadyStateSolverUnitTest.cpp")
## FixedPointIteration not finished due to difficulties getting it to work.
#add_test_executable(
#        FixedPointIterationTests test_targets
#        FixedPointIterationUnitTests.cpp SundialsSteadyStateSolverUnitTest ${SharedTestFiles}
#)

add_test_executable(
        test_sundials_steady_state_approx test_targets
        ApproxSteadyStateDecoratorTests.cpp ${SharedTestFiles}
)

add_test_executable(
        test_sundials_automatic_moiety_conservation test_targets
        AutomaticMoietyConservationTests.cpp
        ${SharedTestFiles}
)


add_test_executable(
        test_sundials_steady_state_unit_basic_newton test_targets
        SundialsNewtonIterationUnitTests.cpp
        ${SharedTestFiles}
)

add_test_executable(
        test_sundials_steady_state_unit_presimulation test_targets
        PresimulationDecoratorTests.cpp ${SharedTestFiles}
)

add_test_executable(
        test_sundials_unit_svd test_targets SVDTests.cpp ${SharedTestFiles}
)

add_test_executable(
        test_sundials_steady_state_integration test_targets
        SteadyStateIntegrationTests.h
        SundialsNewtonIntegrationTests.cpp
        DecoratorIntegrationTests.cpp
        SundialsLinesearchIntegrationTests.cpp
        NLEQ1IntegrationTests.cpp
        NLEQ2IntegrationTests.cpp
        ${SharedTestFiles}
)


# make test_targets list global to all tests
set(test_targets ${test_targets} PARENT_SCOPE)
