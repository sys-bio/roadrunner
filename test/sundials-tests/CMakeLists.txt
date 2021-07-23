add_subdirectory(CVODEIntegratorTests)
add_subdirectory(SundialsSteadyStateTests)
add_subdirectory(ForwardSensitivitySolverTests)

# make test_targets list global to all tests
set(test_targets ${test_targets} PARENT_SCOPE)


























