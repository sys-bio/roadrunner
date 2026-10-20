add_test_executable(test_c_api_core test_targets
        rrSBMLTestSuiteSimulation_CAPI.cpp
        ${SharedTestFiles}
        exceptions.cpp
        CAPICoreTest.cpp
        libstruct.cpp
        model_editing.cpp
        SimulateOptionsCAPITests.cpp
        )



# make test_targets list global to all tests
set(test_targets ${test_targets} PARENT_SCOPE)
