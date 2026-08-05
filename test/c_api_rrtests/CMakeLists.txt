add_test_executable(test_c_api_rrtests test_targets
        ${SharedTestFiles} NomTests.cpp RRTestFileTests.cpp OtherRRTestFileTests.cpp
        )

# make test_targets list global to all tests
set(test_targets ${test_targets} PARENT_SCOPE)









