# Defines a function for querying defined cmake variables.
# Takes 1 argument which is a regular expression for matching
# subset of cmake variables.
function(QueryCMakeVariables)
    message(STATUS "Querying CMake variables. Matching for pattern \"${ARGV0}\"")
    get_cmake_property(_variableNames VARIABLES)
    list (SORT _variableNames)
    foreach (_variableName ${_variableNames})
        if (ARGV0)
            unset(MATCHED)
            string(REGEX MATCH ${ARGV0} MATCHED ${_variableName})
            if (NOT MATCHED)
                continue()
            endif()
        endif()
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
endfunction()