# A simple configuration summary for roadrunner.
macro(ConfigurationSummaryPython)
    set(PythonSummary "   BUILD_PYTHON                               ${BUILD_PYTHON}
    Python_ROOT_DIR                            ${Python_ROOT_DIR}
    Python_FOUND                               ${Python_FOUND}
    Python_Interpreter_FOUND                   ${Python_Interpreter_FOUND}
    Python_EXECUTABLE                          ${Python_EXECUTABLE}
    Python_INTERPRETER_ID                      ${Python_INTERPRETER_ID}
    IronPython                                 ${IronPython}
    Python_STDLIB                              ${Python_STDLIB}
    Python_STDARCH                             ${Python_STDARCH}
    Python_SITELIB                             ${Python_SITELIB}
    Python_SITEARCH                            ${Python_SITEARCH}
    Python_SOABI                               ${Python_SOABI}
    Python_Compiler_FOUND                      ${Python_Compiler_FOUND}
    Python_COMPILER                            ${Python_COMPILER}
    Python_COMPILER_ID                         ${Python_COMPILER_ID}
    Python_DOTNET_LAUNCHER                     ${Python_DOTNET_LAUNCHER}
    Python_Development_FOUND                   ${Python_Development_FOUND}
    Python_Development                         ${Python_Development}                     .Module_FOUND
    Python_Development                         ${Python_Development}                     .Embed_FOUND
    Python_INCLUDE_DIRS                        ${Python_INCLUDE_DIRS}
    Python_LINK_OPTIONS                        ${Python_LINK_OPTIONS}
    Python_LIBRARY                             ${Python_LIBRARY}
    Python_LIBRARIES                           ${Python_LIBRARIES}
    Python_LIBRARY_DIRS                        ${Python_LIBRARY_DIRS}
    Python_RUNTIME_LIBRARY_DIRS                ${Python_RUNTIME_LIBRARY_DIRS}
    Python_VERSION                             ${Python_VERSION}
    Python_VERSION_MAJOR                       ${Python_VERSION_MAJOR}
    Python_VERSION_MINOR                       ${Python_VERSION_MINOR}
    Python_VERSION_PATCH                       ${Python_VERSION_PATCH}
    Python_PyPy_VERSION                        ${Python_PyPy_VERSION}
    Python_NumPy_FOUND                         ${Python_NumPy_FOUND}
    Python_NumPy_INCLUDE_DIRS                  ${Python_NumPy_INCLUDE_DIRS}
    Python_NumPy_VERSION                       ${Python_NumPy_VERSION}
")
endmacro()
