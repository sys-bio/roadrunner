set(target ExamplesCAPIGetCCode)

add_executable(${target}
        main.c
#        ${CMAKE_SOURCE_DIR}/source/rrFileLog.cpp
#        ${CMAKE_SOURCE_DIR}/source/rrLogger.cpp
#        ${CMAKE_SOURCE_DIR}/source/rrLogFile.cpp
#        ${CMAKE_SOURCE_DIR}/source/rrLoggerUtils.cpp
#        ${CMAKE_SOURCE_DIR}/source/rrLogOutput.cpp
#        ${CMAKE_SOURCE_DIR}/source/rrObject.cpp
        ${CMAKE_SOURCE_DIR}/source/rrStringUtils.cpp
        ${CMAKE_SOURCE_DIR}/source/rrUtils.cpp
        )

add_compile_definitions(${target} PRIVATE STATIC_RR)
target_link_libraries(${target} PRIVATE roadrunner_c_api roadrunner-static)
target_include_directories(${target} PRIVATE ${INCLUDE_DIRECTORIES})

install(
        TARGETS ${target}
        DESTINATION bin
        COMPONENT compiled_examples
)



