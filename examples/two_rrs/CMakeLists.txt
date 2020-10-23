set(target ExampleTwoRRs)
add_executable(${target} main.cpp)
target_compile_definitions(${target} PRIVATE STATIC_RR)
target_include_directories(${target} PRIVATE ${INCLUDE_DIRECTORIES})
target_link_libraries(${target} PRIVATE roadrunner-static)


install(TARGETS ${target}
        DESTINATION bin
        COMPONENT examples)
