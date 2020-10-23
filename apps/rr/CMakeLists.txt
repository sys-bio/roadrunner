cmake_minimum_required(VERSION 2.8)
set(target rr)

add_executable(${target}
        main.cpp
        Args.cpp Args.h
        )

target_compile_definitions(${target} PRIVATE STATIC_RR)
target_include_directories(${target} PRIVATE ${INCLUDE_DIRECTORIES})
target_link_libraries(${target} PRIVATE roadrunner-static ${LINK_LIBRARIES})


install(TARGETS ${target}
        DESTINATION bin
        COMPONENT apps
        )
