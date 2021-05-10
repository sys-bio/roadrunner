cmake_minimum_required(VERSION 2.8)
set(target tel_example_property_meta_data)

set(sources
    main.cpp
)

add_executable(${target} ${sources})

include_directories(
    "$<TARGET_PROPERTY:roadrunner,INCLUDE_DIRECTORIES>"
    "$<TARGET_PROPERTY:roadrunner_c_api,INCLUDE_DIRECTORIES>"
)

add_definitions(
    -DPOCO_STATIC
)

target_link_libraries (${target}
    roadrunner-static
    telplugins_core-static
)
if(APPLE)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-no_implicit_dylibs -undefined error")
else()
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")
endif()

install (TARGETS ${target}
    DESTINATION bin
    COMPONENT examples
)

