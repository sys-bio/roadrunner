cmake_minimum_required(VERSION 2.8)
set(target tel_example_property_container)

include_directories(
${THIRD_PARTY_INSTALL_FOLDER}/include
)

set(sources
main
)

add_executable(${target} ${sources})

add_definitions(
-DPOCO_STATIC
)

target_include_directories(${target} PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ${RR_PLUGINS_ROOT}/rrplugins/common
    )

message(STATUS "RR_PLUGINS_ROOT: ${RR_PLUGINS_ROOT}")

target_link_libraries (${target}
    roadrunner-static
    telplugins_c_api
)

if(APPLE)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-no_implicit_dylibs -undefined error")
else()
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")
endif()

#    set_property(TARGET ${target}
#        PROPERTY  COMPILE_DEFINITIONS
#        LIBSBML_STATIC
#        LIBLAX_STATIC
#        STATIC_PUGI
#        STATIC_LIBSTRUCT
#        )

install (TARGETS ${target}
    DESTINATION bin
    COMPONENT examples
)

