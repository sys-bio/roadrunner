cmake_minimum_required(VERSION 2.8)
set(target tel_example_cpp_plugin_demo)

set(sources
    cpp_plugin_demo.cpp
    cpp_plugin_demo.h
)

add_library(${target} STATIC ${sources})

add_definitions(
    -DPOCO_STATIC
)

target_link_libraries (${target}
    roadrunner-static
    telplugins_core-static
)

if(UNIX)
    if(APPLE)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-no_implicit_dylibs -undefined error")
    else()
        set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")
    endif()

    set_property(TARGET ${target}
        PROPERTY  COMPILE_DEFINITIONS
        LIBSBML_STATIC
        LIBLAX_STATIC
        STATIC_PUGI
        STATIC_LIBSTRUCT
        )
endif()

if(${MSVC})
#set (CMAKE_SHARED_LINKER_FLAGS     "${CMAKE_SHARED_LINKER_FLAGS} \"/DEF:${CMAKE_CURRENT_SOURCE_DIR}/add_noise.def\"")
endif()

if(INSTALL_CPP_EXAMPLES)
    install (TARGETS ${target}
             DESTINATION plugins
             COMPONENT plugins
    )
endif()

