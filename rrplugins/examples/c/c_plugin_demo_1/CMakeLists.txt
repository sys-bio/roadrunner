cmake_minimum_required(VERSION 2.8)
project(tel_c_plugin_demo C)
ENABLE_LANGUAGE (C)
set(target tel_example_c_plugin_demo_lib)

include_directories(
    ${RR_PLUGINS_ROOT}/rrplugins/core
    ${RR_PLUGINS_ROOT}/rrplugins/common
    ${RR_PLUGINS_ROOT}/wrappers/C
    ${THIRD_PARTY_INSTALL_FOLDER}/include
    ${THIRD_PARTY_INSTALL_FOLDER}/include/c_api
)

SET_SOURCE_FILES_PROPERTIES( c_plugin_demo.c PROPERTIES LANGUAGE C)
set(sources
c_plugin_demo.c
c_plugin_demo.h
)

add_library(${target} MODULE ${sources})

target_link_libraries (${target}
    roadrunner
    telplugins_c_api
    roadrunner_c_api
)

add_definitions(
-DEXPORT_TEL_PLUGIN
-DPOCO_STATIC
)


#Also build the main executable
set(main_target tel_example_c_plugin_demo)

include_directories(
${TLP_ROOT}/source
${TLP_ROOT}/wrappers/C
${THIRD_PARTY_INSTALL_FOLDER}/include
)

set(main_sources
main.cpp
)

add_executable(${main_target} ${main_sources})

add_definitions(
-DPOCO_STATIC
)


if(WIN32)
target_link_libraries (${main_target}
roadrunner
roadrunner_c_api
telplugins_c_api
)
endif()

if(INSTALL_C_EXAMPLES)
    install (TARGETS ${target}
        DESTINATION plugins
        COMPONENT examples
    )

    install (TARGETS ${main_target}
             DESTINATION bin
             COMPONENT examples
    )
endif()

