cmake_minimum_required(VERSION 2.8)
set(target tel_example_one_rr)

add_executable(${target} 
    main.cpp
)

add_definitions(
    -DPOCO_STATIC
    -DLIBSBML_STATIC 
    -DSTATIC_LIBSTRUCT
    -DLIBLAX_STATIC  
    -DSTATIC_NLEQ
)

include_directories(
    ${RR_PLUGINS_ROOT}/rrplugins/core
    ${RR_PLUGINS_ROOT}/rrplugins/common
    ${RR_PLUGINS_ROOT}/wrappers/C
)

target_link_libraries (${target}
    roadrunner-static
    telplugins_c_api
)

if(WIN32)
add_definitions(
    -DWIN32
)
endif()

if(INSTALL_CPP_EXAMPLES)
    install (TARGETS ${target}
             DESTINATION plugins
             COMPONENT plugins
    )
endif()
install (TARGETS ${target} 		
        DESTINATION bin 
        COMPONENT examples)
