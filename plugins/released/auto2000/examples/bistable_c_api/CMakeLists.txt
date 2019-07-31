cmake_minimum_required(VERSION 2.8)
set(target bistable)

add_executable(${target} 
main.cpp 
)

add_definitions(
-DSTATIC_RR
-DSTATIC_PLUGIN_API
-DPOCO_STATIC
-DLIBSBML_STATIC 
-DSTATIC_LIBSTRUCT
-DLIBLAX_STATIC  
-DSTATIC_NLEQ
)

if(WIN32)
add_definitions(
-DWIN32
)

target_link_libraries (${target} 
roadrunner-static
telplugins_api-static
roadrunner
)
endif()

install (TARGETS ${target} 		
        DESTINATION bin 
        COMPONENT examples)
