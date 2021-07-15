# M Totte Karlsson
# J Kyle Medley

project(lmfit)

set(target lmfit) 

set(sources 
lmmin.c
lmmin.h
)

add_library(lmfit-static STATIC 
${sources}
)

set_target_properties(lmfit-static PROPERTIES PREFIX "")

if (WIN32)
    target_compile_definitions(lmfit-static PUBLIC STATIC_RR)
endif ()

set_target_properties(lmfit-static PROPERTIES POSITION_INDEPENDENT_CODE True )

