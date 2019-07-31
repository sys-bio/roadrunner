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

set_target_properties( lmfit-static PROPERTIES POSITION_INDEPENDENT_CODE True )

