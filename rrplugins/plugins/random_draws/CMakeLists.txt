project(random_draws)
set(target tel_random_draws)

set(RRP_RANDOM_DRAWS_SOURCES
  random_draws.cpp
  random_draws.h
)

add_library(${target} MODULE ${RRP_RANDOM_DRAWS_SOURCES})

target_link_libraries(${target} PRIVATE
        telplugins_c_api
)

target_compile_definitions(${target} PRIVATE
     -DEXPORT_TEL_PLUGIN
     -DEXPORT_RANDOM_DRAWS
     )

if (UNIX)
  set_target_properties(${target} PROPERTIES INSTALL_RPATH "$ORIGIN/./")
endif ()
