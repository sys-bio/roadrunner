# Debashish Roy

project(add_noise)
set(target tel_add_noise)

set(RRP_ADD_NOISE_SOURCES
        add_noise.cpp
        add_noise_worker.cpp
        add_noise_doc.cpp
        noise.cpp
        add_noise.h
        add_noise_worker.h
        add_noise_doc.h
        noise.h
        )

add_library(${target} MODULE ${RRP_ADD_NOISE_SOURCES})

target_link_libraries(${target} PRIVATE
        telpluginsBaseClass-static
        telplugins_math-static
        telplugins_common-static
        ${RRPLUGINS_COMMON_SHARED_LIBS}
        )

target_compile_definitions(${target} PRIVATE
        -DEXPORT_TEL_PLUGIN
        -DEXPORT_ADD_NOISE
        )


if (UNIX)
    set_target_properties(${target} PROPERTIES INSTALL_RPATH "$ORIGIN/./")
endif ()

install(TARGETS ${target}
        DESTINATION ${RR_PLUGINS_INSTALL_PREFIX}
        COMPONENT plugins
        )

