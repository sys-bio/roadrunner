# M Totte Karlsson
# J Kyle Medley
# Debashish Roy

cmake_minimum_required(VERSION 3.2)
project(rrplugins)

set(RR_PLUGINS_ROOT ${CMAKE_CURRENT_SOURCE_DIR})


if (UNIX)
    set(RRPLUGINS_COMMON_SHARED_LIBS ${LIBROADRUNNER_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} dl)
else ()
    if (${MSVC})
        set(RRPLUGINS_COMMON_SHARED_LIBS ${LIBROADRUNNER_LIBRARIES})
    else ()
        set(RRPLUGINS_COMMON_SHARED_LIBS ${LIBROADRUNNER_LIBRARIES} iphlpapi)
    endif ()
endif ()

add_subdirectory(rrplugins)

add_subdirectory(plugins)

add_subdirectory(wrappers)

