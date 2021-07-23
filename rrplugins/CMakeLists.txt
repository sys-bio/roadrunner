# M Totte Karlsson
# J Kyle Medley
# Debashish Roy

cmake_minimum_required(VERSION 3.16)
project(rrplugins)

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.20")
    # Source file extensions (.h/.cpp) do not need to be explicitly
    #   specified if CMP0115 set to OLD.
    #   New behavior requires all extensions be explicit
    cmake_policy(SET CMP0115 OLD)
endif ()

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

