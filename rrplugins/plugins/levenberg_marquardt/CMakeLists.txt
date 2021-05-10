# M Totte Karlsson
# J Kyle Medley
# Debashish Roy

project( levenberg_marquardt )
set(target tel_levenberg_marquardt )

# This is where the Levenberg lib is built
add_subdirectory(lib)


set(RRP_LEVENBERG_MARQUARDT_SOURCES
  lm.cpp
  lmWorker.cpp
  lmUtils.cpp
  lmObjectiveFunction.cpp
  levenberg_marquardt_doc.cpp
  lm.h
  lmWorker.h
  lmUtils.h
  lmObjectiveFunction.h
  levenberg_marquardt_doc.h
)


add_library(${target} MODULE ${RRP_LEVENBERG_MARQUARDT_SOURCES})

target_compile_definitions(${target} PRIVATE
        EXPORT_TEL_PLUGIN
        EXPORT_LEVENBERG_MARQUARDT
        )


target_include_directories(${target}
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/libAutoTelluriumInterface"
        "$<TARGET_PROPERTY:roadrunner,INCLUDE_DIRECTORIES>"
        )

target_link_libraries (${target} PRIVATE
        lmfit-static
        telpluginsBaseClass-static
        telplugins_math-static
        telplugins_common-static
        telplugins_c_api
        #libTelluriumAutoInterface-static
)

if(UNIX)
  if(APPLE)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-no_implicit_dylibs -undefined error")
  else()
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")
  endif()
endif()

if(BUILD_PYTHON)
    install (TARGETS ${target}
    DESTINATION ${RR_PLUGINS_PYLIB_INSTALL_PREFIX}
    COMPONENT plugins
    )
    file(GLOB EXAMPLES docs/Examples/*)
    install(FILES ${EXAMPLES} DESTINATION "${RR_PLUGINS_PYTHON_INSTALL_PREFIX}" COMPONENT plugins)
    unset(EXAMPLES)
endif()
