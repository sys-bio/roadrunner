# M Totte Karlsson
# J Kyle Medley

project( levenberg_marquardt )

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

# This is where the Levenberg lib is built
add_subdirectory(lib)

add_definitions(
  -DPOCO_STATIC
  -DLIBSBML_STATIC # FIXME: using shared libsbml
  -DEXPORT_TEL_PLUGIN
  -DSTATIC_LIBSTRUCT
)

add_library(tel_levenberg_marquardt MODULE ${RRP_LEVENBERG_MARQUARDT_SOURCES})

target_link_libraries (tel_levenberg_marquardt
  lmfit-static
  #telplugins_core
  #telplugins_c_api
  #${LIBROADRUNNER_C_API_LIBRARY}
  #${RRPLUGINS_COMMON_SHARED_LIBS}
  roadrunner
)

if(UNIX)
  if(APPLE)
    #set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-no_implicit_dylibs -undefined error")
    SET_TARGET_PROPERTIES(tel_levenberg_marquardt PROPERTIES INSTALL_RPATH "@loader_path/../python2.7/roadrunner/;@loader_path/../lib" INSTALL_NAME_DIR "@rpath")
  else()
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")
  endif()

  set_property(TARGET tel_levenberg_marquardt
    PROPERTY COMPILE_DEFINITIONS
    LIBSBML_STATIC
    LIBLAX_STATIC
    STATIC_PUGI
    STATIC_LIBSTRUCT
  )
endif()

install (TARGETS tel_levenberg_marquardt
  DESTINATION ${RRP_PLUGINS_DIR}
  COMPONENT plugins
)
