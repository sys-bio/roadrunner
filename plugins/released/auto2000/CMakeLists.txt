# M Totte Karlsson
# J Kyle Medley

project( auto2000 )

add_subdirectory(libAuto)
add_subdirectory(libAutoTelluriumInterface)

if (TLP_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# Auto2k lib includes
include_directories(libAutoTelluriumInterface)

set(RRP_AUTO2000_SOURCES
  telAutoPlugin.cpp
  telAutoWorker.cpp
  telAutoDataParser.cpp
  auto_utils.cpp
  auto2000_doc.cpp
  telAutoPlugin.h
  telAutoWorker.h
  telAutoDataParser.h
  auto_utils.h
  auto2000_doc.h
)

add_definitions(
  -DEXPORT_TEL_PLUGIN
  -DPOCO_STATIC
  -DLIBSBML_STATIC # FIXME: using shared libsbml
  -DSTATIC_AUTO
  -DSTATIC_TEL_AUTO
  -DSTATIC_LIBSTRUCT
  -DLIBLAX_STATIC
  -DSTATIC_NLEQ
)

add_library(tel_auto2000 MODULE ${RRP_AUTO2000_SOURCES})

target_link_libraries (tel_auto2000
  #telplugins_core
  libTelluriumAutoInterface-static
  libAuto-static
  #${RRPLUGINS_COMMON_SHARED_LIBS}
  roadrunner
)

if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set_target_properties( tel_auto2000 PROPERTIES INSTALL_RPATH "$ORIGIN/./" )
endif ()

install (TARGETS tel_auto2000
  DESTINATION ${RRP_PLUGINS_DIR}
  COMPONENT plugins
)
