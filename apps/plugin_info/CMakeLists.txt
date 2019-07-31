# M Totte Karlsson
# J Kyle Medley

project(plugin_info)

add_executable(plugin_info main.cpp)

target_link_libraries (plugin_info roadrunner)

if(WIN32)
  add_definitions (-DWIN32)
endif()

if(UNIX)
  add_definitions (-DPOCO_STATIC)
endif()


install (TARGETS plugin_info
  DESTINATION bin 
  COMPONENT apps
)
