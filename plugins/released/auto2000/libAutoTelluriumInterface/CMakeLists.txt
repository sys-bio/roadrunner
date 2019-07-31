# M Totte Karlsson
# J Kyle Medley

set(RRP_LIB_AUTO_INTERFACE_SRC
  telAutoConstants.cpp
  telAutoTelluriumInterface.cpp
  telAutoUtils.cpp
  telAutoConstants.h
  telAutoTelluriumInterface.h
  telAutoUtils.h
)

add_library( libTelluriumAutoInterface-static STATIC ${RRP_LIB_AUTO_INTERFACE_SRC} )

set_target_properties(
  libTelluriumAutoInterface-static
  PROPERTIES
  POSITION_INDEPENDENT_CODE True
)

if(WIN32)
  set_property(TARGET libTelluriumAutoInterface-static 
    PROPERTY  COMPILE_DEFINITIONS 
    STATIC_AUTO
    STATIC_TEL_AUTO
  )
    
  target_link_libraries ( libTelluriumAutoInterface-static
    roadrunner
  )
endif()

install(   
  TARGETS libTelluriumAutoInterface-static
  RUNTIME DESTINATION bin 
  ARCHIVE DESTINATION lib 
  LIBRARY DESTINATION lib 
)

