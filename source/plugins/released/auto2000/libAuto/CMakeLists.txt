# M Totte Karlsson
# J Kyle Medley

include_directories(
  libf2c
)

set(RRP_LIB_AUTO_SRC
  theMain.cpp
  autlab.cpp
  autlib1.cpp
  autlib2.cpp
  autlib3.cpp
  autlib4.cpp
  autlib5.cpp
  conpar.cpp
  dmatrix.cpp
  eispack.cpp
  getopt.cpp
  reduce.cpp
  setubv.cpp
  vsAuto.cpp
  worker.cpp
  libf2c/z_log.cpp
  libf2c/z_exp.cpp
  libf2c/z_abs.cpp
  libf2c/r_lg10.cpp
  libf2c/pow_ii.cpp
  libf2c/pow_di.cpp
  libf2c/pow_dd.cpp
  libf2c/i_nint.cpp
  libf2c/i_dnnt.cpp
  libf2c/d_sign.cpp
  libf2c/d_lg10.cpp
  libf2c/d_imag.cpp
  libf2c/cabs.cpp
  auto_api.h
  auto_c.h
  auto_exporter.h
  auto_mpi.h
  auto_types.h
  fcon.h
  libAuto_pch.h
  vsAuto.h
  libf2c/auto_f2c.h
)

##------- SHARED TARGET
#add_library(${target} SHARED
#${src}
#)
#
#if(WIN32)
#    set_property(TARGET ${target} 
#        PROPERTY  COMPILE_DEFINITIONS 
#        EXPORT_AUTO
#        )
#    
#endif()
#----------------------

#------- STATIC target
add_library( libAuto-static STATIC ${RRP_LIB_AUTO_SRC} )

set_target_properties(
  libAuto-static
  PROPERTIES
  POSITION_INDEPENDENT_CODE True
)

if(WIN32)
  set_property(TARGET libAuto-static 
    PROPERTY  COMPILE_DEFINITIONS 
    STATIC_AUTO
  )
endif()

##=============== INSTALL =====================
install(
  FILES 
  vsAuto.h 
  libf2c/auto_f2c.h
  auto_exporter.h
  DESTINATION include/auto_api 
)

install(   
  TARGETS libAuto-static
  RUNTIME DESTINATION bin 
  ARCHIVE DESTINATION lib 
  LIBRARY DESTINATION lib 
)

