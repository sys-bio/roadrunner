# M Totte Karlsson
# J Kyle Medley

project( rrplugins_python_api )

#=============== INSTALL =====================

set (RRP_PYTHON_API_SOURCE
  __init__.py
  telplugins.py
  telplugins_c_api.py
)

install(
  FILES
  ${RRP_PYTHON_API_SOURCE}
  README.txt
  DESTINATION site-packages/rrplugins
  COMPONENT telplugins
)


