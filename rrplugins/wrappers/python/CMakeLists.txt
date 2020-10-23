# M Totte Karlsson
# J Kyle Medley

project( rrplugins_python_api )

#=============== INSTALL =====================

# question: how was the Python API generated? If this was swig or some other automated
# system then really the code to regenerate the telplugins_c_api.py needs to be included
# in the repository. Otherwise, if we make changes in future we cannot regenerate the
# python api!

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


