# M Totte Karlsson
# J Kyle Medley

project( rrplugins_python_api )

#=============== INSTALL =====================

# question: how was the Python API generated? If this was swig or some other automated
# system then really the code to regenerate the telplugins_c_api.py needs to be included
# in the repository. Otherwise, if we make changes in future we cannot regenerate the
# python api!

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/__init__.py
)

# For testing, also install in the build directory:
configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py.in
        ${RR_PLUGINS_PYTHON_BUILD_PREFIX}/__init__.py
)

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/telplugins.py
        ${RR_PLUGINS_PYTHON_BUILD_PREFIX}/telplugins.py
)

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/telplugins_c_api.py
        ${RR_PLUGINS_PYTHON_BUILD_PREFIX}/telplugins_c_api.py
)

set (RRP_PYTHON_API_SOURCE
  ${CMAKE_CURRENT_BINARY_DIR}/__init__.py
  telplugins.py
  telplugins_c_api.py
)

install(
  FILES
  ${RRP_PYTHON_API_SOURCE}
  README.txt
  DESTINATION "${RR_PLUGINS_PYTHON_INSTALL_PREFIX}"
  COMPONENT rrplugins
)

install(
  FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/setup_rrplugins.py
  DESTINATION .
  COMPONENT rrplugins
)

