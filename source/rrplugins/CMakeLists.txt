# M Totte Karlsson
# J Kyle Medley

include_directories(
  ${CMAKE_SOURCE_DIR}/source/rrplugins/core
  ${CMAKE_SOURCE_DIR}/source/rrplugins/common
  ${THIRD_PARTY_INSTALL_FOLDER}/include
  #${THIRD_PARTY_INSTALL_FOLDER}/include/rr # deprecated
  #${THIRD_PARTY_INSTALL_FOLDER}/include/roadrunner
  ${THIRD_PARTY_INSTALL_FOLDER}/include/sbml
  ${THIRD_PARTY_INSTALL_FOLDER}/include/cvode
  ${THIRD_PARTY_INSTALL_FOLDER}/include/clapack
  ${Poco_INCLUDE_DIRS}
  ${LIBXML2_INCLUDE_DIR}
)

#add_subdirectory(common)
#add_subdirectory(core)
#add_subdirectory(math)

