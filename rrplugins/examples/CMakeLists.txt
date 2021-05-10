cmake_minimum_required(VERSION 2.8)
project(plugins_examples)

#Give access to headers in the following folders
include_directories(
        "$<TARGET_PROPERTY:roadrunner,INCLUDE_DIRECTORIES>"
        "$<TARGET_PROPERTY:roadrunner_c_api,INCLUDE_DIRECTORIES>"
#	${TLP_ROOT}/source
#	${TLP_ROOT}/wrappers/C
# 	${THIRD_PARTY_INSTALL_FOLDER}/include
# 	${THIRD_PARTY_INSTALL_FOLDER}/include/rr
# 	${THIRD_PARTY_INSTALL_FOLDER}/include/rr/C
)

OPTION (INSTALL_C_EXAMPLES              "Install Tellurium Plugin C Examples"         OFF)
OPTION (INSTALL_CPP_EXAMPLES            "Install Tellurium Plugin CPP Examples"       OFF)
OPTION (INSTALL_PYTHON_EXAMPLES         "Install Tellurium Plugin PYTHON Examples"    OFF)

add_subdirectory(c)
add_subdirectory(cpp)

if(INSTALL_PYTHON_EXAMPLES)
add_subdirectory(python)
endif()

install(
    FILES README.txt
    DESTINATION examples
)
