include(GNUInstallDirs)
set(target roadrunner_c_api)

set(
        HEADERS
        rrcStringList.h
        rrArrayList.h
        rrArrayListItem.h
        rrArrayListItemBase.h
        rrc_api.h
        rrc_logging_api.h
        rrc_nom_api.h
        rrc_libstruct_api.h
        rrc_utilities.h
        rrc_cpp_support.h
#=======
#set(C_API_INCLUDE_ROOT "${RR_ROOT}/wrappers/C")


#include_directories(
#${RR_GENERATED_HEADER_PATH}     #we have to do this to get the generated header, NEEDED FOR rrVersionInfo.h file
#${C_API_INCLUDE_ROOT}
#${RR_ROOT}/source
#${THIRD_PARTY_INSTALL_FOLDER}/include
#>>>>>>> 32624909156eb38d84edf40566b07980f7435fca
)

set(
        SOURCES
        rrcStringList.cpp
        rrArrayList.cpp
        rrArrayListItem.cpp
        rrArrayListItemBase.cpp
        rrc_api.cpp
        rrc_logging_api.cpp
        rrc_nom_api.cpp
        rrc_libstruct_api.cpp
        rrc_utilities.cpp
        rrc_cpp_support.cpp
)

#The C-api is built as a shared library..
add_library(${target} SHARED ${HEADERS} ${SOURCES})

target_include_directories(${target} PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>" # for #include "C/rrc_api.h"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        )

# link statically so users of the c library will not also need the shared.
# Note: properties of roadrunner-static are transitive. Link libraries and
# include_directories for roadrunner-static all come with the
# roadrunner-static target automatically.
target_link_libraries(${target} PUBLIC roadrunner-static)

add_dependencies(${target} roadrunner-static)

target_compile_definitions(${target} PRIVATE EXPORT_RRC STATIC_RR)

install(
        TARGETS ${target}
        EXPORT ${target}
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT ${target}
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT ${target}
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT ${target}
        PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/rr" COMPONENT ${target}
)
install(FILES ${HEADERS} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/rr")

# install export info
install(
        EXPORT ${target}
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake"
        NAMESPACE ${target}::
        FILE ${target}-config.cmake
        COMPONENT ${target}
        EXPORT_LINK_INTERFACE_LIBRARIES
)



