# Debashish Roy

project(auto2000)
set(target tel_auto2000)

add_subdirectory(libAuto)
add_subdirectory(libAutoTelluriumInterface)


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


add_library(${target} MODULE ${RRP_AUTO2000_SOURCES})

target_compile_definitions(${target} PRIVATE
        EXPORT_TEL_PLUGIN
        EXPORT_AUTO2000
        )


target_include_directories(${target}
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/libAutoTelluriumInterface"
        "$<TARGET_PROPERTY:roadrunner,INCLUDE_DIRECTORIES>"
        )

target_link_libraries(${target} PRIVATE
        telplugins_c_api
        libTelluriumAutoInterface-static
        )

add_dependencies(${target} libTelluriumAutoInterface-static libAuto-static)
if (UNIX)
    set_target_properties(${target} PROPERTIES INSTALL_RPATH "$ORIGIN/./")
endif ()

install(
        TARGETS ${target}
        RUNTIME DESTINATION bin COMPONENT rrplugins
        ARCHIVE DESTINATION lib COMPONENT rrplugins
        LIBRARY DESTINATION lib COMPONENT rrplugins
)

if(BUILD_PYTHON)
    install(TARGETS ${target}
        DESTINATION ${RR_PLUGINS_PYLIB_INSTALL_PREFIX}
        COMPONENT plugins
    )
    file(GLOB EXAMPLES docs/Examples/*)
    install(FILES ${EXAMPLES} DESTINATION "${RR_PLUGINS_PYTHON_INSTALL_PREFIX}" COMPONENT plugins)
    unset(EXAMPLES)
endif()
