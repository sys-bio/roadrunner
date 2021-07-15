# Debashish Roy

set(target tel_test_model)

set(RRP_TEST_MODEL_SOURCES
        TestModel.h
        test_model_doc.h
        TestModel.cpp
        test_model_doc.cpp
        )

add_library(${target} MODULE ${RRP_TEST_MODEL_SOURCES})

target_compile_definitions(${target} PRIVATE -DEXPORT_TEL_PLUGIN -DEXPORT_TEST_MODEL)

target_link_libraries(${target} PRIVATE
        telpluginsBaseClass-static
        telplugins_math-static
)

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
    install(TARGETS ${target} DESTINATION ${RR_PLUGINS_PYLIB_INSTALL_PREFIX} COMPONENT plugins)
    file(GLOB EXAMPLES docs/Examples/*)
    install(FILES ${EXAMPLES} DESTINATION "${RR_PLUGINS_PYTHON_INSTALL_PREFIX}" COMPONENT plugins)
    unset(EXAMPLES)
endif()
