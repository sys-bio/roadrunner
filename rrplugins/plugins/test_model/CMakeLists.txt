# Debashish Roy

set(target teltest_model)

set(RRP_TEST_MODEL_SOURCES
        TestModel.h
        test_model_doc.h
        TestModel.cpp
        test_model_doc.cpp
        ../add_noise/add_noise.cpp
        ../add_noise/add_noise_worker.cpp
        ../add_noise/noise.cpp
        ../add_noise/add_noise.h
        ../add_noise/add_noise_worker.h
        ../add_noise/noise.h
        )

add_library(${target} MODULE ${RRP_TEST_MODEL_SOURCES})

target_compile_definitions(${target} PRIVATE -DEXPORT_TEL_PLUGIN -DEXPORT_TEST_MODEL)

target_link_libraries(${target} PRIVATE
        telpluginsBaseClass
        telplugins_math
        ${RRPLUGINS_COMMON_SHARED_LIBS})

#target_include_directories(${target} PRIVATE
#        "D:/rrbuild/roadrunner/rrplugins/plugins/released/add_noise")

if (UNIX)
    set_target_properties(${target} PROPERTIES INSTALL_RPATH "$ORIGIN/./")
endif ()


install(TARGETS ${target} DESTINATION ${RR_PLUGINS_INSTALL_PREFIX} COMPONENT plugins)
