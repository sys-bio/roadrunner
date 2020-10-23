# M Totte Karlsson
# J Kyle Medley
set(target libTelluriumAutoInterface-static)

set(RRP_LIB_AUTO_INTERFACE_SRC
        telAutoConstants.cpp
        telAutoTelluriumInterface.cpp
        telAutoUtils.cpp
        telAutoConstants.h
        telAutoTelluriumInterface.h
        telAutoUtils.h
        )

add_library(${target} STATIC ${RRP_LIB_AUTO_INTERFACE_SRC})

if (WIN32)
    target_compile_definitions(${target} PUBLIC
            STATIC_AUTO
            STATIC_TEL_AUTO
            STATIC_LIBSTRUCT
            )
endif ()

target_link_libraries(${target} PUBLIC
        rr-libstruct::rr-libstruct-static
        telplugins_core
        libAuto-static
        )
add_dependencies(${target} telplugins_common telplugins_core)
target_include_directories(${target} PUBLIC $<TARGET_PROPERTY:roadrunner-static,INCLUDE_DIRECTORIES>)
install(
        TARGETS libTelluriumAutoInterface-static
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
)

