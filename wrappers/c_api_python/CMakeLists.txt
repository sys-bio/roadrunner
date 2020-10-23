
if (${BUILD_TESTS})
    add_subdirectory(Testing)

    if (INSTALL_EXAMPLES)
        add_subdirectory(Examples)
    endif (INSTALL_EXAMPLES)

    #=============== install =====================
    install(
            FILES
            rrPython.py README.txt
            DESTINATION c_api_python
            COMPONENT testing
    )
endif (${BUILD_TESTS})
