##################################################################
# Usage
# -----
# Build the roadrunner-docs-doxygen and roadrunner-docs-sphinx targets
#

##################################################################
# Details
# -------
# Use a combination of Doxygen, Breathe and Sphinx to build the roadrunner documentation. Doxygen
# handles creating documentation from C/C++ doc strings and storing them in xml. Sphinx can read
# this xml using directives provided by Breathe. Python docs and rendering the documentation
# into html is handled by sphinx.
#
# To build the documentation set
#     -DBUILD_DOCS=ON
# when you configure the cmake project. This defines two cmake targets
#    1) roadrunner-docs-doxygen
#    2) roadrunner-docs-sphinx
# You need to build the first and then build the second.  At one point, the second was dependent on the first, but that seems to have broken at some point, but building them both explicitly does work.
#

##################################################################
# Dependencies
# ------------
# You will need Sphinx and some python packages, all listed in docs/requirements.txt
#   - pip install -r docs/requirements.txt
# and doxygen. Make sure you either install doxygen in a default location, so that
# cmake can find it or add the path to the doxygen executable to the PATH environment
# variable so cmake can find it.
#

##################################################################
# Troubleshooting
# ---------------
# Sometimes cmake finds a different version of Python installed on your system
# compared to what you were expecting. Therefore, if cmake complains that it cannot
# locate "sphinx-build" you can provide this manually with:
#   -DSPHINX_EXECUTABLE=/full/path/to/sphinx/build
# Make sure that this sphinx-build is from the same python version you are using
# (particularly important when building with conda). You can control which version
# of Python is found by using the
#   -DPython_ROOT_DIR=/full/path/to/python/root
# For example, when you want to use a conda environment, the python exe is here:
#       /Users/ciaranwelsh/miniconda/envs/py38/bin/python
# and its root is
#       /Users/ciaranwelsh/miniconda/envs/py38


set(DOCS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")


############################################################
#   doxygen
#


# https://devblogs.microsoft.com/cppblog/clear-functional-c-documentation-with-sphinx-breathe-doxygen-cmake/
find_package(Doxygen REQUIRED)

set(DOXYGEN_INPUT_DIR "${RR_ROOT}/source")
set(DOXYGEN_INPUT_DIR ${DOXYGEN_INPUT_DIR} "${RR_ROOT}/source/llvm")
set(DOXYGEN_OUTPUT_DIR "${RR_ROOT}/docs/doxygen-output")
set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/html/index.xml)
set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

# Doxygen won't create this for us
file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR})

# todo Only regenerate Doxygen when the Doxyfile or public headers change
add_custom_command(
        OUTPUT ${DOXYGEN_INDEX_FILE}
        DEPENDS ${rrHeaders}
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/source"
        MAIN_DEPENDENCY ${DOXYFILE_OUT}
        COMMENT "Generating docs"
        VERBATIM
)

add_custom_target(roadrunner-docs-doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE})

#############################################################
#   Sphinx
#
find_package(Python COMPONENTS Interpreter)
if (NOT ${Python_FOUND})
    message(STATUS "Python cannot be found")
endif ()


message(STATUS "Python_EXECUTABLE ${Python_EXECUTABLE}")
get_filename_component(PYTHON_ROOT ${Python_EXECUTABLE} DIRECTORY)
find_program(SPHINX_EXECUTABLE
        NAMES sphinx-build sphinx-build.exe
        PATHS
        "${PYTHON_ROOT}/Scripts/sphinx-build.exe"
        "${PYTHON_ROOT}/Scripts/sphinx-build"
        REQUIRED
        COMMENT
        "absolute path to the sphinx-build executable. If you do not have sphinx-build
you will need to install it. In your conda environment do: $conda activate <env>;
pip install breathe sphinx sphinxcontrib-bibtex sphinx-tabs sphinx_rtd_theme; which sphinx-build. Then copy the value of which sphinx-build into
-DSPHINX_EXECUTABLE"
        )
message(STATUS "SPHINX_EXECUTABLE: ${SPHINX_EXECUTABLE}")

set(SPHINX_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/source")
set(SPHINX_BUILD "${CMAKE_CURRENT_SOURCE_DIR}/docs-build")
set(SPHINX_INDEX_FILE ${SPHINX_BUILD}/index.html)

# todo Only regenerate Sphinx when:
# - Doxygen has rerun
# - Our doc files have been updated
# - The Sphinx config has been updated
add_custom_command(
        OUTPUT ${SPHINX_INDEX_FILE}
        COMMAND
        # Tell Breathe where to find the Doxygen output
        ${SPHINX_EXECUTABLE} -b html -Dbreathe_projects.roadrunner=${DOXYGEN_OUTPUT_DIR}/xml
        ${SPHINX_SOURCE} ${SPHINX_BUILD}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS
        # Other docs files you want to track should go here (or in some variable)
        ${SPHINX_SOURCE}/index.rst
#        ${DOXYGEN_INDEX_FILE}DOXYGEN_INDEX_FILE
        MAIN_DEPENDENCY ${SPHINX_SOURCE}/conf.py
        COMMENT "Generating documentation with Sphinx")


add_custom_target(roadrunner-docs-sphinx DEPENDS ${SPHINX_INDEX_FILE})


# Add an install target to install the docs
include(GNUInstallDirs)
install(DIRECTORY ${SPHINX_BUILD} DESTINATION docs)
install(FILES index.html DESTINATION docs)


