Building the Documentation
==========================

Quick instructions
-------------------

Configure the roadrunner cmake system using the ``-DBUILD_DOCS=ON`` flag and build the ``roadrunner-docs-sphinx`` target.
This basically will run the commands for running doxygen and sphinx.

For example:

.. code-block:: bash

    cmake -DCMAKE_INSTALL_PREFIX=../install-mac-rel \
        -DRR_DEPENDENCIES_INSTALL_PREFIX=/Users/ciaranwelsh/Documents/libroadrunner-deps/install-clang13-rel \
        -DLLVM_INSTALL_PREFIX=/Users/ciaranwelsh/Documents/llvm-6.x/llvm-6.x-clang11.0.3-x64-release \
        -DBUILD_TESTS=ON \
        -DPython_ROOT_DIR=/Users/ciaranwelsh/miniconda3/envs/py39 \
        -DBUILD_PYTHON=ON \
        -DSWIG_EXECUTABLE=/Users/ciaranwelsh/Documents/swig-4.0.2/swig \
        -DBUILD_DOCS=ON \
        -DSPHINX_EXECUTABLE=/Users/ciaranwelsh/miniconda3/envs/py39/bin/sphinx-build
    cmake --build . --target roadrunner-docs-sphinx --config Release

Details
----------

Roadrunner is a C++ library with a C and Python front end. The C and C++ code is documented using the standard
``doxygen`` tool, whilst the Python front end is documented using Sphinx. Furthermore, we make use of a
Python package called `Breathe <https://breathe.readthedocs.io/en/latest/>`_ for bridging the gap between
``Doxygen`` and ``Sphinx`` which are the two standard documentation systems for C/C++ and Python respectively.

In short, ``doxygen`` reads the docstrings from C/C++ code and saves them to XML which is pulled into documentation
that can be rendered by ``Sphinx`` using directives from ``Breathe``. Whilst rendering the documentation
into html is handled by sphinx, we still maintain and provide access to the doxygen generated documentation.

When roadrunner is built with the ``-DBUILD_DOCS=ON`` flag, two new cmake targets are presented to the developer:
    * roadrunner-docs-doxygen
    * roadrunner-docs-sphinx.

Since the ``roadrunner-docs-sphinx`` target depends on the ``roadrunner-docs-doxygen`` target, cmake will always
make sure to build the ``roadrunner-docs-doxygen`` target before it builds the ``roadrunner-docs-doxygen`` target.

Dependencies
------------

You will need Sphinx, `Doxygen <https://www.doxygen.nl/download.html>`_ and some python packages
  * pip install sphinx breathe sphinxcontrib-bibtex sphinx-tabs sphinx_rtd_theme

.. warning::

    Make sure you either install doxygen in a default location (i.e. already on the PATH)  or that you add the path to the
    doxygen executable to the PATH environment variable so cmake can find it.



Troubleshooting
---------------

Sometimes cmake finds a different version of Python installed on your system
compared to what you were expecting. Therefore, if cmake complains that it cannot
locate "sphinx-build" you can provide this manually with:

.. code:block::

    -DSPHINX_EXECUTABLE=/full/path/to/sphinx/build

Make sure that this `sphinx-build` is from the same python version you are using to build roadrunner.
This is particularly important when building with conda. You can control which version
of Python is found by using ``-DPython_ROOT_DIR=/full/path/to/python/root``. For example, if you want
to use the python inside the conda environment here at ``/Users/ciaranwelsh/miniconda/envs/py38/bin/python``
then the Python executable is located at
      ``/Users/ciaranwelsh/miniconda/envs/py38/bin/python``
and its root is
      ``/Users/ciaranwelsh/miniconda/envs/py38``















