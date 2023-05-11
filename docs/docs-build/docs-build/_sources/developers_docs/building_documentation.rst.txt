Building the Documentation
==========================

.. warning::

    Please familiarize yourself with our :ref:`Git Workflow` before attempting to add to
    or build the docs yourself. If you are not, there is a high probability that you
    will break something, or just get frustrated.

Quick instructions
-------------------

1. Make sure you are on the ``gh-pages`` branch. Changes to the docs source should have been merged into this branch from develop.

.. warning::

    Adding documentation source code to the `gh-pages` branch means that the `gh-pages`
    branch will need to be merged with develop. Failure to do so dramatically increases
    the likelihood of a merge conflict somewhere down the line, since the `gh-pages` will
    contain

2. Configure the roadrunner cmake system using the ``-DBUILD_DOCS=ON`` flag
3. Build the ``roadrunner-docs-sphinx`` target.

This basically will run the commands needed for running doxygen and sphinx.

For example:

.. code-block:: bash

    cmake -DCMAKE_INSTALL_PREFIX=../install-mac-rel \
        -DRR_DEPENDENCIES_INSTALL_PREFIX=/Users/ciaranwelsh/Documents/libroadrunner-deps/install-clang13-rel \
        -DLLVM_INSTALL_PREFIX=/Users/ciaranwelsh/Documents/llvm-13.x/llvm-13.x-clang11.0.3-x64-release \
        -DBUILD_TESTS=ON \
        -DPython_ROOT_DIR=/Users/ciaranwelsh/miniconda3/envs/py39 \
        -DBUILD_PYTHON=ON \
        -DSWIG_EXECUTABLE=/Users/ciaranwelsh/Documents/swig-4.0.2/swig \
        -DBUILD_DOCS=ON \
        -DSPHINX_EXECUTABLE=/Users/ciaranwelsh/miniconda3/envs/py39/bin/sphinx-build
    cmake --build . --target roadrunner-docs-sphinx --config Release

Details
----------
Github hosts our documentation. It does so off of the `gh-pages` branch, and specifically
the `<roadrunner-root>/docs` subfolder. The homepage of our documentation is the `index.html`
of the `docs` subfolder, but this only redirects to `<roadrunner-root>/docs/docs-build/index.html`.
Therefore, the docs-build folder is where all the action happens. When you build the
`roadrunner-docs-sphinx` target, you are using Python's Sphinx library
to create a documentation website (using `sphinx-build`) and storing it in the `docs-build`
directory, which is served as a website by github.

Because our documentation is hosted on the `gh-pages` branch, care must be taken with
version control when updating the documentation. Specifically, you should commit
documentation source code to your feature branch and this should gradually make its
way to the `gh-pages` branch after merging with develop. Please see our :ref:`Git Workflow`
page if any of this sounds unfamiliar.

Roadrunner is a C++ library with a C and Python front end. The C and C++ code is documented using the standard
``doxygen`` tool, whilst the Python front end is documented using ``Sphinx``. Furthermore, we make use of a
Python package called `Breathe <https://breathe.readthedocs.io/en/latest/>`_ for bridging the gap between
``Doxygen`` and ``Sphinx`` which are the two standard documentation systems for C/C++ and Python respectively.

In short, ``doxygen`` reads the docstrings from C/C++ code and saves them to XML which is pulled into documentation
that can be rendered by ``Sphinx`` using directives from ``Breathe``. Whilst rendering the documentation
into html is handled by sphinx, we still maintain and provide access to the `doxygen generated documentation <http://sys-bio.github.io/roadrunner/OriginalDoxygenStyleDocs/html/index.html>`_.

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

.. code-block::

    -DSPHINX_EXECUTABLE=/full/path/to/sphinx/build

Make sure that this `sphinx-build` is from the same python version you are using to build roadrunner.
This is particularly important when building with conda. You can control which version
of Python is found by using ``-DPython_ROOT_DIR=/full/path/to/python/root``. For example, if you want
to use the python inside the conda environment here at ``/Users/ciaranwelsh/miniconda/envs/py38/bin/python``
then the Python executable is located at:

.. code-block::

      ``/Users/ciaranwelsh/miniconda/envs/py38/bin/python``

and its root is

.. code-block::

      ``/Users/ciaranwelsh/miniconda/envs/py38``















