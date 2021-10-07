Build roadrunner using Python's debug build
--------------------------------------------

Get the CPython source code:

.. code-block:: bash

    git clone https://github.com/python/cpython.git

Windows
-------

Build CPython in debug mode:

.. code-block:: powerhsell

    .\PCbuild\build.bat -e -d -p x64

from the CPython source directory. This builds the Python binaries
but does not produce the layout needed to be useful. For this
we can use:

.. code-block:: powershell

    .\python.bat .\PC\layout\ -h

for example:

.. code-block:: powershell

    .\python.bat .\PC\layout\ -d -s . -b .\PCbuild\amd64\ -vv --copy install-cpython/x64/dbg --preset-default

Now build roadrunner like normal but passing ``-DPython_ROOT_DIR=D:\cpython\install-cpython\x64\dbg``












