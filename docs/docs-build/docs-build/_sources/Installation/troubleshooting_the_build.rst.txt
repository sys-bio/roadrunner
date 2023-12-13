Troubleshooting the Build
==========================

#. Cannot find CMake

If you have downloaded or installed cmake but cmake doesn't run in the terminal/powershell it probably means that
you have not configured your `PATH` environment variables correctly. The PATH variable exists on all three major
operating systems. Do a google search for "Add to path <operating system>" replacing the contents of "<operating system>"
with your operating system and follow the instructions of the first hit you find to add the cmake
`bin` directory to the PATH environment variable.

.. code-block:: bash

    .
    ├── bin                 <--------------- You want to add this directory to the PATH environment variable.
    │       ├── cmake-gui.exe
    │       ├── cmake.exe
    │       ├── cmcldeps.exe
    │       ├── cpack.exe
    │       └── ctest.exe
    ├── doc
    │       └── cmake
    ├── man
    │       ├── man1
    │       └── man7
    └── share
        ├── aclocal
        └── cmake-3.17













