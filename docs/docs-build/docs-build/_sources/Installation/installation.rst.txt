=====================
Installing RoadRunner
=====================

Python Front End
===================

The recommended way to use roadrunner is by installing the Python front end and writing Python scripts.

.. code-block:: bash

   $ pip install libroadrunner


Building Roadrunner from Source
==================================


.. toctree::
    :maxdepth: 1

    windows_warning
    llvm_dependency
    troubleshooting_the_build

You will need:

- `CMake <https://cmake.org/download/>`_ version > 3.14. The build system was developed with version 3.18.4.
- `Visual studio <https://visualstudio.microsoft.com/downloads/>`_ (windows only) with the C++ package installed.
- `git <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_

If you are new to CMake, we recommend you read the `cmake tutorial <https://cmake.org/cmake/help/latest/guide/tutorial/index.html>`_ to get the basics before continuing.

The following instructions apply to Windwos, Mac OS and Linux. Where they diverge we will provide the specfic platform instruction. 

.. note::

    If at any point you get stuck or encounter an error please first read the :ref:`Troubleshooting the build <Troubleshooting the Build>`
    page and then if the problem is not resolved, submit a github issue and we'll be more than willing to help.

On windows, you should use the "x64 Native Tools Command Prompt for VS 2019" shell which you can find by opening the start menu and begin typing "x64 Native Tools Command Prompt for VS 2019". This will ensure you are building x64 bit application.

If you want to use new `Windows Terminal <https://www.microsoft.com/en-us/p/windows-terminal/9n0dx20hk701?activetab=pivot:overviewtab>`_ you can set up the same environment by editing the windows terminal settings file: `Add Developer Command Prompt <https://stackoverflow.com/questions/57925428/add-developer-command-prompt-for-visual-studio-to-windows-terminal>`_

.. warning::

    Failure to use the "X64 Native Tools Command Prompt for VS 2019" shell may result in 32-bit binaries.

1) Where to put Everything
--------------------------

On Windows it is recommended that you place all the files as close to a root on a drive as possible. This is because Windows has a 260 char limit on the length of a path which can be easily exceeded if you have a deep folder structure. One possibility is this. If for example you have a D drive (works equally well with the C drive), create a folder:

.. code-block:: bash

    cd d:/
    mkdir buildroadrunner  # You can call this whatever you like
    cd buildroadrunner

Everything we need will go into buildroadrunner.

2) Install LLVM
---------------

Install the :ref:`llvm-13.x dependency<LLVM-13.x dependency>`.

.. note::

    For those wanting to build roadrunner in Debug mode, remember that you will need to download or build the llvm debug binaries, not release.

If you download the precompiled binaries for LLVM you get a top level folder called something
like llvm-13.x-msvc2022-x64-release or llvm-13.x-msvc2022-x64-debug.

Place this folder (which contains all the LLVM code) into your top level directory we crated in step 1) called buildroadrunner.

3) Install the Roadrunner Dependency Package
---------------------------------------------

Install the `roadrunner dependency package <https://github.com/sys-bio/libroadrunner-deps>`_

See below for the script to use.

.. note::

    In the following set of instructions you can replace every instance of the word "Release" with the word "Debug"
    (`or another configuration <https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html>`_) to build
    roadrunner in Debug mode.

Execute these commands in a windows/linux/mac shell. 

Note that the first thing we do is
cd to the root folder we created in Step 1, namely buildroadrunner. 


.. code-block:: bash

    cd d:/buildroadrunner
    git clone https://github.com/sys-bio/libroadrunner-deps.git --recurse-submodules 	# get the dependency package using git
    cd libroadrunner-deps
    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX="../install-Release" -DCMAKE_BUILD_TYPE="Release" .. # configure the dependency package
    cmake --build . --target install --config Release # build the install target
    cd ../../ # return to directory containing libroadrunner-deps

.. note::

    Take note of where you put the roadrunner deps install tree (`../install-Release` here). It is required as
    argument to `-DRR_DEPENDENCIES_INSTALL_PREFIX` below.

At this point you will have a folder called buildroadrunner and inside that you'll have
at least two folders, one for the LLVM binaries (two of these if you downloaded the release and debug versions) and the folder libroadrunner-deps. This is shown below.

buildroadrunner
|  --- llvm-13.x-msvc2022-x64-debug
|  --- llvm-13.x-msvc2022-x64-debug
|  --- libroadrunner-deps

4) Install Roadrunner
----------------------

Install roadrunner using the following commands from a windows/linux/mac shell.

If you have not done so already, :ref:`download or build llvm-13.x <LLVM-13.x dependency>`. The folder containing llvm's `bin`, `include` and `lib` directories is the argument to `LLVM_INSTALL_PREFIX` below.

.. code-block:: bash

    .. note::

        The following `cmake` command assumes that you are following these instruction exactly so the relative
        path `-DRR_DEPENDENCIES_INSTALL_PREFIX="../../libroadrunner-deps/install-Release"` will point to
        the `-DCMAKE_INSTALL_PREFIX` argument from step 2, i.e. the libroadrunner-deps cmake command.

    git clone https://github.com/sys-bio/roadrunner.git # get roadrunner
    cd roadrunner
    mkdir build-release
    cd build-release
    cmake -DCMAKE_INSTALL_PREFIX="../install-Release" \
        -DLLVM_INSTALL_PREFIX="/full/path/to/where/you/put/llvm-13.x-release" \
        -DRR_DEPENDENCIES_INSTALL_PREFIX="../../libroadrunner-deps/install-Release" \
        -DCMAKE_BUILD_TYPE="Release" ..
    cmake --build . --target install --config Release

The first cmake line above is quite long and you won't be able to copy and paste it directly into the windows terminal. To avoid this, copy the long cmake line into an editor (such as notepad) and make it one line long. Then copy and paste into the Windows terminal.

Note: if you want to create a debug version, just swap instances of the phrase Release (or release) for Debug (or debug).

Roadrunner Optional Features
=============================

These are controlled by turning on or off desired options in cmake.

.. note::

    Generally speaking, the `cmake-gui <https://cmake.org/cmake/help/latest/manual/cmake-gui.1.html>`_  tool
    is the best way to get a good oversight of the available options.

Build the Python Bindings
-------------------------

    1) Download `SWIG version 3.0.0 <https://sourceforge.net/projects/swig/files/>`_ for linux, macos or windows. It is just a zip file - decompress it and put it where you want it. SWIG is now installed and the swig executable is under the top level swig directory.

    2) Configure or reconfigure cmake using the `-DBUILD_PYTHON=ON` option. If you installed swig
    globally or add the swig directory to the PATH environment variable, you will *not* need
    the `-DSWIG_EXECUTABLE` argument as `swig.exe` will be found
    automatically. However in most circumstances you will need to provide the path to the `swig.exe` using
    `-DSWIG_EXECUTABLE`. As per the instructions for [FindPython](https://cmake.org/cmake/help/git-stage/module/FindPython.html)
    you can build for specific Python environments by pointing -DPython_ROOT_DIR to the root of your Python installation. Using Conda
    environments and this variable makes it easy to build Python wheels for different Python versions.

.. code-block:: bash

    cmake -DCMAKE_INSTALL_PREFIX="D:\roadrunner\install-msvc2019" \
        -DLLVM_INSTALL_PREFIX="/full/path/to/where/you/put/llvm-13.x-release" \
        -DRR_DEPENDENCIES_INSTALL_PREFIX="D:\libroadrunner-deps\libroadrunner-deps-install" \
        -DCMAKE_BUILD_TYPE="Release" \
        -DBUILD_PYTHON=ON \
        -DPython_ROOT_DIR="/path/to/miniconda/env/root"
        -DSWIG_EXECUTABLE="path/to/swig3.0.0/executable/swig.exe" ..
    cmake --build . --target install --config Release


Building the Roadrunner Plugins
-------------------------------

To build the plugins use the `-DBUILD_RR_PLUGINS=ON`. When set to on, the `BUILD_RR_PLUGINS`
option automatically turns on a set of variables. These include:

    * RR_PLUGINS_BUILD_SHARED_LIB
    * RR_PLUGINS_BUILD_C_API
    * RR_PLUGINS_BUILD_PYTHON_API
    * RR_PLUGINS_BUILD_TEST_MODEL_PLUGIN
    * RR_PLUGINS_BUILD_ADD_NOISE_PLUGIN
    * RR_PLUGINS_BUILD_CHISQUARE_PLUGIN
    * RR_PLUGINS_BUILD_AUTO2000_PLUGIN
    * RR_PLUGINS_BUILD_HELLO_PLUGIN

while the `RR_PLUGINS_BUILD_STATIC_LIB` option remains OFF.

.. code-block:: bash

    $ cmake -DCMAKE_INSTALL_PREFIX="D:\roadrunner\install-msvc2019" \
        -DLLVM_INSTALL_PREFIX="/full/path/to/where/you/put/llvm-13.x-release" \
        -DRR_DEPENDENCIES_INSTALL_PREFIX="D:\libroadrunner-deps\libroadrunner-deps-install" \
        -DCMAKE_BUILD_TYPE="Release" \
        -DBUILD_RR_PLUGINS=ON
    $ cmake --build . --target install --config Release


Building roadrunner tests
--------------------------

I you want the tests compiled you need to add an extra option to the cmake command that was used in step 4 but is repeated here for convenience.  

Use the `-DBUILD_TESTS=ON` option. Remember that in the following script to change Release to Debug is you are creating the debug version. 

.. code-block:: bash

    $ cmake -DCMAKE_INSTALL_PREFIX="../install-Release" \
        -DLLVM_INSTALL_PREFIX="/full/path/to/where/you/put/llvm-13.x-release" \
        -DRR_DEPENDENCIES_INSTALL_PREFIX="../../libroadrunner-deps/install-Release" \
        -DCMAKE_BUILD_TYPE="Release" \
        -DBUILD_TESTS=ON ..
    $ cmake --build . --target install --config Release

You can run the tests using ctest on the command line, ensuring you are in the top level
of your build tree (e.g D`:\\buildroadrunner\\roadrunner\\build-debug` here).

.. code-block:: bash

    $ ctest .

This will give you a short summary of the results of the tests. If you want detailed information you need to enter instead:

.. code-block:: bash

    $ ctest --verbose .
    
Alternatively you can build the `ctest-run` target, which is more convenient
when you want to run the test from within an IDE since you use the mouse to
build the `ctest-run` target. In the background, this will run the following
command, which you may also run manually.

.. code-block:: bash

    $ cmake --build . --target ctest-run


Packaging Roadrunner
------------------------

* Windows: produces a zip file.

* Linux: produces tar.gz and a debian package.

* Macos: produces a tar.gz file.


First follow the instructions for installing roadrunner, including options for
any other desired options such as tests, plugins or language bindings. Then rerun
cmake using using the `-DBUILD_PACKAGING=ON` option.

.. code-block::

    $ cmake -DCMAKE_INSTALL_PREFIX="D:\roadrunner\install-msvc2019" \
        -DLLVM_INSTALL_PREFIX="/full/path/to/where/you/put/llvm-13.x-release" \
        -DRR_DEPENDENCIES_INSTALL_PREFIX="D:\libroadrunner-deps\libroadrunner-deps-install" \
        -DCMAKE_BUILD_TYPE="Release" \
        -DBUILD_PACKAGING=ON
    $ cmake --build . --target install --config Release
    $ cmake --build . --target packaging --config Release





