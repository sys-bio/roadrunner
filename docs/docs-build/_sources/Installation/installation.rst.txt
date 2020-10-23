=====================
Installing RoadRunner
=====================

Python Front End
===================

The recommended way to use roadrunner is by installing the Python front end and writing Python scripts.

.. code-block:: bash

   $ pip install libroadrunner

.. todo:

    Find out whether we support conda package?

Prebuilt binaries (C/C++ Front End)
===================================

If you want to use roadrunner from C++ or C, you may be able to use a prebuilt binary distribution
if one is suitable. To determine suitability, ensure your compiler, architecture (i.e. x86 vs x64)
and build configuration (i.e. Release or Debug) matches what was used to build the binaries.

.. todo::

    Put roadrunner binaries on sourceforce and update the links below

* `VisualStudio 2019 x64 release <>`_
* `VisualStudio 2019 x64 debug<>`_
* `GCC7.5.0 x64 release<>`_
* `GCC7.5.0 x64 debug<>`_
* `AppleClang11.0.3 darwin19.4.0 x86-64 release<>`_
* `AppleClang11.0.3 darwin19.4.0 x86-64 debug<>`_

If you cannot use the prebuilt binaries or want to built from source please see the following instructions.

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

.. note::

    If at any point you get stuck or encounter an error please first read the :ref:`Troubleshooting the build <Troubleshooting the Build>`
    page and then if the problem is not resolved, submit a github issue and we'll be more than willing to help.

On windows, you should use the "x64 Native Tools Command Prompt for VS 2019" shell which you can find by
opening the start menu and begin typing "x64 Native Tools Command Prompt for VS 2019". This will
ensure you are building x64 bit application.

.. warning::

    Failure to use the "X64 Native Tools Command Prompt for VS 2019" shell may result in 32-bit binaries.

1) Install LLVM
---------------

Install the :ref:`llvm-6.x dependency<LLVM-6.x dependency>`.

.. note::

    For those wanting to build roadrunner in Debug mode, remember that you will need to download or build the llvm
    debug binaries, not release.

2) Install the Roadrunner Dependency Package
---------------------------------------------

Install the `roadrunner dependency package <https://github.com/CiaranWelsh/roadrunner-deps>`_ by executing the following commands in a windows/linux/mac shell.

.. note::

    In the following set of instructions you can replace every instance of the word "Release" with the word "Debug"
    (`or another configuration <https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html>`_) to build
    roadrunner in Debug mode.

.. code-block:: bash

    git clone https://github.com/CiaranWelsh/roadrunner-deps.git --recurse-submodules 	# get the dependency package using git
    cd roadrunner-deps
    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX="../install-Release" -DCMAKE_BUILD_TYPE="Release" .. # configure the dependency package
    cmake --build . --target install --config Release # build the install target
    cd ../../ # return to directory containing roadrunner-deps

.. note::

    Take note of where you put the roadrunner deps install tree (`../install-Release` here). It is required as
    argument to `-DRR_DEPENDENCIES_INSTALL_PREFIX` below.

3) Install Roadrunner
----------------------

Install roadrunner using the following commands from a windows/linux/mac shell.

If you have not done so already, :ref:`download or build llvm-6.x <LLVM-6.x dependency>`. The folder containing
llvm's `bin`, `include` and `lib` directories is the argument to `LLVM_INSTALL_PREFIX` below.

.. code-block:: bash

    .. note::

        The following `cmake` command assumes that you are following these instruction exactly so the relative
        path `-DRR_DEPENDENCIES_INSTALL_PREFIX="../../roadrunner-deps/install-Release"` will point to
        the `-DCMAKE_INSTALL_PREFIX` argument from step 2, i.e. the roadrunner-deps cmake command.

    git clone https://github.com/CiaranWelsh/roadrunner.git # get roadrunner
    cd roadrunner
    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX="../install-Release" \
        -DLLVM_INSTALL_PREFIX="/full/path/to/where/you/put/llvm-6.x-release" \
        -DRR_DEPENDENCIES_INSTALL_PREFIX="../../roadrunner-deps/install-Release" \
        -DCMAKE_BUILD_TYPE="Release" ..
    cmake --build . --target install --config Release

Roadrunner Optional Features
=============================

These are controlled by turning on or off desired options in cmake.

.. note::

    Generally speaking, the cmake-gui tool
    is the best way to get a good oversight of the available options.

Build the Python Bindings
-------------------------

    1) Download `SWIG version 3.0.0 <https://sourceforge.net/projects/swig/files/>`_ for linux,
    macos or windows. It is just a zip file - decompress it and put it where you want it. SWIG is now installed and
    the swig executable is under the top level swig directory.

    2) Configure or reconfigure cmake using the `-DBUILD_PYTHON=ON` option. If you installed swig
    globally or add the swig directory to the PATH environment variable, you will *not* need
    the `-DSWIG_EXECUTABLE` argument as `swig.exe` will be found
    automatically. However in most circumstances you will need to provide the path to the `swig.exe` using
    `-DSWIG_EXECUTABLE`.

.. code-block:: bash

    cmake -DCMAKE_INSTALL_PREFIX="D:\roadrunner\install-msvc2019" \
        -DLLVM_INSTALL_PREFIX="D:\llvm-6.x\llvm\llvm-6.x-msvc-x64-release" \
        -DRR_DEPENDENCIES_INSTALL_PREFIX="D:\roadrunner-deps\roadrunner-deps-install" \
        -DCMAKE_BUILD_TYPE="Release" \
        -DBUILD_PYTHON=ON
        -DSWIG_EXECUTABLE="C:\Users\Ciaran\Documents\swigwin-3.0.0\swig.exe" ..
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
        -DLLVM_INSTALL_PREFIX="D:\llvm-6.x\llvm\llvm-6.x-msvc-x64-release" \
        -DRR_DEPENDENCIES_INSTALL_PREFIX="D:\roadrunner-deps\roadrunner-deps-install" \
        -DCMAKE_BUILD_TYPE="Release" \
        -DBUILD_RR_PLUGINS=ON
    $ cmake --build . --target install --config Release


Building roadrunner tests
--------------------------

Use the `-DBUILD_TESTS=ON` option

.. code-block:: bash

    $ cmake -DCMAKE_INSTALL_PREFIX="D:\roadrunner\install-msvc2019" \
        -DLLVM_INSTALL_PREFIX="D:\llvm-6.x\llvm\llvm-6.x-msvc-x64-release" \
        -DRR_DEPENDENCIES_INSTALL_PREFIX="D:\roadrunner-deps\roadrunner-deps-install" \
        -DCMAKE_BUILD_TYPE="Release" \
        -DBUILD_TESTS=ON
    $ cmake --build . --target install --config Release

You can run the tests using ctest on the command line, ensuring you are in the top level
of your build tree (D`:\roadrunner\build` here).

.. code-block:: bash

    $ ctest .

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
        -DLLVM_INSTALL_PREFIX="D:\llvm-6.x\llvm\llvm-6.x-msvc-x64-release" \
        -DRR_DEPENDENCIES_INSTALL_PREFIX="D:\roadrunner-deps\roadrunner-deps-install" \
        -DCMAKE_BUILD_TYPE="Release" \
        -DBUILD_PACKAGING=ON
    $ cmake --build . --target install --config Release
    $ cmake --build . --target packaging --config Release





