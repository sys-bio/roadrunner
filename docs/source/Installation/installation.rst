=====================
Installing RoadRunner
=====================

Python Front End
===================

The recommended way to use roadrunner is by installing the Python front end and writing Python scripts, for Python 3.10 or later:

.. code-block:: bash

   $ pip install libroadrunner

Installing Roadrunner from Precompiled Binaries
===============================================

Precompiled binaries for many platforms are available at https://github.com/sys-bio/roadrunner/releases

To use, see the tested installation and use script at https://github.com/sys-bio/roadrunner/tree/develop/test/cmake_consumers

If you prefer to build roadrunner from source, `scripts/build_roadrunner.sh <https://github.com/sys-bio/roadrunner/blob/develop/scripts/build_roadrunner.sh>`__ does all of the steps described below for you: it downloads the prebuilt LLVM and dependency binaries for your platform, then configures, builds and installs roadrunner. Clone the roadrunner repository and run the script from inside it:

.. code-block:: bash

    git clone https://github.com/sys-bio/roadrunner.git
    cd roadrunner
    bash scripts/build_roadrunner.sh -c

Everything it creates goes into a `roadrunner-build` folder in the current directory (the installation is in `roadrunner-build/install`), and `-c` makes it finish by running the tests in `test/cmake_consumers` against that installation. The comments at the top of the script describe the other options, such as `-t Debug` and `-w` to choose the folder. On Windows, run it with bash (such as Git Bash) from the "x64 Native Tools Command Prompt".


Building Roadrunner from Source
==================================


.. toctree::
    :maxdepth: 1

    windows_warning
    llvm_dependency

You will need:

- `CMake <https://cmake.org/download/>`_ version 3.20 or newer.
- `Visual studio <https://visualstudio.microsoft.com/downloads/>`_ (windows only) with the C++ package installed.
- Linux only: a C++17 compiler, either GCC (`g++`) or Clang (`clang++`). The manylinux builds use GCC 11.
- macOS only: the Xcode command line tools, which provide the Apple Clang compiler (install them with `xcode-select --install`).
- `git <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_
- Linux and macOS: `make` or `Ninja <https://ninja-build.org/>`_ to run the build. The build script uses Ninja if it finds it, and otherwise `make`, which comes with the Xcode command line tools on macOS.
- `curl` and `unzip`, which the build script uses to download and unpack the prebuilt LLVM and dependency binaries. macOS has both, and so does Git Bash on Windows.

If you are new to CMake, we recommend you read the `cmake tutorial <https://cmake.org/cmake/help/latest/guide/tutorial/index.html>`_ to get the basics before continuing.

The following instructions apply to Windows, Mac OS and Linux. Where they diverge we will provide the specific platform instruction. 

.. note::

    If at any point you get stuck or encounter an error, please submit a github issue and we'll be more than willing to help.

On windows, you should use the "x64 Native Tools Command Prompt for VS 20xx" shell which you can find by opening the start menu and begin typing "x64 Native Tools Command Prompt". This will ensure you are building x64 bit application.

If you want to use new `Windows Terminal <https://www.microsoft.com/en-us/p/windows-terminal/9n0dx20hk701?activetab=pivot:overviewtab>`_ you can set up the same environment by editing the windows terminal settings file: `Add Developer Command Prompt <https://stackoverflow.com/questions/57925428/add-developer-command-prompt-for-visual-studio-to-windows-terminal>`_

.. warning::

    Failure to use the "X64 Native Tools Command Prompt for VS 20xx" shell may result in 32-bit binaries.

1) Where to put Everything
--------------------------

On Windows it is recommended that you place all the files as close to a root on a drive as possible. This is because Windows has a 260 char limit on the length of a path which can be easily exceeded if you have a deep folder structure. One possibility is this. If for example you have a D drive (works equally well with the C drive), create a folder:

.. code-block:: bash

    cd d:/
    mkdir buildroadrunner  # You can call this whatever you like
    cd buildroadrunner

Everything we need will go into buildroadrunner.  See `scripts/versions.env <https://github.com/sys-bio/roadrunner/blob/develop/scripts/versions.env>`__ for the version of LLVM and libroadrunner-deps known to be compatible with your version of roadrunner.

2) Install LLVM
---------------

Install the `LLVM dependency <https://github.com/sys-bio/llvm-build/releases>`__ for your system.

.. note::

    For those wanting to build roadrunner in Debug mode on Windows, remember that you will need to download or build the llvm debug binaries, not release.  Downloading the debug binaries is optional on other systems.

The precompiled LLVM binaries are zip files that unpack directly into `bin`, `include` and `lib` folders,
so extract the one you want into a folder, for example llvm-23.1.4-release/ (or llvm-23.1.4-debug/ for the Debug binaries)

Place these folders into your top level directory we created in step 1) called buildroadrunner.


Alternatively, you can build LLVM from source.  Follow the build instructions for LLVM for your platform, and use the configuration flags found at `sys-bio/llvm-build <https://github.com/sys-bio/llvm-build/blob/main/config/llvm-flags.cmake>`__.  In particular, RTTI and exceptions must be on.

3) Install the Roadrunner Dependency Package
---------------------------------------------

Pre-built binaries are available for most platforms from `sys-bio/libroadrunner-deps <https://github.com/sys-bio/libroadrunner-deps/releases>`__.  
Again, to build a Windows debug version of roadrunner will require the Windows debug binaries; on other platforms, you may use either binaries.  Extract the binaries into libroadrunner-deps/install-release/ (or libroadrunner-deps/install-debug/ for Debug).  Again, there will be `bin`, `include`, and `lib` subfolders (or `lib64` on manylinux).

For convenience, place the libroadrunner-deps folder in the top level directory we created in step 1) called buildroadrunner.

Alternatively, you can again build all the libraries from source from https://github.com/sys-bio/libroadrunner-deps

To do that, use the commands below. If you downloaded the prebuilt binaries, skip ahead to step 4.

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
    cmake -DCMAKE_INSTALL_PREFIX="../install-release" -DCMAKE_BUILD_TYPE="Release" .. # configure the dependency package
    cmake --build . --target install --config Release # build the install target
    cd ../../ # return to directory containing libroadrunner-deps

.. note::

    Take note of where you put the roadrunner deps install tree (`../install-release` here). It is required as
    argument to `-DRR_DEPENDENCIES_INSTALL_PREFIX` below.

At this point you will have a folder called buildroadrunner and inside that you'll have
at least two folders, one for the LLVM binaries (two of these if you downloaded the release and debug versions) and the folder libroadrunner-deps. This is shown below.

buildroadrunner
|  --- llvm-23.1.4-release
|  --- libroadrunner-deps


4) Install Roadrunner
----------------------

Install roadrunner using the following commands from a windows/linux/mac shell.

If you have not done so already, :ref:`download or build LLVM <LLVM dependency>`. The folder containing llvm's `bin`, `include` and `lib` directories is the argument to `LLVM_INSTALL_PREFIX` below.

.. note::

    The following `cmake` command assumes that you are following these instruction exactly so the relative
    path `-DRR_DEPENDENCIES_INSTALL_PREFIX="../../libroadrunner-deps/install-release"` will point to
    the `-DCMAKE_INSTALL_PREFIX` argument from step 3, i.e. the libroadrunner-deps cmake command.

.. code-block:: bash

    git clone https://github.com/sys-bio/roadrunner.git # get roadrunner
    cd roadrunner
    mkdir build-release
    cd build-release
    cmake -DCMAKE_INSTALL_PREFIX="../install-release" \
        -DLLVM_INSTALL_PREFIX="/full/path/to/where/you/put/llvm-23.1.4-release" \
        -DRR_DEPENDENCIES_INSTALL_PREFIX="../../libroadrunner-deps/install-release" \
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

    1) Download `SWIG version 4.4.1 <https://sourceforge.net/projects/swig/files/>`_ (or later) for your platform.  (You can also use `pip install swig>=4.4.1` if this is more convenient.)

    2) Configure or reconfigure cmake using the `-DBUILD_PYTHON=ON` option. If you installed swig
    globally or add the swig directory to the PATH environment variable, you will *not* need
    the `-DSWIG_EXECUTABLE` argument as `swig.exe` will be found
    automatically. However in most circumstances you will need to provide the path to the `swig.exe` using
    `-DSWIG_EXECUTABLE`. As per the instructions for `FindPython <https://cmake.org/cmake/help/git-stage/module/FindPython.html>`__
    you can build for specific Python environments by pointing -DPython_ROOT_DIR to the root of your Python installation.

.. code-block:: bash

    cmake -DCMAKE_INSTALL_PREFIX="../install-release" \
        -DLLVM_INSTALL_PREFIX="/full/path/to/where/you/put/llvm-23.1.4-release" \
        -DRR_DEPENDENCIES_INSTALL_PREFIX="../../libroadrunner-deps/install-release" \
        -DCMAKE_BUILD_TYPE="Release" \
        -DBUILD_PYTHON=ON \
        -DPython_ROOT_DIR="/path/to/miniconda/env/root" \
        -DSWIG_EXECUTABLE="path/to/swig4.x/executable/swig.exe" ..
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

    $ cmake -DCMAKE_INSTALL_PREFIX="../install-release" \
        -DLLVM_INSTALL_PREFIX="/full/path/to/where/you/put/llvm-23.1.4-release" \
        -DRR_DEPENDENCIES_INSTALL_PREFIX="../../libroadrunner-deps/install-release" \
        -DCMAKE_BUILD_TYPE="Release" \
        -DBUILD_RR_PLUGINS=ON \
        ..
    $ cmake --build . --target install --config Release


Building roadrunner tests
--------------------------

If you want the tests compiled you need to add an extra option to the cmake command that was used in step 4 but is repeated here for convenience.  

Use the `-DBUILD_TESTS=ON` option. Remember that in the following script to change Release to Debug if you are creating the debug version. 

.. code-block:: bash

    $ cmake -DCMAKE_INSTALL_PREFIX="../install-release" \
        -DLLVM_INSTALL_PREFIX="/full/path/to/where/you/put/llvm-23.1.4-release" \
        -DRR_DEPENDENCIES_INSTALL_PREFIX="../../libroadrunner-deps/install-release" \
        -DCMAKE_BUILD_TYPE="Release" \
        -DBUILD_TESTS=ON ..
    $ cmake --build . --target install --config Release

You can run the tests using ctest on the command line, ensuring you are in the top level
of your build tree (e.g. `D:\\buildroadrunner\\roadrunner\\build-release` here).

.. code-block:: bash

    $ ctest .

This will give you a short summary of the results of the tests. If you want detailed information you need to enter instead:

.. code-block:: bash

    $ ctest --verbose .
    
Alternatively you can build the `run-ctest` target, which is more convenient
when you want to run the tests from within an IDE, since you only have to
build that one target. It first builds any test programs that are out of date,
and then runs ctest on your build tree with the options `--output-on-failure --parallel 8 --verbose`.

.. code-block:: bash

    $ cmake --build . --target run-ctest

The targets `run-ctest-exclude-sbml-test-suite` and `run-ctest-only-sbml-test-suite` do the same for
everything except the SBML test suite, and for the SBML test suite alone.


Packaging Roadrunner
------------------------

* Windows: produces a zip file.

* Linux: produces tar.gz and a debian package.

* Macos: produces a tar.gz file.


First follow the instructions for installing roadrunner, including options for
any other desired options such as tests, plugins or language bindings. Then rerun
cmake using using the `-DBUILD_PACKAGING=ON` option.

.. code-block::

    $ cmake -DCMAKE_INSTALL_PREFIX="../install-release" \
        -DLLVM_INSTALL_PREFIX="/full/path/to/where/you/put/llvm-23.1.4-release" \
        -DRR_DEPENDENCIES_INSTALL_PREFIX="../../libroadrunner-deps/install-release" \
        -DCMAKE_BUILD_TYPE="Release" \
        -DBUILD_PACKAGING=ON \
         ..
    $ cmake --build . --target install --config Release
    $ cmake --build . --target packaging --config Release


Checking that everything worked
-------------------------------

There's a test script `run_consumers.sh <https://github.com/sys-bio/roadrunner/blob/develop/test/cmake_consumers/run_consumers.sh>`__ that will build test programs using your installation. Give it the folder roadrunner was installed to, the folder LLVM was installed to, and the build type (on Windows, run it with bash, such as Git Bash):

.. code-block:: bash

    $ bash test/cmake_consumers/run_consumers.sh /full/path/to/roadrunner/install-release /full/path/to/where/you/put/llvm-23.1.4-release Release

.. note::

    The test programs include ones that use the roadrunner plugins, so roadrunner must have been built with the plugins
    (`-DBUILD_RR_PLUGINS=ON`, see `Building the Roadrunner Plugins`_) for the check to pass.
