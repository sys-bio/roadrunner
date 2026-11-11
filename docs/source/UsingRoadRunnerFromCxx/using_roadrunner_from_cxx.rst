Using RoadRunner from C++
============================
Despite RoadRunner's user interface being primarily in Python,
RoadRunner is first and foremost a C++ library. It is therefore is possible to
link roadrunner into an executable or library.

Requirements
------------

- RoadRunner installed into a directory on your system. See :ref:`Installing RoadRunner`.
  The install folder also holds the libraries RoadRunner depends on (zlib, libSBML, Poco,
  SUNDIALS and others), so they do not have to be installed separately.
- The LLVM that RoadRunner was built with. It is not part of the RoadRunner install folder,
  so its location is given with ``-DLLVM_INSTALL_PREFIX``. See :ref:`LLVM dependency`.
- `CMake <https://cmake.org/download/>`_ 3.20 or newer, with its "bin" directory in the PATH
  environment variable. You will know this is done if the command `cmake --help` works on the
  terminal/powershell.
- A C++17 compiler. On Windows, use the Visual Studio that RoadRunner was built with, and build
  your program in the same configuration (Debug or Release) as the RoadRunner install.

.. note::

    On Windows, the LLVM package needs the "DIA SDK" that comes with the "Desktop development
    with C++" workload of Visual Studio. It is found automatically for the Visual Studio in
    use. If CMake reports that it was not found, set ``-DMSVC_DIA_SDK_DIR`` to the "DIA SDK"
    folder of your Visual Studio installation, or run CMake from a Visual Studio command prompt.

A first program
---------------
The following is a very simple example of how to link
roadrunner into an executable and run a simulation from C++.
Create a new folder with whatever name seems appropriate and
create the following files:

.. code-block:: bash

    .
    ├── CMakeLists.txt
    └── main.cpp

The `CMakeLists.txt` file should contain the following:

.. literalinclude:: CMakeLists.txt
    :linenos:
    :language: cmake

Now you can use RoadRunner from C++.

.. literalinclude:: main.cpp
    :linenos:
    :language: C++

Building and running it
-----------------------
From the new folder, configure, build and run the program. Replace the two paths with the folder
RoadRunner was installed to and the folder LLVM was installed to, and use `Release` or `Debug` to
match your RoadRunner install:

.. code-block:: bash

    cd [new folder]
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DROADRUNNER_INSTALL_PREFIX=/path/to/roadrunner-install -DLLVM_INSTALL_PREFIX=/path/to/llvm
    cmake --build build --config Release
    ./build/UseRoadRunnerFromCxx

On Windows, the program is in a folder named after the configuration, so the last line is
`build\Release\UseRoadRunnerFromCxx.exe`. The `-DCMAKE_BUILD_TYPE` option has no effect with the
Visual Studio generator, which takes the configuration from `--config`.

If your program links one of the shared libraries (`roadrunner::roadrunner` or
`roadrunner::roadrunner_c_api`), Windows also needs the `bin` folder of the RoadRunner install in
the PATH environment variable when the program runs.

Targets
-------
The `roadrunner` package provides the following targets. The include directories and all of the libraries the target needs comes with it.

- `roadrunner::roadrunner`              : the shared C++ library.
- `roadrunner::roadrunner-static`       : the static C++ library.
- `roadrunner::roadrunner_c_api`        : the shared C API.
- `roadrunner::roadrunner_c_api-static` : the static C API.

RoadRunner plugins
------------------
If RoadRunner was built with the plugins (`BUILD_RR_PLUGINS`), the install folder also holds a second
CMake package, `rrplugins`. The `CMakeLists.txt` above looks for it as well, and if it is there it also
builds a second program, `UseRRPluginsFromCxx`, from the following file. The program loads the plugins and lists them, so there is nothing to change
in `CMakeLists.txt`.

.. literalinclude:: plugins_main.cpp
    :linenos:
    :language: C++

Build and run it with the same commands as above, with the same two `-D` options. The last line is
`./build/UseRRPluginsFromCxx`, or `build\Release\UseRRPluginsFromCxx.exe` on Windows.

The `rrplugins` package includes `roadrunner` itself, so a program that only uses the plugins does not need its own
`find_package(roadrunner ...)` line, just `find_package(rrplugins CONFIG REQUIRED)`.

The `rrplugins` package provides the following targets. The include directories and all of the libraries
the target needs come with it, including `roadrunner`.

- `rrplugins::telplugins_core-static`        : the static C++ plugin library: the `PluginManager` that loads plugins, and the plugin API. Includes the two libraries below.
- `rrplugins::telplugins_common-static`      : the static library of classes shared by the others: `TelluriumData`, properties, logging, and string and file utilities.
- `rrplugins::telpluginsBaseClass-static`    : the static library holding the base classes for writing a plugin (`Plugin`, `CPlugin`, `CPPPlugin`).
- `rrplugins::telplugins_math-static`        : the static math library: matrices, random numbers and other numerical helpers.
- `rrplugins::telplugins_c_api`              : the shared C API.

The plugins themselves (`tel_add_noise`, `tel_auto2000`, `tel_chisquare`, `tel_hello`,
`tel_levenberg_marquardt`, `tel_monte_carlo_bs`, `tel_nelder_mead` and `tel_test_model`) are not CMake
targets, and there is no target per plugin. They are modules that the plugin manager loads while the program runs, so
nothing links to them. They are installed in the `lib` folder of the RoadRunner install (the `bin` folder on
Windows). A `PluginManager` created with that folder (`tpCreatePluginManager` in the C API) loads them, as the program above does.

The older way
-------------
Earlier releases were found with ``include(ImportRoadrunnerAndDependencies)`` or
``find_package(roadrunner-static)``. Both still work, and print a deprecation message, but will be
removed in a future release. Use `find_package(roadrunner CONFIG REQUIRED)` and the `roadrunner::`
targets above instead.

.. note::

    The example on this page and the commands above are built and run on every supported
    platform by the `test_cmake_consumer` job of the continuous integration workflow, using
    `test/cmake_consumers/run_consumers.sh`.
