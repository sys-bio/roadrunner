# RoadRunner Julia bindings

A pure-Julia package that calls the roadrunner C API. It includes the [Antimony](https://github.com/sys-bio/antimony) Julia bindings as `RoadRunner.Antimony`, so models can be loaded from Antimony strings with `RoadRunner.loada`.

The bindings began as [RoadRunner.jl](https://github.com/sys-bio/RoadRunner.jl) by Jin Xu and Adel Heydarabadipour (Copyright (c) 2020 Jin Xu, MIT license, see `NOTICE.txt`).

## Installing a prebuilt package

1. Download the `roadrunner-julia-<platform>` asset for your system from the roadrunner releases page: `windows-latest`, `macos-14` (Apple silicon), `macos-15-intel`, `manylinux_2_28` or `ubuntu-latest`. The package is platform-specific because it carries the libraries.
2. Unzip it to get a `RoadRunner` folder, and leave that folder where you want to use it from.
3. In Julia (1.6 or newer):

```julia
using Pkg
Pkg.develop(path="/path/to/RoadRunner")

using RoadRunner
rr = RoadRunner.loada("S1 -> S2; k1*S1; S1 = 10; S2 = 0; k1 = 0.5")
RoadRunner.getFloatingSpeciesIds(rr)
```

## Building from source

Configure roadrunner with `-DBUILD_JULIA=ON`. This needs the Antimony Julia package, which CMake downloads from the latest Antimony release (set `ANTIMONY_JULIA_VERSION` to a version such as `3.3.0` to use a specific release). The download is kept in the build tree; delete its `antimony-julia-download` folder to fetch a newer one. To use a copy you already have, set `ANTIMONY_JULIA_PACKAGE_DIR` to its `Antimony` folder (for example `bindings/julia/Antimony` in an Antimony install built with `-DWITH_JULIA=ON`).

After installing, the package is in `<prefix>/bindings/julia/RoadRunner`, with the C API library in its `lib` folder and the Antimony package in its `Antimony` folder, so the folder can be moved or zipped on its own. Use it with `Pkg.develop` as above.

To use a different copy of the C API library, set the `ROADRUNNER_C_API_LIBRARY` environment variable to its full path.

Tests are registered with CTest as `julia_roadrunner_bindings` (label `julia`) when `BUILD_TESTS` is on and `julia` is on the path.

On macOS the Julia tests need the C API library to be linked with `-DRR_C_API_HIDE_LLVM_SYMBOLS=ON`. Julia has its own LLVM, and without this the library's statically linked LLVM clashes with it and crashes. This option stops the roadrunner C++ tests from working, so build the C++ tests and the Julia package with separate settings, as the CI workflow does: build and test normally, then reconfigure with the option ON, build the `roadrunner_c_api` and `roadrunner_julia_package` targets, and run `ctest -L julia`.
