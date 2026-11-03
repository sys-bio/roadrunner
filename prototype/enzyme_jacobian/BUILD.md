# Building the Enzyme Jacobian prototype

Everything is built on linux (Ubuntu 24.04, gcc 13.3, cmake, `make -j20`) in a local workspace `~/rr-enzyme/`. Release builds throughout.

## Workspace

```bash
mkdir -p ~/rr-enzyme/src
# LLVM 23.1.4 and the other dependencies, from the libroadrunner-deps release archives
#   llvm-23.1.4-ubuntu-24-x86_64-Release.zip       -> ~/rr-enzyme/llvm
#   libroadrunner-deps-ubuntu-24-x86_64-Release.zip -> ~/rr-enzyme/deps
git -C ~/git/roadrunner worktree add ~/git/roadrunner-enzyme -b enzyme-jacobian origin/develop
```

The LLVM install is static (`llvm-config --shared-mode` = static), built with RTTI, and has no clang and no `opt`.

## Enzyme

Tag `v0.0.303` (commit `06da3b7`), built as a static library against the LLVM install above. The clang plugin and the bitcode loader need clang and are switched off.

```bash
git clone --depth 1 --branch v0.0.303 https://github.com/EnzymeAD/Enzyme.git ~/rr-enzyme/src/Enzyme
cmake -S ~/rr-enzyme/src/Enzyme/enzyme -B ~/rr-enzyme/build-enzyme \
  -DLLVM_DIR=$HOME/rr-enzyme/llvm/lib/cmake/llvm \
  -DENZYME_CLANG=OFF -DENZYME_STATIC_LIB=ON -DENZYME_BC_LOADER=OFF \
  -DCMAKE_BUILD_TYPE=Release
make -C ~/rr-enzyme/build-enzyme -j20 EnzymeStatic-23
# -> ~/rr-enzyme/build-enzyme/Enzyme/libEnzymeStatic-23.a
```

## SuiteSparse (KLU) and SUNDIALS with KLU

SuiteSparse `v7.14.1` (commit `e202a5b`, KLU 2.3.6), only the packages KLU needs, static and position independent. SUNDIALS `v7.9.0` (commit `312fc0f`, the version of the libroadrunner-deps archive) rebuilt with KLU and installed over the SUNDIALS of `~/rr-enzyme/deps` (a copy of the original is in `~/rr-enzyme/deps-orig`). Index size 64 and logging level 2 as in the archive. Ubuntu's own SuiteSparse in `/usr/lib` must not be picked up: SUNDIALS finds the static install through `CMAKE_PREFIX_PATH`, roadrunner through `RR_SUITESPARSE_PREFIX`.

```bash
git clone --depth 1 --branch v7.14.1 https://github.com/DrTimothyAldenDavis/SuiteSparse.git ~/rr-enzyme/src/SuiteSparse
cmake -S ~/rr-enzyme/src/SuiteSparse -B ~/rr-enzyme/build-suitesparse -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/rr-enzyme/suitesparse \
  -DSUITESPARSE_ENABLE_PROJECTS="suitesparse_config;amd;colamd;btf;klu" \
  -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DKLU_USE_CHOLMOD=OFF -DSUITESPARSE_USE_FORTRAN=OFF -DSUITESPARSE_USE_OPENMP=OFF \
  -DSUITESPARSE_DEMOS=OFF -DBUILD_TESTING=OFF
make -C ~/rr-enzyme/build-suitesparse -j20 install

git clone --depth 1 --branch v7.9.0 https://github.com/LLNL/sundials.git ~/rr-enzyme/src/sundials
cmake -S ~/rr-enzyme/src/sundials -B ~/rr-enzyme/build-sundials -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/rr-enzyme/deps -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_PREFIX_PATH=$HOME/rr-enzyme/suitesparse \
  -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DSUNDIALS_INDEX_SIZE=64 -DSUNDIALS_LOGGING_LEVEL=2 \
  -DBUILD_ARKODE=OFF -DBUILD_IDA=OFF -DBUILD_IDAS=OFF -DBUILD_CVODE=ON -DBUILD_CVODES=ON -DBUILD_KINSOL=ON \
  -DEXAMPLES_ENABLE_C=OFF -DEXAMPLES_INSTALL=OFF -DSUNDIALS_ENABLE_KLU=ON
make -C ~/rr-enzyme/build-sundials -j20 install
```

## roadrunner

```bash
cmake -S ~/git/roadrunner-enzyme -B ~/rr-enzyme/build \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON \
  -DLLVM_INSTALL_PREFIX=$HOME/rr-enzyme/llvm \
  -DRR_DEPENDENCIES_INSTALL_PREFIX=$HOME/rr-enzyme/deps \
  -DCMAKE_INSTALL_PREFIX=$HOME/rr-enzyme/install \
  -DRR_USE_ENZYME=ON \
  -DRR_ENZYME_LIBRARY=$HOME/rr-enzyme/build-enzyme/Enzyme/libEnzymeStatic-23.a \
  -DRR_USE_KLU=ON -DRR_SUITESPARSE_PREFIX=$HOME/rr-enzyme/suitesparse
make -C ~/rr-enzyme/build -j20
ctest --test-dir ~/rr-enzyme/build -j20
```

## Known test failures that are not caused by the prototype

With `ctest -j20`, about 90 `ModelEditingTests` and 20 `StateSavingTests` fail on `develop` as well (110 failures with the same dependencies and `develop` at `2876ab2ed`). They all pass with `ctest -j1`, so they interfere with each other when run in parallel.

## Go/no-go gate

Passed on 2026-11-03 with the tagged release `v0.0.303` and no patches to Enzyme: `EnzymeJacobianTests.GateHctzDirectionalDerivative` differentiates the generated `evalReactionRates` of `hctz_body_flat.xml` in roadrunner's JIT (MCJit) and matches the central difference. Enzyme needed IR bodies for the helpers `rr_min` and `rr_max`, which the generated code calls and which are C++ functions otherwise. Enzyme reports failures as error diagnostics of the LLVM context; roadrunner collects them with its own diagnostic handler, so a failure is an exception rather than the termination of the process.
