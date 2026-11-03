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

## roadrunner

```bash
cmake -S ~/git/roadrunner-enzyme -B ~/rr-enzyme/build \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON \
  -DLLVM_INSTALL_PREFIX=$HOME/rr-enzyme/llvm \
  -DRR_DEPENDENCIES_INSTALL_PREFIX=$HOME/rr-enzyme/deps \
  -DCMAKE_INSTALL_PREFIX=$HOME/rr-enzyme/install \
  -DRR_USE_ENZYME=ON \
  -DRR_ENZYME_LIBRARY=$HOME/rr-enzyme/build-enzyme/Enzyme/libEnzymeStatic-23.a
make -C ~/rr-enzyme/build -j20
ctest --test-dir ~/rr-enzyme/build -j20
```

## Known test failures that are not caused by the prototype

With `ctest -j20`, about 90 `ModelEditingTests` and 20 `StateSavingTests` fail on `develop` as well (110 failures with the same dependencies and `develop` at `2876ab2ed`). They all pass with `ctest -j1`, so they interfere with each other when run in parallel.

## Go/no-go gate

Passed on 2026-11-03 with the tagged release `v0.0.303` and no patches to Enzyme: `EnzymeJacobianTests.GateHctzDirectionalDerivative` differentiates the generated `evalReactionRates` of `hctz_body_flat.xml` in roadrunner's JIT (MCJit) and matches the central difference. Enzyme needed IR bodies for the helpers `rr_min` and `rr_max`, which the generated code calls and which are C++ functions otherwise. Enzyme reports failures as error diagnostics of the LLVM context; roadrunner collects them with its own diagnostic handler, so a failure is an exception rather than the termination of the process.
