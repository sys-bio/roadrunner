#!/usr/bin/env bash
# Builds and runs projects that use an installed roadrunner through its CMake packages.
#
#   run_consumers.sh <roadrunner install folder> <LLVM install folder> <build type> [work folder]
#
# The projects are the example in docs/source/UsingRoadRunnerFromCxx and the ones next to this
# script. Each one is configured, built and run with the commands the documentation shows.
set -euo pipefail

if [ $# -lt 3 ]; then
    echo "usage: $0 <roadrunner install folder> <LLVM install folder> <build type> [work folder]" >&2
    exit 2
fi

prefix=$1
llvm=$2
build_type=$3
mkdir -p "${4:-consumer-build}"
work=$(cd "${4:-consumer-build}" && pwd)

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
docs_example="$here/../../docs/source/UsingRoadRunnerFromCxx"

# On Windows the shared libraries are found through PATH.
if command -v cygpath > /dev/null; then
    export PATH="$(cygpath -u "$prefix")/bin:$PATH"
fi

# build_and_run <name> <source folder> <executable>...
build_and_run() {
    local name=$1 source=$2
    shift 2
    echo "::group::$name"
    cmake -S "$source" -B "$work/$name" \
        -DCMAKE_BUILD_TYPE="$build_type" \
        -DROADRUNNER_INSTALL_PREFIX="$prefix" \
        -DLLVM_INSTALL_PREFIX="$llvm"
    cmake --build "$work/$name" --config "$build_type"
    echo "::endgroup::"

    local exe candidate output
    for exe in "$@"; do
        # Visual Studio puts the programs in a folder named after the configuration.
        for candidate in "$work/$name/$exe" "$work/$name/$exe.exe" \
                         "$work/$name/$build_type/$exe" "$work/$name/$build_type/$exe.exe"; do
            [ -f "$candidate" ] && break
            candidate=
        done
        if [ -z "$candidate" ]; then
            echo "ERROR: $exe was not built in $work/$name" >&2
            exit 1
        fi
        echo "--- running $candidate"
        output=$(cd "$work/$name" && "$candidate")
        echo "$output"
        if [ -z "$output" ]; then
            echo "ERROR: $exe printed nothing" >&2
            exit 1
        fi
        # What the output of the programs in the documentation must contain, one item per line:
        # the simulation result, and all of the plugins.
        case "$exe" in
            UseRoadRunnerFromCxx) expected_text='S1' ;;
            use_klu) expected_text='KLU in use: yes' ;;
            UseRRPluginsFromCxx) expected_text='Loaded 8 plugins
AddNoise
Auto-2000
ChiSquare
Hello
Levenberg-Marquardt
MonteCarlo-Bootstrap
Nelder-Mead
TestModel' ;;
            *) expected_text= ;;
        esac
        while IFS= read -r expected; do
            [ -z "$expected" ] && continue
            if ! grep -qF -- "$expected" <<< "$output"; then
                echo "ERROR: the output of $exe does not contain '$expected'" >&2
                exit 1
            fi
        done <<< "$expected_text"
    done
}

build_and_run docs_example "$docs_example" UseRoadRunnerFromCxx UseRRPluginsFromCxx
build_and_run package "$here/package" use_static use_shared use_c_static use_c_shared use_klu
build_and_run legacy "$here/legacy" legacy_static legacy_shared legacy_c_static legacy_c_shared
build_and_run rrplugins "$here/rrplugins" plugins_cpp plugins_c

echo "All consumer projects built and ran."
