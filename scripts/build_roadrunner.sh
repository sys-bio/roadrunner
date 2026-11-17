#!/usr/bin/env bash
# Builds and installs roadrunner from the checkout this script is in, against the prebuilt
# dependencies and LLVM that CI uses (their versions are in scripts/versions.env).
#
#   scripts/build_roadrunner.sh [-t Release|Debug] [-w workspace] [-c] [-- extra cmake options]
#
#   -t  build type (default Release)
#   -w  folder for the downloads, the build tree and the install (default ./roadrunner-build)
#   -c  afterwards, build and run the projects in test/cmake_consumers against the install
#
# Needs bash, curl, unzip, cmake and a C++17 compiler; Ninja is used if it is found. On Windows,
# run it with bash from the "x64 Native Tools Command Prompt for Visual Studio".
# For the Python bindings add:  -- -DBUILD_PYTHON=ON -DSWIG_EXECUTABLE=/path/to/swig
set -euo pipefail

die() { echo "error: $*" >&2; exit 1; }

type=Release ws=roadrunner-build check=no
while getopts t:w:c opt; do
    case $opt in
        t) type=$OPTARG ;;
        w) ws=$OPTARG ;;
        c) check=yes ;;
        *) exit 2 ;;
    esac
done
shift $((OPTIND - 1))   # anything left over goes to cmake

# Absolute paths, in the form cmake expects on Windows too (pwd -W exists only there).
abspath() { (cd "$1" && { pwd -W 2>/dev/null || pwd; }); }
src=$(abspath "$(dirname "${BASH_SOURCE[0]}")/..")
mkdir -p "$ws"
ws=$(abspath "$ws")

eval "$(tr -d '\r' < "$src/scripts/versions.env")"   # DEPS_RELEASE, LLVM_NAME, LLVM_RELEASE
for tool in curl unzip cmake; do command -v $tool > /dev/null || die "$tool is not installed"; done

# The binaries are named <name>-<os>-<architecture>-<build type>.zip. The Linux dependencies
# work on every distribution. LLVM comes in an Ubuntu and a manylinux build, Release only.
arch=$(uname -m)
llvm_type=$type
case "$(uname -s)" in
    Linux)
        deps_os=manylinux_2_28
        llvm_os=manylinux_2_28
        llvm_type=Release
        if grep -qx 'ID=ubuntu' /etc/os-release; then
            llvm_os=ubuntu-$(. /etc/os-release; echo "${VERSION_ID%%.*}")
        fi ;;
    Darwin)
        deps_os=macos-$(sw_vers -productVersion | cut -d. -f1)
        llvm_os=$deps_os ;;
    MINGW* | MSYS*)
        deps_os=windows-msvc$(ls "C:\Program Files\Microsoft Visual Studio")
        llvm_os=$deps_os ;;
    *) die "there are no prebuilt binaries for $(uname -s)" ;;
esac

# fetch <repository> <release> <name> <os> <build type> <folder>
fetch() {
    if [ -d "$6" ]; then return; fi
    local url=https://github.com/sys-bio/$1/releases/download/$2/$3-$4-$arch-$5.zip
    echo "Downloading $url"
    curl -fsSL "$url" -o "$6.zip" || die "cannot download $url (see the LLVM dependency page of the documentation)"
    unzip -qo "$6.zip" -d "$6.tmp"
    mv "$6.tmp" "$6"
    rm "$6.zip"
}
fetch libroadrunner-deps "$DEPS_RELEASE" libroadrunner-deps "$deps_os" "$type" "$ws/deps"
fetch llvm-build "$LLVM_RELEASE" "$LLVM_NAME" "$llvm_os" "$llvm_type" "$ws/llvm"
chmod a+x "$ws"/llvm/bin/llvm-config*

if command -v ninja > /dev/null; then export CMAKE_GENERATOR=Ninja; fi
cmake -S "$src" -B "$ws/build" \
    -DCMAKE_BUILD_TYPE="$type" \
    -DCMAKE_INSTALL_PREFIX="$ws/install" \
    -DRR_DEPENDENCIES_INSTALL_PREFIX="$ws/deps" \
    -DLLVM_INSTALL_PREFIX="$ws/llvm" \
    -DBUILD_RR_PLUGINS=ON \
    "$@"
cmake --build "$ws/build" --target install --config "$type" --parallel

if [ "$check" = yes ]; then
    bash "$src/test/cmake_consumers/run_consumers.sh" "$ws/install" "$ws/llvm" "$type" "$ws/consumer-build"
fi
echo "Installed roadrunner into $ws/install"
