#!/usr/bin/env bash
#
# build_roadrunner.sh - build libroadrunner from source (C++ core + Python bindings).
#
# The recipe follows the upstream CI workflow, .github/workflows/main.yml: the
# C++ sources are built from a git checkout, while the third party dependencies
# (libroadrunner-deps) and LLVM are downloaded as prebuilt binaries from the
# sys-bio GitHub releases, exactly as the workflow does. The dependency and LLVM
# versions are read from the workflow file of the checkout, so this script stays
# in sync with upstream automatically.
#
# The script is self contained. Run it in an empty directory; it creates
#
#   <workspace>/roadrunner/                    source checkout (git)
#   <workspace>/libroadrunner-deps-binaries/   prebuilt dependencies
#   <workspace>/llvm-<version>-binaries/       prebuilt LLVM
#   <workspace>/venv/                          Python env used for the bindings
#   <workspace>/build-roadrunner/              CMake build tree
#   <workspace>/install-roadrunner/            install tree (the result)
#
# Bindings: Python only. The Julia and Java wrappers are explicitly off.
#
# Hosts: Linux and macOS. Windows needs MSVC and is out of scope; follow the
# workflow directly there.
#
# Re-running the script in the same workspace is incremental: the checkout is
# updated, downloads are reused unless their version changed, and the build tree
# is kept (use --clean to start over).

set -euo pipefail

# --------------------------------------------------------------------------- #
# Defaults (every one of them can be overridden from the command line)
# --------------------------------------------------------------------------- #

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$PWD"
REPO_URL="https://github.com/sys-bio/roadrunner.git"
BRANCH="develop"
BUILD_TYPE="Release"
BUILD_PYTHON="ON"
BUILD_TESTS="ON"
BUILD_PLUGINS="ON"
BUILD_EXAMPLES="ON"
RUN_TESTS="OFF"
LINK_EXAMPLE="OFF"
UPDATE_SOURCE="ON"
CLEAN="OFF"
INSTALL_SYSTEM_DEPS="OFF"
SOURCE_DIR=""
INSTALL_DIR=""
DEPS_PREFIX=""
LLVM_PREFIX=""
PYTHON_EXE=""
JOBS=""
DEPS_RELEASE_OVERRIDE=""
LLVM_RELEASE_OVERRIDE=""

# Settings that decide whether the result links. See configure_and_build().
LLVM_ENABLE_RTTI="ON"
HIDE_LLVM_SYMBOLS="OFF"
CMAKE_EXTRA_ARGS=()

# Fallbacks, used only when the workflow file cannot be parsed.
DEPS_OWNER="sys-bio"
DEPS_REPO="libroadrunner-deps"
DEPS_NAME="libroadrunner-deps"
DEPS_RELEASE="v2.3.9"
LLVM_OWNER="sys-bio"
LLVM_REPO="llvm-build"
LLVM_NAME="llvm-23.1.4"
LLVM_RELEASE="llvm-23.1.4-r2"

usage() {
    cat <<'USAGE'
Build libroadrunner from source (C++ core + Python bindings).

Usage: build_roadrunner.sh [options]

Options:
  -w, --workspace DIR     Directory to build in (default: current directory)
  -b, --branch NAME       Branch, tag or commit of roadrunner (default: develop)
      --repo URL          Git URL to clone (default: sys-bio/roadrunner)
  -s, --source-dir DIR    Use this existing checkout instead of cloning
      --no-update         Do not fetch/update an existing checkout
  -t, --build-type TYPE   Release (default), Debug, RelWithDebInfo, MinSizeRel
  -p, --prefix DIR        Install prefix (default: <workspace>/install-roadrunner)
  -j, --jobs N            Parallel build jobs (default: all cores)
      --no-python         Do not build the Python bindings (C++ only)
      --no-tests          Do not build the test suite
      --no-plugins        Do not build the rrplugins package
      --no-examples       Do not build the C++ examples
      --run-tests         Run ctest after installing
      --link-example      Build and run link-example/ against the install tree,
                          a minimal program that links roadrunner and simulates
      --python-exe PATH   Python interpreter the bindings are built against
      --deps-prefix DIR   Use an existing libroadrunner-deps install tree
      --deps-release TAG  libroadrunner-deps release to download
      --llvm-prefix DIR   Use an existing LLVM install tree
      --llvm-release TAG  LLVM release to download
      --no-rtti           Do not require an LLVM built with RTTI (see below)
      --hide-llvm-symbols Do not export the LLVM symbols from the C API library
                          (macOS only, needed when Julia loads that library)
      --cmake-arg ARG     Extra argument for the roadrunner cmake call; may be
                          repeated and wins over the settings above
      --install-system-deps  Install compiler/cmake/ninja/ccache via apt or brew
      --clean             Remove the build and install trees first
  -h, --help              Show this help

Examples:
  mkdir ~/roadrunner-build && cd ~/roadrunner-build
  /path/to/build_roadrunner.sh                     # develop, Release, with Python
  /path/to/build_roadrunner.sh --no-python         # C++ core only
  /path/to/build_roadrunner.sh -t Debug --run-tests
USAGE
}

# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #

while [[ $# -gt 0 ]]; do
    # Accept --option=value and -jN as well as the space separated forms.
    case "$1" in
        --*=*)     set -- "${1%%=*}" "${1#*=}" "${@:2}" ;;
        -j[0-9]*)  set -- -j "${1#-j}" "${@:2}" ;;
    esac

    case "$1" in
        -w|--workspace)   WORKSPACE="$2"; shift 2 ;;
        -b|--branch)      BRANCH="$2"; shift 2 ;;
        --repo)           REPO_URL="$2"; shift 2 ;;
        -s|--source-dir)  SOURCE_DIR="$2"; shift 2 ;;
        --no-update)      UPDATE_SOURCE="OFF"; shift ;;
        -t|--build-type)  BUILD_TYPE="$2"; shift 2 ;;
        -p|--prefix)      INSTALL_DIR="$2"; shift 2 ;;
        -j|--jobs)        JOBS="$2"; shift 2 ;;
        --no-python)      BUILD_PYTHON="OFF"; shift ;;
        --no-tests)       BUILD_TESTS="OFF"; shift ;;
        --no-plugins)     BUILD_PLUGINS="OFF"; shift ;;
        --no-examples)    BUILD_EXAMPLES="OFF"; shift ;;
        --run-tests)      RUN_TESTS="ON"; shift ;;
        --link-example)   LINK_EXAMPLE="ON"; shift ;;
        --python-exe)     PYTHON_EXE="$2"; shift 2 ;;
        --deps-prefix)    DEPS_PREFIX="$2"; shift 2 ;;
        --deps-release)   DEPS_RELEASE_OVERRIDE="$2"; shift 2 ;;
        --llvm-prefix)    LLVM_PREFIX="$2"; shift 2 ;;
        --llvm-release)   LLVM_RELEASE_OVERRIDE="$2"; shift 2 ;;
        --no-rtti)        LLVM_ENABLE_RTTI="OFF"; shift ;;
        --hide-llvm-symbols) HIDE_LLVM_SYMBOLS="ON"; shift ;;
        --cmake-arg)      CMAKE_EXTRA_ARGS+=("$2"); shift 2 ;;
        --install-system-deps) INSTALL_SYSTEM_DEPS="ON"; shift ;;
        --clean)          CLEAN="ON"; shift ;;
        -h|--help)        usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; echo "Try --help." >&2; exit 2 ;;
    esac
done

# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '\033[1;33m[warning] %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31m[error] %s\033[0m\n' "$*" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

# A release asset exists? Fetch a single byte instead of using HEAD; the GitHub
# release redirect answers range requests reliably and costs no API quota.
url_exists() { curl -fsSL --retry 2 -o /dev/null -r 0-0 "$1"; }

# Newest release of a repository that carries a given asset name.
latest_release_with_asset() {
    local owner="$1" repo="$2" asset="$3" json=""
    local -a auth=()
    if [[ -n "${GITHUB_TOKEN:-}" ]]; then
        auth=(-H "Authorization: Bearer ${GITHUB_TOKEN}")
    elif [[ -n "${GH_TOKEN:-}" ]]; then
        auth=(-H "Authorization: Bearer ${GH_TOKEN}")
    fi
    json=$(curl -fsSL --retry 2 "${auth[@]+"${auth[@]}"}" \
        "https://api.github.com/repos/${owner}/${repo}/releases?per_page=50" 2>/dev/null) || return 1
    if have python3; then
        printf '%s' "$json" | python3 -I -c '
import json, sys
wanted = sys.argv[1]
for release in json.load(sys.stdin):
    if any(a.get("name") == wanted for a in release.get("assets", [])):
        print(release["tag_name"])
        break
' "$asset"
    elif have jq; then
        printf '%s' "$json" | jq -r --arg a "$asset" \
            'first(.[] | select(any(.assets[]; .name == $a)) | .tag_name) // empty'
    else
        return 1
    fi
}

# Download and unpack a release asset into a directory, skipping the work when
# the directory already holds that exact asset.
fetch_asset() {
    local url="$1" dest="$2" asset="$3"
    local stamp="${dest}/.asset-stamp"

    if [[ -f "$stamp" && "$(cat "$stamp")" == "$asset" ]]; then
        info "already present: $asset"
        return 0
    fi

    info "downloading $asset"
    rm -rf "${dest:?}"
    mkdir -p "$dest"
    curl -fL --retry 3 --progress-bar -o "${dest}/${asset}" "$url"
    unzip -q "${dest}/${asset}" -d "$dest"
    rm -f "${dest:?}/${asset:?}"

    # Some archives carry a single top level directory; flatten it so that the
    # directory itself is a usable install prefix.
    if [[ ! -d "${dest}/include" && ! -d "${dest}/bin" ]]; then
        local -a entries=()
        mapfile -t entries < <(find "$dest" -mindepth 1 -maxdepth 1 ! -name '.asset-stamp')
        if [[ ${#entries[@]} -eq 1 && -d "${entries[0]}" ]]; then
            info "flattening ${entries[0]##*/}/"
            shopt -s dotglob
            mv "${entries[0]}"/* "$dest"/
            shopt -u dotglob
            rmdir "${entries[0]}"
        fi
    fi

    printf '%s' "$asset" > "$stamp"
}

# --------------------------------------------------------------------------- #
# Host platform, named the way the CI workflow names it
# --------------------------------------------------------------------------- #

detect_platform() {
    HOST_ARCH="$(uname -m)"
    OSX_ARCHITECTURES=""

    case "$(uname -s)" in
        Linux)
            OS_TYPE="linux"
            local id="" version=""
            if [[ -r /etc/os-release ]]; then
                id=$(. /etc/os-release && printf '%s' "${ID:-}")
                version=$(. /etc/os-release && printf '%s' "${VERSION_ID:-}")
            fi
            if [[ "$id" == "ubuntu" && -n "$version" ]]; then
                OS_AND_COMPILER="ubuntu-${version%%.*}"
            else
                # No prebuilt binaries are published for other distributions;
                # the manylinux fallback below covers them.
                OS_AND_COMPILER=""
                info "distribution '${id:-unknown}' has no matching release assets"
            fi
            ;;
        Darwin)
            OS_TYPE="macos"
            OS_AND_COMPILER="macos-$(sw_vers -productVersion | cut -d. -f1)"
            OSX_ARCHITECTURES="$HOST_ARCH"
            ;;
        *)
            die "unsupported host '$(uname -s)'. Windows builds need MSVC; follow .github/workflows/main.yml there."
            ;;
    esac

    info "host: ${OS_AND_COMPILER:-$OS_TYPE} ${HOST_ARCH}, build type ${BUILD_TYPE}"
}

# Resolve the release and asset to download for one component. The pinned
# release is tried first; when it has nothing for this host (upstream moves the
# CI runners to new OS versions and drops the old assets) the newest release
# that does is used, and the manylinux build is the last resort on Linux.
resolve_asset() {
    local owner="$1" repo="$2" name="$3" pinned_tag="$4" flavour="$5"
    local asset tag

    RESOLVED_TAG=""
    RESOLVED_ASSET=""

    local -a candidates=()
    [[ -n "$OS_AND_COMPILER" ]] && candidates+=("${name}-${OS_AND_COMPILER}-${HOST_ARCH}-${flavour}.zip")
    if [[ "$OS_TYPE" == "linux" && "$HOST_ARCH" == "x86_64" ]]; then
        candidates+=("${name}-manylinux_2_28-x86_64-${flavour}.zip")
    fi
    [[ ${#candidates[@]} -eq 0 ]] && die "no known ${name} asset for this host"

    for asset in "${candidates[@]}"; do
        if url_exists "https://github.com/${owner}/${repo}/releases/download/${pinned_tag}/${asset}"; then
            if [[ "$asset" == *manylinux* && -n "$OS_AND_COMPILER" ]]; then
                warn "${pinned_tag} has no ${OS_AND_COMPILER} build of ${name}; using the manylinux build"
            fi
            RESOLVED_TAG="$pinned_tag"
            RESOLVED_ASSET="$asset"
            return 0
        fi
    done

    # The pinned release has nothing for this host: look for an older one that has.
    asset="${candidates[0]}"
    if [[ -n "$OS_AND_COMPILER" ]]; then
        tag=$(latest_release_with_asset "$owner" "$repo" "$asset" || true)
        if [[ -n "$tag" ]]; then
            warn "${pinned_tag} has no ${asset}; falling back to release ${tag}"
            RESOLVED_TAG="$tag"
            RESOLVED_ASSET="$asset"
            return 0
        fi
    fi

    die "no ${name} binaries for ${OS_AND_COMPILER:-$OS_TYPE}-${HOST_ARCH}-${flavour}.
Either point --deps-prefix/--llvm-prefix at an existing install tree, or build
them from source: https://github.com/${owner}/${repo}"
}

# --------------------------------------------------------------------------- #
# Toolchain
# --------------------------------------------------------------------------- #

install_system_deps() {
    log "Installing system build tools"
    if have apt-get; then
        local -a pkgs=(build-essential cmake ninja-build ccache git curl unzip)
        [[ "$BUILD_PYTHON" == "ON" ]] && pkgs+=(python3-dev python3-venv)
        sudo apt-get update
        sudo apt-get install -y "${pkgs[@]}"
    elif have brew; then
        brew install cmake ninja ccache
    else
        die "no apt-get or brew found; install the build tools manually"
    fi
}

check_toolchain() {
    log "Checking the toolchain"

    local -a missing=()
    local tool
    for tool in git cmake curl unzip; do
        have "$tool" || missing+=("$tool")
    done
    have c++ || have g++ || have clang++ || missing+=("a C++ compiler")
    [[ ${#missing[@]} -eq 0 ]] || die "missing: ${missing[*]} (try --install-system-deps)"

    local cmake_version
    cmake_version=$(cmake --version | head -1 | awk '{print $3}')
    info "cmake ${cmake_version}"
    printf '%s\n%s\n' "3.16" "$cmake_version" | sort -V -C ||
        die "cmake >= 3.16 is required, found ${cmake_version}"

    if have ninja; then
        GENERATOR="Ninja"
    else
        GENERATOR="Unix Makefiles"
        warn "ninja not found, using 'Unix Makefiles' (the CI builds with Ninja)"
    fi
    info "generator: ${GENERATOR}"

    if have ccache; then
        COMPILER_LAUNCHER="ccache"
        info "ccache $(ccache --version | head -1 | awk '{print $3}')"
    else
        COMPILER_LAUNCHER=""
        info "ccache not found, building without a compiler cache"
    fi

    if [[ -z "$JOBS" ]]; then
        if have nproc; then JOBS=$(nproc)
        elif have sysctl; then JOBS=$(sysctl -n hw.ncpu)
        else JOBS=4
        fi
    fi
    info "parallel jobs: ${JOBS}"
}

# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #

get_sources() {
    if [[ -n "$SOURCE_DIR" ]]; then
        SOURCE_DIR="$(cd "$SOURCE_DIR" && pwd)"
        log "Using the existing checkout ${SOURCE_DIR}"
        [[ -f "${SOURCE_DIR}/CMakeLists.txt" ]] || die "${SOURCE_DIR} is not a roadrunner checkout"

        # Some of the installed Python tests find their data by walking up from
        # the installed package to a sibling source tree, for example
        # ../../../../../roadrunner/test/sbml-test-suite/stochastic. That only
        # works when the checkout sits in the workspace, so link it there.
        local link="${WORKSPACE}/roadrunner"
        if [[ ! -e "$link" ]]; then
            ln -s "$SOURCE_DIR" "$link"
            info "linked ${link} -> ${SOURCE_DIR} so the data driven tests find their models"
        fi
        return 0
    fi

    SOURCE_DIR="${WORKSPACE}/roadrunner"
    if [[ -d "${SOURCE_DIR}/.git" ]]; then
        if [[ "$UPDATE_SOURCE" == "ON" ]]; then
            log "Updating the roadrunner checkout (${BRANCH})"
            git -C "$SOURCE_DIR" fetch --tags origin
            git -C "$SOURCE_DIR" checkout "$BRANCH"
            git -C "$SOURCE_DIR" pull --ff-only || warn "could not fast-forward ${BRANCH}"
        else
            log "Keeping the roadrunner checkout as it is"
        fi
    else
        log "Cloning roadrunner (${BRANCH})"
        git clone --branch "$BRANCH" "$REPO_URL" "$SOURCE_DIR"
    fi

    if [[ -f "${SOURCE_DIR}/.gitmodules" ]]; then
        git -C "$SOURCE_DIR" submodule update --init --recursive
    fi
    info "at $(git -C "$SOURCE_DIR" rev-parse --short HEAD) ($(git -C "$SOURCE_DIR" log -1 --format=%s | cut -c1-60))"
}

# The workflow matrix is the single source of truth for the versions.
read_versions_from_workflow() {
    local workflow="${SOURCE_DIR}/.github/workflows/main.yml"
    if [[ ! -f "$workflow" ]]; then
        warn "no workflow file in the checkout; using the built-in versions"
    else
        local key value
        for key in libroadrunner_deps_owner libroadrunner_deps_repo libroadrunner_deps_name \
                   libroadrunner_deps_release_version llvm_owner llvm_repo llvm_name llvm_release_version; do
            value=$(sed -n "s/^[[:space:]]*${key}:[[:space:]]*\[[[:space:]]*\"\([^\"]*\)\".*/\1/p" "$workflow" | head -1)
            [[ -z "$value" ]] && continue
            case "$key" in
                libroadrunner_deps_owner)           DEPS_OWNER="$value" ;;
                libroadrunner_deps_repo)            DEPS_REPO="$value" ;;
                libroadrunner_deps_name)            DEPS_NAME="$value" ;;
                libroadrunner_deps_release_version) DEPS_RELEASE="$value" ;;
                llvm_owner)                         LLVM_OWNER="$value" ;;
                llvm_repo)                          LLVM_REPO="$value" ;;
                llvm_name)                          LLVM_NAME="$value" ;;
                llvm_release_version)               LLVM_RELEASE="$value" ;;
            esac
        done
    fi

    [[ -n "$DEPS_RELEASE_OVERRIDE" ]] && DEPS_RELEASE="$DEPS_RELEASE_OVERRIDE"
    [[ -n "$LLVM_RELEASE_OVERRIDE" ]] && LLVM_RELEASE="$LLVM_RELEASE_OVERRIDE"
    info "dependencies ${DEPS_OWNER}/${DEPS_REPO} ${DEPS_RELEASE}, LLVM ${LLVM_OWNER}/${LLVM_REPO} ${LLVM_RELEASE}"
}

# --------------------------------------------------------------------------- #
# Prebuilt dependencies
# --------------------------------------------------------------------------- #

get_dependencies() {
    if [[ -n "$DEPS_PREFIX" ]]; then
        DEPS_DIR="$(cd "$DEPS_PREFIX" && pwd)"
        log "Using the existing dependency tree ${DEPS_DIR}"
    else
        log "Dependencies: ${DEPS_NAME}"
        DEPS_DIR="${WORKSPACE}/${DEPS_NAME}-binaries"
        resolve_asset "$DEPS_OWNER" "$DEPS_REPO" "$DEPS_NAME" "$DEPS_RELEASE" "$BUILD_TYPE"
        fetch_asset "https://github.com/${DEPS_OWNER}/${DEPS_REPO}/releases/download/${RESOLVED_TAG}/${RESOLVED_ASSET}" \
            "$DEPS_DIR" "$RESOLVED_ASSET"
    fi
    [[ -d "${DEPS_DIR}/include" ]] ||
        die "${DEPS_DIR} has no include/ directory, so it is not a dependency install tree"
}

get_llvm() {
    if [[ -n "$LLVM_PREFIX" ]]; then
        LLVM_DIR="$(cd "$LLVM_PREFIX" && pwd)"
        log "Using the existing LLVM tree ${LLVM_DIR}"
    else
        log "LLVM: ${LLVM_NAME}"
        LLVM_DIR="${WORKSPACE}/${LLVM_NAME}-binaries"
        # As in the workflow, Linux links the Release build of LLVM even for a
        # Debug roadrunner; the other platforms match the build type.
        local flavour="$BUILD_TYPE"
        [[ "$OS_TYPE" == "linux" ]] && flavour="Release"
        resolve_asset "$LLVM_OWNER" "$LLVM_REPO" "$LLVM_NAME" "$LLVM_RELEASE" "$flavour"
        fetch_asset "https://github.com/${LLVM_OWNER}/${LLVM_REPO}/releases/download/${RESOLVED_TAG}/${RESOLVED_ASSET}" \
            "$LLVM_DIR" "$RESOLVED_ASSET"
        chmod a+x "${LLVM_DIR}/bin/llvm-config" 2>/dev/null || true
    fi
    local llvm_config="${LLVM_DIR}/bin/llvm-config"
    [[ -x "$llvm_config" ]] ||
        die "${llvm_config} is missing, so it is not an LLVM install tree"

    # roadrunner strips -fno-rtti from the flags llvm-config reports (see
    # cmake/FindLLVM.cmake) and compiles its LLVM back end with RTTI. An LLVM
    # built with LLVM_ENABLE_RTTI=OFF has no type_info for its classes, so the
    # link of roadrunner-static fails with undefined typeinfo/vtable symbols.
    local has_rtti
    has_rtti=$("$llvm_config" --has-rtti 2>/dev/null || echo UNKNOWN)
    info "llvm $("$llvm_config" --version), rtti ${has_rtti}"
    if [[ "$LLVM_ENABLE_RTTI" == "ON" && "$has_rtti" != "YES" ]]; then
        die "this LLVM reports --has-rtti=${has_rtti}, so linking roadrunner against it
would fail with undefined typeinfo symbols. Rebuild that LLVM with
-DLLVM_ENABLE_RTTI=ON, use the sys-bio/llvm-build release instead of
--llvm-prefix, or pass --no-rtti to skip this check."
    fi
}

# --------------------------------------------------------------------------- #
# Python environment for the bindings
# --------------------------------------------------------------------------- #

# An interpreter is usable when it can create virtual environments and ships
# the development headers the extension module is compiled against.
python_is_usable() {
    local exe="$1" header
    "$exe" -I -c 'import ensurepip, venv' >/dev/null 2>&1 || return 1
    header=$("$exe" -I -c 'import os, sysconfig; print(os.path.join(sysconfig.get_paths()["include"], "Python.h"))' 2>/dev/null) || return 1
    [[ -f "$header" ]]
}

select_python() {
    if [[ -n "$PYTHON_EXE" ]]; then
        have "$PYTHON_EXE" || [[ -x "$PYTHON_EXE" ]] || die "no such interpreter: ${PYTHON_EXE}"
        python_is_usable "$PYTHON_EXE" ||
            die "${PYTHON_EXE} cannot build the bindings: it needs the venv/ensurepip modules
(Debian/Ubuntu: python3-venv) and the development headers (python3-dev)."
        return 0
    fi

    # The extension uses the stable ABI with 3.10 as its floor, so the oldest
    # usable interpreter gives the most portable module.
    local candidate
    for candidate in python3 python3.10 python3.11 python3.12 python3.13 python3.14 python3.15; do
        have "$candidate" || continue
        if python_is_usable "$candidate"; then
            PYTHON_EXE="$(command -v "$candidate")"
            return 0
        fi
        [[ "$candidate" == "python3" ]] && info "python3 cannot create virtual environments, looking for another interpreter"
    done

    die "no usable Python found. The bindings need an interpreter with the venv
and ensurepip modules (Debian/Ubuntu: sudo apt install python3-venv) and the
development headers (python3-dev). Pass --python-exe to choose one, or
--no-python to build the C++ core only."
}

setup_python() {
    log "Python environment for the bindings"

    select_python
    info "interpreter ${PYTHON_EXE}"

    VENV_DIR="${WORKSPACE}/venv"
    # A half created environment from an interrupted run is replaced rather than
    # reused, so check for a working pip instead of just the interpreter.
    if ! "${VENV_DIR}/bin/python" -m pip --version >/dev/null 2>&1; then
        info "creating ${VENV_DIR}"
        rm -rf "${VENV_DIR:?}"
        "$PYTHON_EXE" -m venv "$VENV_DIR"
    fi
    local venv_python="${VENV_DIR}/bin/python"

    info "installing the build requirements"
    "$venv_python" -m pip install --quiet --upgrade pip
    # The bindings use the Python stable ABI, which needs SWIG >= 4.4.1; the
    # distribution packages are usually older, so take the PyPI wheel.
    "$venv_python" -m pip install --quiet "swig>=4.4.1"
    "$venv_python" -m pip install --quiet -r "${SOURCE_DIR}/requirements.txt"
    "$venv_python" -m pip install --quiet build
    if [[ "$BUILD_TESTS" == "ON" ]]; then
        "$venv_python" -m pip install --quiet -r "${SOURCE_DIR}/test-requirements.txt"
    fi

    SWIG_EXECUTABLE="${VENV_DIR}/bin/swig"
    [[ -x "$SWIG_EXECUTABLE" ]] || SWIG_EXECUTABLE="$(command -v swig || true)"
    [[ -x "$SWIG_EXECUTABLE" ]] || die "no swig executable found after installing it"
    info "$("$SWIG_EXECUTABLE" -version | sed -n 's/^SWIG Version/swig/p')"
    info "python $("$venv_python" -c 'import platform; print(platform.python_version())')"
}

# --------------------------------------------------------------------------- #
# Configure, build, install
# --------------------------------------------------------------------------- #

configure_and_build() {
    log "Configuring roadrunner"

    local -a args=(
        "$SOURCE_DIR"
        -G "$GENERATOR"
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"
        -DRR_DEPENDENCIES_INSTALL_PREFIX="$DEPS_DIR"
        -DLLVM_INSTALL_PREFIX="$LLVM_DIR"
        -DBUILD_TESTS="$BUILD_TESTS"
        -DBUILD_RR_PLUGINS="$BUILD_PLUGINS"
        -DBUILD_EXAMPLES="$BUILD_EXAMPLES"
        -DBUILD_PYTHON="$BUILD_PYTHON"
        -DBUILD_JULIA=OFF
        -DBUILD_JAVA=OFF

        # --- settings that decide whether everything links together ---------
        # roadrunner compiles its LLVM back end with RTTI (cmake/FindLLVM.cmake
        # removes -fno-rtti from the flags llvm-config reports), so the LLVM it
        # links has to be built with LLVM_ENABLE_RTTI=ON or the link ends in
        # undefined typeinfo symbols. get_llvm() verifies this with
        # 'llvm-config --has-rtti'; the variable is passed on so that an LLVM
        # configured inside this tree is built the same way. roadrunner's own
        # cmake does not read it, so cmake lists it as unused on a fresh
        # configure.
        -DLLVM_ENABLE_RTTI="$LLVM_ENABLE_RTTI"
        # Every static archive here ends up inside a shared object: libroadrunner,
        # the C API library, the Python extension and the plugins. All of it has
        # to be position independent or the shared link fails on x86_64 with
        # "relocation R_X86_64_PC32 against symbol ... can not be used when making
        # a shared object". roadrunner sets this internally as well; passing it
        # also covers the subprojects configured from this tree.
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        # The C API library exports the symbols of the LLVM it statically links
        # unless this is ON. Hiding them is only needed on macOS when a host
        # process brings its own LLVM (Julia), and breaks nothing else here.
        -DRR_C_API_HIDE_LLVM_SYMBOLS="$HIDE_LLVM_SYMBOLS"
        # Nothing is passed for the install RPATH on purpose: roadrunner sets it
        # itself ($ORIGIN, @loader_path on macOS, plus the relative path to lib),
        # so the installed binaries already find their libraries next to
        # themselves. Overriding it here would only break the macOS case.
    )
    if [[ -n "$COMPILER_LAUNCHER" ]]; then
        args+=(
            -DCMAKE_C_COMPILER_LAUNCHER="$COMPILER_LAUNCHER"
            -DCMAKE_CXX_COMPILER_LAUNCHER="$COMPILER_LAUNCHER"
        )
    fi
    [[ -n "$OSX_ARCHITECTURES" ]] && args+=(-DCMAKE_OSX_ARCHITECTURES="$OSX_ARCHITECTURES")
    if [[ "$BUILD_PYTHON" == "ON" ]]; then
        args+=(
            -DPython_ROOT_DIR="$VENV_DIR"
            -DSWIG_EXECUTABLE="$SWIG_EXECUTABLE"
        )
    fi
    # Last, so that an explicit --cmake-arg overrides anything set above.
    args+=("${CMAKE_EXTRA_ARGS[@]+"${CMAKE_EXTRA_ARGS[@]}"}")

    mkdir -p "$BUILD_DIR"
    ( cd "$BUILD_DIR" && cmake "${args[@]}" )

    log "Building and installing roadrunner (this takes a while)"
    cmake --build "$BUILD_DIR" --target install --config "$BUILD_TYPE" --parallel "$JOBS"

    have ccache && ccache --show-stats | head -8 || true
}

run_tests() {
    log "Running the C++ test suite"
    local status=0
    ( cd "$BUILD_DIR" && ctest -C "$BUILD_TYPE" --output-on-failure --exclude-regex ython ) || status=1

    if [[ "$BUILD_PYTHON" == "ON" ]]; then
        log "Running the Python test suite"
        ( cd "$BUILD_DIR" && ctest -C "$BUILD_TYPE" --output-on-failure --tests-regex ython ) || status=1
    fi
    [[ $status -eq 0 ]] || die "tests failed"
}

# Build and run the standalone example next to this script. It is a consumer of
# the install tree like any other project, so it also checks that the installed
# cmake package files and headers are usable from outside this build.
build_link_example() {
    log "Building the link example against the install tree"

    local example_dir="${SCRIPT_DIR}/link-example"
    [[ -f "${example_dir}/CMakeLists.txt" ]] ||
        die "${example_dir} is missing; keep link-example/ next to this script"

    local example_build="${WORKSPACE}/build-link-example"
    cmake -S "$example_dir" -B "$example_build" \
        -G "$GENERATOR" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DROADRUNNER_INSTALL_PREFIX="$INSTALL_DIR" \
        -DRR_DEPENDENCIES_INSTALL_PREFIX="$DEPS_DIR" \
        -DLLVM_INSTALL_PREFIX="$LLVM_DIR"
    cmake --build "$example_build" --config "$BUILD_TYPE" --parallel "$JOBS"

    log "Running the link example"
    "${example_build}/simulate"
}

summary() {
    local version="unknown"
    [[ -f "${INSTALL_DIR}/VERSION.txt" ]] && version="$(cat "${INSTALL_DIR}/VERSION.txt")"

    log "Done: roadrunner ${version} (${BUILD_TYPE})"
    cat <<EOF
    source       ${SOURCE_DIR}
    build        ${BUILD_DIR}
    install      ${INSTALL_DIR}
    dependencies ${DEPS_DIR}
    llvm         ${LLVM_DIR}

    C++:    headers in ${INSTALL_DIR}/include/rr, libraries in ${INSTALL_DIR}/lib.
            In a consuming CMake project:
              set(CMAKE_PREFIX_PATH "${INSTALL_DIR}")
              list(APPEND CMAKE_MODULE_PATH "${INSTALL_DIR}/lib/cmake")
              set(LLVM_INSTALL_PREFIX "${LLVM_DIR}")
              include("${INSTALL_DIR}/lib/cmake/ImportRoadrunnerAndDependencies.cmake")
              target_link_libraries(your_target PRIVATE roadrunner-static::roadrunner-static)
            With pkg-config: ${INSTALL_DIR}/lib/pkgconfig/roadrunner.pc
            Compile consumers with C++17 or later and with RTTI enabled, the
            same way this tree was built, or the link will not resolve.
EOF
    if [[ "$BUILD_PYTHON" == "ON" ]]; then
        cat <<EOF
    Python: export PYTHONPATH=${INSTALL_DIR}/site-packages
            ${VENV_DIR}/bin/python -c "import roadrunner; print(roadrunner.__version__)"
            or build a wheel: (cd ${INSTALL_DIR} && ${VENV_DIR}/bin/python -m build --wheel)
EOF
    fi
    if [[ "$RUN_TESTS" == "OFF" && "$BUILD_TESTS" == "ON" ]]; then
        echo "    Tests:  (cd ${BUILD_DIR} && ctest --output-on-failure)"
    fi
    if [[ "$LINK_EXAMPLE" == "OFF" ]]; then
        cat <<EOF
    Linking a program against this install tree, start to finish:
            cmake -S ${SCRIPT_DIR}/link-example -B ${WORKSPACE}/build-link-example \\
                  -DROADRUNNER_WORKSPACE=${WORKSPACE}
            cmake --build ${WORKSPACE}/build-link-example
            ${WORKSPACE}/build-link-example/simulate
            or pass --link-example to have this script do it.
EOF
    fi
}

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

main() {
    mkdir -p "$WORKSPACE"
    WORKSPACE="$(cd "$WORKSPACE" && pwd)"
    BUILD_DIR="${WORKSPACE}/build-roadrunner"
    [[ -n "$INSTALL_DIR" ]] || INSTALL_DIR="${WORKSPACE}/install-roadrunner"

    log "roadrunner build in ${WORKSPACE}"

    case "$BUILD_TYPE" in
        Release|Debug|RelWithDebInfo|MinSizeRel) ;;
        *) die "unknown build type '${BUILD_TYPE}'" ;;
    esac

    if [[ "$CLEAN" == "ON" ]]; then
        log "Removing the build and install trees"
        rm -rf "${BUILD_DIR:?}" "${INSTALL_DIR:?}"
    fi

    [[ "$INSTALL_SYSTEM_DEPS" == "ON" ]] && install_system_deps
    detect_platform
    check_toolchain
    get_sources
    read_versions_from_workflow
    get_dependencies
    get_llvm
    [[ "$BUILD_PYTHON" == "ON" ]] && setup_python
    configure_and_build
    [[ "$RUN_TESTS" == "ON" ]] && run_tests
    [[ "$LINK_EXAMPLE" == "ON" ]] && build_link_example
    summary
}

main "$@"
