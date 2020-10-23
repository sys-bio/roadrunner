#!/bin/sh

set -e
#set -x

if [ $# -lt 3 ]; then
    cat >&2 <<EOF
Usage: $0 simgrid.jar java_command entry_point strip_command [-so file.so...] [-txt file.txt...]
    simgrid.jar    SimGrid jar file
    java_command   path to the Java runtime
    entry_point    entry point for java
    strip_command  path to the command used to strip libraries
    file.so        library file to stript and bundle into the archive
    file.txt       other file  to bundle into the archive
EOF
    exit 1
fi

SIMGRID_JAR=$1
JAVA=$2
ENTRY_POINT=$3
STRIP=$4
shift 4

echo "$JAVA" -classpath "$SIMGRID_JAR" $ENTRY_POINT $1 $2

JSG_BUNDLE=$("$JAVA" -classpath "$SIMGRID_JAR" $ENTRY_POINT)

# sanity check
case "$JSG_BUNDLE" in
    NATIVE/*)
        cat >&2 <<EOF
-- [Java] Native libraries bundled into: ${JSG_BUNDLE}
EOF
        ;;
    *)
        cat >&2 <<EOF
-- [Java] Native libraries NOT bundled into invalid directory: ${JSG_BUNDLE}
EOF
        exit 1
        ;;
esac

# prepare directory
rm -rf NATIVE
rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi
mkdir -p "$JSG_BUNDLE"

if [ "$1" = "-so" ]; then
    shift
    for file; do
        [ "$file" != "-txt" ] || break
        cp -f "$file" "$JSG_BUNDLE"
        "$STRIP" -S "$JSG_BUNDLE/${file##*/}"
        shift
    done
fi

if [ "$1" = "-txt" ]; then
    shift
    for file; do
        cp -f "$file" "$JSG_BUNDLE"
        shift
    done
fi
