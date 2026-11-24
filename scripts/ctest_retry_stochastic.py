#!/usr/bin/env python3
"""Run ctest; if only known-stochastic tests failed, re-run all of them once.

Usage: ctest_retry_stochastic.py [--list FILE] [--] <ctest arguments>

The list file holds ctest test names (default: test/stochastic_tests.txt).
If ctest fails and every failed test is on the list, every listed test that
the ctest arguments select is run again, not just the ones that failed. The
result is that second run's exit status. If any unlisted test failed, the
first run's exit status is returned without a retry.
"""
import os
import re
import subprocess
import sys

FILTER_OPTIONS = ("-R", "--tests-regex", "-E", "--exclude-regex")


def read_list(path):
    names = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                names.add(line)
    return names


def run_ctest(args):
    return subprocess.call(["ctest"] + args)


def selected_tests(args):
    out = subprocess.run(["ctest", "-N"] + args, stdout=subprocess.PIPE,
                         universal_newlines=True).stdout
    return set(re.findall(r"^\s*Test\s+#\d+:\s+(\S+)", out, re.MULTILINE))


def failed_tests():
    path = os.path.join("Testing", "Temporary", "LastTestsFailed.log")
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return set(line.strip().split(":", 1)[1] for line in f if ":" in line)


def without_filters(args):
    kept = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
        elif arg in FILTER_OPTIONS:
            skip_next = True
        elif not arg.startswith(("--tests-regex=", "--exclude-regex=")):
            kept.append(arg)
    return kept


def main():
    default_list = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "test", "stochastic_tests.txt")
    args = sys.argv[1:]
    list_path = default_list
    if args[:1] == ["--list"]:
        list_path, args = args[1], args[2:]
    if args[:1] == ["--"]:
        args = args[1:]
    ctest_args = args

    stochastic = read_list(list_path)
    status = run_ctest(ctest_args)
    if status == 0:
        return 0

    failed = failed_tests()
    unlisted = sorted(failed - stochastic)
    if not failed or unlisted:
        print("ctest failed; not retrying (unlisted or unidentified failures: %s)"
              % (", ".join(unlisted) or "none identified"), flush=True)
        return status

    retry = sorted(selected_tests(ctest_args) & stochastic)
    print("Stochastic test(s) failed: %s" % ", ".join(sorted(failed)), flush=True)
    print("Re-running all %d stochastic tests once." % len(retry), flush=True)
    pattern = "^(" + "|".join(re.escape(t) for t in retry) + ")$"
    return run_ctest(without_filters(ctest_args) + ["-R", pattern])


if __name__ == "__main__":
    sys.exit(main())
