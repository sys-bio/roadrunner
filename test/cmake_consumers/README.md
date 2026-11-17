# CMake consumer tests

<!-- TODO: Write the final text. Summary of what belongs here:
- Purpose: build and run small projects that use an installed roadrunner through its CMake
  packages, so that the install tree and the documentation stay working.
- Folders: package (find_package(roadrunner): use_static, use_shared, use_c_static, use_c_shared,
  use_klu), legacy (include(ImportRoadrunnerAndDependencies): legacy_static, legacy_shared,
  legacy_c_static, legacy_c_shared), rrplugins (find_package(rrplugins): plugins_cpp, plugins_c).
  run_consumers.sh also builds the example in docs/source/UsingRoadRunnerFromCxx.
- Usage: run_consumers.sh <roadrunner install folder> <LLVM install folder> <build type> [work folder]
- Requirements: an installed roadrunner (not a build tree), the LLVM it was built with, CMake and a
  C++17 compiler; on Windows the script uses cygpath to put the install bin folder on PATH.
- CI: the test_cmake_consumer job in .github/workflows/main.yml, and build_script.yml, which runs
  it on the result of scripts/build_roadrunner.sh (-c).
- How to add a consumer: new folder, add its build_and_run line in run_consumers.sh, and add the
  expected output text it checks.
- Links: using_roadrunner_from_cxx.rst and the Installation page.
-->
