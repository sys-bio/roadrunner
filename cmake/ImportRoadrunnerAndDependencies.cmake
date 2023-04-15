# CMake module for importing roadrunner targets with dependencies
# into consuming libraries cmake system.
#
# Clients need roadrunner but must also import the dependencies
# with find_package. Its easy to bundle these together
# and document it.
#
find_package(Threads) # for libxml2, FindThreads.cmake is shipped with cmake
find_package(LibLZMA) # for libxml2, LibLZMA.cmake is shipped with cmake
find_package(zlib CONFIG REQUIRED)
find_package(bzip2 CONFIG REQUIRED)
#find_package(iconv CONFIG REQUIRED)
#find_package(LibXml2 CONFIG REQUIRED)
find_package(expat CONFIG REQUIRED)
find_package(sbml-static CONFIG REQUIRED)
find_package(rr-libstruct CONFIG REQUIRED)
find_package(clapack CONFIG REQUIRED)
find_package(nleq1 CONFIG REQUIRED)
find_package(nleq2 CONFIG REQUIRED)
find_package(PocoFoundation CONFIG REQUIRED)
find_package(PocoNet CONFIG REQUIRED)
find_package(PocoXML CONFIG REQUIRED)
find_package(SUNDIALS CONFIG REQUIRED)
find_package(LLVM REQUIRED)
find_package(roadrunner-static CONFIG REQUIRED)
find_package(roadrunner CONFIG REQUIRED)
find_package(roadrunner_c_api CONFIG REQUIRED)

