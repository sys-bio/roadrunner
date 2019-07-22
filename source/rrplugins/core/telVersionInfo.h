#ifndef telVersionInfoH
#define telVersionInfoH
#include "telCoreExporter.h"
#include <string>

namespace rr{

#define TLP_VERSION_STR "1.1.1"

#if defined(__llvm__)
    #define TLP_COMPILER "clang"
#endif

#if defined(__GNUC__) && !defined(__llvm__)
    #define TLP_COMPILER "gcc"
#endif

#if defined (_MSC_VER)
    #if (_MSC_VER == 1700)
        #define TLP_COMPILER "Microsoft Visual Studio 2012"
    #elif (_MSC_VER == 1600)
        #define TLP_COMPILER "Microsoft Visual Studio 2010"
    #elif (_MSC_VER == 1500)
        #define TLP_COMPILER "Microsoft Visual Studio 2008"
    #elif (_MSC_VER == 1400)
        #define TLP_COMPILER "Microsoft Visual Studio 2005"
    #elif (_MSC_VER == 1310)
        #define TLP_COMPILER "Microsoft Visual Studio 2003"
    #else
        #define TLP_COMPILER "Microsoft Visual Studio Unknown Version"
    #endif
#endif

#if defined (__BORLANDC__)
    #define TLP_COMPILER "Borland"
#endif

#if !defined(TLP_COMPILER)
    #define TLP_COMPILER "Unknown compiler"
#endif


/**
 * get the API version string.
 */
CORE_DECLSPEC std::string getVersion();

/**
 * get the copyright string.
 */
CORE_DECLSPEC std::string getCopyright();


}

#endif
