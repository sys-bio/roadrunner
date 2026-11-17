#include "telVersionInfo.h"

// The version of roadrunner, which the build passes in.
#ifndef TLP_VERSION_STR
#error "TLP_VERSION_STR must be defined by the build."
#endif

namespace tlp {

std::string getVersion()
{
    return TLP_VERSION_STR;
}

std::string getCopyright()
{
    return "(C) 2009-2021  Totte Karlsson, H. M. Sauro";
}


}


