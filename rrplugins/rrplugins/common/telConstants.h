#ifndef telConstantsH
#define telConstantsH
#include <limits>
#include <string>
#include <cstdlib>
#include "telCommonExporter.h"

typedef unsigned int    u_int;

namespace tlp
{

//Useful constants...
COMMON_DECLSPEC extern const char           gPathSeparator;
COMMON_DECLSPEC extern const std::string    gExeSuffix;

COMMON_DECLSPEC extern const char* 			gDoubleFormat;
COMMON_DECLSPEC extern const char* 			gIntFormat;
COMMON_DECLSPEC extern const char* 			gComma;
COMMON_DECLSPEC extern const std::string 	gDefaultSupportCodeFolder;
COMMON_DECLSPEC extern const std::string 	gDefaultCompiler;
COMMON_DECLSPEC extern const std::string 	gDefaultTempFolder;
COMMON_DECLSPEC extern const std::string 	gNoneString;
COMMON_DECLSPEC extern const std::string	gEmptyString;

//Messages
COMMON_DECLSPEC extern const std::string    gEmptyModelMessage;


// Constants
COMMON_DECLSPEC extern const char           gTab;
COMMON_DECLSPEC extern const char           gNL;
COMMON_DECLSPEC extern const double         gDoubleNaN;
COMMON_DECLSPEC extern const float          gFloatNaN;
COMMON_DECLSPEC extern const int            gMaxPath;

// Enums...
enum SBMLType {stCompartment = 0, stSpecies, stParameter};    //Species clashes with class Species, prefix enums with st, for SbmlType

}
#endif
