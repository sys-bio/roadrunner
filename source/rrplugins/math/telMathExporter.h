#ifndef telMathExporterH
#define telMathExporterH

#if defined(_WIN32)
    #if defined(STATIC_PLUGIN_API_MATH)
        #define MATH_DECLSPEC
    #else
        #if defined(EXPORT_MATH_API)
            #define MATH_DECLSPEC __declspec(dllexport)
        #else
            #define MATH_DECLSPEC __declspec(dllimport)
        #endif
    #endif
#else
    #define MATH_DECLSPEC
#endif


#endif

