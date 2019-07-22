#ifndef telCommonExporterH
#define telCommonExporterH

#if defined(_WIN32)
    #if defined(STATIC_PLUGIN_API_COMMON)
        #define COMMON_DECLSPEC
    #else
        #if defined(EXPORT_COMMON_API)
            #define COMMON_DECLSPEC __declspec(dllexport)
        #else
            #define COMMON_DECLSPEC __declspec(dllimport)
        #endif
    #endif
#else
    #define COMMON_DECLSPEC
#endif


#endif

