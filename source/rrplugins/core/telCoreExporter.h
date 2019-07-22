#ifndef telCoreExporterH
#define telCoreExporterH

#if defined(_WIN32)
    #if defined(STATIC_PLUGIN_API_CORE)
        #define CORE_DECLSPEC
    #else
        #if defined(EXPORT_CORE_API)
            #define CORE_DECLSPEC __declspec(dllexport)
        #else
            #define CORE_DECLSPEC __declspec(dllimport)
        #endif
    #endif
#else
    #define CORE_DECLSPEC
#endif

//Helper macros for plugins
#if defined(_WIN32)
    #if defined(EXPORT_TEL_PLUGIN)
        #define TLP_DS __declspec(dllexport)
    #else
        #define TLP_DS __declspec(dllimport)
    #endif
#else
    #define TLP_DS
#endif

//Function calling conventions
#if defined(__BORLANDC__)
    #define call_conv __cdecl
#elif defined(_MSC_VER)
    #define call_conv __cdecl
#else
    #define call_conv
#endif


#endif

