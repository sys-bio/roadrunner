#ifndef rrAutoPluginExporterH
#define rrAutoPluginExporterH

#if defined(_WIN32)
    #if defined(STATIC_AUTO_PLUGIN)
        #define AUTO_PLUGIN_DECLSPEC
    #else
        #if defined(EXPORT_AUTO_PLUGIN)
            #define AUTO_PLUGIN_DECLSPEC __declspec(dllexport)
        #else
            #define AUTO_PLUGIN_DECLSPEC __declspec(dllimport)
        #endif
    #endif
#else
    #define AUTO_PLUGIN_DECLSPEC
#endif

#if defined(_MSC_VER) || defined(__BORLANDC__)
    #define autoCallConv __cdecl
#else
    #define autoCallConv
#endif

#endif
