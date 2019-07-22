#ifndef telPluginsAPISettingsH
#define telPluginsAPISettingsH

//C function calling conventions
#if defined(_MSC_VER)
    #define plugins_cc   __cdecl
    #define event_cc __cdecl      //For plugin event functions
#elif defined(__BORLANDC__)
    #define plugins_cc __cdecl
    #define event_cc  __cdecl
#else
    #define plugins_cc
    #define event_cc
#endif



#endif

