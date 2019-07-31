//Tellurium-Auto-Interface-exporter

#ifndef tai_exporterH
#define tai_exporterH

#if defined(_WIN32)
    #if defined(STATIC_TEL_AUTO)
        #define TA_DS
    #else
        #if defined(EXPORT_TEL_AUTO)
            #define TA_DS __declspec(dllexport)
        #else
            #define TA_DS __declspec(dllimport)
        #endif
    #endif
#else
    #define TA_DS
#endif

#endif
