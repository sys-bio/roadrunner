#ifndef telJobsExporterH
#define telJobsExporterH

#if defined(_WIN32)
    #if defined(STATIC_PLUGIN_API)
        #define JOBS_DECLSPEC
    #else
        #if defined(EXPORT_JOBS_API)
            #define JOBS_DECLSPEC __declspec(dllexport)
        #else
            #define JOBS_DECLSPEC __declspec(dllimport)
        #endif
    #endif
#else
    #define JOBS_DECLSPEC
#endif


#endif

