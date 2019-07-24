#ifndef telplugins_settingsH
#define telplugins_settingsH

//C function calling conventions
#if defined(_MSC_VER)
    #define tlp_cc    __cdecl
    #define event_cc  __cdecl
#elif defined(__BORLANDC__)
    #define tlp_cc      __cdecl
    #define event_cc    __cdecl
#else
    #define tlp_cc
    #define event_cc
#endif

#if !defined(__cplusplus)
    #if !defined(_MSC_VER)
        #include "stdbool.h"
    #else   //VS don't have stdbool.h
        /*!
         \brief Typedef for boolean type in C
        */
        typedef int bool;
        #define false 0
        #define true 1
    #endif
#endif

#endif
