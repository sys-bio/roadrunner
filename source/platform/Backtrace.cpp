// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

/*
 * Backtrace.cpp
 *
 *  Created on: Aug 13, 2014
 *      Author: JKM
 */

// == INCLUDES ================================================

#include "Backtrace.h"
#include "nullptr.h"

// backtrace dumping in libstdc++
#if defined(__GLIBCXX__)
#include <execinfo.h>
#include <cxxabi.h>
#endif

// == CODE ====================================================

namespace rr
{

void dumpBacktrace(std::ostream& s) {
    void** addrbuf = nullptr;
    int max = 32;
    int size;
    do {
        if(addrbuf)
            free(addrbuf);

        addrbuf = (void**)malloc(max*sizeof(*addrbuf));
        size = backtrace(addrbuf, max);
        std::cerr << "size = " << size << ", max = " << max << std::endl;

    } while((size == max) && (max <<= 1));
}

}
