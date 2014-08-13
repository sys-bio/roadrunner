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

#include <sstream>
#include <cassert>
#include <string.h>

// == CODE ====================================================

namespace rr
{

auto Backtrace::fromHere() -> BacktracePtr {
#ifdef WINVER
    assert(0 && "FATAL: implement backtrace on Windows");
#endif
#if !defined(__GLIBC__)
    assert(0 && "FATAL: no backtrace support on this platform");
#endif
#   ifdef __GLIBC__
    return BacktracePtr(new libstdcxx_Backtrace());
#   endif
}

#   ifdef __GLIBC__

void libstdcxx_Backtrace::dump(std::ostream& s) const {
    // see http://panthema.net/2008/0901-stacktrace-demangled/

    void** addrbuf = nullptr;
    int max = 32;
    int size;
    do {
        free(addrbuf);

        addrbuf = (void**)malloc(max*sizeof(*addrbuf));
        size = backtrace(addrbuf, max);
//         s << "size = " << size << ", max = " << max << std::endl;

    } while((size == max) && (max <<= 1));

    char** symbols = backtrace_symbols(addrbuf, size);

    assert(symbols && "Internal: failed to allocate");

    for(int i=start_; i<size-1; ++i) {
//         s << "sym " << symbols[i] << std::endl;
        std::string demangled = demangle(symbols[i]);
        if (demangled.size())
            s << "* " << demangled << "\n";
        else
            s << "* " << symbols[i] << "\n";
        s << "   in " << getLocation(symbols[i]) << "\n";

    }

    free(symbols);

    free(addrbuf);
}

std::string libstdcxx_Backtrace::demangle(char* btsym) const {
    char* newstr = (char*)malloc((strlen(btsym)+1)*sizeof(char));
    btsym = strcpy(newstr, btsym);
    char* x = btsym;
    while(x) {
        if (*x == '(') {
            ++x;
            break;
        }
        ++x;
    }
    assert(x && "Internal: not valid backtrace output");

    char* start = x;

    while(x && *x != '+') {
        ++x;
    }
    if (!x) {
        free(btsym);
        return "";
    }
    assert(*x == '+' && "Internal: not valid backtrace output");

    *x = '\0';

    int status;
    char* p = abi::__cxa_demangle(start, NULL, NULL, &status);
    if(status) {
        free(btsym);
        return "";
    }

    std::string result(p);
    free(p);

    free(btsym);

    return result;
}

std::string libstdcxx_Backtrace::getLocation(char* btsym) const {
    // http://blog.bigpixel.ro/2010/09/stack-unwinding-stack-trace-with-gcc/
    char* newstr = (char*)malloc((strlen(btsym)+1)*sizeof(char));
    btsym = strcpy(newstr, btsym);

    while (btsym && *btsym != '[')
        ++btsym;
    if (!btsym) {
        free(newstr);
        return "";
    }
    if (*btsym != '[') {
        free(newstr);
        return "";
    }
    ++btsym;
    if (!btsym) {
        free(newstr);
        return "";
    }

    char* p = btsym;
    ++p;
    while (p && *p != ']')
        ++p;
    if (!p) {
        free(newstr);
        return "";
    }
    if (*p != ']') {
        free(newstr);
        return "";
    }
    *p = '\0';

    char sbuf[256];
    sprintf(sbuf, "addr2line %s -e /home/poltergeist/home/devel/install/roadrunner/bin/rr",  btsym);
//     std::cerr << sbuf << "\n";

    FILE* f = popen(sbuf,  "r");
    assert(f && "Internal: backtrace failed");

    fgets(sbuf, 256, f);
    fgets(sbuf, 256, f);

    if (sbuf[0] == '?') {
        free(newstr);
        return "unknown";
    }

//     std::cerr << sbuf << "\n";

    p = sbuf;
    int n=0;
    while (*p && *p != ':') {
        ++p;
        ++n;
    }

    std::string file(sbuf, n);

    ++p;
    char* q = p;
    n=0;
    while (*p && *p != ' ' && *p != '\n' && *p != '\r') {
        ++p;
        ++n;
    }
    std::string line(q, n);

    pclose(f);

    free(newstr);

    return file + ":" + line;
}

void libstdcxx_Backtrace::setStartDepth(int start) {
    start_ = start;
}

#     endif // __GLIBC__

std::string btStringFromDepth(int depth) {
    std::stringstream ss;
    Backtrace::BacktracePtr bt = Backtrace::fromHere();
    bt->setStartDepth(2+depth);
    bt->dump(ss);
    return ss.str();
}

std::string btStringFromHere() {
    return btStringFromDepth(0);
}

} // namespace rr
