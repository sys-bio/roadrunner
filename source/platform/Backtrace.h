// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

/*
 * Backtrace.h
 *
 *  Created on: Aug 13, 2014
 *      Author: JKM
 */

#ifndef rrBacktraceH
#define rrBacktraceH

// == INCLUDES ================================================

#include "rrExporter.h"

#include <iostream>

#if (__cplusplus >= 201103L) || defined(_MSC_VER)
#include <memory>
#define cxx11_ns std
#else
#include <tr1/memory>
#define cxx11_ns std::tr1
#endif

// == CODE ====================================================

#ifdef __cplusplus

namespace rr
{

/**
 * @author JKM
 * @brief Class to encapsulate a backtrace
 */
class Backtrace {
public:
    typedef std::unique_ptr<Backtrace> BacktracePtr;

    /// Factory method for whatever the current platform is
    static BacktracePtr fromHere();

    /**
    * @brief Dump a backtrace of the caller
    * @param[out] s Steam to dump backtrace to
    */
    virtual void dump(std::ostream& s) const = 0;

    /**
    * @brief Set the start depth of the stack trace
    * @param[out] start Desired start depth
    */
    virtual void setStartDepth(int start) = 0;
};

#   ifdef __GLIBC__

/**
 * @author JKM
 * @brief libstdc++ specialization of Backtrace
 */
class libstdcxx_Backtrace : public Backtrace {
public:
    /**
    * @brief Dump a backtrace of the caller
    * @param[out] s Steam to dump backtrace to
    */
    void dump(std::ostream& s) const;

    /**
    * @brief Set the start depth of the stack trace
    * @param[out] start Desired start depth
    */
    void setStartDepth(int start);

private:
    std::string demangle(char* btsym) const;

    std::string getLocation(char* btsym) const;

    int start_ = 1;
};

#     endif // __GLIBC__

std::string btStringFromHere();

/**
 * @author JKM
 * @brief Print backtrace starting at given depth
 */
std::string btStringFromDepth(int depth);

} // namespace rr

#endif // __cplusplus

#endif // rrBacktraceH
