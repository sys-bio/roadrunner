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

// == CODE ====================================================

#ifdef __cplusplus

#include <iostream>

namespace rr
{

/**
 * @author JKM
 * @brief Dump a backtrace of the caller
 * @param[out] s Steam to dump backtrace to
 */
void dumpBacktrace(std::ostream& s);

} // namespace rr

#endif // __cplusplus

#endif // rrBacktraceH
