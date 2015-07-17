// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

/** @file nullptr.h
 * @brief Defines nullptr for pre C++11 compilers
 */

/*
 * Backtrace.h
 *
 *  Created on: Aug 13, 2014
 *      Author: JKM
 */

#ifndef rr_platfomr_nullptrH
#define rr_platfomr_nullptrH

// == CODE ====================================================

#if __cplusplus <= 199711L
#define nullptr NULL
#endif

#endif // rrBacktraceH
