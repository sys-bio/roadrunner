/*
 * PyUtils.h
 *
 *  Created on: Apr 27, 2014
 *      Author: andy
 */

#ifndef PYUTILS_H_
#define PYUTILS_H_

#include "Variant.h"
#include <Python.h>

namespace rr
{

/**
 * @brief Convert a Python string object to a @a std::string
 * @details Python 2/3 API independent.
 * Supports either Python 2 strings or Python 3 unicode
 * objects
 */
std::string rrPyString_getCPPString(PyObject* uni);

/**
 * @brief Returns true if strings are equal, false otherwise
 * @details Python 2/3 API independent
 */
int rrPyCompareString(PyObject* uni, const char* str);

/**
 * @brief Creates a Python string/unicode object from a given UTF-8 buffer
 * @details Python 2/3 API independent.
 * If using Python 2, will return a string object.
 * If using Python 3, will return a unicode object.
 */
PyObject* rrPyString_FromString(const char* s);

/**
 * @brief Creates a Python string/unicode object from a given UTF-8 buffer
 * @details Python 2/3 API independent.
 * Similar to @ref rrPyString_FromString but allows the user to
 * specify the length of the buffer.
 */
PyObject* rrPyString_FromStringAndSize(const char* s, Py_ssize_t size);

/**
 * @brief Convert a variant to a Python object
 * @details Python 2/3 API independent.
 * This conversion handles most simple types
 * (strings and basic numeric types).
 * The exact type of the resultant Python object
 * is dependent on the Python version, since
 * some types were removed/altered in Python 2->3
 */
PyObject *Variant_to_py(const Variant& var);

/**
 * @brief Inverse of @ref Variant_from_py
 */
Variant Variant_from_py(PyObject *py);

} /* namespace rr */

#endif /* PYUTILS_H_ */
