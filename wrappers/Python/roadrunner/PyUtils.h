/** @file PyUtils.h
  * @author Andy
  * @author JKM
  * @date 04/27/2014 Creation date
  * @date 09/04/2014 Add support for Python 3
  * @copyright Apache License, Version 2.0
  * @brief Low-level functions used in the Python wrapper
**/


#ifndef PYUTILS_H_
#define PYUTILS_H_

#include "Setting.h"
#include <Python.h>
#include <rr-libstruct/lsMatrix.h>
#include <cstdint>
#include "Matrix.h"
#include "Matrix3D.h"
#include <complex>
#include <vector>
#include "rrLogger.h"

namespace rr {

/**
 * @brief Convert a Python string object to a @a std::string
 * @details Python 2/3 wrappers independent.
 * Supports either Python 2 strings or Python 3 unicode
 * objects
 */
    std::string rrPyString_getCPPString(PyObject *uni);

/**
 * @brief Returns true if strings are equal, false otherwise
 * @details Python 2/3 wrappers independent
 */
    int rrPyCompareString(PyObject *uni, const char *str);

/**
 * @brief Creates a Python string/unicode object from a given UTF-8 buffer
 * @details Python 2/3 wrappers independent.
 * If using Python 2, will return a string object.
 * If using Python 3, will return a unicode object.
 */
    PyObject *rrPyString_FromString(const char *s);

/**
 * @brief Creates a Python string/unicode object from a given UTF-8 buffer
 * @details Python 2/3 wrappers independent.
 * Similar to @ref rrPyString_FromString but allows the user to
 * specify the length of the buffer.
 */
    PyObject *rrPyString_FromStringAndSize(const char *s, Py_ssize_t size);

/**
 * @brief Creates a Python string/unicode object from a given UTF-8 buffer
 * @details Python 2/3 wrappers independent.
 * Similar to @ref rrPyString_FromString but allows the user to
 * specify the length of the buffer.
 */
    char *rrPyString_AsString(PyObject *s);

/**
 * @brief Gets the current Python error message as C++ string
 * @details You take ownership of returned string.
 */
    char *rrGetPyErrMessage();

    class Dictionary;

/**
 * @brief Convert a variant to a Python object
 * @details Python 2/3 wrappers independent.
 * This conversion handles most simple types
 * (strings and basic numeric types).
 * The exact type of the resultant Python object
 * is dependent on the Python version, since
 * some types were removed/altered in Python 2->3
 */
    PyObject *Variant_to_py(const Setting &var);

    PyObject *Variant_to_py(Setting *var);

/**
 * @brief Inverse of @ref Variant_from_py
 */
    Setting Variant_from_py(PyObject *py);

    PyObject *dictionary_keys(const Dictionary *dict);

    PyObject *dictionary_values(const Dictionary *dict);

    PyObject *dictionary_items(const Dictionary *dict);

    PyObject *dictionary_getitem(const Dictionary *dict, const char *key);

    PyObject *dictionary_setitem(Dictionary *dict, const char *key, PyObject *value);

    void dictionary_delitem(Dictionary *dict, const char *key);

    PyObject *dictionary_contains(const Dictionary *dict, const char *key);

    PyObject *doublematrix_to_py(const ls::DoubleMatrix *mat, bool structured_result, bool copy_result);

/**
 * Casts a rr::Matrix<double> to its superclass ls::DoubleMatrix
 * and reuses doublematrix_to_py
 */
    PyObject *rrDoubleMatrix_to_py(const rr::Matrix<double> *m, bool copy_result);

    PyObject *stringvector_to_py(const std::vector<std::string> &vec);

    std::vector<std::string> py_to_stringvector(PyObject *obj);

/**
 * @brief serialization a python list using the C API
 * @details Imports serialization from C and uses dump. Returns a
 * PyObject serialization.
 * @details returned reference is owned
 */
    PyObject *PyList_toPickle(PyObject *list);


    Dictionary *Dictionary_from_py(PyObject *py);

/**
 * @brief Copy a buffer of doubles into a new numpy float64 array.
 * @details The values go through a Python bytearray and numpy.frombuffer,
 * so the compiled module does not need the numpy C API.
 * Returns a new reference, or nullptr with a Python error set.
 */
    PyObject *doubles_to_py(const double *data, size_t count);

/**
 * @brief Same as @ref doubles_to_py for complex128 values.
 */
    PyObject *complexes_to_py(const std::complex<double> *data, size_t count);

/**
 * @brief Wrap a bytearray of raw values as a numpy array of the given shape.
 * @details Steals the reference to buffer. dtype is a numpy dtype name
 * such as "float64". The array shares the bytearray's memory.
 */
    PyObject *buffer_to_array(PyObject *buffer, const char *dtype, const std::vector<Py_ssize_t> &shape);

/**
 * @brief Copy a Python sequence of numbers (list, tuple, numpy array) into a vector.
 * @details Returns false with a Python error set when the conversion fails.
 */
    bool py_to_doubles(PyObject *sequence, std::vector<double> &out);

    bool py_to_ints(PyObject *sequence, std::vector<int> &out);


/**
 * @brief convert a rr::Matrix3D<double, double>
 */
    class Matrix3DToNumpy {
    public:
        //using DoubleMatrix3D = rr::Matrix3D<double, double>;

        explicit Matrix3DToNumpy(rr::Matrix3D<double, double>* matrix);

        /**
         * @brief converts the index data field of the Double3DMatrix
         * to a 1D numpy array
         */
        PyObject *convertIndex();

        /**
         * @brief converts the data from the Double3DMatrix
         * to a 3D numpy array with the same dimensions.
         */
        PyObject *convertData();

        /**
         * @brief converts the rownames for the Matrix3D into a Python
         * list of strings.
         */
        PyObject *convertRowNames();

        /**
         * @brief converts the rownames for the Matrix3D into a Python
         * list of strings.
         */
        PyObject *convertColNames();

    private:
        /**
         * @brief the Matrix3D<double, double> to convert
         */
        rr::Matrix3D<double, double>* matrix_;
    };





//PyMODINIT_FUNC
//initcheese(void)
//{
//   (void)Py_InitModule("cheese", mymethods);
//   import_array();
//}

} /* namespace rr */

#endif /* PYUTILS_H_ */
