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
#include <numpy/arrayobject.h>
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

    // docs: https://numpy.org/devdocs/user/c-info.beyond-basics.html#subtyping-the-ndarray-in-c
    struct NamedArrayObject {
        /**
         * Note, access to the PyArrayObject is deprecated as of numpy v1.7
         * See PyArrayObject_fields in numpy ndarraytypes.h
         */
        PyArrayObject array;
        PyObject *rowNames = nullptr;
        PyObject *colNames = nullptr;

        /**
         * @brief create a binary string from the array field of NamedArrayObject.
         * @returns PyObject* of type Bytes
         */
        PyObject *saveToBytes() {
            PyArrayObject *arr = &array;
            PyObject *arrayBytes = PyArray_ToString(arr, NPY_CORDER);
            if (arrayBytes == nullptr) {
                std::ostringstream err;
                err << "Could not create bytes object the array (" << PyBytes_Size(arrayBytes)
                    << " bytes) field of the NamedArrayObject";
                PyErr_SetString(PyExc_ValueError, err.str().c_str());
            }
            return arrayBytes;
        }

        void loadFromBytes(PyObject *bytesObject, long* dim1, long* dim2 = nullptr) {
            if (strcmp(bytesObject->ob_type->tp_name, "bytes") != 0) {
                std::ostringstream err;
                err << "Cannot accept object of type " << bytesObject->ob_type->tp_name
                    << " expected a bytes object" << std::endl;
                PyErr_SetString(PyExc_TypeError, err.str().c_str());
            }

            int size = 0;
            if (dim2){
                size = *dim1 * *dim2;
            } else {
                size = *dim1;
            }
            /**
             * We can load our bytes object back into a numpy array
             * with:
             *      PyObject *PyArray_FromBuffer(PyObject *buf, PyArray_Descr *dtype, npy_intp count, npy_intp offset)
             *
             */
            PyArray_Descr *descr = PyArray_DescrFromType(NPY_DOUBLE);
            if (!descr) {
                PyErr_SetString(PyExc_ValueError, "Could not create PyArray_Descr in NamedArray.__setstate__");
                return ;
            }

            PyArrayObject *newArrFromBytes = (PyArrayObject *) PyArray_FromBuffer(bytesObject, descr, -1, 0);
            if (!newArrFromBytes) {
                PyErr_SetString(PyExc_ValueError,
                                "Could not create a PyArrayObject from a bytes buffer using PyArray_FromBuffer");
                return ;
            }

            double *dataFromBytesArray = (double *) PyArray_DATA(newArrFromBytes);
            double *dataFromSelfArray = (double *) PyArray_DATA(&array);
            for (int i = 0; i < size; i++) {
                std::swap(dataFromBytesArray[i], dataFromSelfArray[i]);
            }
            PyArray_Type.tp_dealloc((PyObject*)newArrFromBytes);
        }
    };


    /**
     * @brief allocates a NamedArrayObject, the underlying struct for
     * the Python NamedArray object.
     * @return new reference. PyObject* with a ref count of 1.
     * @details A NamedArrayObject is allocated using PyObject_New
     * and then initialized with PyObject_Init. The PyObject*
     * returned is cast from a NamedArrayObject and
     * the array field of this object has the same address.
     * In other words
     *  &namedArrayObject == &(namedArrayObject->array);
     */
    static PyObject *NamedArrayObject_alloc(PyTypeObject *type, Py_ssize_t nitems);

    /**
     *@brief Finalize the instantiation of a NamedArray
     *
     * @details numpy types call __array_finalize_ex__ after
     * allocating and initializing. There are three ways in which
     * a numpy array can be created
     *  1) Using the constructor, like normal python objects
     *  2) Using a "view" of a supertype as a subtype (i.e. np.ndarray as NamedArray)
     *  3) Using a "template" of a type, which is where the user slices an array and
     *  get a copy returned.
     *
     * In 1, the argument to __finalize_ex__ is None, and the function does nothing.
     * In 2, the argument the argument is of type np.ndarray and the
     *  return type is a NamedArray
     * In 3, both the input and output to __finalize_ex__ is a NamedArray
     *
     * @see https://numpy.org/devdocs/user/basics.subclassing.html
     */
    static PyObject *NamedArrayObject_Finalize(NamedArrayObject *self, PyObject *args);

    PyObject *NamedArrayObject_Finalize_FromConstructor(NamedArrayObject *self);

    PyObject *NamedArrayObject_Finalize_FromPyArray(NamedArrayObject *self);

    PyObject *NamedArrayObject_Finalize_FromNamedArray(NamedArrayObject *self, PyObject *rhs);


    static PyObject *NamedArray_repr(NamedArrayObject *self);

    static PyObject *NamedArray_str(NamedArrayObject *self);

    static PyObject *
    NamedArray___getstate__(NamedArrayObject *self, PyObject *Py_UNUSED(ignored));

    /**
     * @brief implementation of __reduce__ method
     * for NamedArray
     * @return tuple[callable, tuple, dict]
     * @details builds the tuple expected from the
     * __reduce__ method. This is a callable
     * which takes a tuple as argument and a
     * dictionary state to be passed onto __setstate__.
     * We use the NamedArray constructor as the callable,
     * with its dimensions for its arguments and
     * the return value of NamedArray___getstate__
     * for the state.
     */
    static PyObject *
    NamedArray___reduce_ex__(NamedArrayObject *self, PyObject *args);

    static PyObject *
    NamedArray___setstate__(NamedArrayObject *self, PyObject *state);

    static PyObject *NamedArray_New(int nd, npy_intp *dims, double *data, int pyFlags,
                                    const ls::DoubleMatrix *mat);

    static void NamedArrayObject_dealloc(NamedArrayObject *self);

/**
 * @brief Initialize the pyutil module inside swig
 * @details This method is called once at inside
 * roadrunerPYTHON_wrap.cxx. The command controlling
 * this action is inside roadrunners swig interface file
 * under the `%init` directive.
 */
    void pyutil_init(PyObject *module);

/**
 * @brief convert a rr::Matrix3D<double, double>
 */
    class Matrix3DToNumpy {
    public:
        using DoubleMatrix3D = rr::Matrix3D<double, double>;

        explicit Matrix3DToNumpy(DoubleMatrix3D &matrix);

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
        DoubleMatrix3D &matrix_;
    };





//PyMODINIT_FUNC
//initcheese(void)
//{
//   (void)Py_InitModule("cheese", mymethods);
//   import_array();
//}

} /* namespace rr */

#endif /* PYUTILS_H_ */
