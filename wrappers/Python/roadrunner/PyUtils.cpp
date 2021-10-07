/*
 * PyUtils.cpp
 *
 *  Created on: Apr 27, 2014
 *      Author: andy
 */

// see discission on import array,
// http://docs.scipy.org/doc/numpy/reference/c-api.array.html#miscellaneous

/**
 * @brief define some macro's to allow numpy api to be used
 * in multiple compilation units.
 *
 * @details We need this because the swig generated roadrunnerPYTOHN_wrap.cxx
 * and PyUtils.cpp both use the numpy C API.
 *
 * By default the numpy C API is stored as an
 * array of pointers (void **) called PyArray_API. This
 * is initialized when import_array() is called. What we'll get is
 *
 *  static void** PyArray_API;
 *
 * There could be problems with having a static PyArray_API when multiple
 * compilations units are in effect. The PY_ARRAY_UNIQUE_SYMBOL macro
 * is defined to deal with this problem. This renames the PyArray_API
 * to whatever PY_ARRAY_UNIQUE_SYMBOL is defined to be, in our case
 * RoadRunner_ARRAY_API (see numpy's __multiarray_api.h for
 * macro definitions). PY_ARRAY_UNIQUE_SYMBOL also makes the
 * declaration of the numpy api non-static.
 *
 * When combined with NO_IMPORT_ARRAY (or equally, NO_IMPORT) the declaration
 * becomes
 *  `extern void** RoadRunner_ARRAY_API;`
 * making the API a global variable. Therefore we have access to the RoadRunner_ARRAY_API
 * from other compilation units. The second consequence of NO_IMPORT_ARRAY
 * is that it removes the definition of import_array, so that it cannot be called
 * from within this compilation unit.
 *
 * So, since import_array() is called from the swig
 * %init we define NO_IMPORT_ARRAY and PY_ARRAY_UNIQUE_SYMBOL
 * to no reimport array and get a handle on the name of the api.
 *
 */
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL RoadRunner_ARRAY_API
//Still using some of deprecated wrappers
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <stdexcept>
#include <string>
#include <algorithm>
#include "rrLogger.h"

// wierdness on OSX clang, this needs to be included before python.h,
// otherwise compile pukes with:
// localefwd.h error: too many arguments provided to function-like macro invocation
#include <sstream>
#include <vector>
#include "PyUtils.h"
#include <rrLogger.h>
#include <Dictionary.h>
#include "rrConfig.h"
#include "rrRoadRunnerOptions.h"
#include <numpy/arrayobject.h>
#include "structmember.h"

// python object.h
#include "object.h"
#include "modsupport.h"

#include <iostream>
#include <iomanip>
#include <Matrix.h>
#include <filesystem>
#include <fstream>

using namespace std;

typedef vector<string> str_vector;

namespace rr {

    //    https://pythonextensionpatterns.readthedocs.io/en/latest/pickle.html#implementing-getstate
    static const char *PICKLE_VERSION_KEY = "_pickle_version";
    static int PICKLE_VERSION = 5;

#define STRINGIFY(x) #x
#define EXPAND_AND_STRINGIFY(x) STRINGIFY(x)

    char *rr_strclone(const char *src) {
        if (!src) {
            assert(0 && "rr_strclone passed null arg");
            return NULL;
        } else {
            size_t size = strlen(src) + 1;
            char *dst = (char *) malloc(size * sizeof(char));

            memcpy(dst, src, size); // copies null char

            return dst;
        }
    }

    void rr_strfree(const char *str) {
        free((void *) str);
    }

/**
 * @brief Get the UTF-8 encoded buffer for a Python string
 * @details Imported from graphfab
 * @note Caller must free the buffer using @ref rr_strfree
 */
    char *rrPyString_getString(PyObject *uni) {
        char *str = NULL;
//     #pragma message "PY_MAJOR_VERSION = " EXPAND_AND_STRINGIFY(PY_MAJOR_VERSION)
#if PY_MAJOR_VERSION == 3
        PyObject *bytes = PyUnicode_AsUTF8String(uni);
        str = rr_strclone(PyBytes_AsString(bytes));
        Py_XDECREF(bytes);
#else
        str = rr_strclone(PyString_AsString(uni));
#endif
        return str;
    }

    std::string rrPyString_getCPPString(PyObject *uni) {
        char *cstr = rrPyString_getString(uni);
        std::string str(cstr);
        rr_strfree(cstr);
        return str;
    }

    int rrPyCompareString(PyObject *uni, const char *str) {
#if SAGITTARIUS_DEBUG_LEVEL >= 2
        {
            printf("PyCompareString started\n");
        }
#endif
        {
            char *s = rrPyString_getString(uni);
            int cmp = !strcmp(s, str);
            rr_strfree(s);

            if (cmp)
                return 1;
            else
                return 0;
        }
    }

    PyObject *rrPyString_FromString(const char *s) {
#if PY_MAJOR_VERSION == 3
        return PyUnicode_FromString(s);
#else
        return PyString_FromString(s);
#endif
    }

    PyObject *rrPyString_FromStringAndSize(const char *s, Py_ssize_t size) {
#if PY_MAJOR_VERSION == 3
        return PyUnicode_FromStringAndSize(s, size);
#else
        return PyString_FromStringAndSize(s, size);
#endif
    }

    char *rrPyString_AsString(PyObject *s) {
#if PY_MAJOR_VERSION == 3
        char *str;
        PyObject *bytes = PyUnicode_AsUTF8String(s);
        str = rr_strclone(PyBytes_AsString(bytes));
        Py_XDECREF(bytes);
        return str;
#else
        return rr_strclone(PyString_AsString(s));
#endif
    }

    char *rrGetPyErrMessage() {
        // It is a bit convoluted to get the Python error message as a string
        // from https://stackoverflow.com/a/1418703
        // Don't need to free these as they are only "references"
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);

        if (ptype == NULL) {
            return NULL;
        }

        //Get error message
        char *pStrErrorMessage = rrPyString_AsString(pvalue);
        return pStrErrorMessage;
    }

    PyObject *Variant_to_py(const Setting &var) {
        PyObject *result = nullptr;

        const std::type_info &type = var.typeInfo();

        if (type == typeid(std::monostate)) {
            Py_RETURN_NONE;
        }

        if (type == typeid(std::string)) {
            return rrPyString_FromString(var.get<string>().c_str());
        }

        if (type == typeid(bool)) {
            return PyBool_FromLong(var.get<bool>());
        }

        if (type == typeid(std::uint64_t)) {
            return PyLong_FromUnsignedLong(var.get<std::uint64_t>());
        }

        if (type == typeid(std::int64_t)) {
            return PyLong_FromLong(var.get<std::int64_t>());
        }

        if (type == typeid(std::int32_t)) {
# if PY_MAJOR_VERSION == 3
            return PyLong_FromLong(var.get<std::int32_t>());
# else
            return PyInt_FromLong(var.get<std::int32_t>());
# endif
        }

        if (type == typeid(std::uint32_t)) {
            return PyLong_FromUnsignedLong(var.get<std::uint32_t>());
        }

        if (type == typeid(char)) {
            char c = var.get<char>();
            return rrPyString_FromStringAndSize(&c, 1);
        }

        if (type == typeid(unsigned char)) {
# if PY_MAJOR_VERSION == 3
            return PyLong_FromLong((long) var.get<std::int64_t>());
# else
            return PyInt_FromLong(var.get<std::int64_t>());
# endif
        }

        if (type == typeid(float)) {
            return PyFloat_FromDouble(var.get<float>());
        }

        if (type == typeid(double)) {
            return PyFloat_FromDouble(var.get<double>());
        }

        if (type == typeid(vector<double>)) {
            PyObject *list = PyList_New((std::int64_t) var.get<vector<double> >().size());
            if (!list) throw logic_error("Unable to allocate memory for Python list");
            for (unsigned int i = 0; i < var.get<vector<double> >().size(); i++) {
                PyObject *num = PyFloat_FromDouble((double) var.get<vector<double> >()[i]);
                if (!num) {
                    Py_DECREF(list);
                    throw logic_error("Unable to allocate memory for Python list");
                }
                PyList_SET_ITEM(list, i, num);
            }

            return list;
        }


        throw invalid_argument("could not convert " + var.get<std::string>() + "to Python object");
    }

    PyObject *Variant_to_py(Setting *var) {
        return Variant_to_py(*var);
    }

    Setting Variant_from_py(PyObject *py) {
        Setting var;

        if (py == Py_None) {
            return var;
        }

# if PY_MAJOR_VERSION == 3
        if (PyUnicode_Check(py))
# else
            if (PyString_Check(py))
# endif
        {
            var = rrPyString_getCPPString(py);
            return var;
        } else if (PyBool_Check(py)) {
            var = (bool) (py == Py_True);
            return var;
        } else if (PyLong_Check(py)) {
            // need to check for overflow.
            var = std::int64_t(PyLong_AsLong(py));

            // Borrowed reference.
            PyObject *err = PyErr_Occurred();
            if (err) {
                char *message = rrGetPyErrMessage();
                std::stringstream ss;
                ss << "Could not convert Python long to C ";
                ss << sizeof(long) * 8 << " bit long: ";
                ss << std::string(message);
                // clear error, raise our own
                PyErr_Clear();
                rr_strfree(message);

                throw invalid_argument(ss.str());
            }

            return var;
        }

# if PY_MAJOR_VERSION == 3
        else if (PyLong_Check(py))
# else
            else if (PyInt_Check(py))
# endif
        {
# if PY_MAJOR_VERSION == 3
            var = std::int64_t(PyLong_AsLong(py));
# else
            var = (std::int32_t)PyInt_AsLong(py);
# endif
            return var;
        } else if (PyFloat_Check(py)) {
            var = PyFloat_AsDouble(py);
            return var;
        } else if (PyList_Check(py)) {
            std::vector<double> data;
            for (Py_ssize_t i = 0; i < PyList_Size(py); i++) {
                PyObject *value = PyList_GetItem(py, i);
                data.push_back(PyFloat_AsDouble(value));
            }
            return data;
        }


        std::string msg = "could not convert Python type to built in type";
        throw invalid_argument(msg);
    }

    PyObject *dictionary_keys(const Dictionary *dict) {
        std::vector<std::string> keys = dict->getKeys();

        size_t size = keys.size();

        PyObject *pyList = PyList_New(size);

        unsigned j = 0;

        for (std::vector<std::string>::const_iterator i = keys.begin(); i != keys.end(); ++i) {
            const std::string &key = *i;
            PyObject *pyStr = rrPyString_FromString(key.c_str());
            PyList_SET_ITEM(pyList, j++, pyStr);
        }

        return pyList;
    }

    PyObject *dictionary_values(const Dictionary *dict) {
        std::vector<std::string> keys = dict->getKeys();

        size_t size = keys.size();

        PyObject *pyList = PyList_New(size);

        unsigned j = 0;

        for (std::vector<std::string>::const_iterator i = keys.begin(); i != keys.end(); ++i) {
            const std::string &key = *i;
            PyObject *pyVal = Variant_to_py(dict->getItem(key));
            PyList_SET_ITEM(pyList, j++, pyVal);
        }

        return pyList;
    }

    PyObject *dictionary_items(const Dictionary *dict) {
        std::vector<std::string> keys = dict->getKeys();

        size_t size = keys.size();

        PyObject *pyList = PyList_New(size);

        unsigned j = 0;

        for (std::vector<std::string>::const_iterator i = keys.begin(); i != keys.end(); ++i) {
            const std::string &key = *i;
            PyObject *pyStr = Variant_to_py(dict->getItem(key));

            PyObject *pyKey = rrPyString_FromString(key.c_str());
            PyObject *pyVal = Variant_to_py(dict->getItem(key));
            PyObject *tup = PyTuple_Pack(2, pyKey, pyVal);

            Py_DECREF(pyKey);
            Py_DECREF(pyVal);

            // list takes ownershipt of tuple
            PyList_SET_ITEM(pyList, j++, tup);
        }

        return pyList;
    }

    PyObject *dictionary_getitem(const Dictionary *dict, const char *key) {
        return Variant_to_py(dict->getItem(key));
    }

    PyObject *dictionary_setitem(Dictionary *dict, const char *key, PyObject *value) {
        dict->setItem(key, Variant_from_py(value));
        Py_RETURN_NONE;
    }

    void dictionary_delitem(Dictionary *dict, const char *key) {
        dict->deleteItem(key);
    }

    PyObject *dictionary_contains(const Dictionary *dict, const char *key) {
        bool contains = dict->hasKey(key);
        return PyBool_FromLong(contains);
    }
/*
 *
 */
#define VERIFY_PYARRAY(p) { \
    assert(p && "PyArray is NULL"); \
    assert((PyArray_NBYTES(p) > 0 ? PyArray_ISCARRAY(p) : true) &&  "PyArray must be C format"); \
}

    PyObject *doublematrix_to_py(const ls::DoubleMatrix *m, bool structured_result, bool copy_result) {
        rrLogDebug << __FUNC__;
        ls::DoubleMatrix *mat = const_cast<ls::DoubleMatrix *>(m);

        // a valid array descriptor:
        // In [87]: b = array(array([0,1,2,3]),
        //      dtype=[('r', 'f8'), ('g', 'f8'), ('b', 'f8'), ('a', 'f8')])


        // are we returning a structured array?
        if (structured_result) {
            rrLogDebug << "Structured result path";

            std::vector<string> names = mat->getColNames();

            unsigned int rows = mat->numRows();
            unsigned int cols = mat->numCols();

            std::cout << "num rows: " << rows;
            std::cout << "num cols: " << cols;

            if (cols == 0) {
                rrLogDebug << "No columns, returning None";
                Py_RETURN_NONE;
            }

            if (cols != names.size()) {
                throw std::logic_error("column names size does not match matrix columns size");
            }

            double *mData = mat->getArray();

            PyObject *list = PyList_New((std::int64_t) names.size());

            for (int i = 0; i < names.size(); ++i) {
                PyObject *col = rrPyString_FromString(names[i].c_str());
                PyObject *type = rrPyString_FromString("f8");
                PyObject *tup = PyTuple_Pack(2, col, type);

                Py_DECREF(col);
                Py_DECREF(type);

                // list takes ownership of tuple
                void PyList_SET_ITEM(list, i, tup);
            }

            PyArray_Descr *descr = nullptr;
            PyArray_DescrConverter(list, &descr);

            // done with list
            Py_CLEAR(list);
            npy_intp dims[] = {rows};

            // steals a reference to descr
            PyObject *pyres = PyArray_SimpleNewFromDescr(1, dims, descr);
//            VERIFY_PYARRAY(pyres);

            if (pyres) {

                assert(PyArray_NBYTES(pyres) == rows * cols * sizeof(double) && "invalid array size");

                double *data = (double *) PyArray_BYTES(pyres);

                memcpy(data, mData, sizeof(double) * rows * cols);
            }

            return pyres;
        }
            // standard array result.
            // this version just wraps the roadrunner owned data.
        else {
            int rows = mat->numRows();
            int cols = mat->numCols();
            PyObject *pArray = NULL;

            if (copy_result) {

                rrLog(rr::Logger::LOG_DEBUG) << "copying result data";

                // passing a NULL for data tells numpy to allocate its own data

                // make a 1D array in this case
                if (cols == 1 && mat->getColNames().size() == 0) {
                    rrLogDebug << "1 column and no column names";
                    int nd = 1;
                    npy_intp dims[1] = {rows};
                    pArray = PyArray_New(
                            &PyArray_Type,       // PyTypeObject* subtype
                            nd,                  // int nd
                            dims,                // npy_intp const* dims
                            NPY_DOUBLE,          // int type_num
                            NULL,                // npy_intp const* strides
                            nullptr,     // void* data
                            0,                   // int itemsize
                            0,                   // int flags
                            NULL                 // PyObject* obj
                    );
                } else {
                    rrLogDebug << "2D array";
                    int nd = 2;
                    npy_intp dims[2] = {rows, cols};
                    pArray = NamedArray_New(nd, dims, nullptr, 0, mat);
                }

                VERIFY_PYARRAY(pArray);

                // copy our data into the numpy array
                double *data = static_cast<double *>(PyArray_DATA(pArray));
                memcpy(data, mat->getArray(), sizeof(double) * rows * cols);

            } else {

                rrLog(rr::Logger::LOG_DEBUG) << "wraping existing data";

                double *data = mat->getArray();

                if (cols == 1 && mat->getColNames().size() == 0) {
                    int nd = 1;
                    npy_intp dims[1] = {rows};
                    pArray = PyArray_New(&PyArray_Type, nd, dims, NPY_DOUBLE,
                                         NULL, data, 0, NPY_ARRAY_CARRAY, NULL);
                } else {
                    int nd = 2;
                    npy_intp dims[2] = {rows, cols};
                    pArray = NamedArray_New(nd, dims, data,
                                            NPY_ARRAY_CARRAY, mat);
                }

                VERIFY_PYARRAY(pArray);

            }
            rrLogDebug << "Done" << std::endl;
            return pArray;
        }
    }

    /**
     * Casts a rr::Matrix<double> to its superclass ls::DoubleMatrix
     * and reuses doublematrix_to_py
     */
    PyObject *rrDoubleMatrix_to_py(const rr::Matrix<double> *m, bool copy_result) {
        rr::Matrix<double> *mat = const_cast<rr::Matrix<double> *>(m);
//        bool structured_result = true;
//        if (mat->rowNames.empty() && mat->colNames.empty()) {
//            structured_result = false;
//        }
        auto superMat = ls::DoubleMatrix(mat->getValues());
        return doublematrix_to_py(&superMat, false, copy_result);
    }


    /**
     * @brief Use the Python C API to convert a string vector to a Python list of
     * strings.
     */
    PyObject *convertStringVectorToPython(const std::vector<std::string> &vec);


#define rrPyNullCheck(object, errMsg)  if (object == nullptr){ \
            std::cerr << (errMsg) << std::endl;                \
            PyErr_Print();                                     \
        }\


#define CSTR(str) const_cast<char*>(str)


    /**
     * @brief Define the data members of the NamedArray type.
     * @details Each entry of this array is a PyMemberDef
     * with the following signature:
     *    typedef struct PyMemberDef {
     *         const char *name;
     *         int         type;
     *         int         offset;
     *         int         flags;
     *         const char *doc;
     *     } PyMemberDef;
     * Each entry of the NamedArray_members array links an attribute of the Python side NamedArray
     * to the NamedArray C struct.
     *
     * The `(Py_ssize_t )offsetof(NamedArrayObject, x)` part returns the starting byte of data member x
     *
     * @see https://docs.python.org/3/extending/newtypes.html#generic-attribute-management
     * @see https://en.cppreference.com/w/cpp/types/offsetof
     * @see Python/include/structmember.h
     */
    static PyMemberDef NamedArray_members[] = {
            {CSTR("rownames"), T_OBJECT_EX, (Py_ssize_t) offsetof(NamedArrayObject, rowNames), 0, CSTR("row names")},
            {CSTR("colnames"), T_OBJECT_EX, (Py_ssize_t) offsetof(NamedArrayObject, colNames), 0, CSTR("column names")},
            {NULL}  /* Sentinel */
    };

    /**
     * @brief Define methods of the NamedArray object.
     * @details The struct PyMethodDef* tp_methods field
     * in the PyTypeObject points to this array of PyMethodDef
     * objects. Each entry in this array is an instance of PyMethodDef.
     * and has the following type:
     * typedef struct PyMethodDef {
     *    const char  *ml_name;        method name
     *    PyCFunction  ml_meth;        implementation function
     *    int          ml_flags;       flags
     *    const char  *ml_doc;         docstring
     * } PyMethodDef;
     * Note the deliberate sentinal entry at the end of the array.
     * @see https://docs.python.org/3/extending/newtypes.html#generic-attribute-management
     */
    static PyMethodDef NamedArray_methods[] = {
            {CSTR("__array_finalize__"), (PyCFunction) NamedArrayObject_Finalize, METH_VARARGS, CSTR("")},
            {CSTR("__reduce_ex__"),      (PyCFunction) NamedArray___reduce_ex__,  METH_O,       CSTR("")},
            {CSTR("__getstate__"),       (PyCFunction) NamedArray___getstate__,   METH_NOARGS,  CSTR("")},
            {CSTR("__setstate__"),       (PyCFunction) NamedArray___setstate__,   METH_O,       CSTR("")},
            {NULL}/* Sentinel */
    };


/**
 * binaryfunc PyMappingMethods.mp_subscript
 * This function is used by PyObject_GetItem() and has the same signature.
 * This slot must be filled for the PyMapping_Check() function to return 1,
 * it can be NULL otherwise.
 */
    static PyObject *NammedArray_subscript(NamedArrayObject *self, PyObject *op) {
        binaryfunc PyArray_subscript = PyArray_Type.tp_as_mapping->mp_subscript;
        int dim = PyArray_NDIM(self);

# if PY_MAJOR_VERSION == 3
        if (dim == 2 && PyUnicode_Check(op)) {
# else
            if (dim == 2 && PyString_Check(op)) {
# endif
            char *keyName = rrPyString_AsString(op);

            PyObject *colSeq = PySequence_Fast(self->colNames, "expected a sequence");
            Py_ssize_t len = PySequence_Size(colSeq);
            for (int col = 0; col < len; col++) {
                PyObject *item = PySequence_Fast_GET_ITEM(colSeq, col);
                char *itemStr = rrPyString_AsString(item);

                if (strcmp(keyName, itemStr) == 0) {

                    npy_intp rows = PyArray_DIM(self, 0);
                    npy_intp cols = PyArray_DIM(self, 1);

                    npy_intp dims[1] = {rows};
                    PyObject *result = PyArray_New(&PyArray_Type, 1, dims, NPY_DOUBLE, NULL, NULL, 0,
                                                   NPY_ARRAY_CARRAY, NULL);

                    // copy data to result array
                    double *data = (double *) PyArray_DATA(self);
                    double *newData = (double *) PyArray_DATA(result);

                    for (int i = 0; i < rows; ++i) {
                        newData[i] = data[(i * cols) + col];
                    }

                    // free the seq and return
                    Py_DECREF(colSeq);
                    return result;
                }

                rr_strfree(itemStr);
            }

            // did not find a col name, free the seq
            Py_DECREF(colSeq);


            // look for row names
            PyObject *rowSeq = PySequence_Fast(self->rowNames, "expected a sequence");
            len = PySequence_Size(rowSeq);
            for (int row = 0; row < len; ++row) {
                PyObject *item = PySequence_Fast_GET_ITEM(rowSeq, row);
                char *itemStr = rrPyString_AsString(item);

                if (strcmp(keyName, itemStr) == 0) {

                    npy_intp rows = PyArray_DIM(self, 0);
                    npy_intp cols = PyArray_DIM(self, 1);

                    npy_intp dims[1] = {cols};
                    PyObject *result = PyArray_New(&PyArray_Type, 1, dims, NPY_DOUBLE, NULL, NULL, 0,
                                                   NPY_ARRAY_CARRAY, NULL);

                    // copy data to result array
                    double *data = (double *) PyArray_DATA(self);
                    double *newData = (double *) PyArray_DATA(result);

                    for (int i = 0; i < cols; ++i) {
                        newData[i] = data[(row * cols) + i];
                    }

                    // free the seq and return
                    Py_DECREF(rowSeq);
                    return result;
                }

                rr_strfree(itemStr);
            }

            // did not find a col name, free the seq
            Py_DECREF(rowSeq);

            rr_strfree(keyName);

        }

        // not found in row or col names, return default numpy func.
        return PyArray_subscript((PyObject *) self, op);
    }


    static PyMappingMethods NamedArray_MappingMethods = {
#if PY_VERSION_HEX >= 0x02050000
            (lenfunc) 0,             /*mp_length*/
#else
            (inquiry)0,             /*mp_length*/
#endif
            (binaryfunc) 0,          /*mp_subscript*/
            (objobjargproc) 0,       /*mp_ass_subscript*/
    };

    static PyTypeObject NamedArray_Type = {
            PyVarObject_HEAD_INIT(NULL, 0)
            "roadrunner._roadrunner.NamedArray",                              /* tp_name */
            sizeof(NamedArrayObject),                              /* tp_basicsize */
            0,                              /* tp_itemsize */
            (destructor) NamedArrayObject_dealloc,                              /* tp_dealloc */
            0,                              /* tp_vectorcall_offset */
            0,                              /* tp_getattr */
            0,                              /* tp_setattr */
            0,                              /* tp_as_async */
            (reprfunc) NamedArray_repr,                              /* tp_repr */
            0,                              /* tp_as_number */
            0,                              /* tp_as_sequence */
            &NamedArray_MappingMethods,                              /* tp_as_mapping */
            0,                              /* tp_hash */
            0,                              /* tp_call */
            (reprfunc) NamedArray_str,                              /* tp_str */
            0,                              /* tp_getattro */
            0,                              /* tp_setattro */
            0,                              /* tp_as_buffer */
            Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,                              /* tp_flags */
            "NamedArray",                              /* tp_doc */
            0,                              /* tp_traverse */
            0,                              /* tp_clear */
            0,                              /* tp_richcompare */
            0,                              /* tp_weaklistoffset */
            0,                              /* tp_iter */
            0,                              /* tp_iternext */
            NamedArray_methods,                              /* tp_methods */
            NamedArray_members,                              /* tp_members */
            0,                              /* tp_getset */
            0,                              /* tp_base */
            0,                              /* tp_dict */
            0,                              /* tp_descr_get */
            0,                              /* tp_descr_set */
            0,                              /* tp_dictoffset */
            0,                              /* tp_init : note we use numpy's ct'r*/
            NamedArrayObject_alloc,                              /* tp_alloc */
            0,                              /* tp_new */
            0,                              /* tp_free */
            0,                              /* tp_is_gc */
            0,                              /* tp_bases */
            0,                              /* tp_mro */
            0,                              /* tp_cache */
            0,                              /* tp_subclasses */
            0,                              /* tp_weaklist */
            0,                              /* tp_del */
            0,                              /* tp_version_tag */
            0,                              /* tp_finalize */
    };

    // Module initialization

    static PyObject *getItemFromDictWithErrChecking(PyObject *dict, const char *string) {
        rrLogDebug << __FUNC__;
        PyDict_CheckExact(dict);
        PyObject *item = PyDict_GetItemString(dict, string);
        if (!item) {
            PyErr_Format(PyExc_KeyError, "Could not find key '%s' in dict", string);
            return nullptr;
        }
        rrLogDebug << "Done" << std::endl;
        return item;
    }

    /* Pickle the object */
    static PyObject *
    NamedArray___getstate__(NamedArrayObject *self, PyObject *Py_UNUSED(ignored)) {
        rrLogDebug << __FUNC__;
        // see https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_Dumps
        PyObject *arrayBytes = self->saveToBytes();
        if (!arrayBytes) {
            PyErr_SetString(PyExc_ValueError, "Could not convert array to bytes");
            return nullptr;
        }
        // number of dimensions
        int nDims = PyArray_NDIM(self);
        std::int64_t dim1 = 0;
        std::int64_t dim2 = 0;

        npy_intp *shape = PyArray_SHAPE((PyArrayObject *) self);
        if (!shape) {
            PyErr_SetString(PyExc_ValueError, "Could not extract shape from array");
            return nullptr;
        }
        if (nDims > 0)
            dim1 = shape[0];
        if (nDims > 1)
            dim2 = shape[1];

        PyObject* rownames;
        if (!self->rowNames){
            Py_IncRef(Py_None);
            rownames = Py_None;
        } else {
            rownames = self->rowNames;
        }
        PyObject* colnames;
        if (!self->colNames){
            Py_IncRef(Py_None);
            colnames = Py_None;
        } else {
            colnames = self->colNames;
        }

        // string, bytes, string, object, string object string int
        // note that this adds 1 to the ref counts of arrayBytes
        // rownames and colnames
        PyObject *dictObj = Py_BuildValue("{sSsisLsLsOsOsi}",
                                          "array", arrayBytes,
                                          "nDims", nDims,
                                          "dim1", dim1,
                                          "dim2", dim2,
                                          "rownames", rownames,
                                          "colnames", colnames,
                                          PICKLE_VERSION_KEY, PICKLE_VERSION);

        if (!dictObj) {
            PyErr_SetString(PyExc_ValueError, "Could not create dict using Py_BuildValue "
                                              "in NamedArray.__getstate__");
            return nullptr;
        }

        // these references had a count of 1 when entering the function (the scope reference)
        // and incremented 1 when they were inserted into the dictionary (the dict reference).
        // we decerment the function scope reference.
        Py_DECREF(self->rowNames);
        Py_DECREF(self->colNames);
        Py_DECREF(arrayBytes);
        if (dictObj->ob_refcnt != 1) {
            PyErr_Format(PyExc_MemoryError, "Expecting reference count to be equal to 1 not '%L", dictObj->ob_refcnt);
        }

        rrLogDebug << "Done" << std::endl;
        return dictObj;
    }

    static PyObject *
    NamedArray___reduce_ex__(NamedArrayObject *self, PyObject *args) {

        rrLogDebug << __FUNC__;

        PyObject *state = NamedArray___getstate__(self, nullptr);
        if (!state) {
            // do not set an error here as it will override the
            // useful error from NamedArray___getstate__
//            PyErr_SetString(PyExc_MemoryError, "state is nullptr");
            return nullptr;
        }

        // verify we have a dict
        PyDict_CheckExact(state);

        // borrowed reference
        PyObject *nDimsObj = getItemFromDictWithErrChecking(state, "nDims");
        PyObject *dim1 = getItemFromDictWithErrChecking(state, "dim1");
        PyObject *dim2 = getItemFromDictWithErrChecking(state, "dim2");

        PyObject *dimensionsTuple = nullptr;
        auto nDims = PyLong_AsLong(nDimsObj);
        if (nDims == 2) {
            dimensionsTuple = PyTuple_Pack(2, dim1, dim2);
            if (!dimensionsTuple) {
                PyErr_SetString(dimensionsTuple, "Could not create dimensions tuple");
                return nullptr;
            }
        } else if (nDims == 1) {
            dimensionsTuple = PyTuple_Pack(1, dim1);
            if (!dimensionsTuple) {
                PyErr_SetString(dimensionsTuple, "Could not create dimensions tuple");
                return nullptr;
            }
        } else {
            PyErr_Format(PyExc_ValueError, "Unexpected number of dimensions %i", nDims);
        }
        // import _roadrunner and grab the NamedArray object
        // from the Python end (new reference)
        PyObject *roadrunnerModule = PyImport_ImportModule("roadrunner._roadrunner");
        if (!roadrunnerModule) {
            PyErr_SetString(PyExc_ImportError, "Could not import roadrunner._roadrunner");
            return nullptr;
        }

        // new reference:
        PyObject *namedArrayConstructor = PyObject_GetAttrString(roadrunnerModule, "NamedArray");
        if (!namedArrayConstructor) {
            PyErr_SetString(PyExc_AttributeError, "Could not find NamedArray in the roadrunner._roadrunner module");
            return nullptr;
        }

        PyObject *nestedDimensionsTuple = PyTuple_Pack(1, dimensionsTuple);

        PyObject *tup = nullptr;
        // Py_BuildValue adds 1 to ref counts of the objects we add to it.
        tup = Py_BuildValue(
                "(OOOOO)",
                /**
                 * A callable object that will be called to
                 * create the initial version of the object.
                 * Aka
                 */
                namedArrayConstructor,

                /**
                 * A tuple of arguments for the callable object.
                 * An empty tuple must be given if the callable
                 * does not accept any argument.
                 *
                 * This is a nested tuple for passing in the shape
                 * of the array we need to namedArrayObj
                 *  In python syntax --> ((3, 4))
                 * where the outer tuple is an argument list passed to NamedArray
                 *  NamedArray(*args)
                 * which expands to the inner tuple being passed for the np.shape parameter.
                 */
                nestedDimensionsTuple,

                /**
                 * Optionally, the object’s state, which will be passed to
                 * the object’s __setstate__() method as previously
                 * described. If the object has no such method then,
                 * the value must be a dictionary and it will be added
                 * to the object’s __dict__ attribute.
                 */
                state,
                Py_None,
                Py_None
        );
        // we've used Py_None twice, so we need to increment its ref count twice
        Py_IncRef(Py_None);
        Py_IncRef(Py_None);
        // the inside tuple does not get incremented by the call to Py_BuildValue
        // (callable, ((3, 4),), state, None, None)
        //              ^^^^<-- Needs incrementing
        Py_IncRef(dimensionsTuple);

        /**
         * Creating a tuple with PyTuple or PyArgs correctly increments the ref count
         * of objects you put into it. However they do not correctly
         * increment the ref count for objects that are nested inside other
         * containers. Therefore, we need to manually increment the contents of
         * the state dictionary.
         */
//        Py_INCREF(PyDict_GetItemString(state, "array"));
        Py_INCREF(PyDict_GetItemString(state, "nDims"));
        Py_INCREF(PyDict_GetItemString(state, "dim1"));
        Py_INCREF(PyDict_GetItemString(state, "dim2"));
        Py_INCREF(PyDict_GetItemString(state, "rownames"));
        Py_INCREF(PyDict_GetItemString(state, "colnames"));
        Py_INCREF(PyDict_GetItemString(state, PICKLE_VERSION_KEY));

        Py_DECREF(roadrunnerModule);
        Py_DECREF(namedArrayConstructor);
        Py_DECREF(nestedDimensionsTuple);
        Py_DECREF(state);
        rrLogDebug << "Done" << std::endl;
        return tup;
    }

    void verifyPickleVersion(PyObject *state) {

        // check serialization version. At the time of writing there is only one
        // serialization version, but this helps to future proof this code
        PyObject *pklVersionPyObj = PyDict_GetItemString(state, PICKLE_VERSION_KEY);
        if (pklVersionPyObj == nullptr) {
            /* PyDict_GetItemString does not set error, so we do it*/
            PyErr_Format(PyExc_KeyError, "No '%s' in pickled data", PICKLE_VERSION_KEY);
            return;
        }
        int pklVersion = (int) PyLong_AsLong(pklVersionPyObj);
        if (pklVersion != PICKLE_VERSION) {
            PyErr_Format(
                    PyExc_ValueError,
                    "Pickle version mismatch. Got version %d but expected version %d.",
                    pklVersion, PICKLE_VERSION);
        }
        return;
    }

    PyObject *deepCopyPyList(PyObject *listObj) {
        PyList_CheckExact(listObj);
        Py_ssize_t N = PyList_Size(listObj);
        PyObject *newList = PyList_New(N);
        if (!newList) {
            PyErr_SetString(PyExc_ValueError, "Could not create a new list "
                                              "for deep copy");
            return nullptr;
        }
        // When you copy.deepcopy a list in Python
        // you create a new list but not new items
        // for inside the list
        // Validated in python console:
        // >>> import copy
        // >>> l = [1,2,3]
        // >>> l2 = copy.deepcopy(l)
        // >>> id(l)
        // 140053629264384
        // >>> id(l2)
        // 140053629372992
        // >>> id(l[0])
        // 140053681346864
        // >>> id(l2[0])
        // >>> # item 0 of l2 is same reference as item 0 of l
        // 140053681346864
        for (int i = 0; i < N; i++) {
            PyObject *item = PyList_GetItem(listObj, i); // borrowed ref
            if (!item) {
                PyErr_SetString(PyExc_ValueError, "Item collected from old list "
                                                  "is nullptr. Cannot create a deep copy "
                                                  "for nullptr");
                return nullptr;
            }
            Py_INCREF(item);
            if (PyList_SetItem(newList, i, item) < 0) {
                PyErr_Format(PyExc_ValueError, "Could not set item '%i'", i);
                return nullptr;
            };
        }
        return newList;
    }


    static PyObject *
    NamedArray___setstate__(NamedArrayObject *self, PyObject *state) {
        /**
         * self is a NamedArrayObject created with a call to
         * the NamedArray constructor.
         * >>> n = NamedArray((4, 5)) # it is the correct size already.
         * We need to
         *  1) read the binary dataFromBytesArray from within the state into a PyArray.
         *     Use PyArray_FromBuffer which will create a 1D array.
         *  2) We now have two arrays, self which is a conveniently shaped shell
         *     but no numbers and the array built from binary containing our numbers
         *     Change the pointer in self to point to the dataFromBytesArray from the other array.
         */
        rrLogDebug << __FUNC__;

        // ensure we have a dict object, ruling out dict subclasses
        if (!PyDict_CheckExact(state)) {
            PyErr_SetString(PyExc_ValueError, "__setstate__ input object is not a dict");
        }

        verifyPickleVersion(state);

        // Grab the bytes object.
        PyObject *bytesObj = getItemFromDictWithErrChecking(state, "array");
        PyObject *nDimsObj = getItemFromDictWithErrChecking(state, "nDims");
        PyObject *dim1Obj = getItemFromDictWithErrChecking(state, "dim1");
        PyObject *dim2Obj = getItemFromDictWithErrChecking(state, "dim2");
        long nDims = PyLong_AsLong(nDimsObj);
        long dim1 = PyLong_AsLong(dim1Obj);
        long dim2 = PyLong_AsLong(dim2Obj);

        // ndims is restricted to 1 and 2 only
        if (nDims == 1)
            self->loadFromBytes(bytesObj, &dim1, nullptr);
        else
            self->loadFromBytes(bytesObj, &dim1, &dim2);

        // This is an alternative implementation that requires copying
        //   PyObject* rowNames = getItemFromDictWithErrChecking(state, "rownames");
        //   self->rowNames = deepCopyPyList(rowNames);
        // and this is the implementation that we are using, but note
        // its harder to trace the memory footprint and may contain bugs.
        self->rowNames = getItemFromDictWithErrChecking(state, "rownames");
        Py_IncRef(self->rowNames);
        for (int i = 0; i < PyList_Size(self->rowNames); i++)
            Py_IncRef(PyList_GetItem(self->rowNames, i));

        self->colNames = getItemFromDictWithErrChecking(state, "colnames");
        Py_IncRef(self->colNames);
        for (int i = 0; i < PyList_Size(self->colNames); i++) {
            Py_IncRef(PyList_GetItem(self->colNames, i));
        }
        rrLogDebug << "Done" << std::endl;
        Py_RETURN_NONE;
    }


    void pyutil_init(PyObject *module) {
        rrLogDebug << __FUNC__;
        // note to developers: turn this to LOG_INFORMATION for details
        // when running the tests
        // Logger::setLevel(Logger::LOG_DEBUG);
        // set up the base class to be the numpy ndarray PyArray_Type
        NamedArray_Type.tp_base = &PyArray_Type;


        // set up the pointers of the NamedArray_MappingMethods to point
        // to the numpy ones
        PyMappingMethods *numpyMappMethods = PyArray_Type.tp_as_mapping;

        assert(numpyMappMethods && "Numpy PyMappingMethods is NULL");

        NamedArray_MappingMethods = *numpyMappMethods;

        // set our getitem pointer
        NamedArray_MappingMethods.mp_subscript = (binaryfunc) NammedArray_subscript;

#if (PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 9)
        // this if for python v3.9 and above. Remove
        // old code when we drop support for py37 and py38
        if (PyModule_AddType(module, &NamedArray_Type) < 0) {
            PyErr_SetString(PyExc_ValueError, "Could not add NamedArray_Type to module roadrunner._roadrunner");
            return;
        }
#else // python 3.9
        int result;

        if ((result = PyType_Ready(&NamedArray_Type)) < 0) {
            std::cout << "PyType_Ready(&NamedArray_Type)) Failed, error: " << result;
            return;
        }

        Py_INCREF(&NamedArray_Type);
        result = PyModule_AddObject(module, "NamedArray", (PyObject *) (&NamedArray_Type));
#endif // python 3.9
    }


/*
/* PyArray_New(&PyArray_Type, nd, dims, NPY_DOUBLE, NULL, data, 0,
                                     NPY_CARRAY | NPY_OWNDATA, NULL);
 */
    PyObject *NamedArray_New(int nd, npy_intp *dims, double *data, int pyFlags, const ls::DoubleMatrix *mat) {
        bool named = Config::getValue(Config::PYTHON_ENABLE_NAMED_MATRIX);
        rrLogDebug << __FUNC__;

        if (named) {

            rrLogInfo << "creating NEW style array";

            //         (*(PyObject * (*)(PyTypeObject *, int, npy_intp const *, int, npy_intp const *, void *, int, int, PyObject *))
            /**
             * calls the NamedArray_New which feeds into NamedArray_Alloc
             * and then NamedArray_Finalize.
             */
            // PyArray_New calls finalize twice, once for the pyarray and
            // again for the subtype
            NamedArrayObject *array = (NamedArrayObject *) PyArray_New(
                    &NamedArray_Type,   // PyTypeObject* subtype,
                    nd,                 // int nd
                    dims,               // npy_intp const* dims,
                    NPY_DOUBLE,         // int type_num
                    NULL,               // npy_intp const* strides,
                    (void *) data,        // void* data,
                    0,                  // int itemsize. If the type always has the same number of bytes, then itemsize is ignored. Otherwise, itemsize specifies the particular size of this array
                    pyFlags,            // int flags,
                    nullptr             // PyObject* obj
            );
            if (array == NULL) {
                const char *error = rrGetPyErrMessage();
                rrLog(Logger::LOG_CRITICAL) << error;
                rr_strfree(error);
                return NULL;
            }

            if (PyList_Size(array->rowNames) != 0) {
                PyErr_SetString(PyExc_ValueError, "Expecting empty initialized list for array->rowNames.");
                return nullptr;
            }

            if (PyList_Size(array->colNames) != 0) {
                PyErr_SetString(PyExc_ValueError, "Expecting empty initialized list for array->colNames.");
                return nullptr;
            }
            array->rowNames = stringvector_to_py(mat->getRowNames());
            array->colNames = stringvector_to_py(mat->getColNames());

            return (PyObject *) array;

        } else {
            rrLogInfo << "creating old style array";
            return PyArray_New(&PyArray_Type, nd, dims, NPY_DOUBLE, NULL, data, 0,
                               pyFlags, NULL);
        }
    }


    PyObject *stringvector_to_py(const std::vector<std::string> &vec) {
        rrLogDebug << __FUNC__;
        size_t size = vec.size();

        PyObject *pyList = PyList_New(size);

        unsigned j = 0;

        for (auto i = vec.begin(); i != vec.end(); ++i) {
            const std::string &str = *i;
            PyObject *pyStr = rrPyString_FromString(str.c_str());
            PyList_SET_ITEM(pyList, j++, pyStr);
        }
        return pyList;
    }

    PyObject *PyList_toPickle(PyObject *list) {
        rrLogDebug << __FUNC__;
        static PyObject *module = NULL;
        PyObject *pickle;

        if (module == NULL && (module = PyImport_ImportModule("serialization")) == NULL)
            return NULL;

        pickle = PyObject_CallMethodObjArgs(module,
                                            PyUnicode_FromString("dumps"),
                                            list,
                                            NULL);
        if (!pickle) {
            std::cerr << "PyList_toPickle returned None";
            Py_RETURN_NONE;
        }
        return pickle;
    }

    std::vector<std::string> py_to_stringvector(PyObject *obj) {
        rrLogDebug << __FUNC__;
        std::vector<std::string> result;
        if (!obj)
            return result;
        PyObject *seq = PySequence_Fast(obj, "expected a sequence");
        Py_ssize_t len = PySequence_Size(obj);
        if (PyList_Check(seq))
            for (unsigned i = 0; i < len; i++) {
                PyObject *item = PyList_GetItem(seq, i);
                if (!PyUnicode_CheckExact(item)) {
                    PyErr_Format(PyExc_TypeError, "Expected a string but got '%s'", item->ob_type->tp_name);
                    return {};
                }
                // Return value: New reference.
                const char *pystr = PyUnicode_AsUTF8(item);
                result.emplace_back(pystr);
            }
        else
            for (unsigned i = 0; i < len; i++) {
                PyObject *item = PyTuple_GetItem(seq, i);
                if (!PyUnicode_CheckExact(item)) {
                    PyErr_Format(PyExc_TypeError, "Expected a string but got '%s'", item->ob_type->tp_name);
                    return {};
                }
                // Return value: New reference.
                const char *pystr = PyUnicode_AsUTF8(item);
                result.emplace_back(pystr);
            }
        Py_XDECREF(seq);

        rrLogDebug << "Done" << std::endl;
        return result;
    }


    Dictionary *Dictionary_from_py(PyObject *py) {
        rrLogDebug << __FUNC__;
        if (PyDict_Check(py) == 0) {
            throw invalid_argument("object is not a dictionary");
        }

        BasicDictionary *dict = new BasicDictionary();

        PyObject *pkey, *pvalue;
        Py_ssize_t pos = 0;

        while (PyDict_Next(py, &pos, &pkey, &pvalue)) {
# if PY_MAJOR_VERSION == 3
            if (PyUnicode_Check(pkey)) {
# else
                if (PyString_Check(pkey)) {
# endif
                std::string key(rrPyString_AsString(pkey));
                Setting value = Variant_from_py(pvalue);

                dict->setItem(key, value);

            } else {
                throw invalid_argument("keys must be strings");
            }
        }
        rrLogDebug << "Done" << std::endl;
        return dict;
    }


    static PyObject *NamedArrayObject_alloc(PyTypeObject *type, Py_ssize_t nitems) {
        rrLogDebug << __FUNC__;
        assert(type->tp_basicsize == sizeof(NamedArrayObject));
        NamedArrayObject *namedArrayObject = PyObject_New(NamedArrayObject, type);
        if (!namedArrayObject) {
            PyErr_SetString(PyExc_MemoryError, "Could allocate object of type 'NamedArray'");
            return nullptr;
        }
        namedArrayObject->rowNames = PyList_New(0);
        namedArrayObject->colNames = PyList_New(0);
        PyObject *obj = PyObject_Init((PyObject *) namedArrayObject, type);
        if (!obj) {
            PyErr_SetString(PyExc_MemoryError, "Could not initialize object of type 'NamedArray'");
            return nullptr;
        }
        rrLogDebug << "namedArrayObject allocated:  " << namedArrayObject;
        rrLogDebug << "Done" << std::endl;
        return obj;
    }

/**
 * @brief the part of NamedArrayObject_Finalize that deals with the situation
 * where the second argument PyObject* args evaluates to Py_None;
 */
    PyObject *NamedArrayObject_Finalize_FromConstructor(NamedArrayObject *self) {
        rrLogDebug << __FUNC__;

        if (!self->rowNames) {
            rrLogDebug << "No rownames in self, using empty list instead";
            PyObject *rows = PyList_New(0);
            if (!rows) {
                PyErr_SetString(PyExc_MemoryError, "Could not allocate a new list for rownames");
                Py_RETURN_NONE;
            }
            self->rowNames = rows;
        }
        if (!self->colNames) {
            rrLogDebug << "No colnames in self, using empty list instead";
            PyObject *cols = PyList_New(0);
            if (!cols) {
                PyErr_SetString(PyExc_MemoryError, "Could not allocate a new list for colnames");
                Py_RETURN_NONE;
            }
            self->colNames = cols;
        }
        // Note it seems that we do not need to increment the ref count
        // for these new lists.

        // and assign
        rrLogDebug << "Done" << std::endl;
        Py_RETURN_NONE;
    }

/**
 * @brief the part of NamedArrayObject_Finalize that deals with the situation
 * where the second argument PyObject* args evaluates to a PyArrayObject;
 * @details since a PyArray does not have row/colnames then we can actually
 * reuse the NamedArrayObject_Finalize_FromConstructor method.
 */
    PyObject *NamedArrayObject_Finalize_FromPyArray(NamedArrayObject *self) {
        rrLogDebug << __FUNC__;
        PyObject *obj = NamedArrayObject_Finalize_FromConstructor(self);
        rrLogDebug << "done " << std::endl;
        return obj;
    }

/**
 * @brief the part of NamedArrayObject_Finalize that deals with the situation
 * where the second argument PyObject* args evaluates to another NamedArray ;
 * In this case rhs contains the original state of the array
 * before any slicing or other modifications that
 * count as "templating"
 * @see https://numpy.org/devdocs/user/basics.subclassing.html
 *
 * The array inside "self" contains a copy of the state
 * of the array after being manipulated by numpy.
 * For instance
 *  >>> n = NamedArray((3, 4))
 *  >>> v = n[1:, :]
 * invokes NamedArray_Finalize where
 *  - `n` maps to `rhs` aka the first rhs of the 1-tuple input args.
 *     Its state is the original
 *  - `v` maps to the `self` arugment. Its a copy of the np array with
 *  any changes implemented
 *
 * Its all very reminiscent of a copy constructor in c++
 */
    PyObject *NamedArrayObject_Finalize_FromNamedArray(NamedArrayObject *self, PyObject *rhs) {
        rrLogDebug << __FUNC__;

        if (self->rowNames == nullptr) {
            PyErr_SetString(PyExc_MemoryError, "self->rownames is nullptr");
            return nullptr;
        }
        if (self->colNames == nullptr) {
            PyErr_SetString(PyExc_MemoryError, "self->colnames is nullptr");
            return nullptr;
        }
        if (((NamedArrayObject *) rhs)->rowNames == nullptr) {
            PyErr_SetString(PyExc_MemoryError, "rhs rownames is nullptr");
            return nullptr;
        }
        if (((NamedArrayObject *) rhs)->colNames == nullptr) {
            PyErr_SetString(PyExc_MemoryError, "rhs rownames is nullptr");
            return nullptr;
        }

        unsigned selfNdim = PyArray_NDIM(&self->array);
        npy_intp selfNRows = selfNdim > 0 ? PyArray_DIM(&self->array, 0) : -1;
        npy_intp selfNCols = selfNdim > 1 ? PyArray_DIM(&self->array, 1) : -1;
        unsigned rhsfNdim = PyArray_NDIM(rhs);
        npy_intp rhsNRows = rhsfNdim > 0 ? PyArray_DIM(rhs, 0) : -1;
        npy_intp rhsNCols = rhsfNdim > 1 ? PyArray_DIM(rhs, 1) : -1;

        rrLogDebug << "Self address: " << self << " rhs addr : " << rhs;
        rrLogDebug << "selfNdim: " << selfNdim;
        rrLogDebug << "selfNRows: " << selfNRows;
        rrLogDebug << "selfNCols: " << selfNCols;
        rrLogDebug << "rhsfNdim: " << rhsfNdim;
        rrLogDebug << "rhsNRows: " << rhsNRows;
        rrLogDebug << "rhsNCols: " << rhsNCols;
        rrLogDebug << "PyList_Size(self->rowNames): " << PyList_Size(self->rowNames);
        rrLogDebug << "PyList_Size(rhs->rowNames): " << PyList_Size(((NamedArrayObject *) rhs)->rowNames);
        rrLogDebug << "PyList_Size(self->colNames): " << PyList_Size(self->colNames);
        rrLogDebug << "PyList_Size(rhs->colNames): " << PyList_Size(((NamedArrayObject *) rhs)->colNames);

        /**
         * This is a bit confusing. Sometimes we want to keep the rhs, and other we want to keep
         * the self object and its not clear why. When using slicing operations from numpy, we are getting
         * another array from a *template*. This method is called and we need
         * to manually take the column/row names from rhs to self, and self
         * is what we keep. However, when creating a NamedArray using
         * the doublematrix_to_py function it seems that rhs contains
         * what we need and self is a row vector (len(3) and no columns).
         *
         * Its not 100% clear whats going on and theres but I cannot get the
         * slicing + row/colnames working at the same time as doublematrix_to_py.
         * Given time constraints and doublematrix_to_py is more important,
         * we settle for just supporting this operation. Will return to this sometime in the
         * future so leave the commented out code intact.
         *
         *     - ciaran
         */

        Py_RETURN_NONE;

        // note this is likley not the best indicator of how the NamedArray was invoked.
//        if (selfNdim == 1) {
//            rrLogDebug << "NamedArray invoked from doublematrix_to_py. "
//                          "Doing nothing, rhs already contains the data,"
//                          " rows and cols we need";
//            Py_RETURN_NONE;
//        } else {
//            rrLogDebug << "NamedArray invoked from template. Collecting row and column"
//                          " information from rhs for self. ";
//            double *data = (double *) PyArray_DATA(self);
//            rrLogDebug << "data[0]: " << data[0];
//            double *selfData = (double *) PyArray_DATA(self);
//            double *rhsData = (double *) PyArray_DATA(rhs);
//            NamedArrayObject *rhsAsNamedArrayObj = (NamedArrayObject *) rhs;
//            if (!rhsAsNamedArrayObj) {
//                PyErr_SetString(PyExc_MemoryError,
//                                "Could not cast argument to NamedArray_Finalize to a NamedArrayObject*");
//                Py_RETURN_NONE;
//            }
//
//            PyObject *selfRownames = self->rowNames;
//            PyObject *selfColnames = self->colNames;
//            PyObject *rhsRownames = self->rowNames;
//            PyObject *rhsColnames = self->colNames;
//
//
//            if (PyList_Size(rhsAsNamedArrayObj->rowNames) != 0) {
//                std::vector<double> selfFirstColumn;
//                for (std::int64_t i = 0; i < selfNCols * selfNRows; i += selfNCols) {
//                    selfFirstColumn.push_back(selfData[i]);
//                }
//                std::vector<double> rhsFirstColumn;
//                for (std::int64_t i = 0; i < rhsNCols * rhsNRows; i += rhsNCols) {
//                    rhsFirstColumn.push_back(rhsData[i]);
//                }
//                std::int64_t firstIdxOfSelfInRhs = std::distance(
//                        rhsFirstColumn.begin(),
//                        std::find(rhsFirstColumn.begin(), rhsFirstColumn.end(), selfFirstColumn[0])
//                );
//                std::int64_t lastIdxOfSelfInRhs = firstIdxOfSelfInRhs + selfNRows;
//                // then we need to slice the rownames list
//                // self->rowNames is a new reference (Ref count = 1)
//                self->rowNames = PyList_GetSlice(rhsAsNamedArrayObj->rowNames, firstIdxOfSelfInRhs, lastIdxOfSelfInRhs);
//                // we therefore need to decrement the ref count of the original copy
//                Py_DECREF(rhsAsNamedArrayObj->rowNames);
//                if (!self->rowNames) {
//                    PyErr_SetString(PyExc_IndexError, "Could not slice NamedArray rownames");
//                    Py_RETURN_NONE;
//                }
//            }
//            if (PyList_Size(rhsAsNamedArrayObj->colNames) != 0) {
//                std::vector<double> selfFirstRow(selfNRows);
//                for (std::int64_t i = 0; i < selfNRows; i++) {
//                    selfFirstRow[i] = selfData[i];
//                }
//
//                std::vector<double> rhsFirstRow(rhsNRows);
//                for (std::int64_t i = 0; i < rhsNRows; i++) {
//                    rhsFirstRow[i] = selfData[i];
//                }
//
//                std::int64_t firstIdxOfSelfInRhs = std::distance(
//                        rhsFirstRow.begin(),
//                        std::find(rhsFirstRow.begin(), rhsFirstRow.end(), selfFirstRow[0])
//                );
//                std::int64_t lastIdxOfSelfInRhs = firstIdxOfSelfInRhs + selfNCols ;
//                rrLogDebug << firstIdxOfSelfInRhs << " " << lastIdxOfSelfInRhs;
//
//                // then we need to slice the rownames list
//                // self->colNames is a new reference (Ref count = 1)
//                Py_DECREF(self->colNames);
//                self->colNames = PyList_GetSlice(rhsAsNamedArrayObj->colNames, firstIdxOfSelfInRhs, lastIdxOfSelfInRhs);
//                Py_INCREF(self->colNames);
//                // we therefore need to decrement the ref count of the original copy
//                Py_DECREF(rhsAsNamedArrayObj->colNames);
//                if (!self->colNames) {
//                    PyErr_SetString(PyExc_IndexError, "Could not slice NamedArray colNames");
//                    return nullptr;
//                }
//            }
//            rrLogDebug << "Done" << std::endl;
//            Py_RETURN_NONE;
//        }
    }


    PyObject *NamedArrayObject_Finalize(NamedArrayObject *self, PyObject *args) {
        // Some docs on why this is needed: https://numpy.org/devdocs/user/basics.subclassing.html
        // when self is passed, the ref count of array goes up.
        rrLogDebug << __FUNC__;

        PyObject *rhs;
        if (PyArg_ParseTuple(args, "O", &rhs) < 0) {
            PyErr_SetString(PyExc_ValueError, "Could not get rhs from tuple");
            return nullptr;
        }
        rrLogDebug << "finalizing object self: " << self << "; args " << rhs;

        // when NamedArray instantiated with constructor
        //   >>> n = NamedArray((3, 4)) # 1
        // or from view of np.array:
        //   >>> n = np.ndarray((3, 4)).view(NamedArray) #2
        // we are guarenteed to not have row or colnames because
        //  #1 syntax is not supported at this time and
        //  #2 np.ndarray does not have row/colnames
        if (rhs == Py_None) {
            rrLogDebug << "NamedArrayObject initialized from constructor. 'None' path taken";
            return NamedArrayObject_Finalize_FromConstructor(self);
        } else if (rhs->ob_type == &PyArray_Type) {
            rrLogDebug << "Taking the PyArray_Type path";
            return NamedArrayObject_Finalize_FromPyArray(self);
        } else if (rhs->ob_type == &NamedArray_Type) {
            rrLogDebug << "Taking the NamedArray_Type path";
            return NamedArrayObject_Finalize_FromNamedArray(self, rhs);
        } else {
            rrLogErr << "Unexpected type passed to NamedArrayObject_Finalize for the args parameter";
            PyErr_SetString(PyExc_ValueError,
                            "Unexpected type passed to NamedArrayObject_Finalize for the args parameter");
        }
        // decrement the refcount as we're done with this scope
        Py_DecRef((PyObject *) &self->array);
        rrLogDebug << "Done" << std::endl;
        Py_RETURN_NONE;
    }


    static void NamedArrayObject_dealloc(NamedArrayObject *self) {
        rrLogDebug << __FUNC__;
        // tp_dealloc is called by Py_DECREF and Py_XDECREF once the ref count
        // gets to 0. Therefore, we expect to see the self ref count equal to 0.

        rrLogWarn << "Remember you have commented out decrementing row/colnames";
        // decrement the rownames, allowing for possibility of being null
        Py_XDECREF(self->rowNames);
        Py_XDECREF(self->colNames);

        PyObject *pself = (PyObject *) self;

        assert(pself->ob_type->tp_base == &PyArray_Type);
        PyArray_Type.tp_dealloc(pself);

        rrLogDebug << "Done" << std::endl;
    }

/*****************************************************************************************
 * Array Printing Stuff
 ****************************************************************************************/



    static int longestStrLen(const str_vector &s) {
        string::size_type longest = 0;
        for (str_vector::const_iterator i = s.begin(); i != s.end(); ++i) {
            longest = std::max(longest, i->length());
        }
        return static_cast<int>(longest);
    }

    static std::string array_format(
            PyArrayObject *arr, const str_vector &rowNames, const str_vector &colNames) {
        unsigned ndim = PyArray_NDIM(arr);
        npy_intp rows = ndim > 0 ? PyArray_DIM(arr, 0) : 0;
        npy_intp cols = ndim > 1 ? PyArray_DIM(arr, 1) : 0;

        assert(rows > 0 && cols > 0);

#define GET_VAL(_row, _col) *((double *) PyArray_GETPTR2(arr, _row, _col))

        // longest row name + " [["
        int rowNameWidth = longestStrLen(rowNames) + 1;

        // find widest element in each column
        vector<unsigned> columnWidths(cols);
        for (unsigned i = 0; i < rows; ++i) {
            for (unsigned j = 0; j < cols; ++j) {
                unsigned colw = columnWidths[j];
                colw = std::max(colw, j < colNames.size() ? (unsigned) colNames[j].length() + 1 : 0);
                stringstream ss;
                ss << GET_VAL(i, j);
                colw = std::max(colw, (unsigned) ss.str().length() + 1);
                columnWidths[j] = colw;
            }
        }

        stringstream ss;

        // first row, column names
        if (colNames.size()) {
            for (unsigned j = 0; j < cols; ++j) {
                ss << setw((j == 0 ? rowNameWidth + 2 : 0) + columnWidths[j]);
                ss << right;
                ss << (j < colNames.size() ? colNames[j] : "");
                if (j < cols - 1) {
                    ss << ",";
                }
            }
            ss << endl;
        }

        // main matrix bit
        for (unsigned i = 0; i < rows; ++i) {
            // row names
            ss << setw(rowNameWidth);
            ss << left;
            ss << (i < rowNames.size() ? rowNames[i] : "");
            ss << ((i > 0 && rows > 1) ? " [" : "[[");
            // columns
            for (unsigned j = 0; j < cols; ++j) {
                ss << setw(columnWidths[j]);
                ss << right;
                ss << GET_VAL(i, j);
                if (j < cols - 1) {
                    ss << ",";
                }
            }
            ss << ((i < rows - 1) ? "]," : "]]");
            ss << endl;
        }
        rrLogDebug << "Done" << std::endl;
        return ss.str();

#undef GET_VAL
    }

    PyObject *NamedArray_repr(NamedArrayObject *self) {
        rrLogDebug << __FUNC__;
        const void * address = static_cast<const void*>(self);
        std::stringstream ss;
        ss << address;
        std::string stringPtrVal = ss.str();
        PyObject *s = PyUnicode_FromFormat("<NamedArray object at %s>", stringPtrVal.c_str());
        rrLogDebug << "Done" << std::endl;
        return s;
    }

    PyObject *NamedArray_str(NamedArrayObject *self) {
        rrLogDebug << __FUNC__;
        PyArrayObject *array = (PyArrayObject *) self;
        str_vector rowNames = py_to_stringvector(self->rowNames);
        str_vector colNames = py_to_stringvector(self->colNames);

        unsigned ndim = PyArray_NDIM(array);
        npy_intp rows = ndim > 0 ? PyArray_DIM(array, 0) : 0;
        npy_intp cols = ndim > 1 ? PyArray_DIM(array, 1) : 0;

        if (rows == 0 || cols == 0) {
            return PyArray_Type.tp_str((PyObject *) self);
        }

        string str = array_format(array, rowNames, colNames);

        rrLogDebug << "Done" << std::endl;
        return rrPyString_FromString(str.c_str());
    }


    PyObject *convertStringVectorToPython(const std::vector<std::string> &vec) {
        rrLogDebug << __FUNC__;
        long long size = (long long) vec.size();

        PyObject *pyList = PyList_New(size);

        unsigned j = 0;

        for (const auto &item : vec) {
            PyObject *pyStr = rrPyString_FromString(item.c_str());
            PyList_SET_ITEM(pyList, j, pyStr);
            j++;
        }

        rrLogDebug << "Done" << std::endl;
        return pyList;
    }

    std::vector<std::string> convertPythonListToStringVector(PyObject *pyList) {
        rrLogDebug << __FUNC__;
        int size = (int) PyList_Size(pyList);
        std::vector<std::string> vec(size);
        for (Py_ssize_t i = 0; i < size; i++) {
            PyObject *item = PyList_GetItem(pyList, i);
            const char *s = PyUnicode_AsUTF8(item);
            vec[(int) i] = std::string(s);
        }

        rrLogDebug << "Done" << std::endl;
        return vec;
    }


    Matrix3DToNumpy::Matrix3DToNumpy(Matrix3DToNumpy::DoubleMatrix3D &matrix)
            : matrix_(matrix) {}

    PyObject *Matrix3DToNumpy::convertData() {
        // collect dimensions. x = width, y=height, z = depth.
        const npy_intp xMax = matrix_.numCols();
        const npy_intp yMax = matrix_.numRows();
        const npy_intp zMax = matrix_.numZ();

        // allocate 1D array with enough space to store linearized 3D matrix
        double *data = new double[yMax * xMax * zMax];

        // populate the 1D array with values from the Matrix3D<double, double> data values
        for (int z = 0; z < zMax; z++) {
            for (int y = 0; y < yMax; y++) {
                for (int x = 0; x < xMax; x++) {
                    // compute the linear index (so loop does idx \in 1, 2, ..., yMax*xMax*zMax)
                    unsigned int linearIdx = x + y * xMax + z * xMax * yMax;
                    data[linearIdx] = matrix_.slice(z, y, x);
                }
            }
        }
        // dimensions for resulting np array
        npy_intp dims[3] = {zMax, yMax, xMax};
        // create the numpy array
        PyObject *result = PyArray_SimpleNewFromData(3, dims, NPY_DOUBLE, (void *) data);

        // we need to cast to a PyArrayObject so that we can enable the NPY_ARRAY_OWNDATA flag
        //  which gives responsibility for handling the memory of `data` to numpy.
        PyArrayObject *arr = reinterpret_cast<PyArrayObject *>(result);
        PyArray_ENABLEFLAGS(arr, NPY_ARRAY_OWNDATA);
        int owned = PyArray_CHKFLAGS(arr, NPY_ARRAY_OWNDATA);
        if (!owned) {
            std::cerr << "PyArrayObject does not own its memory" << std::endl;
        }
        return result;
    }

    PyObject *Matrix3DToNumpy::convertIndex() {

        // matrix_.index dimensions == depth of Matrix3D
        const npy_intp zMax = matrix_.numZ();

        // allocate 1D array with enough space to store linearized 3D matrix
        double *data = new double[zMax];

        // populate the 1D array with values from the Matrix3D<double, double> data values
        auto &index = matrix_.getIndex();
        for (int z = 0; z < zMax; z++) {
            data[z] = index[z];
        }
        // dimensions for resulting np array
        npy_intp dims = zMax;

        // create the numpy array
        PyObject *result = PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, (void *) data);

        // we need to cast to a PyArrayObject so that we can enable the NPY_ARRAY_OWNDATA flag
        //  which gives responsibility for handling the memory of `data` to numpy.
        PyArrayObject *arr = reinterpret_cast<PyArrayObject *>(result);
        PyArray_ENABLEFLAGS(reinterpret_cast< PyArrayObject *>(result), NPY_ARRAY_OWNDATA);
        int owned = PyArray_CHKFLAGS(reinterpret_cast< PyArrayObject *>(result), NPY_ARRAY_OWNDATA);
        if (!owned) {
            std::cerr << "PyArrayObject does not own its memory" << std::endl;
        }
        return result;
    }

    PyObject *Matrix3DToNumpy::convertRowNames() {
        return convertStringVectorToPython(matrix_.getRowNames());
    }

    PyObject *Matrix3DToNumpy::convertColNames() {
        return convertStringVectorToPython(matrix_.getColNames());
    }


} /* namespace rr */

