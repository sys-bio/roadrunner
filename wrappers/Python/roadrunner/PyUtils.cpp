/*
 * PyUtils.cpp
 *
 *  Created on: Apr 27, 2014
 *      Author: andy
 */


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
#include <climits>
#include <complex>

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
        PyObject *bytes = PyUnicode_AsUTF8String(uni);
        str = rr_strclone(PyBytes_AsString(bytes));
        Py_XDECREF(bytes);
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
        return PyUnicode_FromString(s);
    }

    PyObject *rrPyString_FromStringAndSize(const char *s, Py_ssize_t size) {
        return PyUnicode_FromStringAndSize(s, size);
    }

    char *rrPyString_AsString(PyObject *s) {
        char *str;
        PyObject *bytes = PyUnicode_AsUTF8String(s);
        str = rr_strclone(PyBytes_AsString(bytes));
        Py_XDECREF(bytes);
        return str;
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
            return PyLong_FromLong(var.get<std::int32_t>());
        }

        if (type == typeid(std::uint32_t)) {
            return PyLong_FromUnsignedLong(var.get<std::uint32_t>());
        }

        if (type == typeid(char)) {
            char c = var.get<char>();
            return rrPyString_FromStringAndSize(&c, 1);
        }

        if (type == typeid(unsigned char)) {
            return PyLong_FromLong((long) var.get<std::int64_t>());
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
                PyList_SetItem(list, i, num);
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

        if (PyUnicode_Check(py)) {
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
            PyList_SetItem(pyList, j++, pyStr);
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
            PyList_SetItem(pyList, j++, pyVal);
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
            PyList_SetItem(pyList, j++, tup);
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
    // Returns numpy.frombuffer (looked up once), or nullptr with a Python error set.
    static PyObject *numpyFrombuffer() {
        static PyObject *frombuffer = nullptr;
        if (!frombuffer) {
            PyObject *numpy = PyImport_ImportModule("numpy");
            if (!numpy) {
                return nullptr;
            }
            frombuffer = PyObject_GetAttrString(numpy, "frombuffer");
            Py_DECREF(numpy);
        }
        return frombuffer;
    }

    PyObject *buffer_to_array(PyObject *buffer, const char *dtype, const std::vector<Py_ssize_t> &shape) {
        PyObject *frombuffer = numpyFrombuffer();
        PyObject *flat = frombuffer ? PyObject_CallFunction(frombuffer, "Os", buffer, dtype) : nullptr;
        Py_DECREF(buffer);
        if (!flat || shape.size() == 1) {
            return flat;
        }

        PyObject *shapeTuple = PyTuple_New((Py_ssize_t) shape.size());
        if (!shapeTuple) {
            Py_DECREF(flat);
            return nullptr;
        }
        for (size_t i = 0; i < shape.size(); ++i) {
            PyTuple_SetItem(shapeTuple, (Py_ssize_t) i, PyLong_FromSsize_t(shape[i]));
        }
        PyObject *result = PyObject_CallMethod(flat, "reshape", "(O)", shapeTuple);
        Py_DECREF(shapeTuple);
        Py_DECREF(flat);
        return result;
    }

    PyObject *doubles_to_py(const double *data, size_t count) {
        PyObject *buffer = PyByteArray_FromStringAndSize(
                reinterpret_cast<const char *>(data), (Py_ssize_t) (count * sizeof(double)));
        if (!buffer) {
            return nullptr;
        }
        return buffer_to_array(buffer, "float64", {(Py_ssize_t) count});
    }

    PyObject *complexes_to_py(const std::complex<double> *data, size_t count) {
        PyObject *buffer = PyByteArray_FromStringAndSize(
                reinterpret_cast<const char *>(data), (Py_ssize_t) (count * sizeof(std::complex<double>)));
        if (!buffer) {
            return nullptr;
        }
        return buffer_to_array(buffer, "complex128", {(Py_ssize_t) count});
    }

    bool py_to_doubles(PyObject *sequence, std::vector<double> &out) {
        if (!PySequence_Check(sequence) || PyUnicode_Check(sequence)) {
            PyErr_SetString(PyExc_TypeError, "expected a sequence of numbers");
            return false;
        }
        Py_ssize_t count = PySequence_Size(sequence);
        if (count < 0) {
            return false;
        }
        out.clear();
        out.reserve((size_t) count);
        for (Py_ssize_t i = 0; i < count; ++i) {
            PyObject *item = PySequence_GetItem(sequence, i);
            if (!item) {
                return false;
            }
            double value = PyFloat_AsDouble(item);
            Py_DECREF(item);
            if (value == -1.0 && PyErr_Occurred()) {
                return false;
            }
            out.push_back(value);
        }
        return true;
    }

    bool py_to_ints(PyObject *sequence, std::vector<int> &out) {
        if (!PySequence_Check(sequence) || PyUnicode_Check(sequence)) {
            PyErr_SetString(PyExc_TypeError, "expected a sequence of integers");
            return false;
        }
        Py_ssize_t count = PySequence_Size(sequence);
        if (count < 0) {
            return false;
        }
        out.clear();
        out.reserve((size_t) count);
        for (Py_ssize_t i = 0; i < count; ++i) {
            PyObject *item = PySequence_GetItem(sequence, i);
            if (!item) {
                return false;
            }
            long value = PyLong_AsLong(item);
            Py_DECREF(item);
            if (value == -1 && PyErr_Occurred()) {
                return false;
            }
            if (value < INT_MIN || value > INT_MAX) {
                PyErr_SetString(PyExc_OverflowError, "integer does not fit in a C int");
                return false;
            }
            out.push_back((int) value);
        }
        return true;
    }

    // The numpy arrays are built by a Python function so that the compiled
    // module does not depend on the numpy C API.
    static PyObject *getMatrixFactory() {
        static PyObject *factory = nullptr;
        if (!factory) {
            PyObject *module = PyImport_ImportModule("roadrunner.named_array");
            if (!module) {
                return nullptr;
            }
            factory = PyObject_GetAttrString(module, "matrix_to_array");
            Py_DECREF(module);
        }
        return factory;
    }

    PyObject *doublematrix_to_py(const ls::DoubleMatrix *m, bool structured_result, bool copy_result) {
        rrLogDebug << __FUNC__;
        ls::DoubleMatrix *mat = const_cast<ls::DoubleMatrix *>(m);

        PyObject *factory = getMatrixFactory();
        if (!factory) {
            return nullptr;
        }

        PyObject *address = PyLong_FromVoidPtr(mat->getArray());
        PyObject *rowNames = stringvector_to_py(mat->getRowNames());
        PyObject *colNames = stringvector_to_py(mat->getColNames());

        PyObject *result = nullptr;
        if (address && rowNames && colNames) {
            bool named = Config::getValue(Config::PYTHON_ENABLE_NAMED_MATRIX);
            result = PyObject_CallFunction(
                    factory, "OiiOOiiiO",
                    address, (int) mat->numRows(), (int) mat->numCols(), rowNames, colNames,
                    (int) copy_result, (int) named, (int) structured_result, Py_None);
        }

        Py_XDECREF(address);
        Py_XDECREF(rowNames);
        Py_XDECREF(colNames);
        return result;
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


    PyObject *stringvector_to_py(const std::vector<std::string> &vec) {
        rrLogDebug << __FUNC__;
        size_t size = vec.size();

        PyObject *pyList = PyList_New(size);

        unsigned j = 0;

        for (auto i = vec.begin(); i != vec.end(); ++i) {
            const std::string &str = *i;
            PyObject *pyStr = rrPyString_FromString(str.c_str());
            PyList_SetItem(pyList, j++, pyStr);
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
                    PyErr_Format(PyExc_TypeError, "Expected a string but got %R", (PyObject *) Py_TYPE(item));
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
                    PyErr_Format(PyExc_TypeError, "Expected a string but got %R", (PyObject *) Py_TYPE(item));
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
            if (PyUnicode_Check(pkey)) {
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


    PyObject *convertStringVectorToPython(const std::vector<std::string> &vec) {
        rrLogDebug << __FUNC__;
        long long size = (long long) vec.size();

        PyObject *pyList = PyList_New(size);

        unsigned j = 0;

        for (const auto &item : vec) {
            PyObject *pyStr = rrPyString_FromString(item.c_str());
            PyList_SetItem(pyList, j, pyStr);
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


    Matrix3DToNumpy::Matrix3DToNumpy(rr::Matrix3D<double, double>* matrix)
            : matrix_(matrix) {}

    PyObject *Matrix3DToNumpy::convertData() {
        // collect dimensions. x = width, y=height, z = depth.
        const Py_ssize_t xMax = matrix_->numCols();
        const Py_ssize_t yMax = matrix_->numRows();
        const Py_ssize_t zMax = matrix_->numZ();

        // linearized 3D matrix
        PyObject *buffer = PyByteArray_FromStringAndSize(
                nullptr, (Py_ssize_t) (zMax * yMax * xMax * sizeof(double)));
        if (!buffer) {
            return nullptr;
        }
        double *data = reinterpret_cast<double *>(PyByteArray_AsString(buffer));

        for (Py_ssize_t z = 0; z < zMax; z++) {
            for (Py_ssize_t y = 0; y < yMax; y++) {
                for (Py_ssize_t x = 0; x < xMax; x++) {
                    data[x + y * xMax + z * xMax * yMax] = matrix_->slice(z, y, x);
                }
            }
        }
        return buffer_to_array(buffer, "float64", {zMax, yMax, xMax});
    }

    PyObject *Matrix3DToNumpy::convertIndex() {
        // matrix_.index dimensions == depth of Matrix3D
        const Py_ssize_t zMax = matrix_->numZ();

        PyObject *buffer = PyByteArray_FromStringAndSize(nullptr, (Py_ssize_t) (zMax * sizeof(double)));
        if (!buffer) {
            return nullptr;
        }
        double *data = reinterpret_cast<double *>(PyByteArray_AsString(buffer));

        auto &index = matrix_->getIndex();
        for (Py_ssize_t z = 0; z < zMax; z++) {
            data[z] = index[z];
        }
        return buffer_to_array(buffer, "float64", {zMax});
    }

    PyObject *Matrix3DToNumpy::convertRowNames() {
        return convertStringVectorToPython(matrix_->getRowNames());
    }

    PyObject *Matrix3DToNumpy::convertColNames() {
        return convertStringVectorToPython(matrix_->getColNames());
    }


} /* namespace rr */

