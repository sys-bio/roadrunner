/*
 * PyUtils.cpp
 *
 *  Created on: Apr 27, 2014
 *      Author: andy
 *      Contrib: JKM 2014
 */

#include <stdexcept>
#include <string>

// wierdness on OSX clang, this needs to be included before python.h,
// otherwise compile pukes with:
// localefwd.h error: too many arguments provided to function-like macro invocation
#include <sstream>
#include <PyUtils.h>
#include "rrConfigure.h"




using namespace std;

namespace rr
{

/// Imported from graphfab
#define STRINGIFY(x) #x
/// Imported from graphfab
#define EXPAND_AND_STRINGIFY(x) STRINGIFY(x)

/// Imported from graphfab
char* gf_strclone(const char* src) {
    if(!src) {
        assert(0 && "gf_strclone passed null arg");
        return NULL;
    } else {
        size_t size = strlen(src)+1;
        char* dst = (char*)malloc(size*sizeof(char));

        memcpy(dst, src, size); // copies null char

        return dst;
    }
}

/// Imported from graphfab
void gf_strfree(char* str) {
    free(str);
}


/**
 * @brief Get the UTF-8 encoded buffer for a Python string
 * @details Imported from graphfab
 * @note Caller must free the buffer using @ref gf_strfree
 */
char* rrPyString_getString(PyObject* uni) {
    char* str = NULL;
//     #pragma message "RR_PYTHON_VERSION = " EXPAND_AND_STRINGIFY(RR_PYTHON_VERSION)
#if RR_PYTHON_VERSION == 3
    PyObject* bytes = PyUnicode_AsUTF8String(uni);
    str = gf_strclone(PyBytes_AsString(bytes));
    Py_XDECREF(bytes);
#else
    str = gf_strclone(PyString_AsString(uni));
#endif
    return str;
}

std::string rrPyString_getCPPString(PyObject* uni) {
    char* cstr = rrPyString_getString(uni);
    std::string str(cstr);
    gf_strfree(cstr);
    return str;
}

/// Imported from graphfab
int rrPyCompareString(PyObject* uni, const char* str) {
    #if SAGITTARIUS_DEBUG_LEVEL >= 2
    {
        printf("PyCompareString started\n");
    }
    #endif
    {
        char* s = rrPyString_getString(uni);
        int cmp = !strcmp(s,str);
        gf_strfree(s);

        if(cmp)
            return 1;
        else
            return 0;
    }
}

/// Imported from graphfab
PyObject* rrPyString_FromString(const char* s) {
#if RR_PYTHON_VERSION == 3
    return PyUnicode_FromString(s);
#else
    return PyString_FromString(s);
#endif
}

/// Imported from graphfab
PyObject* rrPyString_FromStringAndSize(const char* s, Py_ssize_t size) {
#if RR_PYTHON_VERSION == 3
    return PyUnicode_FromStringAndSize(s, size);
#else
    return PyString_FromStringAndSize(s, size);
#endif
}


PyObject* Variant_to_py(const Variant& var)
{
    PyObject *result = 0;

    const std::type_info &type = var.type();

    if (var.isEmpty()) {
        Py_RETURN_NONE;
    }

    if (type == typeid(std::string)) {
        return rrPyString_FromString(var.convert<string>().c_str());
    }

    if (type == typeid(bool)) {
        return PyBool_FromLong(var.convert<bool>());
    }

    if (type == typeid(unsigned long)) {
        return PyLong_FromUnsignedLong(var.convert<unsigned long>());
    }

    if (type == typeid(long)) {
        return PyLong_FromLong(var.convert<long>());
    }

    if (type == typeid(int)) {
# if RR_PYTHON_VERSION == 3
        // http://python3porting.com/cextensions.html
        return PyLong_FromLong(var.convert<long>());
# else
        return PyInt_FromLong(var.convert<long>());
# endif
    }

    if (type == typeid(unsigned int)) {
        return PyLong_FromUnsignedLong(var.convert<unsigned long>());
    }

    if (type == typeid(char)) {
        char c = var.convert<char>();
        return rrPyString_FromStringAndSize(&c, 1);
    }

    if (type == typeid(unsigned char)) {
# if RR_PYTHON_VERSION == 3
        // http://python3porting.com/cextensions.html
        return PyLong_FromLong(var.convert<long>());
# else
        return PyInt_FromLong(var.convert<long>());
# endif
    }

    if (type == typeid(float) || type == typeid(double)) {
        return PyFloat_FromDouble(var.convert<double>());
    }


    throw invalid_argument("could not convert " + var.toString() + "to Python object");
}

Variant Variant_from_py(PyObject* py)
{
    Variant var;

    if(py == Py_None)
    {
        return var;
    }

# if RR_PYTHON_VERSION == 3
    if (PyUnicode_Check(py))
# else
    if (PyString_Check(py))
# endif
    {
        var = rrPyString_getStdString(py);
        return var;
    }

    else if (PyBool_Check(py))
    {
        var = (bool)(py == Py_True);
        return var;
    }

    else if (PyLong_Check(py))
    {
        // need to check for overflow.
        var = (long)PyLong_AsLong(py);

        // Borrowed reference.
        PyObject* err = PyErr_Occurred();
        if (err) {
            std::stringstream ss;
            ss << "Could not convert Python long to C ";
            ss << sizeof(long) * 8 << " bit long: ";
            ss << rrPyString_getStdString(err);

            // clear error, raise our own
            PyErr_Clear();

            invalid_argument(ss.str());
        }

        return var;
    }

# if RR_PYTHON_VERSION == 3
    else if (PyLong_Check(py))
# else
    else if (PyInt_Check(py))
# endif
    {
# if RR_PYTHON_VERSION == 3
        var = PyLong_AsLong(py);
# else
        var = (int)PyInt_AsLong(py);
# endif
        return var;
    }

    else if (PyFloat_Check(py))
    {
        var = (double)PyFloat_AsDouble(py);
        return var;
    }

    string msg = "could not convert Python type to built in type";
    throw invalid_argument(msg);
}

} /* namespace rr */

