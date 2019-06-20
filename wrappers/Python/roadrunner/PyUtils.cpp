/*
 * PyUtils.cpp
 *
 *  Created on: Apr 27, 2014
 *      Author: andy
 */

// see discission on import array,
// http://docs.scipy.org/doc/numpy/reference/c-api.array.html#miscellaneous
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL RoadRunner_ARRAY_API

#include <stdexcept>
#include <string>
#include <algorithm>

// wierdness on OSX clang, this needs to be included before python.h,
// otherwise compile pukes with:
// localefwd.h error: too many arguments provided to function-like macro invocation
#include <sstream>
#include <vector>
#include <PyUtils.h>
#include <rrLogger.h>
#include <Dictionary.h>
#include "rrConfig.h"
#include "rrRoadRunnerOptions.h"
#include <numpy/arrayobject.h>
#include "structmember.h"

// python object.h
#include "object.h"

#include <iostream>
#include <iomanip>

using namespace std;

typedef vector<string> str_vector;

namespace rr
{

/// Imported from graphfab
#define STRINGIFY(x) #x
/// Imported from graphfab
#define EXPAND_AND_STRINGIFY(x) STRINGIFY(x)

/// Imported from graphfab
char* rr_strclone(const char* src) {
    if(!src) {
        assert(0 && "rr_strclone passed null arg");
        return NULL;
    } else {
        size_t size = strlen(src)+1;
        char* dst = (char*)malloc(size*sizeof(char));

        memcpy(dst, src, size); // copies null char

        return dst;
    }
}

/// Imported from graphfab
void rr_strfree(const char* str) {
    free((void*)str);
}

/**
 * @brief Get the UTF-8 encoded buffer for a Python string
 * @details Imported from graphfab
 * @note Caller must free the buffer using @ref rr_strfree
 */
char* rrPyString_getString(PyObject* uni) {
    char* str = NULL;
//     #pragma message "PY_MAJOR_VERSION = " EXPAND_AND_STRINGIFY(PY_MAJOR_VERSION)
#if PY_MAJOR_VERSION == 3
    PyObject* bytes = PyUnicode_AsUTF8String(uni);
    str = rr_strclone(PyBytes_AsString(bytes));
    Py_XDECREF(bytes);
#else
    str = rr_strclone(PyString_AsString(uni));
#endif
    return str;
}

std::string rrPyString_getCPPString(PyObject* uni) {
    char* cstr = rrPyString_getString(uni);
    std::string str(cstr);
    rr_strfree(cstr);
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
        rr_strfree(s);

        if(cmp)
            return 1;
        else
            return 0;
    }
}

/// Imported from graphfab
PyObject* rrPyString_FromString(const char* s) {
#if PY_MAJOR_VERSION == 3
    return PyUnicode_FromString(s);
#else
    return PyString_FromString(s);
#endif
}

/// Imported from graphfab
PyObject* rrPyString_FromStringAndSize(const char* s, Py_ssize_t size) {
#if PY_MAJOR_VERSION == 3
    return PyUnicode_FromStringAndSize(s, size);
#else
    return PyString_FromStringAndSize(s, size);
#endif
}

char* rrPyString_AsString(PyObject* s) {
#if PY_MAJOR_VERSION == 3
    char* str;
    PyObject* bytes = PyUnicode_AsUTF8String(s);
    str = rr_strclone(PyBytes_AsString(bytes));
    Py_XDECREF(bytes);
    return str;
#else
    return rr_strclone(PyString_AsString(s));
#endif
}

PyObject* Variant_to_py(const Variant& var)
{
    PyObject *result = 0;

    const std::type_info &type = var.typeInfo();

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
# if PY_MAJOR_VERSION == 3
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
# if PY_MAJOR_VERSION == 3
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

	if (py == Py_None)
	{
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
			ss << rrPyString_getCPPString(err);

			// clear error, raise our own
			PyErr_Clear();

			invalid_argument(ss.str());
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

	else if (PyList_Check(py))
	{
		vector<double> data;
		for (Py_ssize_t i = 0; i < PyList_Size(py); i++) {
			PyObject* value = PyList_GetItem(py, i);
			data.push_back(PyFloat_AsDouble(value));
		}
		return data;
	}

    string msg = "could not convert Python type to built in type";
    throw invalid_argument(msg);
}

PyObject* dictionary_keys(const Dictionary* dict)
{
    std::vector<std::string> keys = dict->getKeys();

    unsigned size = keys.size();

    PyObject* pyList = PyList_New(size);

    unsigned j = 0;

    for (std::vector<std::string>::const_iterator i = keys.begin(); i != keys.end(); ++i)
    {
        const std::string& key  = *i;
        PyObject* pyStr = rrPyString_FromString(key.c_str());
        PyList_SET_ITEM(pyList, j++, pyStr);
    }

    return pyList;
}

PyObject* dictionary_values(const Dictionary* dict)
{
    std::vector<std::string> keys = dict->getKeys();

    unsigned size = keys.size();

    PyObject* pyList = PyList_New(size);

    unsigned j = 0;

    for (std::vector<std::string>::const_iterator i = keys.begin(); i != keys.end(); ++i)
    {
        const std::string& key  = *i;
        PyObject* pyVal = Variant_to_py(dict->getItem(key));
        PyList_SET_ITEM(pyList, j++, pyVal);
    }

    return pyList;
}

PyObject* dictionary_items(const Dictionary* dict)
{
    std::vector<std::string> keys = dict->getKeys();

    unsigned size = keys.size();

    PyObject* pyList = PyList_New(size);

    unsigned j = 0;

    for (std::vector<std::string>::const_iterator i = keys.begin(); i != keys.end(); ++i)
    {
        const std::string& key  = *i;
        PyObject* pyStr = Variant_to_py(dict->getItem(key));

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

PyObject* dictionary_getitem(const Dictionary* dict, const char* key)
{
    return Variant_to_py(dict->getItem(key));
}

PyObject* dictionary_setitem(Dictionary* dict, const char* key, PyObject* value)
{
    dict->setItem(key, Variant_from_py(value));
    Py_RETURN_NONE;
}

void dictionary_delitem(Dictionary* dict, const char* key)
{
    dict->deleteItem(key);
}

PyObject* dictionary_contains(const Dictionary* dict, const char* key)
{
    bool contains = dict->hasKey(key);
    return PyBool_FromLong(contains);
}

#define VERIFY_PYARRAY(p) { \
    assert(p && "PyArray is NULL"); \
    assert((PyArray_NBYTES(p) > 0 ? PyArray_ISCARRAY(p) : true) &&  "PyArray must be C format"); \
}

/*  original %typemap(out) ls::DoubleMatrix {
 *
    PyObject *pArray = NULL;
    int rows = ($1).numRows();
    int cols = ($1).numCols();
    double *data = (double*)malloc(sizeof(double)*rows*cols);
    memcpy(data, ($1).getArray(), sizeof(double)*rows*cols);

    if(cols == 1) {
        int nd = 1;
        npy_intp dims[1] = {rows};
        pArray = PyArray_New(&PyArray_Type, nd, dims, NPY_DOUBLE, NULL, data, 0,
                             NPY_CARRAY | NPY_OWNDATA, NULL);

    }
    else {
        int nd = 2;
        npy_intp dims[2] = {rows, cols};
        pArray = PyArray_New(&PyArray_Type, nd, dims, NPY_DOUBLE, NULL, data, 0,
                             NPY_CARRAY | NPY_OWNDATA, NULL);
    }

    VERIFY_PYARRAY(pArray);
    $result  = pArray;
 */

/*  original %typemap(out) const ls::DoubleMatrix* {
 *
    PyObject *pArray = NULL;
    int rows = ($1)->numRows();
    int cols = ($1)->numCols();
    double *data = ($1)->getArray();

    if(cols == 1) {
        int nd = 1;
        npy_intp dims[1] = {rows};
        pArray = PyArray_New(&PyArray_Type, nd, dims, NPY_DOUBLE, NULL, data, 0,
                             NPY_CARRAY, NULL);
    }
    else {
        int nd = 2;
        npy_intp dims[2] = {rows, cols};
        pArray = PyArray_New(&PyArray_Type, nd, dims, NPY_DOUBLE, NULL, data, 0,
                             NPY_CARRAY, NULL);
    }

    VERIFY_PYARRAY(pArray);
    $result  = pArray;
 */


static PyObject* NamedArray_New(int nd, npy_intp *dims, double *data, int pyFlags,
        const ls::DoubleMatrix* mat);


PyObject* doublematrix_to_py(const ls::DoubleMatrix* m, bool structured_result, bool copy_result)
{
    ls::DoubleMatrix *mat = const_cast<ls::DoubleMatrix*>(m);


    // a valid array descriptor:
    // In [87]: b = array(array([0,1,2,3]),
    //      dtype=[('r', 'f8'), ('g', 'f8'), ('b', 'f8'), ('a', 'f8')])


    // are we returning a structured array?
    if (structured_result) {

        // get the column names
        //const std::vector<SelectionRecord>& sel = ($self)->getSelections();
        //std::vector<string> names(sel.size());

        //for(int i = 0; i < sel.size(); ++i) {
        //    names[i] = sel[i].to_string();
        //}

        std::vector<string> names = mat->getColNames();


        int rows = mat->numRows();
        int cols = mat->numCols();

        if (cols == 0) {
            Py_RETURN_NONE;
        }

        if (cols != names.size()) {
            throw std::logic_error("column names size does not match matrix columns size");
        }

        double* mData = mat->getArray();

        PyObject* list = PyList_New(names.size());

        for(int i = 0; i < names.size(); ++i)
        {
            PyObject *col = rrPyString_FromString(names[i].c_str());
            PyObject *type = rrPyString_FromString("f8");
            PyObject *tup = PyTuple_Pack(2, col, type);

            Py_DECREF(col);
            Py_DECREF(type);

            // list takes ownershipt of tuple
            void PyList_SET_ITEM(list, i, tup);
        }

        PyArray_Descr* descr = 0;
        PyArray_DescrConverter(list, &descr);

        // done with list
        Py_CLEAR(list);
        npy_intp dims[] = {rows};

        // steals a reference to descr
        PyObject *pyres = PyArray_SimpleNewFromDescr(1, dims,  descr);
        VERIFY_PYARRAY(pyres);

        if (pyres) {

            assert(PyArray_NBYTES(pyres) == rows*cols*sizeof(double) && "invalid array size");

            double* data = (double*)PyArray_BYTES(pyres);

            memcpy(data, mData, rows*cols*sizeof(double));
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

            Log(rr::Logger::LOG_DEBUG) << "copying result data";

        // passing a NULL for data tells numpy to allocate its own data

            // make a 1D array in this case
            if(cols == 1 && mat->getColNames().size() == 0) {
                int nd = 1;
                npy_intp dims[1] = {rows};
                pArray = PyArray_New(&PyArray_Type, nd, dims, NPY_DOUBLE,
                        NULL, NULL, 0, 0, NULL);
            }
            else {
                int nd = 2;
                npy_intp dims[2] = {rows, cols};
                pArray = NamedArray_New(nd, dims, NULL,
                                     0, mat);
            }

            VERIFY_PYARRAY(pArray);

            // copy our data into the numpy array
            double *data = static_cast<double*>(PyArray_DATA(pArray));
            memcpy(data, mat->getArray(), sizeof(double)*rows*cols);

        }
        else {

            Log(rr::Logger::LOG_DEBUG) << "wraping existing data";

            double *data = mat->getArray();

            if(cols == 1 && mat->getColNames().size() == 0) {
                int nd = 1;
                npy_intp dims[1] = {rows};
                pArray = PyArray_New(&PyArray_Type, nd, dims, NPY_DOUBLE,
                        NULL, data, 0, NPY_CARRAY, NULL);
            }
            else {
                int nd = 2;
                npy_intp dims[2] = {rows, cols};
                pArray = NamedArray_New(nd, dims, data,
                        NPY_CARRAY, mat);
            }

            VERIFY_PYARRAY(pArray);

        }
        return pArray;
    }
}


struct NamedArrayObject {
    PyArrayObject array;
    PyObject *rowNames;
    PyObject *colNames;
    int test1;
    int test2;
    int test3;
} ;

static PyObject* NamedArrayObject_Finalize(NamedArrayObject * self, PyObject *parent);

static PyObject *NamedArray_repr(NamedArrayObject *self);

static PyObject *NamedArray_str(NamedArrayObject *self);

static void NamedArrayObject_dealloc(NamedArrayObject *self)
{
    Log(Logger::LOG_INFORMATION) << __FUNC__;
    Py_XDECREF(self->rowNames);
    Py_XDECREF(self->colNames);

    PyObject *pself = (PyObject*)self;

    assert(pself->ob_type->tp_base == &PyArray_Type);
    PyArray_Type.tp_dealloc(pself);

    Log(Logger::LOG_INFORMATION) <<  __FUNC__ << ", Done";
}



static PyObject *NamedArrayObject_alloc(PyTypeObject *type, Py_ssize_t nitems)
{
    Log(Logger::LOG_INFORMATION) << __FUNC__;
    PyObject *obj;

    assert(type->tp_basicsize == sizeof(NamedArrayObject));

    obj = (PyObject *)PyArray_malloc(type->tp_basicsize);
    PyObject_Init(obj, type);
    ((NamedArrayObject *)obj)->rowNames = NULL;
    ((NamedArrayObject *)obj)->colNames = NULL;
    ((NamedArrayObject *)obj)->test1 = 10;
    ((NamedArrayObject *)obj)->test2 = 11;
    ((NamedArrayObject *)obj)->test3 = 12;

    Log(Logger::LOG_INFORMATION) << "created obj: " << obj;
    return obj;
}

#define CSTR(str) const_cast<char*>(str)


static PyMemberDef NamedArray_members[] = {
        {CSTR("rownames"), T_OBJECT_EX, offsetof(NamedArrayObject, rowNames), 0, CSTR("row names")},
        {CSTR("colnames"), T_OBJECT_EX, offsetof(NamedArrayObject, colNames), 0, CSTR("column names")},
        {CSTR("test1"), T_INT, offsetof(NamedArrayObject, test1), 0, CSTR("test1 number")},
        {CSTR("test2"), T_INT, offsetof(NamedArrayObject, test2), 0, CSTR("test2 number")},
        {CSTR("test3"), T_INT, offsetof(NamedArrayObject, test3), 0, CSTR("test3 number")},
        {NULL}  /* Sentinel */
};

static PyMethodDef NamedArray_methods[] = {
    { CSTR("__array_finalize__"), (PyCFunction)NamedArrayObject_Finalize, METH_VARARGS,  CSTR("")},
    {0}
};

/*
static PyMethodDef DatetimeArray_methods[] = {
    { "__array_finalize__", (PyCFunction)DatetimeArray_finalize, METH_VARARGS,
      ""},
//    {"__getitem__", (PyCFunction)DatetimeArray_getitem, METH_VARARGS, ""},
    {"has_dups", (PyCFunction)DatetimeArray_has_dups, METH_VARARGS, ""},
    {"has_missing", (PyCFunction)DatetimeArray_has_missing, METH_VARARGS, ""},
    {"is_chrono", (PyCFunction)DatetimeArray_is_chrono, METH_VARARGS, ""},
    {"is_full", (PyCFunction)DatetimeArray_is_full, METH_VARARGS, ""},
    {"is_valid", (PyCFunction)DatetimeArray_is_valid, METH_VARARGS, ""},
    {"date_to_index", (PyCFunction)DatetimeArray_date_to_index, METH_VARARGS, ""},
    {"tovalues", (PyCFunction)DatetimeArray_tovalues, METH_VARARGS, ""},
    {"toordinals", (PyCFunction)DatetimeArray_toordinals, METH_VARARGS, ""},
    {"tolist", (PyCFunction)DatetimeArray_tolist, METH_VARARGS, ""},
    {"fill_missing_dates", (PyCFunction)DatetimeArray_fill_missing_dates, METH_KEYWORDS, ""},
    {"get_missing_dates_mask", (PyCFunction)DatetimeArray_get_missing_dates_mask, METH_VARARGS, ""},
    {"convert", (PyCFunction)DatetimeArray_convert, METH_KEYWORDS, ""},
    {0}
};
*/



//static PyMappingMethods NamedArray_MappingMethods = {
//#if PY_VERSION_HEX >= 0x02050000
//    (lenfunc)array_length,              /*mp_length*/
//#else
//    (inquiry)array_length,              /*mp_length*/
//#endif
//    (binaryfunc)array_subscript_nice,       /*mp_subscript*/
//    (objobjargproc)array_ass_sub,       /*mp_ass_subscript*/
//};


/**
 * binaryfunc PyMappingMethods.mp_subscript
 * This function is used by PyObject_GetItem() and has the same signature.
 * This slot must be filled for the PyMapping_Check() function to return 1,
 * it can be NULL otherwise.
 */
static PyObject *NammedArray_subscript(NamedArrayObject *self, PyObject *op)
{
    binaryfunc PyArray_subscript = PyArray_Type.tp_as_mapping->mp_subscript;
    int dim = PyArray_NDIM(self);

# if PY_MAJOR_VERSION == 3
    if (dim == 2 && PyUnicode_Check(op)) {
# else
    if (dim == 2 && PyString_Check(op)) {
# endif
        char* keyName = rrPyString_AsString(op);

        PyObject *colSeq = PySequence_Fast(self->colNames, "expected a sequence");
        int len = PySequence_Size(colSeq);
        for (int col = 0; col < len; col++) {
            PyObject *item = PySequence_Fast_GET_ITEM(colSeq, col);
            char* itemStr = rrPyString_AsString(item);

            if(strcmp(keyName, itemStr) == 0) {

                int rows = PyArray_DIM(self, 0);
                int cols = PyArray_DIM(self, 1);

                npy_intp dims[1] = {rows};
                PyObject *result = PyArray_New(&PyArray_Type, 1, dims, NPY_DOUBLE, NULL, NULL, 0,
                        NPY_CARRAY, NULL);

                // copy data to result array
                double* data = (double*)PyArray_DATA(self);
                double* newData = (double*)PyArray_DATA(result);

                for(int i = 0; i < rows; ++i) {
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
            char* itemStr = rrPyString_AsString(item);

            if(strcmp(keyName, itemStr) == 0) {

                int rows = PyArray_DIM(self, 0);
                int cols = PyArray_DIM(self, 1);

                npy_intp dims[1] = {cols};
                PyObject *result = PyArray_New(&PyArray_Type, 1, dims, NPY_DOUBLE, NULL, NULL, 0,
                        NPY_CARRAY, NULL);

                // copy data to result array
                double* data = (double*)PyArray_DATA(self);
                double* newData = (double*)PyArray_DATA(result);

                for(int i = 0; i < cols; ++i) {
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
    return PyArray_subscript((PyObject*)self, op);
}

static PyMappingMethods NamedArray_MappingMethods = {
#if PY_VERSION_HEX >= 0x02050000
    (lenfunc)0,             /*mp_length*/
#else
    (inquiry)0,             /*mp_length*/
#endif
    (binaryfunc)0,          /*mp_subscript*/
    (objobjargproc)0,       /*mp_ass_subscript*/
};



static PyTypeObject NamedArray_Type = {
#if PY_MAJOR_VERSION == 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                        /* ob_size */
#endif
    "NamedArray",                             /* tp_name */
    sizeof(NamedArrayObject),                 /* tp_basicsize */
    0,                                        /* tp_itemsize */
    (destructor)NamedArrayObject_dealloc,     /* tp_dealloc */
    0,                                        /* tp_print */
    0,                                        /* tp_getattr */
    0,                                        /* tp_setattr */
    0,                                        /* tp_compare */
    (reprfunc)NamedArray_repr,                /* tp_repr */
    0,                                        /* tp_as_number */
    0,                                        /* tp_as_sequence */
    &NamedArray_MappingMethods,               /* tp_as_mapping */
    0,                                        /* tp_hash */
    0,                                        /* tp_call */
    (reprfunc)NamedArray_str,                 /* tp_str */
    0,                                        /* tp_getattro */
    0,                                        /* tp_setattro */
    0,                                        /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    "NamedArray",                             /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                                        /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    NamedArray_methods,                       /* tp_methods */
    NamedArray_members,                       /* tp_members */
    0,                                        /* tp_getset */
    0,                                        /* tp_base, this gets set to &PyArray_Type when module is initialized */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    0,                                        /* tp_init */
    NamedArrayObject_alloc,                  /* tp_alloc */
    0,                        /* tp_new */
};


PyObject* NamedArrayObject_Finalize(NamedArrayObject * self, PyObject *parent)
{
    Log(Logger::LOG_INFORMATION) << __FUNC__;

    if(parent != NULL && parent->ob_type == &NamedArray_Type)
    {
        NamedArrayObject *p = (NamedArrayObject *)parent;
        if (p->rowNames != NULL) {
            Py_INCREF(p->rowNames);
            self->rowNames = p->rowNames;
        }

        if (p->colNames != NULL) {
            Py_INCREF(p->colNames);
            self->colNames = p->colNames;
        }

        self->test1 = p->test1;
        self->test2 = p->test2;
        self->test3 = p->test3;
    }
    Py_RETURN_NONE;
}


/*
NPY_NO_EXPORT PyObject *
PyArray_New(PyTypeObject *subtype, int nd, npy_intp *dims, int type_num,
            npy_intp *strides, void *data, int itemsize, int flags,
            PyObject *obj)
{
    PyArray_Descr *descr;
    PyObject *new;

    descr = PyArray_DescrFromType(type_num);
    if (descr == NULL) {
        return NULL;
    }
    if (descr->elsize == 0) {
        if (itemsize < 1) {
            PyErr_SetString(PyExc_ValueError,
                            "data type must provide an itemsize");
            Py_DECREF(descr);
            return NULL;
        }
        PyArray_DESCR_REPLACE(descr);
        descr->elsize = itemsize;
    }
    new = PyArray_NewFromDescr(subtype, descr, nd, dims, strides,
                               data, flags, obj);
    return new;
}
 */

/* PyArray_New(&PyArray_Type, nd, dims, NPY_DOUBLE, NULL, data, 0,
                                     NPY_CARRAY | NPY_OWNDATA, NULL);
 */
PyObject* NamedArray_New(int nd, npy_intp *dims, double *data, int pyFlags,
        const ls::DoubleMatrix* mat)
{
    bool named = Config::getValue(Config::PYTHON_ENABLE_NAMED_MATRIX);

    if(named) {
        Log(Logger::LOG_INFORMATION) << "creating NEW style array";

        NamedArrayObject *array = (NamedArrayObject*)PyArray_New(
                &NamedArray_Type, nd, dims, NPY_DOUBLE, NULL, data, 0,
                        pyFlags, NULL);

        if (array == NULL) {
            PyObject* pystr = PyObject_Str(PyErr_Occurred());
            const char* error = rrPyString_AsString(pystr);
            Log(Logger::LOG_CRITICAL) << error;
            return NULL;
        }

        array->rowNames = stringvector_to_py(mat->getRowNames());
        array->colNames = stringvector_to_py(mat->getColNames());
        array->test1 = 1;
        array->test2 = 2;
        array->test3 = 3;
        return (PyObject*)array;

    } else {
        Log(Logger::LOG_INFORMATION) << "creating old style array";
        return PyArray_New(&PyArray_Type, nd, dims, NPY_DOUBLE, NULL, data, 0,
                pyFlags, NULL);
    }
}

PyObject* stringvector_to_py(const std::vector<std::string>& vec)
{
    unsigned size = vec.size();

    PyObject* pyList = PyList_New(size);

    unsigned j = 0;

    for (std::vector<std::string>::const_iterator i = vec.begin(); i != vec.end(); ++i)
    {
        const std::string& str  = *i;
        PyObject* pyStr = rrPyString_FromString(str.c_str());
        PyList_SET_ITEM(pyList, j++, pyStr);
    }

    return pyList;
}

std::vector<std::string> py_to_stringvector(PyObject* obj)
{
    str_vector result;
    if(obj) {
        PyObject *seq = PySequence_Fast(obj, "expected a sequence");
        if(obj) {
            unsigned len = PySequence_Size(obj);
            if (PyList_Check(seq))
                for (unsigned i = 0; i < len; i++) {
                    PyObject *item = PyList_GET_ITEM(seq, i);
                    // Return value: New reference.
                    PyObject *pystr = PyObject_Str(item);
                    result.push_back(rrPyString_AsString(pystr));
                    Py_XDECREF(pystr);
                }
            else
                for (unsigned i = 0; i < len; i++) {
                    PyObject *item = PyTuple_GET_ITEM(seq, i);
                    // Return value: New reference.
                    PyObject *pystr = PyObject_Str(item);
                    result.push_back(rrPyString_AsString(pystr));
                    Py_XDECREF(pystr);
                }
            Py_XDECREF(seq);
        }
    }

    return result;
}



Dictionary *Dictionary_from_py(PyObject *py)
{
    if (PyDict_Check(py) == 0) {
        throw invalid_argument("object is not a dictionary");
    }

    BasicDictionary* dict = new BasicDictionary();

    PyObject *pkey, *pvalue;
    Py_ssize_t pos = 0;

    while (PyDict_Next(py, &pos, &pkey, &pvalue)) {
# if PY_MAJOR_VERSION == 3
        if (PyUnicode_Check(pkey)) {
# else
        if (PyString_Check(pkey)) {
# endif
            std::string key(rrPyString_AsString(pkey));
            Variant value = Variant_from_py(pvalue);

            dict->setItem(key, value);

        } else {
            throw invalid_argument("keys must be strings");
        }
    }

    return dict;
}


void pyutil_init(PyObject *module)
{
    // set up the base class to be the numpy ndarray PyArray_Type
    NamedArray_Type.tp_base = &PyArray_Type;


    // set up the pointers of the NamedArray_MappingMethods to point
    // to the numpy ones
    PyMappingMethods *numpyMappMethods = PyArray_Type.tp_as_mapping;

    assert(numpyMappMethods && "Numpy PyMappingMethods is NULL");

    NamedArray_MappingMethods = *numpyMappMethods;

    // set our getitem pointer
    NamedArray_MappingMethods.mp_subscript = (binaryfunc)NammedArray_subscript;

    int result;

    if ((result = PyType_Ready(&NamedArray_Type)) < 0) {
        std::cout << "PyType_Ready(&NamedArray_Type)) Failed, error: " << result;
        return;
    }

    Py_INCREF(&NamedArray_Type);
    result = PyModule_AddObject(module, "NamedArray", (PyObject *)(&NamedArray_Type));
}


/*****************************************************************************************
 * Array Printing Stuff
 ****************************************************************************************/



static int longestStrLen(const str_vector& s) {
    string::size_type longest = 0;
    for(str_vector::const_iterator i = s.begin(); i != s.end(); ++i) {
        longest = std::max(longest, i->length());
    }
    return longest;
}

static std::string array_format(PyArrayObject *arr,
        const str_vector& rowNames, const str_vector& colNames) {

    unsigned ndim = PyArray_NDIM(arr);
    unsigned rows = ndim > 0 ? PyArray_DIM(arr, 0) : 0;
    unsigned cols = ndim > 1 ? PyArray_DIM(arr, 1) : 0;

    assert(rows > 0 && cols > 0);

    #define GET_VAL(_row, _col) *((double *) PyArray_GETPTR2(arr, _row, _col))

    // longest row name + " [["
    int rowNameWidth = longestStrLen(rowNames) + 1;

    // find widest element in each column
    vector<unsigned> columnWidths(cols);
    for(unsigned i = 0; i < rows; ++i) {
        for(unsigned j = 0; j < cols; ++j) {
            unsigned colw = columnWidths[j];
            colw = std::max(colw, j < colNames.size() ? (unsigned)colNames[j].length() + 1 : 0);
            stringstream ss;
            ss << GET_VAL(i, j);
            colw = std::max(colw, (unsigned)ss.str().length() + 1);
            columnWidths[j] = colw;
        }
    }

    stringstream ss;

    // first row, column names
    if(colNames.size()) {
        for(unsigned j = 0; j < cols; ++j) {
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
    for(unsigned i = 0; i < rows; ++i) {
        // row names
        ss << setw(rowNameWidth);
        ss << left;
        ss << (i < rowNames.size() ? rowNames[i] : "");
        ss << ((i > 0 && rows > 1) ? " [" : "[[");
        // columns
        for(unsigned j = 0; j < cols; ++j) {
            ss << setw(columnWidths[j]);
            ss << right;
            ss << GET_VAL(i, j);
            if (j < cols-1) {
                ss << ",";
            }
        }
        ss << ((i < rows - 1) ? "]," : "]]");
        ss << endl;
    }
    return ss.str();

    #undef GET_VAL
}

PyObject *NamedArray_repr(NamedArrayObject *self)
{
    PyArrayObject *array = (PyArrayObject*)self;
    str_vector rowNames = py_to_stringvector(self->rowNames);
    str_vector colNames = py_to_stringvector(self->colNames);

    unsigned ndim = PyArray_NDIM(array);
    unsigned rows = ndim > 0 ? PyArray_DIM(array, 0) : 0;
    unsigned cols = ndim > 1 ? PyArray_DIM(array, 1) : 0;

    if(rows == 0 || cols == 0) {
        return PyArray_Type.tp_str((PyObject*)self);
    }

    string str = array_format(array, rowNames, colNames);

    return rrPyString_FromString(str.c_str());
}

PyObject *NamedArray_str(NamedArrayObject *self)
{
    return NamedArray_repr(self);
}




} /* namespace rr */

