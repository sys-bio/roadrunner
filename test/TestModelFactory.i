%module TestModelFactory

%{
    #define SWIG_FILE_WITH_INIT
    #define PY_ARRAY_UNIQUE_SYMBOL RoadRunner_ARRAY_API // for the NamedArrayObject (see roadrunner.i)

    #include "PyUtils.h" // for variant_to_py and back.
    #include "Matrix.h"
    #include "Matrix3D.h"
    #include <vector>
    #include "TestModelFactory.h"
    using namespace rr;

%}

//// since we be using the ls::Matrix from a Python
//// we'll be using the same API needed by the main roadrunner bindings
%include "numpy.i"
%init %{
import_array();
rr::pyutil_init(m);
%}
//
#include <vector>
%include <typemaps.i>
%include "std_vector.i"
%include "std_string.i"
%include "std_pair.i"
%include "std_unordered_map.i"


/**
 * Converts a C++ vector of strings to a Python list of strings
 */
%template() std::vector<int>;
%template() std::vector<std::string>;

%apply std::vector<std::string> INOUT { // note the INOUT which is important for pass by reference
    std::vector<std::string>&,
    const std::vector<std::string>&
}

// make a Python Tuple from a C++ DoublePair
%typemap(out) std::pair<double, double>*
{
    $result = PyTuple_New((Py_ssize_t)2); // new tuple, 2 elements
    if (!$result){
        std::cerr << "Failed to create PyTuple with 2 elements " << std::endl;
    }
    int err = PyTuple_SetItem($result, (Py_ssize_t)0, PyFloat_FromDouble((*$1).first));
    if (err < 0){
        std::cerr << "Failed to add item to tuple " << std::endl;
    }
    err = PyTuple_SetItem($result, (Py_ssize_t)1, PyFloat_FromDouble((*$1).second ));
    if (err < 0){
        std::cerr << "Failed to add item to tuple " << std::endl;
    }
}

%apply std::pair<double, double>* {
        std::pair<double, double>,
        std::pair<double, double>&,
        const std::pair<double, double>&
    };

/**
 * note to future developers: This double map is never actually used.
 * This was part of my learning swig, since a double: double map
 * is simpler than a string Variant map (i.e. to convert to Python dict).
 * Keeping this typemap for future reference (cw)
 */
%typemap(out) std::unordered_map<double, double>* {
    $result = PyDict_New();
    if (!result){
        std::cerr << "Could not create Python Dict" << std::endl;
    }
    for (const auto& item: *$1){
        int err = PyDict_SetItem($result, PyFloat_FromDouble(item.first), PyFloat_FromDouble(item.second));
        if (err < 0){
            std::cout << "Could not create item in Python Dict" << std::endl;
        }
    }
}

%apply std::unordered_map<double, double> * {
    std::unordered_map<double, double> ,
    const std::unordered_map<double, double>&,
    std::unordered_map<double, double>&
    };

%typemap(out) rr::Setting {
    $result = Variant_to_py($1);
}

%apply rr::Setting {
    rr::Setting*
};


/**
 * Converts a rr::Matrix<double> to a numpy array,
 * using the same functions/methods as for ls::Matrix<double>
 * (proxy via rrDoubleMatrix_to_py)
 */
%typemap(out) rr::Matrix<double> {
    // marker for rrDoubleMatrix typemap. Look for me in TestModelFactoryPYTHON_wrap.cxx
    const rr::Matrix<double>* mat = &($1);
    $result = rrDoubleMatrix_to_py(mat, true);
}

%apply rr::Matrix<double> {
    const rr::Matrix<double>,
    const rr::Matrix<double>&,
    rr::Matrix<double>&
};


/**
 * @brief typemap to convert a string : Setting map into a Python dict.
 * Used in the "settings" map of tmf.
 */
%typemap(out) std::unordered_map< std::string,rr::Setting,std::hash< std::string >,std::equal_to< std::string >,std::allocator< std::pair< std::string const,rr::Setting > > >* {
    // Marker for Setting typemap
    $result = PyDict_New();
    if (!result){
        std::cerr << "Could not create Python Dict" << std::endl;
    }

    for (const auto& item: *$1){
        int err = PyDict_SetItem($result, PyUnicode_FromString(item.first.c_str()), Variant_to_py(item.second));
        if (err < 0){
            std::cout << "Could not create item in Python Dict" << std::endl;
        }
    }
}

// non pointer version
%typemap(out) std::unordered_map< std::string,rr::Setting,std::hash< std::string >,std::equal_to< std::string >,std::allocator< std::pair< std::string const,rr::Setting > > > {
    $result = PyDict_New();
    if (!$result){
        std::cerr << "Could not create Python Dict" << std::endl;
    }

    // $1 is a SwigValueWrapper. The "&" operator returns the pointer to the underlying wrapped map
    auto val = &$1;

    // we can now use val as a pointer to our map
    for (const auto& item: *val){
        int err = PyDict_SetItem($result, PyUnicode_FromString(item.first.c_str()), Variant_to_py(item.second));
        if (err < 0){
            std::cout << "Could not create item in Python Dict" << std::endl;
        }
    }
}


/**************************************************
 * How to create a numpy array from a std::vector<double>?
 */

/**
 * Convert a vector of doubles to a 1D numpy array
 */
%typemap(out) std::vector<double> {
    // marker for a std::vector<double> typemap
    const npy_intp length = $1.size();

    // create a numpy type
    PyArray_Descr *desc = PyArray_DescrFromType(NPY_DOUBLE);

    // copy the data from the std::vector<double> so that we can give the
    // ptr to the numpy array
    // hopefully this will be cleaned up by numpy...
    double* data = new double[length];
    std::copy($1.begin(), $1.end(), data);
    $result = PyArray_SimpleNewFromData(1, &length, NPY_DOUBLE, (void*)data);

    PyArrayObject * arr = reinterpret_cast<PyArrayObject*>($result);

    PyArray_ENABLEFLAGS(arr, NPY_ARRAY_OWNDATA);
    int owned = PyArray_CHKFLAGS(arr, NPY_ARRAY_OWNDATA);
    if (!owned){
        std::cerr << "PyArrayObject does not own its memory" << std::endl;
    }
}

/**
 * Convert a rr::Matrix3D<double, double> to a
 * numpy Tuple[np.ndarray, np.ndarray]
 */
%typemap(out) rr::Matrix3D<double, double> {
    // marker for a rr::Matrix3D<double, double> typemap
    Matrix3DToNumpy matrix3DtoNumpy($1);
    PyObject* npArray3D = matrix3DtoNumpy.convertData();
    PyObject* idx = matrix3DtoNumpy.convertIndex();
    PyObject* colnames = matrix3DtoNumpy.convertColNames();
    PyObject* rownames = matrix3DtoNumpy.convertRowNames();

    $result = PyTuple_Pack(4, idx, npArray3D, rownames, colnames);
}

// Typemap for a C++ string
%typemap(in) str {
    // typemap for Python -> C string
  if (PyString_Check($input)) {
    $1 = std::string(PyString_AsString($input));
  } else {
    SWIG_exception(SWIG_TypeError, "string expected");
  }
}
// Copy the typecheck code for "char *".
%typemap(typecheck) std::string = char *;


/**
 * Converts an unordered_map<string, DoublePair>
 *
 */
%typemap(out) std::unordered_map< std::string,std::pair< double,double >,std::hash< std::string >,std::equal_to< std::string >,std::allocator< std::pair< std::string const,std::pair< double,double > > > >  {
    $result = PyDict_New();
    if (!$result){
        std::cerr << "Could not create Python Dict" << std::endl;
    }
    // swig create a SwigValueWrapper here. In order to
    // iterate over the map, we extract the pointer
    auto valPtr = &$1;

    for (const auto& item: *valPtr){
        // make tuple from item.second
        PyObject* tup = PyTuple_New((Py_ssize_t)2); // new tuple, 2 elements
        if (!tup){
            std::cerr << "Failed to create PyTuple with 2 elements " << std::endl;
        }
        int err = PyTuple_SetItem(tup, (Py_ssize_t)0, PyFloat_FromDouble(item.second.first));
        if (err < 0){
            std::cerr << "Failed to add item to tuple " << std::endl;
        }
        err = PyTuple_SetItem(tup, (Py_ssize_t)1, PyFloat_FromDouble(item.second.second ));
        if (err < 0){
            std::cerr << "Failed to add item to tuple " << std::endl;
        }
        // now create the dictionary
        err = PyDict_SetItem($result, PyUnicode_FromString(item.first.c_str()), tup);
        if (err < 0){
            std::cout << "Could not create item in Python Dict" << std::endl;
        }
    }
}


/**
 * Converts an unordered_map<string, double>
 * to python dict
 *
 */
%typemap(out) std::unordered_map< std::string,double>  {
    $result = PyDict_New();
    if (!$result){
        std::cerr << "Could not create Python Dict" << std::endl;
    }
    // swig create a SwigValueWrapper here. In order to
    // iterate over the map, we extract the pointer
    auto valPtr = &$1;

    for (const auto& item: *valPtr){
        // now create the dictionary
        int err = PyDict_SetItem($result, PyUnicode_FromString(item.first.c_str()), PyFloat_FromDouble(item.second));
        if (err < 0){
            std::cout << "Could not create item in Python Dict" << std::endl;
        }
    }
}


/**
 * Convert ls::Complex from C --> Python
 * Copy of the one in roadrunner.i
 */
%typemap(out) std::vector<ls::Complex> {
    typedef std::complex<double> cpx;

    std::vector<cpx>& vec = $1;

    bool iscpx = false;

    // small number
    double epsilon = 2 * std::numeric_limits<double>::epsilon();

    // check to see whether we have any imaginary parts greater than small number epsilon
    for (std::vector<cpx>::const_iterator i = vec.begin(); i != vec.end(); ++i)
    {
        iscpx = iscpx || (std::imag(*i) >= epsilon);
        if (iscpx) break;
    }

    if (iscpx) {
        // if needed, we use complex
        size_t len = $1.size();
        npy_intp dims[1] = {static_cast<npy_intp>(len)};

        PyObject *array = PyArray_SimpleNew(1, dims, NPY_COMPLEX128);

        if (!array) {
            // TODO error handling.
            return 0;
        }

        cpx *data = (cpx*)PyArray_DATA((PyArrayObject*)array);

        memcpy(data, &vec[0], sizeof(std::complex<double>)*len);

        $result  = array;
    } else {
        // otherwise we just use double
        size_t len = $1.size();
        npy_intp dims[1] = {static_cast<npy_intp>(len)};

        PyObject *array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);

        if (!array) {
            // TODO error handling.
            return 0;
        }

        double *data = (double*)PyArray_DATA((PyArrayObject*)array);
        for (int i = 0; i < vec.size(); ++i) {
            data[i] = std::real(vec[i]);
        }

        $result  = array;
    }
}

%apply std::vector<ls::Complex> {
    std::vector<std::complex<double>>
};


// allows polymorphism to work correctly in python
// (Define before including decls)
%typemap(out) TestModel *TestModelFactory{
    // TestModelFactory downcast marker
    const std::string lookup_typename = *arg1 + " *";
    swig_type_info * const outtype = SWIG_TypeQuery(lookup_typename.c_str());
    $result = SWIG_NewPointerObj(SWIG_as_voidptr($1), outtype, $owner);
}


/**
 * Whilst developing the TestModelFactory, a number of functions
 * were defined for testing swig. Since the swig learning curve
 * can be quite steep, these have been left in
 * as examples to future developers (and future me).
 * Some of these methods allocate new memory on the heap.
 * In order to tell swig that these are new pointers (and thus
 * in need of deleting) we can use the %newobject directive.
 * See section 14.2 of the swig docs (version 4.0.2)
 */
%newobject _testDoublePair;
%newobject _testDoubleMap;
%newobject _testVariantMap;
%newobject _testVariant;
%newobject test_doubleArrayToNumpyArray;

%newobject TestModelFactory;

//#include "TestModelFactory.h"
%include "TestModelFactory.h"


