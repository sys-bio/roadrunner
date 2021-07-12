%module TestModelFactory

%{
    #define SWIG_FILE_WITH_INIT
    #define PY_ARRAY_UNIQUE_SYMBOL RoadRunner_ARRAY_API // for the NamedArrayObject (see roadrunner.i)

    #include <unordered_map>
    #include <utility>
    #include <iostream>

    #include "PyUtils.h" // for variant_to_py and back.

    #include "TestModelFactory.h"
    using namespace rr;
%}

#include <iostream>
#include <unordered_map>

// since we be using the ls::Matrix from a Python
// we'll be using the same API needed by the main roadrunner bindings
%include "numpy.i"
%init %{
import_array();
rr::pyutil_init(m);
%}

%include "std_string.i"
%include "std_vector.i"
%include "std_pair.i"
%include "std_unordered_map.i"

%include "PyUtils.h"


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
 * @brief typemap to convert a string : Variant map into a Python dict.
 * Used in the "settings" map of tmf.
 */
%typemap(out) std::unordered_map< std::string,rr::Setting,std::hash< std::string >,std::equal_to< std::string >,std::allocator< std::pair< std::string const,rr::Setting > > >* {
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



// allows polymorphism to work correctly in python
// (Define before including decls)
%typemap(out) TestModel *TestModelFactory{
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

%newobject TestModelFactory;

#include "TestModelFactory.h"
%include "TestModelFactory.h"








