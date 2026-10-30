%module SettingTestsSwigAPI

%begin %{
#if defined(_WIN32) && !defined(NOMINMAX)
#define NOMINMAX
#endif
%}

%{
#include <unordered_map>
#include "Setting.h"
#include <variant> // for visit
#include "PyUtils.h"
#include "SettingTestsSwigAPI.h"
%};
#include "PyUtils.h"

%include "exception.i"

/**
 * See section 11.4.3 swig docs 4.0
 * When using the STL it is advisable to
 * add in an exception handler to catch
 * all STL exceptions. The %exception
 * directive can be used by placing the
 * following code before any other methods or
 * libraries to be wrapped:
 */
%exception {
  try {
    $action
  } catch (const std::exception& e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  }
}

%include "std_string.i"


// converts a C++ rr::Setting to a Python variable, depending on its type
%typemap(out) rr::Setting{
    try {
        // I'm a marker rr::Setting(out). Look for me in the swig_wrap.cxx file
        // to verify that this type map is being properly applied
        $result = rr::Variant_to_py($1);
    } catch (const std::exception& e){
        SWIG_exception(SWIG_RuntimeError, e.what());
    }
}
%apply rr::Setting{
    const rr::Setting&,
    rr::Setting&
};

// converts a Python Variable to a C++ rr::Setting
%typemap(in) const rr::Setting&(PyObject* settingObjFromPython){
    try {
        // I'm a marker rr::Setting(in). Look for me in the swig_wrap.cxx file
        // to verify that this type map is being properly applied
        $1 = rr::Variant_from_py($input);

    } catch (const std::exception& e){
        SWIG_exception(SWIG_RuntimeError, e.what());
    }
}
%apply const rr::Setting&{
    const rr::Setting&,
    rr::Setting&,
    rr::Setting
};



// This is the C++ code that we want to wrap.
// Its purpose is to isolate the Setting functionality
// for testing
%include "SettingTestsSwigAPI.h"

















