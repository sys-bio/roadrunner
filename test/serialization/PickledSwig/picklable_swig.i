%module picklable_swig

%{
    #define SWIG_FILE_WITH_INIT
    #include "CPPObjectForPickling.h"
    #include <iostream>
%};

%include "attribute.i"
%include "std_string.i"
%include "CPPObjectForPickling.h"

%attribute(CPPObjectForPickling, int, int_, getInt, setInt);
%attribute(CPPObjectForPickling, double, double_, getDouble, setDouble);
%attribute(CPPObjectForPickling, std::string, string_, getString, setString);

