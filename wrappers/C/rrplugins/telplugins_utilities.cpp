#pragma hdrstop
#include "telplugins_utilities.h"
#include "telplugins_cpp_support.h"
//---------------------------------------------------------------------------

char* gLastError = 0;


char*  tpCreateText(const char* str)
{
    char* textHandle = tpCreateText(string(str));
    return (char*) gHM.registerHandle(textHandle, typeid(textHandle).name());
}

