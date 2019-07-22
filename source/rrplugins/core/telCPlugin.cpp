#pragma hdrstop
#include "../../rrUtils.h"
#include "telCPlugin.h"
//---------------------------------------------------------------------------

namespace rr
{

CPlugin::CPlugin(const string& name, const string& cat, RoadRunner* aRR)
:
Plugin(name, cat, aRR, "C"),
executeFunction(NULL)
{}

CPlugin::~CPlugin()
{
    //Designed to de-allocate any data in the C plugin
    if(destroyFunction)
    {
        destroyFunction();
    }
}

string CPlugin::getImplementationLanguage()
{
    return "C";
}

bool CPlugin::execute(bool useThread)
{
    if(executeFunction)
    {
        return executeFunction(useThread);
    }
    return false;
}

rr::StringList CPlugin::getPropertyNames()
{
    char* propNames = getCPropertyNames();
    string _names(propNames);
    rr::StringList names(_names, ",");
    freeText(propNames);
    return names;
}

PropertyBase* CPlugin::getProperty(const string& prop)
{
    PropertyBase* baseProp = (PropertyBase*) getCProperty(prop.c_str());
    return baseProp;
}

string CPlugin::getLastError()
{

    char* msg = getCLastError();
    if(msg)
    {
        string msgStr(msg);
        //free text...

        return msgStr;
    }
    return "No error";
}

}


