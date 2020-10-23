#pragma hdrstop
#include "telUtils.h"
#include "telCPlugin.h"
//---------------------------------------------------------------------------

namespace tlp
{

CPlugin::CPlugin(const string& name, const string& cat):// rrc::THostInterface* aRR):
Plugin(name, cat,"C"),//NULL,"C"),
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

tlp::StringList CPlugin::getPropertyNames()
{
    char* propNames = getCPropertyNames();
    string _names(propNames);
    tlp::StringList names(_names, ",");
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


