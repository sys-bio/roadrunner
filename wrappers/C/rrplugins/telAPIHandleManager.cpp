#pragma hdrstop

#include "telAPIHandleManager.h"

#include "rrLogger.h"
#include "rrException.h"
#include "rrplugins/common/telProperties.h"
#include "rrplugins/common/telProperty.h"

#include <sstream>
#include <exception>
//---------------------------------------------------------------------------

using namespace std;
using namespace rr;
APIHandleManager::APIHandleManager()
{}

APIHandleManager::~APIHandleManager()
{
    //Report existence of any handles in here.. that could be a memory leak if so
}

TELHandle APIHandleManager::validate(TELHandle handle, const char* type, const char* fnc)
{
    HandleMap::iterator it = mHandles.find(handle);
    if(it !=  mHandles.end()) //There is something like this registered
    {
        if(string(it->second) == string(type))
        {
            return handle;
        }
        else
        {
            //Todo later: if an object of type B, derived from A is registered in the handles container, a passed handle of
            //type A should be validated as OK.

            if(it !=  mHandles.end()) //Found a registered handle with proper address, but types differ.
            {
                string allowed("Property");
                if(strstr(it->second, allowed.c_str()) != NULL)
                {
                    //For now don't check ParameterBase types. See todo above
                    // msg<<"Received handle of type: "<<it->second<<" but expected type: "<<type;
                    // Log(lError)<<msg.str();
                    // throw(BadHandleException(msg.str())); // so much for type safety
                    return handle;
                }
                else
                {
                    stringstream msg;
                    msg<<"Questionable Handle passed to API function: " << fnc << "; ";
                    msg<<"Received handle of type: "<<it->second<<" but expected type: "<<type;
                    Log(lError)<<msg.str();
                    throw(BadHandleException(msg.str()));
                }
            }
        }
    }
    else
    {
        //Before throwing, check property containers for handles that are dynamically created. If found, return
        TELHandle oneMoreTry =   searchFor(handle);
        if(oneMoreTry)
        {
            return handle;
        }
        stringstream msg;
        msg<<"Invalid Handle passed to API function: "<<fnc<<endl;
        msg<<"No such handle is registered. "<<endl;
        Log(lError)<<msg.str();
        throw(BadHandleException(msg.str()));
    }
    return NULL;
}

TELHandle APIHandleManager::registerHandle(TELHandle handle, const char* type)
{
    //Could check for duplicates and signal if that happens
    if(handle)
    {
        mHandles[handle] = type;
    }

    return handle;
}

bool APIHandleManager::unRegisterHandle(TELHandle handle)
{
    HandleMap::iterator it = mHandles.find(handle);
    mHandles.erase ( it, mHandles.end() );
    return true;
}

TELHandle APIHandleManager::searchFor(TELHandle handle)
{
    HandleMap::iterator it = mHandles.begin();
    while(it != mHandles.end())
    {
        string allowed("Properties");
        if(strstr(it->second, allowed.c_str()) != NULL)
        {
            Properties* propContainer = static_cast<Properties*>(it->first);
            for(int i = 0; i < propContainer->count(); i++)
            {
                PropertyBase* aProperty = (*propContainer)[i];
                if(aProperty == handle)
                {
                    return handle;
                }
            }
        }
            it++;
    }
    return NULL;
}
