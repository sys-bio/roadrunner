#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "rrplugins/core/tel_api.h"
//#include "telplugins_types.h"
#include "c_plugin_demo.h"

char* gLastError = 0;
//void setError(const char* err);

const char* call_conv getImplementationLanguage()
{
    return "C";
}

bool call_conv destroyPlugin(TELHandle plugin)
{
    //Free any data that the plugin generated..
    tpFreeProperty(gDemoProperty);
    tpFreeProperties(gProperties);
    return true;
}

const char* call_conv getName()
{
    return "C Plugin Demo";
}

const char* call_conv getCategory()
{
    return "Demos";
}

char* call_conv getListOfCPluginPropertyNames(TELHandle plugin)
{
    char* names = tpGetNamesFromPropertyList(gProperties);
    return names;
}

void*  call_conv getCPluginProperty(const char* name)
{
    if(gPlugin)
    {
        return tpGetProperty(gProperties, name);
    }
    else
    {
        return 0;
    }
}

bool call_conv setupCPlugin(TELHandle aPlugin)
{
    gPlugin = aPlugin;
    gDemoProperty   = tpCreateProperty("DemoProperty", "string", "Demo Hint", 0);
    if(!gDemoProperty)
    {
        //Failed to create property! Set last error and return false
        //setError("Failed creating Property in c_plugin_demo");
        return false;
    }

    if(!tpSetPropertyByString(gDemoProperty, "Intial Demo Property Value"))
    {
        //setError("Failed setting Property in c_plugin_demo");
        return false;
    }

    //Add the property to the property container
    gProperties = tpCreatePropertyList();
    if(!tpAddPropertyToList(gProperties, gDemoProperty))
    {
        //setError("Failed adding property to Property list in c_plugin_demo");
        return false;
    }

    return true;
}

bool call_conv execute(bool inThread)
{
    char *text1, *text2;
    TELHandle rrHandle;
    text1       = (char*) malloc(4096 * sizeof(char));
    rrHandle    = createRRInstance();

    strcpy(text1, "CPP API Version: ");
    text2 = getCPPAPIVersion(rrHandle);
    if(text2)
    {
        strcat(text1, text2);
        tpSetPropertyByString(gDemoProperty, text1);
    }
    else
    {
        return false;
    }

    //cleanup
    freeRRInstance(rrHandle);
    tpFreeText(text2);
    free(text1);
    return true;
}

char* getCLastError()
{
    return gLastError;
}

//When this function was called, it was failing for some reason.  So, I ditched it!
// It was only there for demo purposes anyway...

//void setError(const char* err)
//{
//    if(gLastError)
//    {
//        tpFreeText(gLastError);
//    }
//
//    char* temp = tpCreateText(err);
//    gLastError = temp;
//}
//
