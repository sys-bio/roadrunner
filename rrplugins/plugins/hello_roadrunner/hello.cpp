#pragma hdrstop
#include "hello.h"
#include "telLogger.h"
#include <iostream>

namespace hello
{
    rrc::THostInterface* gHostInterface = NULL;

    Hello::Hello():CPPPlugin("Hello","General Plugin")//Construct Base
    {}

    bool Hello::execute(bool inThread)
    {
        if (!gHostInterface)
        {
            return false;
        }
        mVersion = gHostInterface->getVersionStr();
        std::cout<< "Executing the Hello plugin by Debashish Roy: " << mVersion << std::endl;
        return true;
    }

    #ifdef EXPORT_HELLO
    // Plugin factory function
    Plugin* plugins_cc createPlugin()
    {
        //allocate a new object and return it
        return new Hello;
    }
    const char* plugins_cc getImplementationLanguage()
    {
        return "CPP";
    }
    void plugins_cc setHostInterface(rrc::THostInterface* _hostInterface) {
        gHostInterface = _hostInterface;
    }
    #endif
}



