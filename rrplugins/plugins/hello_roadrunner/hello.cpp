#pragma hdrstop
#include "hello.h"
#include "telLogger.h"
#include <iostream>

namespace hello
{
    rrc::THostInterface* mhostInterface;

    Hello::Hello():CPPPlugin("Hello","General Plugin")//Construct Base
    {}

    bool Hello::execute(bool inThread)
    {
        if (mhostInterface)mVersion= mhostInterface->getVersionStr();
        else return false;
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
        mhostInterface = _hostInterface;
    }
    #endif
}



