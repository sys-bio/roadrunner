#pragma hdrstop
#include "property_meta_data.h"
#include "rrRoadRunner.h"
#include <iostream>
//---------------------------------------------------------------------------

using namespace tlp;
using namespace std;

TestPlugin::TestPlugin(int testParameter)
:
CPPPlugin("PropertyMetaDataDemo",       "Demo"),
mTestProperty(12345, "NrOfIterations", "Hint for Nr of iterations", "Description for the property")
{
    mProperties.add( &mTestProperty );
}

TestPlugin::~TestPlugin()
{}

bool TestPlugin::execute(bool inThread)
{
    cout<<"Executing plugin...\n";
    if(mRR)
    {
        StringList aList("S2, S1",", ");
        mRR->setSelections(aList);
    }
    return true;
}

// Plugin raw C functions
TestPlugin* plugins_cc createPlugin()
{
    //allocate a new object and return it
    return new TestPlugin();
}

const char* plugins_cc getImplementationLanguage()
{
    return "CPP";
}



