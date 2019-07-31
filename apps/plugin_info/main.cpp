#include <iostream>
#include "rrException.h"
#include "rrUtils.h"
#include "telPluginManager.h"
#include "telPlugin.h"
#pragma hdrstop

using namespace std;
using namespace rr;
int main()
{
    try
    {
        string fldr(joinPath("..", "plugins"));
        PluginManager pm(fldr);

        cout<<"==== Loading Plugins =====\n";
        int nrOfLoadedPlugins = pm.load();
        if(nrOfLoadedPlugins == 0)
        {
            cout<<"No plugins were loaded";
        }
        else
        {
            cout<<"Loaded "<<nrOfLoadedPlugins<<" plugins\n";
        }
        cout<<"\n==== End of Loading Plugins =====\n\n";

        cout<<"\n\n Plugin Loading Info ============\n\n"<<pm;

        //Get info about each plugin
        cout<<"\n\nIndividual Plugin Info ============\n\n";
        for(int i =0; i < pm.getNumberOfPlugins(); i++)
        {
            Plugin* aPlugin = pm[i];
            if(aPlugin)
            {
                cout<<aPlugin->getInfo();
            }
        }
        pm.unload();
        return 0;

    }
    catch(const Exception& ex)
    {
        cout<<ex.what();
    }
}
