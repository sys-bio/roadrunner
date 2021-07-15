/**
 * @file main.cpp
 * @brief Example demonstrating usage of a plugin written in C
 * @author Totte Karlsson
 *
 * <hr>
 * This file is part of cRoadRunner.
 * See http://code.google.com/p/roadrunnerlib for more details.
 *
 * Copyright (C) 2012-2013
 *   University of Washington, Seattle, WA, USA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * In plain english this means:
 *
 * You CAN freely download and use this software, in whole or in part, for personal,
 * company internal, or commercial purposes;
 *
 * You CAN use the software in packages or distributions that you create.
 *
 * You SHOULD include a copy of the license in any redistribution you may make;
 *
 * You are NOT required include the source of software, or of any modifications you may
 * have made to it, in any redistribution you may assemble that includes it.
 *
 * YOU CANNOT:
 *
 * redistribute any piece of this software without proper attribution;
*/
#include <iostream>
#include "telPluginManager.h"
#include "telPlugin.h"
#include "telStringList.h"
#include "telplugins_c_api.h"
#include "telplugins_types.h"
using namespace tlp;
using namespace tlpc;
using namespace std;

int main()
{
    try
    {
        PluginManager pm("..\\plugins");
        pm.load();

        int nrOfPlugins = pm.getNumberOfPlugins();
        if(!nrOfPlugins)
        {
            throw("No plugins were loaded");
        }

        //Retrieve a handle to the C plugin demo
        Plugin* plugin = pm.getPlugin("C Plugin Demo");
        if(!plugin)
        {
            throw("Demo plugin could not be loaded");
        }

        string pluginInfo = plugin->getExtendedInfo();
        cout<<pluginInfo;

        //Retrieve any properties that the plugin has
        StringList properties = plugin->getPropertyNames();
        if(!properties.count())
        {
            throw("Plugin do not have any properties. For this demo, this is an error!");
        }

        for(int i = 0; i < properties.count(); i++)
        {
            cout << properties[i] << endl;
        }

        //We know the name of the Demo property, being "Demo Property"
        PropertyBase* prop = plugin->getProperty("DemoProperty");
        if(prop)
        {
            cout << "Before execute: "<<prop->getValueAsString()<<endl;
        }

        if(!plugin->execute())
        {
            throw("There was a problem executing the plugin...");
        }

        if(prop)
        {
            cout << "After execute: "<<prop->getValueAsString()<<endl;
        }

        //This will unload all plugins..
        pm.unload();
    }
    catch(const char* msg)
    {
        clog<<"There was a problem: "<<msg;
    }
    return 0;
}



