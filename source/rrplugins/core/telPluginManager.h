/**
 * @file telPluginManager.h
 * @brief The plugin manager implementation
 * @author Totte Karlsson & Herbert M Sauro
 *
 * <--------------------------------------------------------------
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
#ifndef telPluginManagerH
#define telPluginManagerH
#include <vector>
#include <string>
#include <sstream>
#include <ostream>
#include "rrplugins/core/telCoreExporter.h"
#include "../../rrConstants.h"
#include "../../c/rrStringList.h"
#include "Poco/SharedLibrary.h"
#include "../../rrException.h"
#include "../../rrLogger.h"
#include "telPlugin.h"
#include "telCPlugin.h"
#include "../../rrUtils.h"

//---------------------------------------------------------------------------
/* A minimalistic Plugin manager. */

namespace rr
{

using std::stringstream;
using std::string;
using std::vector;
using std::pair;
using std::ostream;
using rr::gEmptyString;

//using rr::StringList;
using Poco::SharedLibrary;

class Plugin;

/**
Typedef for a plugins record in the plugins container
*/
typedef pair< Poco::SharedLibrary*, Plugin* > telPlugin;

/**
 * The PluginManager class is responsible for loading and unloading plugins.
 * Each plugin is validated and categorized during the load process. A plugin that don't pass validation
 * will not be loaded.
 *
 * The plugin manager provide various functions to assist a client in
 * retrieving plugins from the manager, e.g.  getFirstPlugin, getNextPlugin, getPreviousPlugin or getPlugin(name).
 *
 * Plugins are by default placed in a Plugins folder, found in RoadRunners install folder.
 */
class CORE_DECLSPEC PluginManager
{
    public:
        /**
         * The constructor of a Plugin manager creates a Plugin manager instance.
         *
         * @param pluginFolder Folder where the plugin manager is looking for plugins. If argument
            is not supplied, the manager will use the default folder, which is "plugins" located in RoadRunners install folder.
         */
                                            PluginManager(const string& pluginFolder = gEmptyString);
        /**
         * The destructor of a Plugin manager will free any memory allocated and also unload any plugins that it loaded.
         */
        virtual                            ~PluginManager();

        /**
            Get information about the current plugin manager object,
            such as number of plugins loaded, their names and where they are loaded from
            @return string String holding plugin manager information
        */
        string                              getInfo() const;

        /**
            Change the directory where the Manager loads plugins
            @param dir Folder to load plugins from
            @return Boolean indicating if folder is valid or not
        */
        bool                                setPluginDir(const string& dir);

        /**
            Get folder where plugins are loaded from
            @return String containg current folder where plugins are loaded from
        */
        string                              getPluginDir() const;

        /**
            Load a specific plugin, or, if no argument supplied, loads ALL plugins in the current plugin folder.
            @return integer Returning number of succesfully loaded plugins
        */
        int                                 load(const string& pluginName = gEmptyString);

        /**
            Unload a specific plugin, or, if no argument supplied, unloads ALL plugins.
            @return Boolean indicating success
        */
        bool                                unload(Plugin* plugin = NULL);

        /**
            Get number of loaded plugins.
        */
        int                                 getNumberOfPlugins() const;

        /**
            Get number of plugin categories.
            \note Not implemented
        */
        int                                 getNumberOfCategories() const;

        /**
            Retrieves the "first" plugin in the managers internal plugin container.
            \return Pointer to a Plugin, or NULL.
            \note This function resets the managers internal plugin iterator to point
            at the "first" plugin.
        */
        Plugin*                             getFirstPlugin() const;

        /**
            Retrieves the "next" plugin in the managers internal plugin container.
            \return Pointer to a Plugin, or NULL.
            \note This function advances the managers internal plugin iterator one step forward.
        */
        Plugin*                             getNextPlugin() const;

        /**
            Retrieves the "previous" plugin in the managers internal plugin container.
            \return Pointer to a Plugin, or NULL.
            \note This function decerements the managers internal plugin iterator one step back.
        */
        Plugin*                             getPreviousPlugin() const;

        /**
            Retrieves the "current" plugin in the managers internal plugin container.
            \return Pointer to a Plugin, or NULL.
        */
        Plugin*                             getCurrentPlugin() const;

        /**
            Retrieves a plugin with name as supplied in the argument.
            \arg name Name of the plugin.
            \return Pointer to a Plugin, or NULL.
        */
        Plugin*                             getPlugin(const string& name) const;

        /**
            Retrieves the names of all loaded plugins as a list of strings.
            \return StringList A Stringlist containing the name of each loaded Plugin.
        */
        rr::StringList                     getPluginNames() const;

        /**
            Retrieves the shared library names of all loaded plugins as a list of strings.
            \return StringList A Stringlist containing the file name of each loaded Plugin.
        */
        rr::StringList                     getPluginLibraryNames() const;

        /**
            Output plugin information to a std ostream
        */
        CORE_DECLSPEC
        friend ostream&                     operator<<(ostream& os, PluginManager& pm);

        /**
            Access a plugin using the [] operator.
        */
        Plugin*                             operator[](const int& i);

        bool                                hasLoadErrors() const;
        string                              getLoadErrors() const;

    private:
        string                              mPluginFolder;
        string                              mPluginExtension;       //Different on different OS's
        string                              mPluginPrefix;          //Different on different OS's
        stringstream                        mLoadPluginErrors;      //Accumulated load errors
        mutable vector< telPlugin >                 mPlugins;
        mutable vector< telPlugin >::iterator       mPluginsIter;

        bool                                loadPlugin(const string& sharedLib);
        bool                                checkImplementationLanguage(SharedLibrary* plugin);
        const char*                         getImplementationLanguage(SharedLibrary* plugin);
        Plugin*                             createCPlugin(SharedLibrary *libHandle);
        Plugin*                             getPlugin(const int& i);
        bool                                unloadAll();
        void                                clearLoadErrors();
};

}
#endif

/*! \mainpage A Plugin Framework for libRoadRunner

 \par
 This document describes libRoadRunners Plugins API. 
 \par
 The Plugins API is centered around the \a PluginManager, \a Plugin, and the \a Property classes. The Plugin manager is 
 responsible for validating, loading and unloading plugins at runtime.

 \par
 The Plugins themselves are self contained shared libraries that may be designed to extend the functionality of the main libRoadRunners core API.
 
 \par
 The capabilities of a Plugin is communicated to a client by the use of \a PluginProperties. A Plugin property is an object 
 able to communicate various types of data betwen the Plugin and the client, such as an int, double or a string. In addition to its value, plugin 
 properties have a name, hint and description field, assisting in communicating the purpose of the property.

 \par
 A client typically loads and configure a plugin, through the use of Properties, and then subsequently \a executes the plugin using the plugins
 execute function. Finally, the client may retrieve some kind of result from the Plugin, again, using a Property.

 \par
 Since a Plugins function will be determined by the actual Plugin author, documentation of a Plugins purpose and on how
 to use the plugin, is ideally bundled with the plugin and retrieved by various Plugin functions, e.g. getPluginInfo() and getPluginManualAsPDF().
 \par

 \author Totte Karlsson (totte@dunescientific.com)
 \author Herbert M. Sauro  (hsauro@u.washington.edu)

 \par
 Redistribution and use in source and binary forms, with or without modification, 
 are permitted provided that the following conditions are met:

 \li Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

 \li Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

 \li Neither the name of University of Washington nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

 \par

 */
