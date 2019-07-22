/**
 * @file telPlugin.h
 * @brief RoadRunner plugin implementation
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
#ifndef telPluginH
#define telPluginH
#include <sstream>
#include <string>
#include "rrplugins/core/telCoreExporter.h"
#include "rrplugins/core/telPluginsManagerSettings.h"
#include "rrplugins/common/telProperties.h"

namespace rr
{
class RoadRunner;
}

namespace rr
{
    using rr::RoadRunner;
    using rr::gEmptyString;
    using rr::gNoneString;
    using std::string;
    class PluginManager;

/**
 * Typedef for a Plugin event function
 */
typedef void    (event_cc *PluginEvent)(void* data1, void* data2);

/**
  The Plugin class is a base class for all RoadRunner plugins.
  A Plugin is implemented in a shared library, e.g. a DLL on Windows, and loaded at runtime by a PluginManager. Any RoadRunner Plugin need to inherit from
  the Plugin class or any of its descendants. Here are the main characteristics of a Plugin:


  -# The Plugin class do contain the bulk of how a Plugin is handled by the Plugin manager and any clients of the plugin, while
  the atcual custom functionality is implmemented in a descendant Plugin.

  -# A plugin may belong to a category and is identified by a name. It has properties like Author, Version, Copyright etc.

  -# A Plugin typically expose various \a Properties, available to a client of a plugin trough the \a Properties conatiner.

  -# The internal work a plugin is designed to do may be executed in a separate thread, and various functions to monitor and manage
  the work of a Plugin is exposed, e.g. isWorking(), terminate(), isBeingTerminated() etc.

  -# A Plugin may implement various \a event functions in order to communicate data or progress to a client. The Plugin class exposes three events,
   \a PluginStarted, \a PluginProgress and a \a PluginFinish event. Each event may communicate two opaque data parameters. Its up to the plugin implementor
  to make use of these events and what data to pass.

  -# A plugin have access to other plugins through the PluginManager. Thus, a plugin may depend on other plugins.

  -# A plugin may embed documentation in various forms, e.g. PDF (getPDFManual(), or simple text getExtendedInfo().
 */
//      CORE_DECLSPEC
class CORE_DECLSPEC Plugin
{
    friend class PluginManager;
    public:
        /**
            Create a plugin
        */
                                        Plugin( const string& name = gEmptyString,
                                                const string& cat = gNoneString,
                                                RoadRunner* aRR = NULL,
                                                const string& language = gNoneString,
                                                const PluginManager* pm = NULL);

        /**
            Destroy the plugin.
        */
        virtual                        ~Plugin();

        /**
            Get the name of the plugin.
        */
        string                          getName();

        /**
            Get the name of the compiled shared library that implemements the plugin.
        */
        string                          getLibraryName();

        /**
            Get the name of plugin author(s)
        */
        string                          getAuthor();

        /**
            Get plugin category
        */
        string                          getCategory();

        /**
            Get plugin description
        */
        string                          getDescription();

        /**
            Get plugin Hint
        */
        string                          getHint();

        /**
            Get plugin version information
        */
        string                          getVersion();

        /**
            Get plugin copyright text
        */
        string                          getCopyright();

        /**
            Retrieves the RoadRunner instance the plugin is using. Can be NULL.
        */
        RoadRunner*                     getRoadRunnerInstance();

        /**
            Retrieves various plugin information
        */
        virtual string                  getInfo();

        /**
            Retrieve more information than getInfo()
        */
        virtual string                  getExtendedInfo();

        /**
            Retrieve a plugins documentation embedded as PDF. May be NULL
        */
        virtual unsigned char*          getManualAsPDF() const;

        /**
            Retrieve the size in bytes of the PDF string.
        */
        virtual unsigned int            getPDFManualByteSize();

        /**
            Retieves the propertys
        */
        virtual Properties*             getProperties(); //Each capability has a set of propertys

        /**
            Retieves the names of the propertys
        */
        virtual rr::StringList          getPropertyNames();

        /**
            Retieves a plugin properties, as XML
        */
        string                          getPluginPropertiesAsXML();

        /**
            Retieves a specific property
        */
        virtual PropertyBase*           getProperty(const string& param);

        /**
            Return a property value as a string
        */
        string                          getPropertyValueAsString(const string& param);

        /**
            Return a handle to a properties value
        */
        void*                           getPropertyValueHandle(const string& param);

        /**
            Sets the value of specified Property with value as specified
        */
        void                            setPropertyByString(const string& nameOf, const char* value);

        /**
            Sets the value of specified Property with value as specified
        */
        void                            setPropertyValue(const string& nameOf, const void* value);
        /**
            If the work of the plugin is carried out in a separate thread, terminate() will
            signal termination of such work.
        */
        void                            terminate();

        /**
            Check if the plugin worker is in the process of being terminated
        */
        bool                            isBeingTerminated() const;

                                        //!check if the plugin was terminated
        bool                            wasTerminated() const;

                                        //!check if the plugin is working
        virtual bool                    isWorking() const;

                                        //!Assign a roadrunner instance for the plugin to use
        virtual bool                    assignRoadRunnerInstance(RoadRunner* rr);

        /**
            Assign function pointer and data the event
        */
        virtual bool                    assignOnStartedEvent(PluginEvent pluginStarted, void* userData1 = NULL, void* userData2 = NULL);

        /**
            Assign function pointer and data the event
        */
        virtual bool                    assignOnProgressEvent(PluginEvent pluginsProgress, void* userData1 = NULL, void* userData2 = NULL);

        /**
            Assign function pointer and data the event
        */
        virtual bool                    assignOnFinishedEvent(PluginEvent pluginsFinished, void* userData1 = NULL, void* userData2 = NULL);

        /**
            Function returnin a plugins "result", as a string
        */
        virtual string                  getResult();

        /**
            Reset the plugin.
        */
        virtual bool                    resetPlugin();

        /**
            Retrieve the status of the plugin.
        */
        virtual string                  getStatus();

        /**
            Retrieve the implementation language of the plugin.
        */
        virtual string                  getImplementationLanguage() = 0;


        /**
            Check if the plugin as an assigned WorkerStartedEvent
        */
        bool                            hasStartedEvent() const;

        /**
            Check if the plugin as an assigned WorkerProgressEvent
        */
        bool                            hasProgressEvent() const;

        /**
            Check if the plugin as an assigned WorkerFinishedEvent
        */
        bool                            hasFinishedEvent() const;

        /**
         * Call WorkStarted event function
         */
        void                            WorkStartedEvent(void* data1, void* data2);

        /**
         * Call WorkProgress event function
         */
        void                            WorkProgressEvent(void* data1, void* data2);

        /**
         * Call WorkFinished event function
         */
        void                            WorkFinishedEvent(void* data1, void* data2);

        /**
         * get WorkStarted event data variables
         */
        pair<void*, void*>              getWorkStartedData();

        /**
         * get WorkProgress event data variables
         */
        pair<void*, void*>              getWorkProgressData();

        /**
         * get WorkFinished event data variables
         */
        pair<void*, void*>              getWorkFinishedData();

        /**
            Execute the plugin
        */
        virtual bool                    execute(bool inAThread = false) = 0;

        /**
            Get a pointer to the plugins manager
        */
        const PluginManager*            getPluginManager();


    protected:                          //! Name of Plugin
        string                          mName;

                                        //! Name of compiled shared library implementing the Plugin
        string                          mLibraryName;

                                        //! Name of plugin author(s)
        string                          mAuthor;

                                        //! Plugin category
        string                          mCategory;

                                        //! Plugin description
        string                          mDescription;

                                        //! Plugin Hint
        string                          mHint;

                                        //! Plugin version
        string                          mVersion;

                                        //! Plugin copyright language
        string                          mCopyright;

                                        //! Plugin implementation language
        string                          mImplementationLanguage;

                                        //! Boolean flag indicating if a user wants to terminate the work in a plugin
        bool                            mTerminate;

                                        //! Boolean flag indicating if the plugin is working
        bool                            mIsWorking;

        /**
         * A pointer to a RoadRunner instance which the plugin
         * uses.
         */
        RoadRunner                     *mRR;

        /**
         * A pointer to the PluginManager that loaded the Plugin
         * uses.
         */
        const PluginManager            *mPM;

        /**
         * Set the name of the shared library
         */
        void                            setLibraryName(const string& libName);

        /**
         * WorkStarted event function pointer
         */
        PluginEvent                     mWorkStartedEvent;

        /**
         * Work In progress event function pointer
         */
        PluginEvent                     mWorkProgressEvent;

        /**
         * WorkFinished event function pointer
         */
        PluginEvent                     mWorkFinishedEvent;

        /**
         * Opaque data parameter 1 passed in the WorkStarted event function
         */
        void                           *mWorkStartedData1;

        /**
         * Opaque data parameter 2 passed in the WorkStarted event function
         */
        void                           *mWorkStartedData2;

        /**
         * Opaque data parameter 1 passed in the WorkProgress event function
         */
        void                           *mWorkProgressData1;

        /**
         * Opaque data parameter 2 passed in the WorkProgress event function
         */
        void                           *mWorkProgressData2;

        /**
         * Opaque data parameter 1 passed in the WorkFinished event function
         */
        void                           *mWorkFinishedData1;

        /**
         * Opaque data parameter 2 passed in the WorkFinished event function
         */
        void                           *mWorkFinishedData2;

        /**
         * Properties container. Descendant add properties to this container, as they wish.
         * Property data values can be exchanged to/from the plugin by accessing individual properties.
         */
         Properties                     mProperties;
};

}

#endif
