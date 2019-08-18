#pragma hdrstop
#include <sstream>
#include "rr-libstruct/lsMatrix.h"
#include "rrRoadRunner.h"
#include "rrLogger.h"
#include "rrplugins/common/telTelluriumData.h"
#include "rrUtils.h"
#include "rrplugins/core/telPluginManager.h"
#include "rrplugins/core/telPlugin.h"
#include "rrplugins/common/telProperty.h"
#include "rrStringUtils.h"
#include "tel_macros.h"
#include "telplugins_c_api.h"
#include "telplugins_cpp_support.h"
#include "rrOSSpecifics.h"
#include "rrException.h"
#include "rrplugins/core/telVersionInfo.h"

using namespace std;
using rr::RoadRunner;
using namespace rr;
using namespace tlpc;
//using tlp::TelluriumData;
//using tlp::StringList;


TELHandle tlp_cc tpCreatePluginManager(const char* _pluginDir)
{
    start_try
        string pluginDir = (_pluginDir != NULL) ? string(_pluginDir) : string(".");

        PluginManager* pm = new PluginManager(pluginDir);
        gHM.registerHandle(pm, typeid(pm).name());

        return pm;
    catch_ptr_macro
}

/**
 * free the plugin manager
 */
bool tlp_cc tpFreePluginManager(TELHandle handle)
{
    start_try
        PluginManager *pm = castHandle<PluginManager>(handle, __FUNC__);
        delete pm;
        return true;
    catch_bool_macro
}

TELHandle tlp_cc tpGetFirstPlugin(TELHandle handle)
{
    start_try
        PluginManager *pm = castHandle<PluginManager>(handle, __FUNC__);
        return pm->getFirstPlugin();
    catch_ptr_macro
}

TELHandle tlp_cc tpGetNextPlugin(TELHandle handle)
{
    start_try
        PluginManager *pm = castHandle<PluginManager>(handle, __FUNC__);
        return pm->getNextPlugin();
    catch_ptr_macro
}

TELHandle tlp_cc tpGetPreviousPlugin(TELHandle handle)
{
    start_try
        PluginManager *pm = castHandle<PluginManager>(handle, __FUNC__);
        return pm->getPreviousPlugin();
    catch_ptr_macro
}

TELHandle tlp_cc tpGetCurrentPlugin(TELHandle handle)
{
    start_try
        PluginManager *pm = castHandle<PluginManager>(handle, __FUNC__);
        return pm->getCurrentPlugin();
    catch_ptr_macro
}

TELHandle tlp_cc tpLoadPlugin(TELHandle handle, const char* pluginName)
{
    start_try

        PluginManager *pm = castHandle<PluginManager>(handle, __FUNC__);
        if(pm->load(pluginName))
        {
            //Register plugins with Handle manager.
            Plugin* handle = pm->getPlugin(pluginName);
            if (!handle)
                throw std::runtime_error("Could not load plugin " + std::string(pluginName));
            return tpRegisterPlugin(handle);
        }
        else
        {
            stringstream s;
            s << "Failed loading plugin: " + string(pluginName);
            s << "\n" <<pm->getLoadErrors();

            tpSetError(s.str());
            return NULL;
        }
    catch_ptr_macro
}

bool tlp_cc tpLoadPlugins(TELHandle handle)
{
    start_try
        PluginManager *pm = castHandle<PluginManager>(handle, __FUNC__);
        pm->load();

        //Register plugins with Handle manager.
        //We also need to register plugin attributes that may be
        //accessed trough the API
        Plugin* pl = pm->getFirstPlugin();
        while(pl)
        {
            tpRegisterPlugin(pl);
            pl = pm->getNextPlugin();
        };

        return pm->hasLoadErrors() ? false : true;
    catch_bool_macro
}

char* tlp_cc tpGetPluginLoadErrors(TELHandle handle)
{

    start_try
        PluginManager *pm = castHandle<PluginManager>(handle, __FUNC__);
        return createText(pm->getLoadErrors());
    catch_ptr_macro
}

bool tlp_cc tpUnLoadPlugins(TELHandle handle)
{
    start_try
        PluginManager *pm = castHandle<PluginManager>(handle, __FUNC__);
        return pm->unload();
    catch_bool_macro
}

bool tlp_cc tpUnLoadPlugin(TELHandle handle, TELHandle plugin)
{
    start_try
        PluginManager *pm = castHandle<PluginManager>(handle, __FUNC__);
        Plugin* aPlugin = castHandle<Plugin>(plugin, __FUNC__);
        if(pm->unload(aPlugin))
        {
            aPlugin = NULL;
            return true;
        }
        else
        {
            return false;
        }
    catch_bool_macro
}

int tlp_cc tpGetNumberOfPlugins(TELHandle handle)
{
    start_try
        PluginManager *pm = castHandle<PluginManager>(handle, __FUNC__);
        return pm->getNumberOfPlugins();
    catch_int_macro
}

TELHandle tlp_cc tpGetPlugin(TELHandle handle, const char* pluginName)
{
    start_try
        PluginManager *pm = castHandle<PluginManager>(handle, __FUNC__);
        Plugin* aPlugin = pm->getPlugin(pluginName);
        return aPlugin;
    catch_ptr_macro
}

TELHandle tlp_cc tpGetRoadRunnerHandleFromPlugin(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return aPlugin->getRoadRunnerInstance();
    catch_ptr_macro
}

char* tlp_cc tpGetPluginNames(TELHandle handle)
{
    start_try
        PluginManager *pm = castHandle<PluginManager>(handle, __FUNC__);
        StringList names = pm->getPluginNames();
        return createText(names.AsString());
    catch_ptr_macro
}

char* tlp_cc tpGetPluginLibraryNames(TELHandle handle)
{
    start_try
        PluginManager *pm = castHandle<PluginManager>(handle, __FUNC__);
        StringList names = pm->getPluginLibraryNames();
        return createText(names.AsString());
    catch_ptr_macro
}

char* tlp_cc tpGetPluginName(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return createText(aPlugin->getName());
    catch_ptr_macro
}

char* tlp_cc tpGetPluginCategory(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return createText(aPlugin->getCategory());
    catch_ptr_macro
}

char* tlp_cc tpGetPluginDescription(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return createText(aPlugin->getDescription());
    catch_ptr_macro
}

TLP_C_DS char* tlp_cc tpGetPluginAuthor(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return createText(aPlugin->getAuthor());
    catch_ptr_macro
}

TLP_C_DS char* tlp_cc tpGetPluginCopyright(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return createText(aPlugin->getCopyright());
    catch_ptr_macro
}

TLP_C_DS char* tlp_cc tpGetPluginVersion(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return createText(aPlugin->getVersion());
    catch_ptr_macro
}

char* tlp_cc tpGetPluginHint(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
    return createText(aPlugin->getHint());
    catch_ptr_macro
}

TELHandle tlp_cc tpGetPluginProperties(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return aPlugin->getProperties();
    catch_ptr_macro
}

char* tlp_cc tpGetListOfPluginPropertyNames(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return createText(aPlugin->getPropertyNames().AsString());
    catch_ptr_macro
}

TELHandle tlp_cc tpGetPluginProperty(TELHandle handle, const char* parameterName)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return aPlugin->getProperty(parameterName);
    catch_ptr_macro
}

TELHandle tlp_cc tpGetPluginPropertyValueHandle(TELHandle handle, const char* parameterName)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        PropertyBase* prop = aPlugin->getProperty(parameterName);
        if(prop)
        {
            return prop->getValueHandle();
        }
        else
        {
            return NULL;
        }
    catch_ptr_macro
}

char* tlp_cc tpGetPluginPropertyValueAsString(TELHandle handle, const char* parameterName)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        PropertyBase* prop = aPlugin->getProperty(parameterName);
        if(prop)
        {
            return createText(prop->getValueAsString());
        }
        else
        {
            return NULL;
        }
    catch_ptr_macro
}

bool tlp_cc tpSetPluginProperty(TELHandle handle, const char* parameterName, const char* value)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        PropertyBase* aProperty = (PropertyBase*) tpGetPluginProperty(aPlugin, parameterName);
        if(!aProperty)
        {
            stringstream msg;
            msg <<"Failed locating property: "<<parameterName<<" in plugin: "<<aPlugin->getName()<<" and function: "<<__FUNC__;
            throw(rr::Exception(msg.str()));
        }
        return tpSetPropertyByString(aProperty, value);
    catch_bool_macro
}

char* tlp_cc tpGetPluginInfo(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return createText(aPlugin->getInfo());
    catch_ptr_macro
}

unsigned int tlp_cc tpGetPluginManualNrOfBytes(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return aPlugin->getPDFManualByteSize();
    catch_uint_macro
}

unsigned char* tlp_cc tpGetPluginManualAsPDF(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return aPlugin->getManualAsPDF();
    catch_ptr_macro
}

char* tlp_cc tpGetPluginStatus(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return createText(aPlugin->getStatus());
    catch_ptr_macro
}

bool tlp_cc tpAssignRoadRunnerInstance(TELHandle handle, TELHandle rrHandle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return (aPlugin) ? aPlugin->assignRoadRunnerInstance(static_cast<RoadRunner*>(rrHandle)) : false;
    catch_bool_macro
}

bool tlp_cc tpExecutePlugin(TELHandle handle)
{
    return tpExecutePluginEx(handle, false);
}

bool tlp_cc tpExecutePluginEx(TELHandle handle, bool inAThread)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return (aPlugin) ? aPlugin->execute(inAThread) : false;
    catch_bool_macro
}

bool tlp_cc tpAssignOnStartedEvent(TELHandle handle, tlpc::PluginEvent theCB, void* userData1, void* userData2)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return (aPlugin) ? aPlugin->assignOnStartedEvent(theCB, userData1, userData2) : false;
    catch_bool_macro
}

bool tlp_cc tpAssignOnProgressEvent(TELHandle handle, tlpc::PluginEvent theCB, void* userData1, void* userData2)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return (aPlugin) ? aPlugin->assignOnProgressEvent(theCB, userData1, userData2) : false;
    catch_bool_macro
}

bool tlp_cc tpAssignOnFinishedEvent(TELHandle handle, tlpc::PluginEvent theCB, void* userData1, void* userData2)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return (aPlugin) ? aPlugin->assignOnFinishedEvent(theCB, userData1, userData2) : false;
    catch_bool_macro
}

char* tlp_cc tpGetPluginResult(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return createText(aPlugin->getResult());
    catch_ptr_macro
}

bool tlp_cc tpResetPlugin(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return aPlugin->resetPlugin();
    catch_bool_macro
}

bool tlp_cc tpIsBeingTerminated(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return aPlugin->isBeingTerminated();
    catch_bool_macro
}

void tlp_cc tpTerminateWork(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        aPlugin->terminate();
    catch_void_macro
}

bool tlp_cc tpWasTerminated(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return aPlugin->wasTerminated();
    catch_bool_macro
}

bool tlp_cc tpIsPluginWorking(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return aPlugin->isWorking();
    catch_bool_macro
}

char* tlp_cc tpGetLastError()
{
    return gLastError;
}

bool tlp_cc tpFreeText(char* text)
{
    return rr::freeText(text);
}

/*char* tlp_cc tpGetPluginPropertiesAsXML(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return createText(aPlugin->getPluginPropertiesAsXML().c_str());
    catch_ptr_macro
}*/

TELHandle tlp_cc tpRegisterHandle(TELHandle handle, char* name)
{
    start_try
        return gHM.registerHandle(handle, name);
    catch_ptr_macro
}

bool tlp_cc tpUnRegisterHandle(TELHandle handle)
{
    start_try
        return gHM.unRegisterHandle(handle);
    catch_bool_macro
}

/*TLP_C_DS char* tlp_cc tpGetVersion()
{
    start_try
        return createText(rr::getVersion());
    catch_ptr_macro
}

TLP_C_DS char* tlp_cc tpGetCopyright()
{
    start_try
        return createText(rr::getCopyright());
    catch_ptr_macro
}*/
