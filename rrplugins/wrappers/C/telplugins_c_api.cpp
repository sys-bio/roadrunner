#pragma hdrstop
#include <sstream>
#include "rr-libstruct/lsMatrix.h"
#include "rrplugins/common/telLogger.h"
#include "rrplugins/common/telTelluriumData.h"
#include "rrplugins/common/telUtils.h"
#include "rrplugins/core/telPluginManager.h"
#include "rrplugins/pluginBaseClass/telPlugin.h"
#include "rrplugins/common/telProperty.h"
#include "rrplugins/common/telStringUtils.h"
#include "tel_macros.h"
#include "telplugins_c_api.h"
#include "telplugins_cpp_support.h"
#include "rrplugins/common/telOSSpecifics.h"
#include "rrplugins/common/telException.h"
#include "rrplugins/core/telVersionInfo.h"

using namespace std;
using namespace tlp;
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
    tel_catch_ptr_macro
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
    tel_catch_bool_macro
}

TELHandle tlp_cc tpGetFirstPlugin(TELHandle handle)
{
    start_try
        PluginManager *pm = castHandle<PluginManager>(handle, __FUNC__);
        return pm->getFirstPlugin();
    tel_catch_ptr_macro
}

TELHandle tlp_cc tpGetNextPlugin(TELHandle handle)
{
    start_try
        PluginManager *pm = castHandle<PluginManager>(handle, __FUNC__);
        return pm->getNextPlugin();
    tel_catch_ptr_macro
}

TELHandle tlp_cc tpGetPreviousPlugin(TELHandle handle)
{
    start_try
        PluginManager *pm = castHandle<PluginManager>(handle, __FUNC__);
        return pm->getPreviousPlugin();
    tel_catch_ptr_macro
}

TELHandle tlp_cc tpGetCurrentPlugin(TELHandle handle)
{
    start_try
        PluginManager *pm = castHandle<PluginManager>(handle, __FUNC__);
        return pm->getCurrentPlugin();
    tel_catch_ptr_macro
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
    tel_catch_ptr_macro
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
    tel_catch_bool_macro
}

char* tlp_cc tpGetPluginLoadErrors(TELHandle handle)
{

    start_try
        PluginManager *pm = castHandle<PluginManager>(handle, __FUNC__);
        return createText(pm->getLoadErrors());
    tel_catch_ptr_macro
}

bool tlp_cc tpUnLoadPlugins(TELHandle handle)
{
    start_try
        PluginManager *pm = castHandle<PluginManager>(handle, __FUNC__);
        return pm->unload();
    tel_catch_bool_macro
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
    tel_catch_bool_macro
}

size_t tlp_cc tpGetNumberOfPlugins(TELHandle handle)
{
    start_try
        PluginManager *pm = castHandle<PluginManager>(handle, __FUNC__);
        return pm->getNumberOfPlugins();
    tel_catch_int_macro
}

TELHandle tlp_cc tpGetPlugin(TELHandle handle, const char* pluginName)
{
    start_try
        PluginManager *pm = castHandle<PluginManager>(handle, __FUNC__);
        Plugin* aPlugin = pm->getPlugin(pluginName);
        return aPlugin;
    tel_catch_ptr_macro
}

/*TELHandle tlp_cc tpGetRoadRunnerHandleFromPlugin(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return aPlugin->getRoadRunnerInstance();
    tel_catch_ptr_macro
}*/

char* tlp_cc tpGetPluginNames(TELHandle handle)
{
    start_try
        PluginManager *pm = castHandle<PluginManager>(handle, __FUNC__);
        StringList names = pm->getPluginNames();
        return createText(names.asString());
    tel_catch_ptr_macro
}

char* tlp_cc tpGetPluginLibraryNames(TELHandle handle)
{
    start_try
        PluginManager *pm = castHandle<PluginManager>(handle, __FUNC__);
        StringList names = pm->getPluginLibraryNames();
        return createText(names.asString());
    tel_catch_ptr_macro
}

char* tlp_cc tpGetPluginName(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return createText(aPlugin->getName());
    tel_catch_ptr_macro
}

char* tlp_cc tpGetPluginCategory(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return createText(aPlugin->getCategory());
    tel_catch_ptr_macro
}

char* tlp_cc tpGetPluginDescription(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return createText(aPlugin->getDescription());
    tel_catch_ptr_macro
}

TLP_C_DS char* tlp_cc tpGetPluginAuthor(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return createText(aPlugin->getAuthor());
    tel_catch_ptr_macro
}

TLP_C_DS char* tlp_cc tpGetPluginCopyright(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return createText(aPlugin->getCopyright());
    tel_catch_ptr_macro
}

TLP_C_DS char* tlp_cc tpGetPluginVersion(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return createText(aPlugin->getVersion());
    tel_catch_ptr_macro
}

char* tlp_cc tpGetPluginHint(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
    return createText(aPlugin->getHint());
    tel_catch_ptr_macro
}

TELHandle tlp_cc tpGetPluginProperties(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return aPlugin->getProperties();
    tel_catch_ptr_macro
}

char* tlp_cc tpGetListOfPluginPropertyNames(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return createText(aPlugin->getPropertyNames().asString());
    tel_catch_ptr_macro
}

TELHandle tlp_cc tpGetPluginProperty(TELHandle handle, const char* parameterName)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return aPlugin->getProperty(parameterName);
    tel_catch_ptr_macro
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
    tel_catch_ptr_macro
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
    tel_catch_ptr_macro
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
            throw(tlp::Exception(msg.str()));
        }
        return tpSetPropertyByString(aProperty, value);
    tel_catch_bool_macro
}

char* tlp_cc tpGetPluginInfo(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return createText(aPlugin->getInfo());
    tel_catch_ptr_macro
}

size_t tlp_cc tpGetPluginManualNrOfBytes(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return aPlugin->getPDFManualByteSize();
    tel_catch_uint_macro
}

unsigned char* tlp_cc tpGetPluginManualAsPDF(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return aPlugin->getManualAsPDF();
    tel_catch_ptr_macro
}

char* tlp_cc tpGetPluginStatus(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return createText(aPlugin->getStatus());
    tel_catch_ptr_macro
}

/*bool tlp_cc tpAssignRoadRunnerInstance(TELHandle handle, TELHandle rrHandle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return (aPlugin) ? aPlugin->assignRoadRunnerInstance(static_cast<RoadRunner*>(rrHandle)) : false;
    tel_catch_bool_macro
}*/

bool tlp_cc tpExecutePlugin(TELHandle handle)
{
    return tpExecutePluginEx(handle, false);
}

bool tlp_cc tpExecutePluginEx(TELHandle handle, bool inAThread)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return (aPlugin) ? aPlugin->execute(inAThread) : false;
    tel_catch_bool_macro
}

bool tlp_cc tpAssignOnStartedEvent(TELHandle handle, tlpc::PluginEvent theCB, void* userData1, void* userData2)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return (aPlugin) ? aPlugin->assignOnStartedEvent(theCB, userData1, userData2) : false;
    tel_catch_bool_macro
}

bool tlp_cc tpAssignOnProgressEvent(TELHandle handle, tlpc::PluginEvent theCB, void* userData1, void* userData2)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return (aPlugin) ? aPlugin->assignOnProgressEvent(theCB, userData1, userData2) : false;
    tel_catch_bool_macro
}

bool tlp_cc tpAssignOnFinishedEvent(TELHandle handle, tlpc::PluginEvent theCB, void* userData1, void* userData2)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return (aPlugin) ? aPlugin->assignOnFinishedEvent(theCB, userData1, userData2) : false;
    tel_catch_bool_macro
}

char* tlp_cc tpGetPluginResult(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return createText(aPlugin->getResult());
    tel_catch_ptr_macro
}

bool tlp_cc tpResetPlugin(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return aPlugin->resetPlugin();
    tel_catch_bool_macro
}

bool tlp_cc tpIsBeingTerminated(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return aPlugin->isBeingTerminated();
    tel_catch_bool_macro
}

void tlp_cc tpTerminateWork(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        aPlugin->terminate();
    tel_catch_void_macro
}

bool tlp_cc tpWasTerminated(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return aPlugin->wasTerminated();
    tel_catch_bool_macro
}

bool tlp_cc tpIsPluginWorking(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return aPlugin->isWorking();
    tel_catch_bool_macro
}

char* tlp_cc tpGetLastError()
{
    return gLastError;
}

bool tlp_cc tpFreeText(char* text)
{
    return tlp::freeText(text);
}

char* tlp_cc tpGetPluginPropertiesAsXML(TELHandle handle)
{
    start_try
        Plugin* aPlugin = castHandle<Plugin>(handle, __FUNC__);
        return createText(aPlugin->getPluginPropertiesAsXML().c_str());
    tel_catch_ptr_macro
}

TELHandle tlp_cc tpRegisterHandle(TELHandle handle, char* name)
{
    start_try
        return gHM.registerHandle(handle, name);
    tel_catch_ptr_macro
}

bool tlp_cc tpUnRegisterHandle(TELHandle handle)
{
    start_try
        return gHM.unRegisterHandle(handle);
    tel_catch_bool_macro
}

TLP_C_DS char* tlp_cc tpGetVersion()
{
    start_try
        return createText(tlp::getVersion());
    tel_catch_ptr_macro
}

TLP_C_DS char* tlp_cc tpGetCopyright()
{
    start_try
        return createText(tlp::getCopyright());
    tel_catch_ptr_macro
}
