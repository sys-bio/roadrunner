#pragma hdrstop
#include <sstream>
#include <string>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include "Poco/Glob.h"
#include "Poco/SharedLibrary.h"
//#include "../../rrRoadRunner.h"
#include "telPluginManager.h"

//---------------------------------------------------------------------------
namespace rr
{
static bool  hasFileExtension(const string& fName);
static std::string getPluginExtension();
static std::string getPluginOSPrefix();
static std::string getPluginLibNamePrefix();

#if defined(__BORLANDC__)
    #define exp_fnc_prefix "_"
#else
    #define exp_fnc_prefix ""
#endif

using namespace std;
using namespace rr;
using Poco::SharedLibrary;
using Poco::Glob;

//Convenient function pointers
typedef Plugin*     (*createRRPluginFunc)(PluginManager*);
typedef char*       (*charStarFnc)();
typedef bool        (*setupCPluginFnc)(Plugin*);
typedef bool        (*destroyRRPluginFunc)(Plugin* );

bool destroyRRPlugin(Plugin *plugin);

PluginManager::PluginManager(const std::string& folder)
:
mPluginFolder(folder),
mPluginExtension(getPluginExtension()),
mPluginPrefix(getPluginOSPrefix())
{}

PluginManager::~PluginManager()
{
    //No matter what.. here shared libs need to be unloaded and deleted
    unloadAll();
}

bool PluginManager::setPluginDir(const string& dir)
{
    mPluginFolder = dir;
    return folderExists(mPluginFolder);
}

string PluginManager::getPluginDir() const
{
    return mPluginFolder;
}

string PluginManager::getInfo() const
{
    stringstream info;
    info<<setw(30)<<left<<"Plugin Folder: "            <<mPluginFolder<<"\n";
    info<<setw(30)<<left<<"Plugin Extensions: "        <<mPluginExtension<<"\n";
    info<<setw(30)<<left<<"Plugin Prefix: "            <<mPluginPrefix<<"\n";
    info<<setw(30)<<left<<"Number of loaded plugins:"  <<getNumberOfPlugins()<<"\n";

    if(getNumberOfPlugins())
    {
        info<<setw(25)<<left<<"Plugin Names"<<setw(25)<<"Plugin Library Names"<<"\n";
        Plugin* p = getFirstPlugin();

        do
        {
            if(p)
            {
                info<<"  "<<setw(25)<<left<<p->getName()<<setw(25)<<left<<p->getLibraryName()<<"\n";
            }

        } while((p = getNextPlugin()));
    }
    return info.str();
}

ostream& operator << (ostream& st, PluginManager& pm)
{
    st<<pm.getInfo();
    return st;
}

Plugin* PluginManager::getFirstPlugin() const
{
    mPluginsIter = mPlugins.begin();
    if(mPluginsIter != mPlugins.end())
    {
        return (*mPluginsIter).second;
    }
    return NULL;
}

Plugin* PluginManager::getCurrentPlugin() const
{
    if(mPluginsIter != mPlugins.end())
    {
        return (*mPluginsIter).second;
    }
    return NULL;
}

Plugin* PluginManager::getNextPlugin() const
{
    if(mPluginsIter != mPlugins.end())
    {
        mPluginsIter++;
        if(mPluginsIter != mPlugins.end())
        {
            return (*mPluginsIter).second;
        }
    }
    return NULL;
}

Plugin* PluginManager::getPreviousPlugin() const
{
    if(mPluginsIter != mPlugins.end())
    {
        mPluginsIter--;
        if(mPluginsIter != mPlugins.end())
        {
            return (*mPluginsIter).second;
        }
    }
    return NULL;
}

Plugin* PluginManager::getPlugin(const int& i)
{
    return (*this)[i];
}

Plugin* PluginManager::operator[](const int& i)
{
    if(i >= 0 && i < mPlugins.size())
    {

        return mPlugins[i].second;
    }
    else
    {
        return NULL;
    }
}

int PluginManager::load(const string& pluginName)
{
    stringstream errors;
    clearLoadErrors();
    int nrOfLoadedPlugins = 0;

    Log(lInfo) << "PluginManager::load: Looking in plugin directory " << mPluginFolder << "\n";

    //Throw if plugin folder don't exist
    if(!folderExists(mPluginFolder))
    {
        errors<<"The plugin folder: \""<<mPluginFolder<<"\" does not exist.";
        Log(lError)<<errors.str();
        mLoadPluginErrors << errors.str();
        throw(Exception(errors.str()));
    }

    set<string> files;
    // FIXME: add platform check
    string globPath =  joinPath(mPluginFolder, getPluginOSPrefix() + getPluginLibNamePrefix() + "*." + mPluginExtension);

    if(pluginName.size())
    {
        string temp = joinPath(mPluginFolder, getPluginOSPrefix() + pluginName + "." + mPluginExtension);
        files.insert(temp);
     }
    else
    {
        //Get all plugins in plugin folder
        Glob::glob(globPath, files, Glob::GLOB_CASELESS);
    }

    set<string>::iterator it = files.begin();
    //Try to load.. failing to load a plugin do not mean we abort. Catch and report the problem
    for (; it != files.end(); ++it)
    {
        string plugin = getFileName(*it);
        Log(lInfo)<<"Loading plugin: "<<plugin;
        try
        {
            bool res = loadPlugin(plugin);
            if(!res)
            {
                Log(lError)<<"There was a problem loading plugin: "<<plugin;
                continue;
            }
            nrOfLoadedPlugins++;
        }
        catch(...)
        {
            Log(lError)<<"There was a serious problem loading plugin: "<<plugin;
        }
    }
    return nrOfLoadedPlugins;
}

bool PluginManager::loadPlugin(const string& _libName)
{
    //This is a private function that catches any throws occuring
    //Catch and save the error message below, in the catch
    stringstream info;
    try
    {
        string libName(_libName);

        //Check if Plugin is already loaded first
        if(getPlugin(libName))
        {
            info<<"The Plugin: "<<libName<<" is already loaded";
            Log(lWarning)<<info.str();
            return true; //Don't make this an error..
        }

        if(!hasFileExtension(libName))
        {
            libName = libName + "." + getPluginExtension();
        }

        SharedLibrary *libHandle = new SharedLibrary;
        string fullName = joinPath(mPluginFolder, libName);

        if(!fileExists(fullName))
        {
            info<<"The Plugin: "<<fullName<<" could not be found";
            throw std::runtime_error(info.str());
        }

        //This one throws if there is a problem..
        libHandle->load(fullName);

        //Validate the plugin
        if(!checkImplementationLanguage(libHandle))
        {
            info<<"The plugin: "<<_libName<<" has not implemented the function getImplementationLanguage properly. Plugin can not be loaded";
            throw std::runtime_error(info.str());
        }

        //Check plugin language
        const char* language = getImplementationLanguage(libHandle);

        if(strcmp(language, "C") == 0)
        {
            //Gather enough library data in order to create a CPlugin object
            //We need at least name, category and an execute function in order to setup a C plugin
            Plugin* aPlugin = createCPlugin(libHandle);
            if(!aPlugin)
            {
                info<<"Failed creating C Plugin";
                throw(info.str());
            }

            aPlugin->setLibraryName(getFileNameNoExtension(libName));
            telPlugin storeMe(libHandle, aPlugin);
            mPlugins.push_back( storeMe );
            return true;
        }
        else if(libHandle->hasSymbol(string(exp_fnc_prefix) + "createPlugin"))
        {
            createRRPluginFunc create = (createRRPluginFunc) libHandle->getSymbol(string(exp_fnc_prefix) + "createPlugin");

            //This plugin
            Plugin* aPlugin = create(this);
            if(aPlugin)
            {
                aPlugin->setLibraryName(getFileNameNoExtension(libName));

                telPlugin storeMe(libHandle, aPlugin);
                mPlugins.push_back( storeMe );
            }
            return true;
        }
        else
        {
            throw("The current Plugin lack the createPlugin(void*) function!");
        }
    }

    //We have to catch exceptions here. Failing to load a plugin should not throw, just return false.
    catch(const string& msg)
    {
        info<<"========== In attempt to load plugin: "<<_libName<<" ==========="<<endl;
        info<<"Plugin loading exception: "<<msg;
        mLoadPluginErrors<<info.str();
        Log(lError)<<info.str();
        return false;
    }
    catch(const Exception& e)
    {
        info<<"========== In attempt to load plugin: "<<_libName<<" ==========="<<endl;
        info<<"Exception: "<<e.what()<<endl;
        mLoadPluginErrors<<info.str();
        Log(lError)<<info.str();
        return false;
    }
    catch(const Poco::Exception& ex)
    {
        info<<"========== In attempt to load plugin: "<<_libName<<" ==========="<<endl;
        info<<"POCO Exception: "<<ex.displayText()<<endl;
        mLoadPluginErrors<<info.str();
        Log(lError)<<info.str();
        return false;
    }
    catch(...)
    {
        info<<"========== In attempt to load plugin: "<<_libName<<" ==========="<<endl;
        info<<"Unknown error occured attempting to load plugin"<<_libName;
        mLoadPluginErrors<<info.str();
        Log(lError)<<info.str();
        return false;
    }
}

//Todo: Clean up the unload process..
bool PluginManager::unloadAll()
{
    bool result(true);
    int nrPlugins = getNumberOfPlugins();
    for(int i = 0; i < nrPlugins; i++)
    {
        telPlugin* aPluginLib = &(mPlugins[i]);
        if(aPluginLib)
        {
            SharedLibrary *pluginLibHandle    = aPluginLib->first;
            Plugin        *aPlugin            = aPluginLib->second;

            destroyRRPlugin(aPlugin);

            //Then unload
            if(pluginLibHandle)
            {
                pluginLibHandle->unload();
                delete pluginLibHandle;
            }
            //And remove from container
            aPluginLib->first = NULL;
            aPluginLib->second = NULL;
            aPluginLib = NULL;
        }
    }

    //Remove all from container...
    mPlugins.clear();
    return result;
}

bool PluginManager::unload(Plugin* plugin)
{
    if(!plugin)
    {
        return unloadAll();
    }

    bool result(false);
    int nrPlugins = getNumberOfPlugins();

    for(vector< telPlugin >::iterator it = mPlugins.begin(); it != mPlugins.end(); it++)
    {
        telPlugin *aPluginLib = &(*it);
        if(aPluginLib)
        {
            SharedLibrary *pluginLibHandle    = aPluginLib->first;
            Plugin        *aPlugin            = aPluginLib->second;

            if(aPlugin == plugin)
            {
                destroyRRPlugin(aPlugin);

                plugin = NULL;
                //Then unload
                if(pluginLibHandle)
                {
                    pluginLibHandle->unload();
                    //Research this delete pluginLibHandle;
                }

                //And remove from container
                aPluginLib->first = NULL;
                aPluginLib->second = NULL;
                aPluginLib = NULL;

                //Reset the plugins iterator to a valid one
                mPluginsIter = mPlugins.erase(it);
                result = true;
                break;
            }
        }
    }
    return result;
}

bool PluginManager::checkImplementationLanguage(Poco::SharedLibrary* plugin)
{
    //Check that the plugin has a getImplementationLanguage function
    try
    {
        plugin->getSymbol(string(exp_fnc_prefix) + "getImplementationLanguage");
        return true;
    }
    catch(const Poco::Exception& ex)
    {
        stringstream msg;
        msg<<"Poco exception: "<<ex.displayText()<<endl;
        Log(lError)<<msg.str();
        return false;
    }
}

const char* PluginManager::getImplementationLanguage(Poco::SharedLibrary* plugin)
{
    //Check that the plugin has a getImplementationLanguage function
    try
    {
        charStarFnc func =     (charStarFnc) plugin->getSymbol(string(exp_fnc_prefix) + "getImplementationLanguage");
        return func();
    }
    catch(const Poco::Exception& ex)
    {
        stringstream msg;
        msg<<"Poco exception: "<<ex.displayText()<<endl;
        Log(lError)<<msg.str();
        return NULL;
    }
}

StringList PluginManager::getPluginNames() const
{
    StringList names;
    int nrPlugins = getNumberOfPlugins();
    for(int i = 0; i < nrPlugins; i++)
    {
        pair< Poco::SharedLibrary*, Plugin* >  *aPluginLib = &(mPlugins[i]);
        if(aPluginLib)
        {
            Plugin* aPlugin     = aPluginLib->second;
            if(aPlugin)
            {
                names.add(aPlugin->getName());
            }
        }
    }
    return names;
}

StringList PluginManager::getPluginLibraryNames() const
{
    StringList names;
    int nrPlugins = getNumberOfPlugins();
    for(int i = 0; i < nrPlugins; i++)
    {
        pair< Poco::SharedLibrary*, Plugin* >  *aPluginLib = &(mPlugins[i]);
        if(aPluginLib)
        {
            Plugin* aPlugin     = aPluginLib->second;
            if(aPlugin)
            {
                names.add(aPlugin->getLibraryName());
            }
        }
    }
    return names;
}

int PluginManager::getNumberOfPlugins() const
{
    return mPlugins.size();
}

int PluginManager::getNumberOfCategories() const
{
    return -1;
}

Plugin* PluginManager::getPlugin(const string& _name) const
{
    //Strip the extension
    string name = getFileNameNoExtension(_name);
    for(int i = 0; i < getNumberOfPlugins(); i++)
    {
        telPlugin aPluginLib = mPlugins[i];
        if(aPluginLib.first && aPluginLib.second)
        {
            Plugin* aPlugin = (Plugin*) aPluginLib.second;
            if(aPlugin && aPlugin->getName() == name)
            {
                   return aPlugin;
            }
            if(aPlugin && (aPlugin->getLibraryName() == name|| aPlugin->getLibraryName() == getPluginOSPrefix() + name))
            {
                   return aPlugin;
            }
        }
    }
    return NULL;
}

Plugin* PluginManager::createCPlugin(SharedLibrary *libHandle)
{
    //Minimum bare bone plugin need these
    charStarFnc         getName                 = (charStarFnc) libHandle->getSymbol(string(exp_fnc_prefix) + "getName");
    charStarFnc         getCategory             = (charStarFnc) libHandle->getSymbol(string(exp_fnc_prefix) + "getCategory");

    //All 'data' that we need to create the plugin
    char* name  = getName();
    char* cat   = getCategory();
    CPlugin* aPlugin = new CPlugin(name, cat);

    aPlugin->executeFunction            = (executeF)         libHandle->getSymbol(string(exp_fnc_prefix) + "execute");
    aPlugin->destroyFunction            = (destroyF)         libHandle->getSymbol(string(exp_fnc_prefix) + "destroyPlugin");
    aPlugin->getCLastError              = (charStarFnc)      libHandle->getSymbol(string(exp_fnc_prefix) + "getCLastError");
    setupCPluginFnc setupCPlugin        = (setupCPluginFnc)  libHandle->getSymbol(string(exp_fnc_prefix) + "setupCPlugin");

    //This give the C plugin an opaque Handle to the CPlugin object
    if(!setupCPlugin(aPlugin))
    {
        //Failed setting up C Plugin!
        //Check for error
        string error = aPlugin->getLastError();
        stringstream  msg;
        msg<<"Failure creating C plugin: "<<error;
        throw(Exception(msg.str()));
    }
    aPlugin->getCPropertyNames  =    (charStarFnc)      libHandle->getSymbol(string(exp_fnc_prefix) + "getListOfCPluginPropertyNames");
    aPlugin->getCProperty       =    (getAPropertyF)    libHandle->getSymbol(string(exp_fnc_prefix) + "getCPluginProperty");
    return aPlugin;

    return NULL;
}

bool PluginManager::hasLoadErrors() const
{
    return mLoadPluginErrors.str().size() != 0 ? true : false;
}

string PluginManager::getLoadErrors() const
{
    return mLoadPluginErrors.str();
}

void PluginManager::clearLoadErrors()
{
    mLoadPluginErrors.str("");
}

// Plugin cleanup function
bool destroyRRPlugin(Plugin *plugin)
{
    //we allocated in the create function with new, delete the passed object
    try
    {
        delete plugin;
        return true;
    }
    catch(...)
    {
        //Bad stuff!
        clog<<"Failed deleting RoadRunner plugin..";
        return false;
    }
}


std::string getPluginExtension()
{

#if defined(_WIN32)
    return "dll";
#elif defined(__linux__)
    return "so";
#else
    // OSX
    return "so";
#endif
}

std::string getPluginOSPrefix()
{

#if defined(_WIN32)
    return "";
#elif defined(__linux__)
    return "lib";
#else
    return "lib";
#endif
}

std::string getPluginLibNamePrefix()
{
    return "tel_";
}

bool hasFileExtension(const string& fName)
{
    return (fName.find_last_of(".") != string::npos) ? true : false;
}

}
