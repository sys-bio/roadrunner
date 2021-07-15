##@Module telplugins_c_api
#This module allows access to the telplugins_api.dll from python"""
import os
import sys
import numpy as np
import tempfile
import ctypes
from ctypes import CDLL, POINTER, CFUNCTYPE, cast, byref
from ctypes import c_double, c_bool, c_int, c_void_p, c_char_p, c_ubyte, c_uint, c_char
import _ctypes
import matplotlib
import matplotlib.pyplot as plt
from os.path import isfile

"""
CTypes Python Bindings to the Tellurium Plugin API.

Currently this is a fairly raw implementation with few Pythonic refinements
"""

__version__ = "1.2.2"

libTitle='telplugins_c_api'

originalWorkingDirectory = os.getcwd()

def rrpPlatformIsWin():
    return sys.platform.startswith('win32')

def rrpPlatformIsOSX():
    return sys.platform.startswith('darwin')

def decodeIfBytes(x):
    if hasattr(x, 'decode'):
        return x.decode('utf-8')
    else:
        return x

# if not os.path.exists(rrplugins_path):

# question: can we turn plugin path into a list of paths?
rrplugins_path = os.path.abspath(os.path.dirname(__file__)) #i.e. 'the directory of this file'

# We can't put the libraries in the same directory as this file, since
#  one of the libraries has the same name, and python on unix gets terribly
#  confused by this.
rrplugins_path = os.path.join(rrplugins_path, "libs")
if not os.path.isdir(rrplugins_path):
    raise NotADirectoryError(f"Looking for plugins in \"{rrplugins_path}\" but not found")

# This is where the plugins manager will look for the plugin libraries by default
#gDefaultPluginsPath = os.path.join(rrplugins_path, "plugins")
gDefaultPluginsPath = rrplugins_path

# bail if the path still hasn't been found
if not os.path.exists(rrplugins_path):
    raise RuntimeError('==== ERROR: path to rrplugin binaries could not be found =====')

# temporarily change into this path so we can load our shared libs
os.chdir(rrplugins_path)

#rrpLib will be our handle returned by ctypes
rrpLib=None

# search for known lib prefixes/suffixes
for name in [
             libTitle + '.dll',
             'lib' + libTitle + '.dylib',
             'lib' + libTitle + '.so',
             libTitle + ".dylib",
             libTitle + ".so",
            ]:
    fullpath = os.path.join(rrplugins_path, name)

    # if the lib file exists, try to load it
    if isfile(fullpath):
        try:
            rrpLib=CDLL(fullpath)
            
        except Exception as e:
            print('Exception when trying to load Tellurium plugins: {}'.format(e))

if rrpLib is None:
    raise RuntimeError('Cannot find telplugins_c_api library in {}'.format(rrplugins_path))

# restore working dir
os.chdir(originalWorkingDirectory)

# custom primitives

c_double_p = POINTER(c_double)
c_bool_p   = POINTER(c_bool)
c_int_p    = POINTER(c_int)

#=======================tel_api========================#
#Type of plugin events, first argument is return type

## \brief Plugin function event type definition
## This is a helper object that a client can use as an argument to a tellurium plugin.
## The exact number of plugins functions required arguments, and their type, is plugin dependent. A client of the
## the plugin needs to get this information from the plugin specific documentation.
## An example of using NotifyEvent is shown below. The NotifyEvent takes no arguments.
##@code
##def myPluginFunction():
##    print 'The plugin can call this function!'
## #The user can assign this function as a plugin event to monitor the start of the plugin as follows.
## #Note, make sure you assign the event to a variable (c_event) so that the Python garbage
## #collector doesn't delete it
## c_event = NotifyEvent(myPluginFunction)
## assignOnStartedEvent(plugin,  c_event)
##@endcode
## \ingroup plugins
NotifyEvent  = CFUNCTYPE(None, c_void_p, c_void_p)
NotifyEventEx  = CFUNCTYPE(None, c_void_p, c_void_p)

## \brief Get Tellurium Plugins copyright.
## \return Returns a string if successful, None otherwise
## \ingroup utilities
rrpLib.tpGetCopyright.restype = c_char_p
def getCopyright():
    data =  rrpLib.tpGetCopyright()
    res = data
    return decodeIfBytes(res)

## \brief Get Tellurium plugin API version.
## \return Returns a string if successful, None otherwise
## \ingroup utilities
rrpLib.tpGetVersion.restype = c_char_p
def getVersion():
    data =  rrpLib.tpGetVersion()
    res = data
    return decodeIfBytes(res)

## \brief Create a new instance of a plugin manager.
## \brief A PluginManager manages a collection of plugins, loaded and unloaded by
##  the load and unload API functions respectively.
## \param pluginDir Full path to folder containing plugins. If None, uses default folder.
## \return On success, a handle to a Plugin manager, on failure, None.
##
## @code
## pm = telPlugins.createPluginManager()
## @endcode
## \htmlonly  <br/>
## \endhtmlonly
## \ingroup plugin_manager
rrpLib.tpCreatePluginManager.restype = c_void_p
rrpLib.tpCreatePluginManager.argtypes = [c_char_p]
def createPluginManager(pluginDir = None):
    if pluginDir == None:
        pluginDir = gDefaultPluginsPath
    return rrpLib.tpCreatePluginManager(pluginDir.encode('utf-8'))

## \brief Free the plugin manager. A call to this function will also unload any loaded plugins.
## \param pm Handle to a plugin manager.
## \return true if success, false otherwise.
##
## \ingroup plugin_manager
rrpLib.tpFreePluginManager.restype = c_bool
rrpLib.tpFreePluginManager.argtypes = [c_void_p]
def freePluginManager(pm):
    return rrpLib.tpFreePluginManager(pm)

## \brief Load plugins. The function will look in the default plugin folder for plugins, and load them.
## \param pm Handle to a PluginManager instance
## \return Returns true if Plugins are loaded, false otherwise
##
## @code sucess = telPlugins.loadPlugins (pm)
## @endcode
## \htmlonly  <br/>
## \endhtmlonly
## \ingroup plugin_manager
rrpLib.tpLoadPlugins.restype = c_bool
rrpLib.tpLoadPlugins.argtypes = [c_void_p]
def loadPlugins(pm):
    return rrpLib.tpLoadPlugins(pm)

## \brief Check if there was any Errors catched during loading of plugins.
## \param pm Handle to a PluginManager instance
## \return Returns true or false indicating if there was errors
## \ingroup plugin_manager
def hasLoadPluginErrors(pm):
    aStr = rrpLib.tpGetPluginLoadErrors(pm)
    if aStr != None and len(aStr) > 0:
        return True
    else:
        return False

## \brief Get any Errors catched during loading of plugins.
## \param pm Handle to a PluginManager instance
## \return Returns a string if there was errors, None otherwise
## \ingroup plugin_manager
rrpLib.tpGetPluginLoadErrors.restype = c_char_p
rrpLib.tpGetPluginLoadErrors.argtypes = [c_void_p]
def getPluginLoadErrors(pm):
    return decodeIfBytes(rrpLib.tpGetPluginLoadErrors(pm))

## \brief Unload all plugins.
## \param pm Handle to a PluginManager instance
## \return Returns true if Plugins are unloaded succesfully, false otherwise
##
## \ingroup plugin_manager
##
rrpLib.tpUnLoadPlugins.restype = c_bool
rrpLib.tpUnLoadPlugins.argtypes = [c_void_p]
def unLoadPlugins(pm):
    return rrpLib.tpUnLoadPlugins(pm)

## \brief Load a particular plugin
## \param pm Handle to a PluginManager instance
## \param pluginName Name of the plugin to load. The plugin name is the plugin's shared library name, without path and extension.
## The library names for the plugins can be obtained by calling getPluginNames()
## \return Returns a handle to a plugin, None if unsuccesful
##
## @code
## lmPlugin = telPlugins.loadPlugin(pm, "tel_lm")
## @endcode
## \htmlonly  <br/>
## \endhtmlonly
## \ingroup plugin_manager
##
rrpLib.tpLoadPlugin.restype = c_void_p
rrpLib.tpLoadPlugin.argtypes = [c_void_p, c_char_p]
def loadPlugin(pm, pluginName):
    return rrpLib.tpLoadPlugin(pm, pluginName.encode('utf-8'))

## \brief Unload a particular plugin
## \param pm Handle to a PluginManager instance
## \param pHandle Handle to a Plugin instance
## \return Returns true if the Plugin are unloaded succesfully, false otherwise
## \ingroup plugin_manager
##
rrpLib.tpUnLoadPlugin.restype = c_bool
rrpLib.tpUnLoadPlugin.argtypes = [c_void_p]
def unLoadPlugin(pm, pHandle):
    return rrpLib.tpUnLoadPlugin(pm, pHandle)

## \brief Get number of loaded plugins.
## \param pm Handle to a PluginManager instance
## \return Returns the number of loaded plugins, -1 if a problem is encountered. Call telPlugins.getLastError() to obtain error message.
## \ingroup plugin_manager
rrpLib.tpGetNumberOfPlugins.restype = c_int
rrpLib.tpGetNumberOfPlugins.argtypes = [c_void_p]
def getNumberOfPlugins(pm):
    return rrpLib.tpGetNumberOfPlugins(pm)

## \brief Function to retrieve the names of all currently loaded plugins.
## \param pm Handle to a PluginManager instance
## \return Returns names for all loaded plugins as a string, None otherwise
##
## @code
## names = telPlugins.getPluginNames(pm)
## print names
## ['AddNoise', 'Levenberg-Marquardt']
## @endcode
## \htmlonly  <br/>
## \endhtmlonly
## \ingroup plugin_manager
rrpLib.tpGetPluginNames.restype = c_void_p
rrpLib.tpGetPluginNames.argtypes = [c_void_p]
def getPluginNames(pm):
    names = rrpLib.tpGetPluginNames(pm)
    res = decodeIfBytes(ctypes.cast(names, ctypes.c_char_p).value)
    if not res:
        return list()
    return res.split(",")

## \brief Function to retrieve the library names of all currently loaded plugins.
## \param pm Handle to a PluginManager instance
## \return Returns library names for all loaded plugins as a string, None otherwise
##
## @code
## names = telPlugins.getPluginLibraryNames(pm)
## print names
## ['tel_add_noise', 'tel_lm']
## @endcode
## \htmlonly  <br/>
## \endhtmlonly
## \ingroup plugin_manager
rrpLib.tpGetPluginLibraryNames.restype = c_char_p
rrpLib.tpGetPluginLibraryNames.argtypes = [c_void_p]
def getPluginLibraryNames(pm):
    names = rrpLib.tpGetPluginLibraryNames(pm)
    if not names:
        return list()
    return decodeIfBytes(names).split(",")

## \brief getFirstPlugin retrieves the "first" plugin in the plugin managers internal list of plugins.
## This function is typically used together with the getNextPlugin and the getPreviousPlugin functions.
## \param pm Handle to a PluginManager instance
## \return Returns a handle to a plugin. Returns None if the plugin is not found
## \ingroup plugin_manager
rrpLib.tpGetFirstPlugin.restype = c_void_p
rrpLib.tpGetFirstPlugin.argtypes = [c_void_p]
def getFirstPlugin(pm):
    return rrpLib.tpGetFirstPlugin(pm)

## \brief getNextPlugin retrieves the "next" plugin in the plugin managers internal list of plugins. This function
## is typically used together with the getFirstPlugin and getPreviousPlugin functions.
## \param pm Handle to a PluginManager instance
## \return Returns a handle to a plugin. Returns None if the plugin is not found
## \ingroup plugin_manager
rrpLib.tpGetNextPlugin.restype = c_void_p
rrpLib.tpGetNextPlugin.argtypes = [c_void_p]
def getNextPlugin(pm):
    return rrpLib.tpGetNextPlugin(pm)

## \brief getPreviousPlugin retrieves the "previous" plugin in the plugin managers internal list of plugins. This function
##    is typically used together with the getFirstPlugin and getNextPlugin functions.
## \param pm Handle to a PluginManager instance
## \return Returns a handle to a plugin. Returns None if the plugin is not found
## \ingroup plugin_manager
rrpLib.tpGetPreviousPlugin.restype = c_void_p
rrpLib.tpGetPreviousPlugin.argtypes = [c_void_p]
def getPreviousPlugin(pm):
    return rrpLib.tpGetPreviousPlugin(pm)

## \brief getCurrentPlugin retrieves the "current" plugin in the plugin managers internal list of plugins. This function
##    is typically used together with the getFirst, Next and getPreviousPlugin functions.
## \param pm Handle to a PluginManager instance
## \return Returns a handle to a plugin. Returns None if the plugin is not found
## \ingroup plugin_manager
rrpLib.tpGetCurrentPlugin.restype = c_void_p
rrpLib.tpGetCurrentPlugin.argtypes = [c_void_p]
def getCurrentPlugin(pm):
    return rrpLib.tpGetCurrentPlugin(pm)

## \brief Get the plugin handle for the named plugin
## \param pm Handle to a PluginManager instance
## \param pluginName A string that holds the name of the plugin
## \return Returns a handle to a plugin, with name as supplied in the property pluginName.
## Returns None if the plugin is not found
## \ingroup plugin_manager
rrpLib.tpGetPlugin.restype = c_void_p
rrpLib.tpGetPlugin.argtypes = [c_void_p, c_char_p]
def getPlugin(pm, pluginName):
    return rrpLib.tpGetPlugin(pm, c_char_p(pluginName.encode('utf-8')))

#---------- PLUGIN HANDLING FUNCTIONS ============================================
## \brief Get the name of a Plugin
## \param pluginHandle Handle to a plugin
## \return Returns the internal name of a plugin. None otherwise
##
## @code
## name = telPlugins.getPluginName(pluginHandle)
## @endcode
## \htmlonly  <br/>
## \endhtmlonly
## \ingroup plugins
rrpLib.tpGetPluginName.restype = c_char_p
rrpLib.tpGetPluginName.argtypes = [c_void_p]
def getPluginName(pluginHandle):
    return decodeIfBytes(rrpLib.tpGetPluginName(pluginHandle))

## \brief Get the Category of a Plugin. This is assigned by the pluging developer
## \param pluginHandle Handle to a plugin
## \return Returns a string if successful, None otherwise
##
## @code
## name = telPlugins.getPluginCategory(pluginHandle)
## @endcode
## \htmlonly  <br/>
## \endhtmlonly
## \ingroup plugins
rrpLib.tpGetPluginCategory.restype = c_void_p
rrpLib.tpGetPluginCategory.argtypes = [c_void_p]
def getPluginCategory(pluginHandle):
    data =  rrpLib.tpGetPluginCategory(pluginHandle)
    res = decodeIfBytes(ctypes.cast(data, ctypes.c_char_p).value)
    return res

## \brief Get the author of a Plugin. This is assigned by the pluging developer
## \param pluginHandle Handle to a plugin
## \return Returns a string if successful, None otherwise
## \ingroup plugins
rrpLib.tpGetPluginAuthor.restype = c_void_p
rrpLib.tpGetPluginAuthor.argtypes = [c_void_p]
def getPluginAuthor(pluginHandle):
    data =  rrpLib.tpGetPluginAuthor(pluginHandle)
    res = decodeIfBytes(ctypes.cast(data, ctypes.c_char_p).value)
    return res

## \brief Get the plugin copyright.
## \param pluginHandle Handle to a plugin
## \return Returns a string if successful, None otherwise
## \ingroup plugins
rrpLib.tpGetPluginCopyright.restype = c_void_p
rrpLib.tpGetPluginCopyright.argtypes = [c_void_p]
def getPluginCopyright(pluginHandle):
    data =  rrpLib.tpGetPluginCopyright(pluginHandle)
    res = decodeIfBytes(ctypes.cast(data, ctypes.c_char_p).value)
    return res

## \brief Get the plugin version.
## \param pluginHandle Handle to a plugin
## \return Returns a string if successful, None otherwise
## \ingroup plugins
rrpLib.tpGetPluginVersion.restype = c_void_p
rrpLib.tpGetPluginVersion.argtypes = [c_void_p]
def getPluginVersion(pluginHandle):
    ptr = rrpLib.tpGetPluginVersion(pluginHandle)
    res = decodeIfBytes(ctypes.cast(ptr, ctypes.c_char_p).value)
    return res

## \brief Get the Description of a Plugin. This is assigned by the pluging developer
## \param pluginHandle Handle to a plugin
## \return Returns a string if successful, None otherwise
##
## @code
## name = telPlugins.getPluginDescription(pluginHandle)
## @endcode
## \htmlonly  <br/>
## \endhtmlonly
## \ingroup plugins
rrpLib.tpGetPluginDescription.restype = c_void_p
rrpLib.tpGetPluginDescription.argtypes = [c_void_p]
def getPluginDescription(pluginHandle):
    data =  rrpLib.tpGetPluginDescription(pluginHandle)
    res = decodeIfBytes(ctypes.cast(data, ctypes.c_char_p).value)
    return res

## \brief Get a plugins Hint. A plugins hint is a short description on what the plugin is doing.This is assigned by the pluging developer
## \param pluginHandle Handle to a plugin
## \return Returns a string if successful, None otherwise
##
## @code
## name = telPlugins.getPluginHint(pluginHandle)
## @endcode
## \htmlonly  <br/>
## \endhtmlonly
## \ingroup plugins
rrpLib.tpGetPluginHint.restype = c_void_p
rrpLib.tpGetPluginHint.argtypes = [c_void_p]
def getPluginHint(pluginHandle):
    data =  rrpLib.tpGetPluginHint(pluginHandle)
    res = decodeIfBytes(ctypes.cast(data, ctypes.c_char_p).value)
    return res

## \brief Returns information about a Plugin.
## \param pluginHandle Handle to a plugin
## \return Returns information as a string for the plugin, None otherwise
## \ingroup plugins
rrpLib.tpGetPluginInfo.restype = c_char_p
rrpLib.tpGetPluginInfo.argtypes = [c_void_p]
def getPluginInfo(pluginHandle):
    data =  rrpLib.tpGetPluginInfo(pluginHandle)
    res = decodeIfBytes(ctypes.cast(data, ctypes.c_char_p).value)
    return res

## \brief Get Plugin manual as PDF. A plugin may embedd a help manual as a PDF.
## Use the function getPluginManualNrOfBytes to get the exact length of this string.
## \param pluginHandle Handle to a plugin
## \return Returns the plugin's manual pdf file as a unsigned char*. If not available, returns None.
##
## @code
## ptr = getPluginManualAsPDF(pluginHandle)
## numOfBytes = getPluginManualNrOfBytes(pluginHandle)
## manual = cast(ptr, POINTER(c_char * numOfBytes))[0]
## if numOfBytes == 0:
##    print 'This plugin does not have a manual.'
##    exit()
## outFName = getPluginName (pluginHandle) + '.pdf'
## with open(outFName, 'wb') as output:
##    output.write(manual)
## os.system('start ' + outFName)
##
## @endcode
## \htmlonly  <br/>
## \endhtmlonly
## \ingroup plugins
rrpLib.tpGetPluginManualAsPDF.restype =  POINTER(c_ubyte)
rrpLib.tpGetPluginManualAsPDF.argtypes = [c_void_p]
def getPluginManualAsPDF(pluginHandle):
    return rrpLib.tpGetPluginManualAsPDF(pluginHandle)

## \brief Get the byte size for the PDF manual.
## \param pluginHandle Handle to a plugin
## \return Returns the number of bytes in the plugin's manual pdf file as an unsigned int.
## \ingroup plugins
rrpLib.tpGetPluginManualNrOfBytes.restype = c_uint
rrpLib.tpGetPluginManualNrOfBytes.argtypes = [c_void_p]
def getPluginManualNrOfBytes(pluginHandle):
    return rrpLib.tpGetPluginManualNrOfBytes(pluginHandle)

## \brief If a plugin has a built-in PDF manual, display it.
## \param pluginHandle Handle to a plugin
## \return Returns False if the plugin has no manual
##
## @code
## success = displayPluginManual(pluginHandle)
##
## @endcode
## \htmlonly  <br/>
## \endhtmlonly
## \ingroup plugins
def displayPluginManual(pluginHandle):
    ptr = getPluginManualAsPDF(pluginHandle)
    numOfBytes = getPluginManualNrOfBytes(pluginHandle)
    manual = cast(ptr, POINTER(c_char * numOfBytes))[0]
    if numOfBytes == 0:
       print('This plugin does not have a manual.')
       return False
    outFName = os.path.join(tempfile.gettempdir(), getPluginName (pluginHandle) + '.pdf')
    print(outFName)
    with open(outFName, 'wb') as output:
      output.write(manual)
    if rrpPlatformIsWin():
        os.system('start ' + outFName)
    elif rrpPlatformIsOSX():
        os.system('open ' + outFName)
    else:
        os.system('xdg-open ' + outFName)

## \brief The executePlugin function is called to start the plugin so that it can carry out its
## function. The call is plugin dependent meaning that it could result in a calculation, starting up a GUI etc.
## \param pluginHandle Handle to a plugin
## \return Returns true or false indicating success/failure
## \note The execute function is a regular blocking function, meaning it won't return to the caller until the task is complete.
## If the plugin is asked to carry out a lengthy calculation, consider using
## the executePluginEx function that has the option to execute the plugin code in the background (in a thread);
## \ingroup plugins
rrpLib.tpExecutePlugin.restype = c_bool
rrpLib.tpExecutePlugin.argtypes = [c_void_p]
def executePlugin(pluginHandle):
    return rrpLib.tpExecutePlugin(pluginHandle)

## \brief The executePluginEx is similar to the executePlugin function, except it takes one extra argument.
## \param pluginHandle Handle to a plugin
## \param inAThread bool indicating if the plugin should be executed in the background (in a thread)
## \return Returns true or false indicating success/failure
## \ingroup plugins
rrpLib.tpExecutePluginEx.restype = c_bool
rrpLib.tpExecutePluginEx.argtypes = [c_void_p, c_bool]
def executePluginEx(pluginHandle, inAThread=False):
    return rrpLib.tpExecutePluginEx(pluginHandle, c_bool(inAThread))

## \brief Get status information from a plugin. This call is plugin dependent, see the plugin documentation for details
## \param pluginHandle Handle to a plugin
## \return Returns plugin status if available, as a string. None otherwise
## \ingroup plugins
rrpLib.tpGetPluginStatus.restype = c_char_p
rrpLib.tpGetPluginStatus.argtypes = [c_void_p]
def getPluginStatus(pluginHandle):
    return decodeIfBytes(rrpLib.tpGetPluginStatus(pluginHandle))

## \brief Returns a plugins result, as a string. This is plugin dependent, and a plugin designer may, or may not, implement
## this function. See the plugin documentation for details.
## \note If a plugin wants to returns specific result, e.g. an Array, or a a float, these are better communicated
## through the use of Plugin properties.
## \param pluginHandle Handle to a plugin
## \return Returns a plugins result if available. None otherwise
## \ingroup plugins
rrpLib.tpGetPluginResult.restype = c_char_p
rrpLib.tpGetPluginResult.argtypes = [c_void_p]
def getPluginResult(pluginHandle):
    return decodeIfBytes(rrpLib.tpGetPluginResult(pluginHandle))

## \brief Reset a Plugin. Plugin dependent. A reset function should bring the internal state of a plugin to a known state
## \param pluginHandle Handle to a plugin
## \return Returns true or false indicating success/failure
## \ingroup plugins
rrpLib.tpResetPlugin.restype = c_bool
rrpLib.tpResetPlugin.argtypes = [c_void_p]
def resetPlugin(pluginHandle):
    return rrpLib.tpResetPlugin(pluginHandle)

## \brief Check if a plugin is actively working. This function is used when the work in the plugin is
## executed in a thread (see executeEx). The isPluginWorking will return true as long work is being active
## and false when the work is done. This is useful in UI environments. Also, see the various event types on
## how to get status back from a plugin during its execution.
## \param pluginHandle Handle to a plugin
## \return Returns true or false indicating if the plugin is busy or not
## \ingroup plugins
rrpLib.tpIsPluginWorking.restype = c_bool
rrpLib.tpIsPluginWorking.argtypes = [c_void_p]
def isPluginWorking(pluginHandle):
    return rrpLib.tpIsPluginWorking(pluginHandle)

## \brief Terminate any work that is in progress in a plugin. If the plugins worker is executed in a thread, this function
## will signal the internals of the plugin to terminate. This function is used when a plugins work is executed in a thread.
## \param pluginHandle Handle to a plugin
## \return Returns true or false indicating success/failure
## \ingroup plugins
rrpLib.tpTerminateWork.restype = c_bool
rrpLib.tpTerminateWork.argtypes = [c_void_p]
def terminateWork(pluginHandle):
    return rrpLib.tpTerminateWork(pluginHandle)

## \brief Check if the work of a plugin is currently being terminated. This function is useful when a plugin is executed in a thread.
## \param pluginHandle Handle to the plugin
## \return Returns true or false indicating if the work within the plugin is in the process of being terminated
## \ingroup plugins
rrpLib.tpIsBeingTerminated.restype = c_bool
rrpLib.tpIsBeingTerminated.argtypes = [c_void_p]
def isBeingTerminated(pluginHandle):
    return rrpLib.tpIsBeingTerminated(pluginHandle)

## \brief Query a plugin if work was terminated succesfully. This function may be used in combination with
## the terminateWork, and isBeingTerminated functions.
## \param pluginHandle Handle to the plugin
## \return Returns true or false indicating if the work in the plugin was terminated or not
## \ingroup plugins
rrpLib.tpWasTerminated.restype = c_bool
rrpLib.tpWasTerminated.argtypes = [c_void_p]
def wasTerminated(pluginHandle):
    return rrpLib.tpWasTerminated(pluginHandle)

## \brief Assigns a plugins OnStartedEvent function.Plugin dependent. Intended usage is to report back on plugin initialization.
## \param pluginHandle Handle to a plugin
## \param pluginEvent Function pointer to OnEvent routine
## \param userData1 void* pointer to user data.
## \param userData2 void* pointer to user data.
## \return Returns true or false indicating success/failure
## \ingroup plugins
rrpLib.tpAssignOnStartedEvent.restype = c_bool
rrpLib.tpAssignOnStartedEvent.argtypes =[c_void_p, NotifyEvent, c_void_p]
def assignOnStartedEvent(pluginHandle, pluginEvent, userData1 = None, userData2 = None):
    return rrpLib.tpAssignOnStartedEvent(pluginHandle, pluginEvent, userData1, userData2)

## \brief Assigns a plugins OnProgressEvent function. Plugin dependent. Intended usage is to report back progress
## \param pluginHandle Handle to a pluginevent routine
## \param userData1 void* pointer to user data.
## \param userData2 void* pointer to user data.
## \return Returns true or false indicating success/failure
## \ingroup plugins
rrpLib.tpAssignOnProgressEvent.restype = c_bool
rrpLib.tpAssignOnProgressEvent.argtypes =[c_void_p, c_void_p, c_void_p]
def assignOnProgressEvent(pluginHandle, pluginEvent, userData1 = None, userData2 = None):
    return rrpLib.tpAssignOnProgressEvent(pluginHandle, pluginEvent, userData1, userData2)

## \brief Assigns a plugins OnFinishedEvent function. Plugin dependent. Intended usage is to report back on plugin finalization.

## \param pluginHandle Handle to a plugin
## \param pluginEvent Function pointer to event routine
## \param userData1 void* pointer to user data.
## \param userData2 void* pointer to user data.
## \return Returns true or false indicating success/failure
## \ingroup plugins
rrpLib.tpAssignOnFinishedEvent.restype = c_bool
rrpLib.tpAssignOnFinishedEvent.argtypes =[c_void_p, NotifyEvent, c_void_p]
def assignOnFinishedEvent(pluginHandle, pluginEvent, userData1 = None, userData2 = None):
    return rrpLib.tpAssignOnFinishedEvent(pluginHandle, pluginEvent, userData1, userData2)

#================ Plugin Property functionality ======================
## \brief Get a handle to the list of properties for a plugin
## \param pluginHandle Handle to a plugin
## \return Returns a handle to a list of Properties on success, None otherwise
## \ingroup plugin_properties
rrpLib.tpGetPluginProperties.restype = c_void_p
rrpLib.tpGetPluginProperties.argtypes = [c_void_p]
def getPluginProperties(pluginHandle):
    return rrpLib.tpGetPluginProperties(pluginHandle)

## \brief Get a list of property names in a plugin
## \param pluginHandle Handle to a plugin
## \return Returns the netire list of top level property names, None otherwise
## \ingroup plugin_properties
rrpLib.tpGetListOfPluginPropertyNames.restype = c_char_p
rrpLib.tpGetListOfPluginPropertyNames.argtypes = [c_void_p]
def getListOfPluginPropertyNames(pluginHandle):
    paraNames = decodeIfBytes(rrpLib.tpGetListOfPluginPropertyNames(pluginHandle))
    if not paraNames:
        return list()
    else:
        names = paraNames.split(',')
        return names

## \brief Clear a list of properties. Some properties exposed by plugins are lists that can hold other properties. New property can be
## added to these lists. clearPropertyList can be used to clear the list is a new list needs to be constructed.
## \param parasHandle Handle to a list of properties
## \return True or false, indicating result. The top level list of properties in a plugin can not be cleared.
## \ingroup plugin_properties
rrpLib.tpClearPropertyList.restype = c_bool
rrpLib.tpClearPropertyList.argtypes = [c_void_p]
def clearPropertyList(propertyListHandle):
    handle = getPropertyValueHandle(propertyListHandle)
    return rrpLib.tpClearPropertyList(handle)

## \brief If the property is a list, this method returns the list of property names in that list
## \param propertyHandle Handle to a property
## \return Returns names for all properties in the list
## \ingroup plugin_properties
rrpLib.tpGetNamesFromPropertyList.restype = c_char_p
rrpLib.tpGetNamesFromPropertyList.argtypes = [c_void_p]
def getNamesFromPropertyList(propertyHandle):
    paraType = getPropertyType(propertyHandle)
    if not paraType:
        print(getLastError())
    if paraType != 'listOfProperties':
        raise Exception('That is not a valid list property')
    listHandle = getPropertyValueHandle(propertyHandle)
    paras = decodeIfBytes(rrpLib.tpGetNamesFromPropertyList(listHandle))
    if not paras:
        return list()
    else:
        names = paras.split(',')
        return names

## \brief Get a the properties of a plugins in xml format.
## \param pluginHandle Handle to a plugin
## \return Returns a string on success, None otherwise
## \ingroup plugin_properties
rrpLib.tpGetPluginPropertiesAsXML.restype = c_char_p
rrpLib.tpGetPluginPropertiesAsXML.argtypes = [c_void_p]
def getPluginPropertiesAsXML(pluginHandle):
    return decodeIfBytes(rrpLib.tpGetPluginPropertiesAsXML(pluginHandle))

## \brief Get the 'first' property handle to a property in a list of properties
## \param paraListHandle Handle to a propertyList
## \return Returns a handle to a property. Returns None if not found
## \ingroup plugin_properties
rrpLib.tpGetFirstProperty.restype = c_void_p
rrpLib.tpGetFirstProperty.argtypes = [c_void_p]
def getFirstProperty(paraListHandle):
    handle = getPropertyValueHandle(paraListHandle)
    return rrpLib.tpGetFirstProperty(handle)

## \brief Get the 'next' property handle to a property in a list of properties
## \param paraListHandle Handle to a propertyList
## \return Returns a handle to a property. Returns None if not found
## \ingroup plugin_properties
rrpLib.tpGetNextProperty.restype = c_void_p
rrpLib.tpGetNextProperty.argtypes = [c_void_p]
def getNextProperty(paraListHandle):
    handle = getPropertyValueHandle(paraListHandle)
    return rrpLib.tpGetNextProperty(handle)

## \brief Get a property handle to a property given the name of the property.
## \param pluginHandle Handle to a plugin
## \param propertyName Name of the property
## \return Returns a propertyHandle to a property. Returns None if not found
## \ingroup plugin_properties
rrpLib.tpGetPluginProperty.restype = c_void_p
rrpLib.tpGetPluginProperty.argtypes = [c_void_p, c_char_p]
def getPluginProperty(pluginHandle, propertyName):
    return rrpLib.tpGetPluginProperty(pluginHandle, propertyName.encode('utf-8'))

## \brief Set the value of a PluginProperty
## \param pluginHandle Handle to a plugin
## \param propertyName Name of property
## \param propertyValue Value of property
## \return true if succesful, false otherwise
## \ingroup plugin_properties
rrpLib.tpSetPluginProperty.restype = c_bool
rrpLib.tpSetPluginProperty.argtypes = [c_void_p, c_char_p, c_char_p]
def setPluginProperty(pluginHandle, propertyName, propertyValue):
    propertyHandle = getPluginProperty(pluginHandle, propertyName)
    if propertyHandle:
        paraType = getPropertyType(propertyHandle)
        if paraType == 'bool':
            return setBoolProperty(propertyHandle, propertyValue)
        if paraType == 'int':
            return setIntProperty(propertyHandle, propertyValue)
        if paraType == 'unsigned long':
            return setUnsignedLongProperty(propertyHandle, propertyValue)
        if paraType == 'double':
            return setDoubleProperty(propertyHandle, propertyValue)
        if paraType == 'string':
            return setStringProperty(propertyHandle, propertyValue)
        if paraType == 'std::string': #Behaves the same in the backend
            return setStringProperty(propertyHandle, propertyValue)
        if paraType == 'listOfProperties':
            return setListProperty(propertyHandle, propertyValue)
        if paraType == 'telluriumData': #The value of this is a handle
            return setTelluriumDataProperty (propertyHandle, propertyValue)
        if paraType == 'stringList':
            return setPropertyByString(propertyHandle, propertyValue)
        else:
           raise TypeError ('Cannot set the value of such property')
    else:
           raise TypeError('Bad Handle')
    return False

## \brief Set the value of a property
## \param property handle Handle to a property
## \param paraValue Value of property
## \return true if successful, false otherwise
## \ingroup plugin_properties
def setProperty(propertyHandle, paraValue):
    if propertyHandle:
        paraType = getPropertyType(propertyHandle)
        if paraType == 'bool':
            return setBoolProperty(propertyHandle, paraValue)
        if paraType == 'int':
            return setIntProperty(propertyHandle, paraValue)
        if paraType == 'double':
            return setDoubleProperty(propertyHandle, paraValue)
        if paraType == 'string':
            return setStringProperty(propertyHandle, paraValue)
        if paraType == 'std::string': #Behaves the same in the backend
            return setStringProperty(propertyHandle, paraValue)
        if paraType == 'listOfProperties':
            return setListProperty(propertyHandle, paraValue)
        if paraType == 'telluriumData': #The value of this is a handle
            return setTelluriumDataProperty(propertyHandle, paraValue)
        if paraType == 'stringList':
            return setPropertyByString(propertyHandle, paraValue)
        else:
           raise TypeError ('Cannot set the value of such property')
    else:
           raise ('Bad Handle')
    return False

## \brief Set the description of a Property
## \param propertyHandle Handle to a Property instance
## \param descr String holding the description
## \return Returns true if successful, false otherwise
## \ingroup plugin_properties
rrpLib.tpSetPropertyDescription.restype = c_bool
rrpLib.tpSetPropertyDescription.argtypes = [c_void_p, c_char_p]
def setPropertyDescription(propertyHandle, descr):
    return rrpLib.tpSetPropertyDescription(propertyHandle, descr.encode('utf-8'))

## \brief Get the description of a Property
## \param propertyHandle Handle to a Property instance
## \param descr String holding the description
## \return Returns the description if successful, None otherwise
## \ingroup plugin_properties
rrpLib.tpGetPropertyDescription.restype = c_char_p
rrpLib.tpGetPropertyDescription.argtypes = [c_void_p]
def getPropertyDescription(propertyHandle):
    descr = rrpLib.tpGetPropertyDescription(propertyHandle)
    if descr is None:
        return None

    val = decodeIfBytes(ctypes.cast(descr, ctypes.c_char_p).value)
    return val

## \brief Set the hint property of a Property
## \param propertyHandle Handle to a Property instance
## \param descr String holding the hint text
## \return Returns true if successful, false otherwise
## \ingroup plugin_properties
rrpLib.tpSetPropertyHint.restype = c_bool
rrpLib.tpSetPropertyHint.argtypes = [c_void_p, c_char_p]
def setPropertyHint(propertyHandle, descr):
    return rrpLib.tpSetPropertyHint(propertyHandle, descr.encode('utf-8'))

## \brief Create a Property of type "type" with a name and hint property
##  Valid types include: 'bool', 'int', 'double', 'string', and 'listOfProperties'
## \param name The properties name as a string
## \param the_type  The properties type as string. Possible values: 'bool', 'int', 'double', 'string', and 'listOfProperties'
## \param hint The properties hint as a string.
## \param value This is an optional argument to set the Property (supports int, double and string).
## \return Returns a handle to a new Property, if succesful, None otherwise
##
## @code
## propertyHandle = telPlugins.createProperty ("k1", string", "A message")
##
##propertyHandle = telPlugins.createProperty ("k1", "double", "A rate constant", 0.3)
## @endcode
## \htmlonly  <br/>
## \endhtmlonly
## \ingroup plugin_properties
rrpLib.tpCreateProperty.restype = c_void_p
rrpLib.tpCreateProperty.argtypes = [c_char_p, c_char_p, c_char_p, c_void_p]
def createProperty(name, the_type, hint="", value=None):
    if value is None:
        ptr = rrpLib.tpCreateProperty(name.encode('utf-8'), the_type.encode('utf-8'), hint.encode('utf-8'), value)
        if not ptr:
            raise TypeError('Unable to create property {}'.format(name))
        return ptr
    else:
        if the_type == 'string':    #Otherwise underlying string type will be char*, don't
            the_type = 'std::string'
        ptr = rrpLib.tpCreateProperty(name.encode('utf-8'), the_type.encode('utf-8'), hint.encode('utf-8'), None)
        if not ptr:
            raise TypeError('Unable to create property {}'.format(name))
        if the_type == "bool":
           setBoolProperty (ptr, value)
        elif the_type == "int":
           setIntProperty (ptr, value)
        elif the_type == "float":
           setDoubleProperty (ptr, value)
        elif the_type == "double":
           setDoubleProperty (ptr, value)
        elif the_type == "string":
           if not setStringProperty (ptr, value):
             raise TypeError('Unable to create property {}'.format(name))
        elif the_type == "std::string":
           if not setStringProperty (ptr, value):
             raise TypeError('Unable to create property {}'.format(name))

        else:
            print("Error: Can't set the value of Property with type: " + the_type)
        return ptr

## \brief Free memory for a Property
## \param propertyHandle Handle to a Property instance
## \return Returns true if successful, false otherwise
## \ingroup plugin_properties
rrpLib.tpFreeProperty.restype = c_bool
rrpLib.tpFreeProperty.argtypes = [c_void_p]
def freeProperty(propertyHandle):
    return rrpLib.tpFreeProperty(propertyHandle)

## \brief Add a Property to a list of Property.
## Some plugins may have Property that
## require list of Property to be specified. For example when deciding what kinetic Property to fit in a model, the list
## of kinetic Property can be pass to the plugin as a list. This method can be used to add the names of the kinetic Property
## to the list.
## \param list A Handle to a Property with type listOfProperty
## \param propertyHandle Handle to the Property to add to the list(see createProperty)
## \return Returns a Boolean indicating success
##
## @code
## paraList = getPluginProperty(plHandle, "SpeciesList");
## newProperty = createProperty("k1", "double", "A Hint", 0.2)
## addPropertyToList(paraList, newProperty)
## @endcode
## \htmlonly  <br/>
## \endhtmlonly
## \ingroup plugin_properties
rrpLib.tpAddPropertyToList.restype = c_bool
rrpLib.tpAddPropertyToList.argtypes = [c_void_p, c_void_p]
def addPropertyToList(propertyHandle, addMe):
    #Make sure the Property is of type list
    if getPropertyType(propertyHandle) == 'listOfProperties':
        listHandle = getPropertyValue(propertyHandle)
        return rrpLib.tpAddPropertyToList(listHandle, addMe)
    else:
        return False

## \brief Set a Property by a string
## \param propertyHandle to a Property instance
## \param value Pointer to string holding the value to assign to the Property, e.g. "0.01" to set a double to 0.01
## \return Returns true if successful, false otherwise
## \ingroup plugin_properties
rrpLib.tpSetPropertyByString.restype = c_bool
rrpLib.tpSetPropertyByString.argtypes = [c_void_p, c_char_p]
def setPropertyByString(PropertyHandle, value):
    return rrpLib.tpSetPropertyByString(PropertyHandle, value.encode('utf-8'))

## \brief Get inforamtion on a Property
## \param propertyHandle Handle to a Property instance
## \return Returns informational text about the Property if successful, None otherwise
## \ingroup plugin_properties
rrpLib.tpGetPropertyInfo.restype = c_char_p
rrpLib.tpGetPropertyInfo.argtypes = [c_void_p]
def getPropertyInfo(propertyHandle):
    return decodeIfBytes(rrpLib.tpGetPropertyInfo(propertyHandle))

## \brief Get a Property value in the form of a string
## \param propertyHandle to a Property instance
## \return Returns the Properties value if successful, None otherwise
## \ingroup plugin_properties
rrpLib.tpGetPropertyValueAsString.restype = c_char_p
rrpLib.tpGetPropertyValueAsString.argtypes = [c_void_p]
def getPropertyValueAsString(propertyHandle):
    return decodeIfBytes(rrpLib.tpGetPropertyValueAsString(propertyHandle))

## \brief Get a handle to a Property value. Such properties could be any type, including a list of Properties.
## Use getlistProperty(propertyaHandle) instead.
## \param propertyHandle to a Property instance
## \return Returns a Handle to the property value if successful, None otherwise
#
## @code
## propertyHandle = telPlugins.getPropertyValueHandle (prophandle)
## @endcode
## \htmlonly  <br/>
## \endhtmlonly
## \ingroup plugin_properties
rrpLib.tpGetPropertyValueHandle.restype = c_void_p
rrpLib.tpGetPropertyValueHandle.argtypes = [c_void_p]
def getPropertyValueHandle(propertyHandle):
    return rrpLib.tpGetPropertyValueHandle(propertyHandle)

## \brief Get the name of a Property
## \param propertyHandle to a Property instance
## \return Returns the Properties name if successful, None otherwise
## \ingroup plugin_properties
rrpLib.tpGetPropertyName.restype = c_char_p
rrpLib.tpGetPropertyName.argtypes = [c_void_p]
def getPropertyName(propertyHandle):
    return decodeIfBytes(rrpLib.tpGetPropertyName(propertyHandle))

## \brief Get the hint text for a Property
## \param propertyHandle to a Property instance
## \return Returns the hint value for a Property if successful, None otherwise
## \ingroup plugin_properties
rrpLib.tpGetPropertyHint.restype = c_char_p
rrpLib.tpGetPropertyHint.argtypes = [c_void_p]
def getPropertyHint(propertyHandle):
    return decodeIfBytes(rrpLib.tpGetPropertyHint(propertyHandle))

## \brief Get the type of a property
## \param propertyHandle to a Property instance
## \return Returns the Properties type as a string if successful, None otherwise
## \ingroup plugin_properties
rrpLib.tpGetPropertyType.restype = c_char_p
rrpLib.tpGetPropertyType.argtypes = [c_void_p]
def getPropertyType(propertyHandle):
    return decodeIfBytes(rrpLib.tpGetPropertyType(propertyHandle))

## \brief Get the Boolean value for a property
## \param propertyHandle to a property instance
## \return Returns a Boolean value. Throws an exception if the property type is not a Boolean
## \ingroup plugin_properties
rrpLib.tpGetBoolProperty.restype = c_bool
rrpLib.tpGetBoolProperty.argtypes = [c_void_p, c_bool_p]
def getBoolProperty (propertyHandle):
    if getPropertyType (propertyHandle) == "bool":
        val = c_bool()
        if rrpLib.tpGetBoolProperty (propertyHandle, byref(val)) == True:
            return val.value
        else:
            raise ('Property value could not be retrieved')
    else:
       raise TypeError ('Property is not a Boolean type')

## \brief Set a boolean property
## \param propertyHandle to a property instance
## \param value to assign to the property.
## \return Returns true if successful, false otherwise
## \ingroup plugin_properties
rrpLib.tpSetBoolProperty.restype = c_bool
rrpLib.tpSetBoolProperty.argtypes = [c_void_p, c_bool]
def setBoolProperty(propertyHandle, value):
    return rrpLib.tpSetBoolProperty (propertyHandle, c_bool(value))

## \brief Get the integer value for a property
## \param propertyHandle to a property instance
## \return Returns an integer value. Throws an exception if the property type is not an integer
## \ingroup plugin_properties
rrpLib.tpGetIntProperty.restype = c_bool
rrpLib.tpGetIntProperty.argtypes = [c_void_p, c_int_p]
def getIntProperty (propertyHandle):
    if getPropertyType (propertyHandle) == "int":
        val = c_int()
        if rrpLib.tpGetIntProperty (propertyHandle, byref(val)) == True:
            return val.value
        else:
            raise ('Property value could not be retrieved')
    else:
       raise TypeError ('Property is not an integer type')

## \brief Set an integer property
## \param propertyHandle to a property instance
## \param value to assign to the property.
## \return Returns true if successful, false otherwise
## \ingroup plugin_properties
rrpLib.tpSetIntProperty.restype = c_bool
rrpLib.tpSetIntProperty.argtypes = [c_void_p, c_int]
def setIntProperty(propertyHandle, value):
    return rrpLib.tpSetIntProperty(propertyHandle, c_int(value))

## \brief Set an unsigned long property
## \param propertyHandle to a property instance
## \param value to assign to the property.
## \return Returns true if successful, false otherwise
## \ingroup plugin_properties
rrpLib.tpSetUnsignedLongProperty.restype = c_bool
rrpLib.tpSetUnsignedLongProperty.argtypes = [c_void_p, c_uint]
def setUnsignedLongProperty(propertyHandle, value):
    return rrpLib.tpSetUnsignedLongProperty(propertyHandle, c_uint(value))

## \brief Get the double value for a property
## \param propertyHandle to a property instance
## \return Returns a double value. Throws an exception if the property type is not a double
## \ingroup plugin_properties
rrpLib.tpGetDoubleProperty.restype = c_bool
rrpLib.tpGetDoubleProperty.argtypes = [c_void_p, c_double_p]
def getDoubleProperty (propertyHandle):
    if getPropertyType (propertyHandle) == "double":
        val = c_double()
        if rrpLib.tpGetDoubleProperty (propertyHandle, byref(val)) == True:
            return val.value
        else:
            raise ('Property value could not be retrieved')
    else:
       raise TypeError ('Property is not a double type')

## \brief Set the value for a double property
## \param propertyHandle Is a property instance
## \param value to assign to the property.
## \return Returns true if successful, false otherwise
## \ingroup plugin_properties
rrpLib.tpSetDoubleProperty.restype = c_bool
rrpLib.tpSetDoubleProperty.argtypes = [c_void_p, c_double]
def setDoubleProperty(propertyHandle, value):
    return rrpLib.tpSetDoubleProperty(propertyHandle, c_double(value))

## \brief Get the string value for a property
## \param propertyHandle to a property instance
## \return Returns a string value. Throws an exception if the property type is not a string
## \ingroup plugin_properties
rrpLib.tpGetStringProperty.restype = c_bool
rrpLib.tpGetStringProperty.argtypes = [c_void_p, c_char_p]
def getStringProperty (propertyHandle):
    if getPropertyType (propertyHandle) == "string" or getPropertyType (propertyHandle) == "std::string":
        val = c_char_p()
        if rrpLib.tpGetStringProperty (propertyHandle, byref(val)) == True:
            return str(val.value)
        else:
            raise ('Property value could not be retrieved')
    else:
       raise TypeError ('Property is not a string type')

## \brief Set a string property
## \param propertyHandle Handle to a Property instance
## \param value Value to assign to the property.
## \return Returns true if successful, false otherwise
## \ingroup plugin_properties
rrpLib.tpSetStringProperty.restype = c_bool
rrpLib.tpSetStringProperty.argtypes = [c_void_p, c_char_p]
def setStringProperty(propertyHandle, value):
    r = rrpLib.tpSetStringProperty(propertyHandle, c_char_p(value.encode('utf-8')))
    if not r:
        raise TypeError('Failed to set string property')
    return r

## \brief Get the list value for a property
## \param propertyHandle to a property instance
## \return Returns a handle to a ListProperty. Throws an exception of the property type is not a list of properties
## \ingroup plugin_properties
rrpLib.tpGetListProperty.restype = c_bool
rrpLib.tpGetListProperty.argtypes = [c_void_p, c_void_p]
def getListProperty (propertyHandle):
    if getPropertyType (propertyHandle) == "listOfProperty":
        return getPropertyValueHandle(propertyHandle)
    else:
       raise TypeError ('Property is not a list type')

## \brief Set a list Property
## \param propertyHandle to a Property instance
## \param value Value to assign to the property (must be a handle to a Property of listOfProperties.
## \return Returns true if successful, false otherwise
## \ingroup plugin_properties
rrpLib.tpSetListProperty.restype = c_bool
rrpLib.tpSetListProperty.argtypes = [c_void_p, c_void_p]
def setListProperty(propertyHandle, value):
    handle = getPropertyValueHandle(value)
    return rrpLib.tpSetListProperty(propertyHandle, c_void_p(handle))

## \brief Get the value of a telluriumData property
## \param propertyHandle A Handle to a property
## \return Returns the value of the property if succesful, None otherwise
## \ingroup plugin_properties
def getTelluriumDataProperty(propertyHandle):
        return getPropertyValue(propertyHandle)

## \brief Set a telluriumData property
## \param propertyHandle Handle to a Property instance
## \param value Value to assign to the property (must be a handle to telluriumData.
## \return Returns true if successful, false otherwise
## \ingroup plugin_properties
rrpLib.tpSetTelluriumDataProperty.restype = c_bool
rrpLib.tpSetTelluriumDataProperty.argtypes = [c_void_p, c_void_p]
def setTelluriumDataProperty(propertyHandle, value):
    return rrpLib.tpSetTelluriumDataProperty(propertyHandle, c_void_p(value))

## \brief Get the value of a property.
## \param propertyHandle A Handle to a property
## \return Returns the value of the property if succesful, None otherwise
## \ingroup plugin_properties
## \todo rename to getPropertyValue (?)
def getProperty(propertyHandle):
    paraType = getPropertyType(propertyHandle)

    if paraType == 'bool':
        paraVoidPtr = getPropertyValueHandle(propertyHandle)
        ptr = cast(paraVoidPtr, POINTER(c_bool))
        return ptr[0]

    if paraType == 'int':
        paraVoidPtr = getPropertyValueHandle(propertyHandle)
        ptr = cast(paraVoidPtr, POINTER(c_int))
        return ptr[0]

    if paraType == 'double':
        paraVoidPtr = getPropertyValueHandle(propertyHandle)
        ptr = cast(paraVoidPtr, POINTER(c_double))
        return ptr[0]

    if paraType == 'std::string':
        return getPropertyValueAsString(propertyHandle)

    if paraType == 'string':
        return getPropertyValueAsString(propertyHandle)

    if paraType == 'stringList':
        return getPropertyValueAsString(propertyHandle)

    if paraType == 'listOfProperties':
        return getPropertyValueHandle(propertyHandle)

    if paraType == 'vector<int>':
        return getPropertyValueAsString(propertyHandle)

    if paraType == 'vector<double>':
        return getPropertyValueAsString(propertyHandle)

    if paraType == 'matrix':
        return getPropertyValueHandle(propertyHandle)

    if paraType == 'telluriumData': #The value of this is a handle
        ptr = getPropertyValueHandle(propertyHandle)
        return ptr
    else:
       raise TypeError ('Property is not a string type: ', paraType)

## \brief Get the value of a property.
## \param propertyHandle A Handle to a property
## \return Returns the value of the property if succesful, None otherwise
## \note Legacy function. Use getProperty() instead.
## \ingroup plugin_properties
def getPropertyValue(propertyHandle):
    paraType = getPropertyType(propertyHandle)
    if paraType == 'bool':
        paraVoidPtr = getPropertyValueHandle(propertyHandle)
        ptr = cast(paraVoidPtr, POINTER(c_bool))
        return ptr[0]

    if paraType == 'int':
        paraVoidPtr = getPropertyValueHandle(propertyHandle)
        ptr = cast(paraVoidPtr, POINTER(c_int))
        return ptr[0]

    if paraType == 'double':
        paraVoidPtr = getPropertyValueHandle(propertyHandle)
        ptr = cast(paraVoidPtr, POINTER(c_double))
        return ptr[0]

    if paraType == 'std::string':
        return getPropertyValueAsString(propertyHandle)

    if paraType == 'string':
        return getPropertyValueAsString(propertyHandle)

    if paraType == 'listOfProperties':
        return getPropertyValueHandle(propertyHandle)

    if paraType == 'matrix':
        return getPropertyValueHandle(propertyHandle)

    if paraType == 'telluriumData': #The value of this is a handle
        ptr = getPropertyValueHandle(propertyHandle)
        return ptr
    else:
       raise TypeError ('Property is not a string type')

## \brief Convert tellurium data to Numpy data
## \param telDataHandle A handle to a tellurium data object
## \return Returns a numpy data object
## \ingroup utilities
def getNumpyData(telDataHandle):
    #colHeader = getTelluriumDataColumnHeader(telDataHandle)
    rowCount = rrpLib.tpGetTelluriumDataNumRows(telDataHandle)
    colCount = rrpLib.tpGetTelluriumDataNumCols(telDataHandle)
    resultArray = np.zeros([rowCount, colCount])
    for row in range(rowCount):
        for col in range(colCount):
                val = c_double()
                if rrpLib.tpGetTelluriumDataElement(telDataHandle, row, col, byref(val)) == True:
                    resultArray[row, col] = val.value
                else:
                    print("problem")
    #resultArray = np.append(resultArray, colHeader.split(","))
    #Not sure how to append the col names.
    return resultArray

def plotTelluriumData(data, colHeaders):
    nrCols = data.shape[1]
    nrRows = data.shape[0]

    if colHeaders == None or len(colHeaders) < 1:
        print("Bad Data ColumnHeader")
        return
    xlbl = colHeaders[0]
    nrOfSeries = nrCols -1
    x = data[:,0]

    for serie in range(nrOfSeries):
        ySeries = np.zeros([nrRows])
        ySeries = data[:,serie + 1]
        plt.plot(x, ySeries, "", label=colHeaders[serie +1])

    plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
    plt.xlabel(xlbl)
    plt.show()

def plotBifurcationData(data, colHeaders, bfPoints, bfLabels, legend=True, cmap=None,
                        xlabel=None, ylabel=None, selections=None):
    nrCols = data.shape[1]
    nrRows = data.shape[0]
    from tellurium import plot, show
    from tellurium.plotting import plot_text

    if colHeaders == None or len(colHeaders) < 1:
        print("Bad Data ColumnHeader")
        return
    if xlabel == None:
        xlbl = colHeaders[0]
    else:
        xlbl = xlabel
    nrOfSeries = nrCols -1
    x = data[:,0]
    ccmap = []
    
    if cmap == None:
        color = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i in range(len(color)):
            ccmap.append(matplotlib.colors.rgb2hex(color[i]))
    else:
        if type(cmap) == list:
            for i in range(len(cmap)):
                ccmap.append(matplotlib.colors.rgb2hex(cmap[i]))
        elif type(cmap) == str:
            try:
                color = matplotlib.cm.get_cmap(cmap)
                for i in range(len(color)):
                    ccmap.append(matplotlib.colors.rgb2hex(color[i]))
            except:
                try:
                    import seaborn
                    ccmap = seaborn.color_palette(cmap).as_hex()
                except:
                    raise Exception("Cannot find colormap named " + str(cmap))
        else:
            try:
                ccmap = cmap.as_hex()
            except:
                raise Exception("Cannot process colormap. Put either a list of RGB ",
                            "tuples or name of colormap")

    previousLbl = ''
    cm_i = 0
    plotargs = {'xtitle': xlbl}
    if ylabel != None:
        plotargs['ytitle'] = ylabel
    for serie in range(nrOfSeries):
        if selections is not None and not colHeaders[serie + 1] in selections:
            continue
        labelNr = 0
        ySeries = np.zeros([nrRows])
        ySeries = data[:,serie + 1]
        xIndx = 0

        for label in bfLabels:
            xPtn = bfPoints[labelNr] - 1
            xSegment = x[xIndx:xPtn]
            ySegment = ySeries[xIndx:xPtn]
            if label == 'EP':
                if xIndx == 0 :
                    plot(xSegment, ySegment, name=colHeaders[serie + 1], tag=colHeaders[serie + 1], show=False, **plotargs)
                else:
                    plot(xSegment, ySegment, name=colHeaders[serie + 1], tag=colHeaders[serie + 1], show=False, **plotargs)
            elif label == 'LP':
                #Check Previous label
                if previousLbl == 'LP':
                    plot(xSegment, ySegment, dash=['dash'], color=['black'], alpha=[0.5], showlegend=False, show=False, **plotargs)
                else:
                    plot(xSegment, ySegment, name=colHeaders[serie + 1], tag=colHeaders[serie + 1], show=False, **plotargs)
            elif label == 'HB':
                #print('plot {} vs {}'.format(xSegment,ySegment))
                plot(xSegment, ySegment, name=colHeaders[serie + 1], tag=colHeaders[serie + 1], show=False, **plotargs)

            xIndx = xPtn
            labelNr = labelNr + 1
            previousLbl = label
        cm_i += 1

    #Plot bifurcation labels
    for serie in range(nrOfSeries):
        if selections is not None and not colHeaders[serie + 1] in selections:
            continue
        labelNr = 0
        ySeries = data[:,serie + 1]
        for label in bfLabels:
            if label != 'EP':
                xPtn = bfPoints[labelNr] - 1
                xCoord = x[xPtn]
                yCoord = ySeries[xPtn]
                plot_text(xCoord, yCoord, text=label, show=False)
            labelNr = labelNr + 1

    show()


## \brief Get column header in tellurium data
## \param telDataHandle A handle to a tellurium data object
## \return Returns a numpy data object
## \ingroup utilities
rrpLib.tpGetTelluriumDataColumnHeader.restype = c_void_p
rrpLib.tpGetTelluriumDataColumnHeader.argtypes = [c_void_p]
def getTelluriumDataColumnHeader(telDataHandle):
    # http://stackoverflow.com/questions/13445568/python-ctypes-how-to-free-memory-getting-invalid-pointer-error
    hdr = rrpLib.tpGetTelluriumDataColumnHeader(telDataHandle)

    if hdr:
        res = decodeIfBytes(ctypes.cast(hdr, ctypes.c_char_p).value)
        return res.split(',')
    else:
        return None

## \brief Get column header item by index
## \param telDataHandle A handle to a tellurium data object
## \param index Index of requested column name
## \return Returns the column name on success, or None on failure
## \ingroup utilities
rrpLib.tpGetTelluriumDataColumnHeaderByIndex.restype = c_char_p
rrpLib.tpGetTelluriumDataColumnHeaderByIndex.argtypes = [c_void_p, c_int]
def getTelluriumDataColumnHeaderByIndex(telDataHandle, index):
    return decodeIfBytes(rrpLib.tpGetTelluriumDataColumnHeaderByIndex(telDataHandle, index))

## \brief Set column header in tellurium data
## \param telDataHandle A handle to a tellurium data object
## \param colHeader A string containing the column header as comma separated values (spaces allowed too)
## \return True or false indicating success
## \ingroup utilities
rrpLib.tpSetTelluriumDataColumnHeader.restype = c_bool
rrpLib.tpSetTelluriumDataColumnHeader.argtypes = [c_void_p, c_char_p]
def setTelluriumDataColumnHeader(telDataHandle, hdr):
    return rrpLib.tpSetTelluriumDataColumnHeader(telDataHandle, hdr.encode('utf-8'))

## \brief Set column header by index in tellurium data
## \param telDataHandle A handle to a tellurium data object
## \param index Index for column name to change
## \param colName A string containing the column name
## \return True or false indicating success
## \ingroup utilities
rrpLib.tpSetTelluriumDataColumnHeaderByIndex.restype = c_bool
rrpLib.tpSetTelluriumDataColumnHeaderByIndex.argtypes = [c_void_p, c_int, c_char_p] #FIXME: Out of order
def setTelluriumDataColumnHeaderByIndex(telDataHandle, index, hdr):
    return rrpLib.tpSetTelluriumDataColumnHeaderByIndex(telDataHandle, hdr.encode('utf-8'), index)

## \brief Get Tellurium data element at row,col
## \param telDataHandle A handle to a tellurium data object
## \return Returns the numeric value at row,col
## \ingroup utilities
rrpLib.tpGetTelluriumDataElement.restype = c_bool
rrpLib.tpGetTelluriumDataElement.argtypes = [c_void_p, c_int, c_int, c_double_p]
def getTelluriumDataElement(telDataHandle, row, col):
    val = c_double()
    if rrpLib.tpGetTelluriumDataElement(telDataHandle, row, col, byref(val)) == True:
        return val.value
    else:
        raise RuntimeError('Failed retrieving data at (row, col) = ({}'.format(row) + ', {}'.format(col) + ')')

## \brief Set Tellurium data element at row,col
## \param telDataHandle A handle to a tellurium data object
## \return Returns the numeric value at row,col
## \ingroup utilities
rrpLib.tpSetTelluriumDataElement.restype = c_bool
rrpLib.tpSetTelluriumDataElement.argtypes = [c_void_p, c_int, c_int, c_double]
def setTelluriumDataElement(telDataHandle, row, col, number):
    return rrpLib.tpSetTelluriumDataElement(telDataHandle, row, col, c_double(number))

## \brief Get Tellurium data element at row,col
## \param telDataHandle A handle to a tellurium data object
## \return Returns the numeric value at row,col
## \ingroup utilities
rrpLib.tpGetTelluriumDataWeight.restype = c_bool
rrpLib.tpGetTelluriumDataWeight.argtypes = [c_void_p, c_int, c_int, c_double_p]
def getTelluriumDataWeight(telDataHandle, row, col):
    val = c_double()
    if rrpLib.tpGetTelluriumDataWeight(telDataHandle, row, col, byref(val)) == True:
        return val.value
    else:
        raise RuntimeError('Failed retrieving weight data at (row, col) = ({}'.format(row) + ', {}'.format(col) + ')')

## \brief Set Tellurium data element at row,col
## \param telDataHandle A handle to a tellurium data object
## \return Returns the numeric value at row,col
## \ingroup utilities
rrpLib.tpSetTelluriumDataWeight.restype = c_bool
rrpLib.tpSetTelluriumDataWeight.argtypes = [c_void_p, c_int, c_int, c_double]
def setTelluriumDataWeight(telDataHandle, row, col, number):
    return rrpLib.tpSetTelluriumDataWeight(telDataHandle, row, col, c_double(number))

## \brief Get number of rows in a tellurium data object
## \param telDataHandle A handle to a tellurium data object
## \return Returns number of rows in the data object
## \ingroup utilities
rrpLib.tpGetTelluriumDataNumRows.restype = c_int
rrpLib.tpGetTelluriumDataNumRows.argtypes = [c_void_p]
def getTelluriumDataNumRows(telDataHandle):
    return rrpLib.tpGetTelluriumDataNumRows(telDataHandle)

## \brief Get number of columns in a tellurium data object
## \param telDataHandle A handle to a tellurium data object
## \return Returns number of cols in the data object
## \ingroup utilities
rrpLib.tpGetTelluriumDataNumCols.restype = c_int
rrpLib.tpGetTelluriumDataNumCols.argtypes = [c_void_p]
def getTelluriumDataNumCols(telDataHandle):
    return rrpLib.tpGetTelluriumDataNumCols(telDataHandle)

## \brief Write TelluriumData to a file
## \param telDataHandle A handle to roadunnerdata
## \param fName Name of output file, including path. If no path is given, the file is written to the
## current working directory
## \return Returns True or false indicating result
## \ingroup utilities
rrpLib.tpWriteTelluriumDataToFile.restype = c_bool
rrpLib.tpWriteTelluriumDataToFile.argtypes = [c_void_p, c_char_p]
def writeTelluriumData(telDataHandle, fName):
    return rrpLib.tpWriteTelluriumDataToFile(telDataHandle, fName.encode('utf-8'))

## \brief Read TelluriumData from a file
## \param telDataHandle A handle to roadunnerdata
## \param fName Name of input file, including path. If no path is given, the file is read
## in current working directory
## \return Returns True or false indicating result
## \ingroup utilities
rrpLib.tpReadTelluriumDataFromFile.restype = c_bool
rrpLib.tpReadTelluriumDataFromFile.argtypes = [c_void_p, c_char_p]
def readTelluriumData(telDataHandle, fName):
    return rrpLib.tpReadTelluriumDataFromFile(telDataHandle, fName.encode('utf-8'))

## \brief Create a TelluriumData object
## \param rows Number of rows in the data to be created
## \param cols Number of columns in the data to be created
## \return Returns a handle to Tellurium data if successful, None otherwise
## \note Use the freeTelluriumData to free memory allocated
## \ingroup utilities
rrpLib.tpCreateTelluriumData.restype = c_void_p
rrpLib.tpCreateTelluriumData.argtypes = [c_int, c_int, c_char_p]
def createTelluriumData(rows, cols):
    #Create a Tellurium data object
    #Create a column header
    nrs = range(cols)
    col_hdr = str(nrs).strip('[]')
    return rrpLib.tpCreateTelluriumData(rows, cols, col_hdr.encode('utf-8'))

## \brief Create TelluriumData from a file
## \param fName Name of input file, including path. If no path is given, the file is read
## in current working directory
## \return Returns a handle to Tellurium data if successful, None otherwise
## \note Use the freeTelluriumData to free memory allocated by the returned data
## \ingroup utilities
def createTelluriumDataFromFile(fName):
    #Create a Tellurium data object
    telDataHandle = rrpLib.tpCreateTelluriumData(0,0, None)
    if rrpLib.tpReadTelluriumDataFromFile(telDataHandle, fName) == False:
        print('Failed to read data')
    return telDataHandle

## \brief Check if tellurium data has weights allocated
## \param dataHandle Handle to a tellurium data object
## \return Returns true or false indicating if the data object has weights or not
## \ingroup utilities
rrpLib.tpHasWeights.restype = c_bool
rrpLib.tpHasWeights.argtypes = [c_void_p, c_bool_p]
def hasWeights(dataHandle):
    hasIt = c_bool()
    if not rrpLib.tpHasWeights(dataHandle, byref(hasIt)):
        raise RuntimeError(getLastError())
    else:
        return hasIt.value

## \brief Allocate weights for tellurium data object
## \param dataHandle Handle to a tellurium data object
## \return Returns true or false indicating if allocating weights were successful or not
## \ingroup utilities
rrpLib.tpAllocateWeights.restype = c_bool
rrpLib.tpAllocateWeights.argtypes = [c_void_p, c_bool_p]
def allocateWeights(dataHandle):
    success = c_bool()
    if not rrpLib.tpAllocateWeights(dataHandle, byref(success)):
        raise RuntimeError(getLastError())
    else:
        return success.value

def getText(fName):
    file = open(fName, 'r')
    return file.read()


## \brief Reads the entire contents of a file and returns it as a string
##@code
## str = readAllText ("mytextfile.txt")
##@endcode
def readAllText(fName):
    file = open(fName, 'r')
    str = file.read()
    file.close()
    return str

## \brief Free TelluriumData
## \param dataHandle Handle to a tellurium data object
## \return Returns True or false indicating result
## \ingroup utilities
rrpLib.tpFreeTelluriumData.restype = c_bool
rrpLib.tpFreeTelluriumData.argtypes = [c_void_p]
def freeTelluriumData(telDataHandle):
    return rrpLib.tpFreeTelluriumData(telDataHandle)

## \brief Get last (API) error. This returns the last error if any.
## \return Returns a string with an error success, None otherwise
## \ingroup utilities
rrpLib.tpGetLastError.restype = c_char_p
def getLastError():
    return decodeIfBytes(rrpLib.tpGetLastError())

## \brief Unload the plugins api shared library
## \ingroup utilities
def unLoadAPI():
    if hasattr(_ctypes, "FreeLibrary"):
        _ctypes.FreeLibrary(rrpLib._handle)
    if hasattr(_ctypes, "dlclose"):
        _ctypes.dlclose(rrpLib._handle)



######### MATRIX WRAPPERS
## \brief Return unerlying data array for a Tellurium matrix
## \return Returns a ctypes pointer to double object on success, None otherwise
## \ingroup utilities
rrpLib.tpGetDataArray.restype = c_int
rrpLib.tpGetDataArray.argtypes = [c_void_p]
def getDataArray(matrixH):
    mat = rrpLib.tpGetDataArray(matrixH)
    if mat:
        return mat
    else:
        return None

rrpLib.tpGetMatrixNumRows.restype = c_int
rrpLib.tpGetMatrixNumRows.argtypes = [c_void_p]
def getMatrixNumRows(matrixH):
    rSize = rrpLib.tpGetMatrixNumRows(matrixH)
    if rSize != -1:
        return rSize
    else:
        return None

rrpLib.tpGetMatrixNumCols.restype = c_int
rrpLib.tpGetMatrixNumCols.argtypes = [c_void_p]
def getMatrixNumCols(matrixH):
    cSize = rrpLib.tpGetMatrixNumCols(matrixH)
    if cSize != -1:
        return cSize
    else:
        return None

rrpLib.tpSetLogLevel.restype = c_bool
rrpLib.tpSetLogLevel.argtypes = [c_char_p]
def setLogLevel(lvl):
    return rrpLib.tpSetLogLevel(lvl.encode('utf-8'))

##\mainpage Plugins for Tellurium
#\section Introduction
#The Tellurium plugin library exposes a simple framework for adding functionality to the RoadRunner core, by means of
#external plugins.
#The code fragment below shows briefly how to load plugins, check for plugins, and use an individual plugin.
#
#\include telGetPluginInformation.py
#
# The above code produces something like the following output (depends on installed plugins):
#@code
##>>>
##Number of Plugins: 3
##Plugin Names: ['AddNoise', 'ChiSquare', 'Levenberg-Marquardt', 'SBMLModel']
##==========================================
##Name: 'AddNoise'
##Author: 'Totte Karlsson'
##Copyright: 'Totte Karlsson, Herbert Sauro, Systems Biology, UW 2012-2014'
##Version: '1.0'
##Category: 'Signal Processing'
##Description:'The AddNoise plugin adds Gaussian noise to synthetic data. The amount of noise is controlled by the plugins Sigma property. Specifically, noise is generated for each single data value, with a probability corresponding to a Gaussian distribution centered around the value, and with a variance equal to (sigma^2). The Plugin accepts Tellurium data as input, in the "InputData" property. Currently only Gaussian noise is supported. The progress of the application of noise can be read in the Progress property. Noise will not be generated onto the first column of data, if its column label is equal to "Time", (not case sensitive). The AddNoise plugin was developed at the University of Washington by Totte Karlsson, 2012-2014.'
##Parameters: ['NoiseType', 'Sigma', 'InputData', 'Progress']
##==========================================
##Name: 'ChiSquare'
##Author: 'Totte Karlsson'
##Copyright: 'Totte Karlsson, Herbert Sauro, Systems Biology, UW 2012-2014'
##Version: '0.8'
##Category: 'Misc'
##Description:None
##Parameters: ['ExperimentalData', 'ModelData', 'NrOfModelParameters', 'ChiSquare', 'ReducedChiSquare']
##==========================================
##Name: 'Levenberg-Marquardt'
##Author: 'Totte Karlsson'
##Copyright: 'Totte Karlsson, Herbert Sauro, Systems Biology, UW 2012-2014'
##Version: '0.8'
##Category: 'Fitting'
##Description:'The Levenberg-Marquardt plugin is used to fit a proposed SBML models parameters to experimental data. The current implementation is based on the lmfit C library by Joachim Wuttke. The Plugin has numerous parameters for fine tuning the algorithm. See the embedded PDF for more information. '
##Parameters: ['SBML', 'ExperimentalData', 'FittedData', 'Residuals', 'InputParameterList', 'OutputParameterList', 'ConfidenceLimits', 'ExperimentalDataSelectionList', 'FittedDataSelectionList', 'Norm', 'Norms', 'NrOfIter', 'StandardizedResiduals', 'NormalProbabilityOfResiduals', 'ChiSquare', 'ReducedChiSquare', 'ftol', 'xtol', 'gtol', 'epsilon', 'stepbound', 'patience']
##==========================================
##Name: 'SBMLModel'
##Author: 'Totte Karlsson'
##Copyright: 'Totte Karlsson, Herbert Sauro, Systems Biology, UW 2012-2014'
##Version: '1.0'
##Category: 'Examples'
##Description:'The SBMLModel plugin exposes one property containing data, as a string, for an ExampleModel. The ExampleModel plugin was developed at the University of Washington by Totte Karlsson, 2012-2014.'
##Parameters: ['Model']
##>>>
#@endcode
#    \section plugins_overview Overview
#    The Tellurium Plugin API is centered around three important concepts:
#    - A Plugin Manager
#    - Plugins
#    - Plugin Properties
#
#    \section plugins_usage How to use plugins
#    A typical use case of the Plugin API may be as follows:
#
#    -# Client creates a PluginManager.
#    -# Client load plugins using the Plugin Manager.
#    -# Client get a handle to a plugin.
#    -# Client get a handle to a specific property in the plugin.
#    -# Client set the value of the property.
#    -# Client excutes the plugin.
#    -# Client retrieve the value of a plugins property, e.g. a "result" property.
#   \subsection pluginEvents PluginEvent functionality
# In addition to data properties that communicate data between a client and the plugin, the framework also support for a variety of  plugin event functions.
# In short, an event is a regular function that is defined and implemented by the client of a plugin, but executed from within the plugin, during the plugins
# execution.
#
# A single plugin may support of up to three event functions. The intended use of these functions are to signal the events of the following:
#   -# Plugin Initialization
#   -# Plugin Progress
#   -# Plugin Finalization
#
# Each event function support up to two opaque data properties. The plugin documentation needs to provide the exact type of these arguments.
# In it simplest form, a plugin may choose to define an event function taking no arguments at all.
# Below are listed a few properties, characteristics of events in the Tellurium Plugin framework.
#   -# A plugin event is a regular function defined by the client of the plugin.
#   -# A plugin event function do not return any value.
#   -# The type and number of arguments needed in the plugin event is defined by the plugin (see plugin docs).
#   -# Plugin events are assigned to the plugin before a plugins execute function.
#   -# Assigning events is optional. A plugins internal work should not be affected wether an event is assigned or not.
#   -# Plugin events are blocking functions. If the work in a plugin is executed in a thread, see executeEx, the plugin event
#   will be executed in the same thread as the plugin worker. Depending on your environment and if the plugin event function is executed in a separate
# thread, regular use of thread synchronization measuress may be needed in order to not create an unstable system.
#
#   See the examples page that provide example code on how to use plugins, properties and event functions.
#    \section plugins_writing How to write plugins
#    \note Writing plugins in Python is not yet supported
#
# \section main_section Using telPlugins.py
# In order to use this wrapper, telplugins need to be installed properly. Typically all neeeded to be done is to run the install_tellurium_plugins.py script, available in the
# root folder of the release.
#
# \defgroup plugin_manager Plugin Manager
# \brief Plugin Manager Library Functions
# The Plugin manager loads and manages one or more plugins. Handles to individual plugins are obtained
# by the getFirstPlugin, getNextPlugin etc functions, or the getPlugin(pluginName) function.
#
# The basic approach to using the plugin system is:
#
# 1. Create a plugin manager first
#
# 2. Load the plugins using loadPlugins (all) or loadPlugin (specific one)
#
# 3. Obtain the plugin handle using getPlugin or directly from loadPlugin (singular)
#
# 4. Using the plugin handle, set values to the plugin properties
#
# 5. Run the plugin method execute(pluginHandle)
#
# 6. Retrieve results from plugin properties
# \defgroup plugins Plugin Functions
# \brief Functions operating on Plugin Handles.
#
# \defgroup plugin_properties Plugin Properties
# \brief Plugins Property related functions
# The plugin system supports property objects, these objects contain a variety of information about a given property, these include:
# the name, value, type, hint, and a description. The following types are currently supported, Booleans, integers, doubles, strings,
# list of property objects and the roadRunner data array format. Properties can also be grouped into convenient categories which can be useful
# for GUI applications. Every plugin exposes as a set of properties than can be inspected and set by a host application.
# The list of plugin properties will be called the pluginProperties. Within the pluginProperties are individual property entries.
# As already aluded to, these property entries can store a variety of different data types, incuding additional lists of properties. Such lists
# are popoluated by creating new properties using the createProperty method and then added to the list using the addPropertyToList method.
# In the followng a propertyHandle points to a single property object.
#
# \defgroup utilities Utility Functions
# \brief Functions to help and assist in the use of the Plugins framework

# \defgroup examples Python Example Scripts
# \brief Scripts illuminating concepts regarding Tellurium Plugins

## \example telGetPluginInformation.py
## This example shows
## -# How to create a plugin manager
## -# Get Plugin Names
## -# Get a handle to a plugin
## -# Obtain some info from the plugin

## \example telPluginProperties.py
## This example shows
## -# Get a handle to a property in a Plugin
## -# Obtain some info about the property
## -# Getting the value of the property
## -# Setting the value of the property

## \example telEvents.py
## This example shows
## -# How to define Python event functions and passing them to a plugin

## \example telPluginDocumentation.py
## This example shows
## -# How to extract a plugins embedded PDF documentation
