import os
import roadrunner
import matplotlib.pyplot as plot
import numpy
import ctypes
from teplugins import *


#Create a plugin manager
pm = createPluginManager()

def pluginStarted():
    print 'The plugin was started'

def pluginIsProgressing(val):    
    pluginHandle = cast(val, ctypes.py_object).value    
    prop = getPluginProperty(pluginHandle, "Progress")
    print '\nPlugin progress:' + `getPropertyValue(prop)` +' %'

def pluginIsFinished():
    print 'The plugin did finish'

try:
    #Create a roadrunner instance
    rr = roadrunner.RoadRunner()

    #Retrieve a SBML model from plugin        
    modelPlugin = Plugin("tel_test_model")        
    sbmlModel= modelPlugin.Model   
    
    #Check if model file exists
    rr.load(sbmlModel)
    
    timeStart = 0
    timeEnd = 10
    numPoints = 500
    data = rr.simulate(timeStart, timeEnd, numPoints)
    
    #Load the 'noise' plugin in order to add some noise to the data
    pluginHandle = loadPlugin(pm, "tel_add_noise")       
    
    #get parameter for noise 'size'
    sigmaHandle = getPluginProperty(pluginHandle, "Sigma")
    aSigma = getPropertyValueAsString(sigmaHandle)
      
    #set size of noise
    setProperty(sigmaHandle, 1.e-5)
    
    cb_func1 =  NotifyEvent(pluginStarted)
    assignOnStartedEvent(pluginHandle,  cb_func1)
    
    progressEvent =  NotifyEventEx(pluginIsProgressing)
    
    #The ID of the plugin is passed as the last argument in the assignOnProgressEvent. 
    #The plugin ID is later on retrieved in the plugin Event handler, see above
    theId = id(pluginHandle)
    assignOnProgressEvent(pluginHandle, progressEvent, theId)
       
    cb_func3 =  NotifyEvent(pluginIsFinished)
    assignOnFinishedEvent(pluginHandle, cb_func3)
    
    # Get the dataseries from roadrunner    
    d = getDataSeries (data)
    
    #Pass data from roadrunner to the plugin         
    setPluginProperty(pluginHandle, "InputData", d._data) #Is _data OK here?
    
    #Execute the noise plugin which will add some noise to the (internal) data
    executePluginEx(pluginHandle)
    
    #Retrieve data from plugin    
    pluginData = getPluginProperty(pluginHandle, "InputData")
    rrData = getNumpyData(getProperty(pluginData))
    colNames = getTelluriumDataColumnHeader(getProperty(pluginData))
    plotTelluriumData(rrData, colNames)
    unLoadPlugins(pm)
    
    print "done"
    
except Exception as e:    
    print 'Exception: ' + `e`    