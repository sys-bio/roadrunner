import roadrunner
import rrplugins


#Create a plugin manager
pm = rrplugins.createPluginManager()

def pluginStarted(ignore1, ignore2):
    print('The plugin was started')

def pluginIsFinished(ignore1, ignore2):
    print('The plugin did finish')

#Create a roadrunner instance
rr = roadrunner.RoadRunner()

#Retrieve a SBML model from plugin        
modelPlugin = rrplugins.Plugin("tel_test_model")        
sbmlModel= modelPlugin.Model   

#Check if model file exists
rr.load(sbmlModel)

timeStart = 0
timeEnd = 10
numPoints = 500
data = rr.simulate(timeStart, timeEnd, numPoints)

#Load the 'noise' plugin in order to add some noise to the data
pluginHandle = rrplugins.loadPlugin(pm, "tel_add_noise")       

#get parameter for noise 'size'
sigmaHandle = rrplugins.getPluginProperty(pluginHandle, "Sigma")
aSigma = rrplugins.getPropertyValueAsString(sigmaHandle)
  
#set size of noise
rrplugins.setProperty(sigmaHandle, 1.e-5)

cb_func1 =  rrplugins.NotifyEvent(pluginStarted)
rrplugins.assignOnStartedEvent(pluginHandle,  cb_func1)

#The ID of the plugin is passed as the last argument in the assignOnProgressEvent. 
#The plugin ID is later on retrieved in the plugin Event handler, see above
theId = id(pluginHandle)
# rrplugins.assignOnProgressEvent(pluginHandle, progressEvent, theId)
   
cb_func3 =  rrplugins.NotifyEvent(pluginIsFinished)
rrplugins.assignOnFinishedEvent(pluginHandle, cb_func3)

# Get the dataseries from roadrunner    
d = rrplugins.getDataSeries (data)

#Pass data from roadrunner to the plugin         
rrplugins.setPluginProperty(pluginHandle, "InputData", d._data) #Is _data OK here?

#Execute the noise plugin which will add some noise to the (internal) data
rrplugins.executePluginEx(pluginHandle)

#Retrieve data from plugin    
pluginData = rrplugins.getPluginProperty(pluginHandle, "InputData")
rrData = rrplugins.getNumpyData(rrplugins.getProperty(pluginData))
colNames = rrplugins.getTelluriumDataColumnHeader(rrplugins.getProperty(pluginData))
rrplugins.plotTelluriumData(rrData, colNames)
rrplugins.unLoadPlugins(pm)

print("done")

