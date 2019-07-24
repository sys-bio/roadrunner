from telPlugins_CAPI import *
import ctypes
import telPlugins as tel

#Acquire the lmfit plugin
lm = tel.Plugin("tel_levenberg_marquardt")

##========== EVENT FUNCTIONS ==================
def pluginIsProgressing(msg, lmP):
    # The plugin don't know what a python object is.
    # We need to cast it here, to a proper python object
    lmObject = cast(lmP, ctypes.py_object).value
    print 'Iterations = ' + `lmObject.getProperty("NrOfIter")` + '\tNorm = ' + `lmObject.getProperty("Norm")`

progressEvent =  NotifyPluginEvent(pluginIsProgressing)
theId = id(lm)

#The identity of the plugin is passed as the last argument in the assignOnProgressEvent. 
#The plugin ID is later on retrieved in the plugin Event handler, see above
assignOnProgressEvent(lm.plugin, progressEvent, None, theId)

#Setup lmfit properties.
lm.setProperty("ftol", 1.e-20)
experimentalData = lm.loadDataSeries ("testData.dat")

lm.setProperty("ExperimentalData", experimentalData)
lm.setProperty("SBML", lm.readAllText("sbml_test_0001.xml"))

# Add the parameters that we're going to fit and a initial 'start' value
lm.setProperty("InputParameterList", ["k1", 5.2])
lm.setProperty("FittedDataSelectionList", "[S1] [S2]")
lm.setProperty("ExperimentalDataSelectionList", "[S1] [S2]")

# Execute the plugin
lm.execute()

## ----------- Uncomment the following to run in a thread (and comment out the above line).
#res = lm.executeEx(True)
#while isPluginWorking(lm.plugin):
#    print 'Working'

print 'Minimization finished. \n==== Result ==== ' 
print getPluginResult(lm.plugin)

# Get the experimental data as a numpy array
experimentalData = experimentalData.AsNumpy

# Get the fitted and residual data
fittedData = lm.getProperty ("FittedData").AsNumpy
residuals  = lm.getProperty ("Residuals").AsNumpy

tel.plot(fittedData[:,[0,1]],       "blue", "-",    "",    "S1 Fitted")
tel.plot(fittedData[:,[0,2]],       "blue", "-",    "",    "S2 Fitted")
tel.plot(residuals[:,[0,1]],        "blue", "None", "x",   "S1 Residual")
tel.plot(residuals[:,[0,2]],        "red",  "None", "x",   "S2 Residual")
tel.plot(experimentalData[:,[0,1]], "red",  "",     "*",   "S1 Data")
tel.plot(experimentalData[:,[0,2]], "blue", "",     "*",   "S2 Data")
tel.plt.show()
