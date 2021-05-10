import ctypes
import rrplugins

#Get a lmfit plugin object
lm          = rrplugins.Plugin("tel_levenberg_marquardt")

#========== EVENT FUNCTION SETUP ===========================
def pluginIsProgressing(lmP, dummy):
    # The plugin don't know what a python object is.
    # We need to cast it here, to a proper python object
    lmObject = ctypes.cast(lmP, ctypes.py_object).value
    print('Iterations = ', lmObject.getProperty("NrOfIter") \
       , '\tNorm = ', lmObject.getProperty("Norm"))

progressEvent =  rrplugins.NotifyEventEx(pluginIsProgressing)

#The ID of the plugin is passed as the last argument in the assignOnProgressEvent. 
#The plugin ID is later on retrieved in the plugin Event handler, see above
theId = id(lm)
rrplugins.assignOnProgressEvent(lm.plugin, progressEvent, theId)
#============================================================
sbmlModel = 'HenrichWilbertFit.xml'       
            
#Setup lmfit properties.
lm.SBML = rrplugins.readAllText(sbmlModel)
experimentalData = rrplugins.DataSeries.readDataSeries ("wilbertData.dat")
lm.ExperimentalData = experimentalData

# Add the parameters that we're going to fit and their initial 'start' value
paraValues = [6.77, 1.01, 1.26, 35.11]

# 'bad' parameter values
#paraValues = [1.77, 1.01, 1.26, 30.11]        

lm.setProperty("InputParameterList", ["p0", paraValues[0]])
lm.setProperty("InputParameterList", ["p1", paraValues[1]])
lm.setProperty("InputParameterList", ["p4", paraValues[2]])
lm.setProperty("InputParameterList", ["p6", paraValues[3]])
            
lm.setProperty("FittedDataSelectionList",       "[y1] [y2]")
lm.setProperty("ExperimentalDataSelectionList", "[y1] [y2]")

# Start minimization
lm.execute()

print('Minimization finished. \n==== Result ====')
print(rrplugins.getPluginResult(lm.plugin))
   
print('Hessian Matrix')
hessian = lm.getProperty("Hessian").toNumPy()
print(hessian)

print('Covariance  Matrix')
covariance = lm.getProperty("CovarianceMatrix").toNumPy()
print(covariance)

#This is a list of parameters
parameters = rrplugins.tpc.getPluginProperty (lm.plugin, "OutputParameterList")
confLimits = rrplugins.tpc.getPluginProperty (lm.plugin, "ConfidenceLimits")    

#Iterate through list of parameters and confidence limits
para  = rrplugins.getFirstProperty(parameters)
limit = rrplugins.getFirstProperty(confLimits)     
while para and limit:
    pid = rrplugins.getPropertyName(para)
    print(pid, ' = ', rrplugins.getPropertyValue(para), ' +/- ', rrplugins.getPropertyValue(limit))
    para  = rrplugins.getNextProperty(parameters)
    limit = rrplugins.getNextProperty(confLimits)                        

# Get the experimental data as a numpy array
experimentalData = experimentalData.toNumPy()

# Get the fitted and residual data
fittedData = lm.getProperty ("FittedData").toNumPy()
residuals  = lm.getProperty ("Residuals").toNumPy()

rrplugins.plot(experimentalData   [:,[0,1]], "blue",  "",     "*",  "y1 Data")
rrplugins.plot(experimentalData   [:,[0,2]], "red", "",     "*",    "y2 Data")
rrplugins.plot(fittedData         [:,[0,1]], "blue", "-",    "",    "y1 Fit")
rrplugins.plot(fittedData         [:,[0,2]], "red", "-",     "",    "y2 Fit")
rrplugins.plot(residuals          [:,[0,1]], "blue", "None", "x",   "y1 Residual")
rrplugins.plot(residuals          [:,[0,2]], "red",  "None", "x",   "y2 Residual")

rrplugins.plt.show()
