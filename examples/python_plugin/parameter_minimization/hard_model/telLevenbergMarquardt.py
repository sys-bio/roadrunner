import ctypes
import numpy as np
from teplugins import *

#Get a lmfit plugin object
chiPlugin   = Plugin("tel_chisquare")
lm          = Plugin("tel_levenberg_marquardt")

#========== EVENT FUNCTION SETUP ===========================
def pluginIsProgressing(lmP):
    # The plugin don't know what a python object is.
    # We need to cast it here, to a proper python object
    lmObject = ctypes.cast(lmP, ctypes.py_object).value
    print 'Iterations = ' + `lmObject.getProperty("NrOfIter")` \
        + '\tNorm = ' + `lmObject.getProperty("Norm")`

try:   
    progressEvent =  NotifyEventEx(pluginIsProgressing)
    
    #The ID of the plugin is passed as the last argument in the assignOnProgressEvent. 
    #The plugin ID is later on retrieved in the plugin Event handler, see above
    theId = id(lm)
    assignOnProgressEvent(lm.plugin, progressEvent, theId)
    #============================================================
    sbmlModel = 'HenrichWilbertFit.xml'       
                
    #Setup lmfit properties.
    lm.SBML = readAllText(sbmlModel)
    experimentalData = DataSeries.readDataSeries ("wilbertData.dat")
    lm.ExperimentalData = experimentalData
    
    # Add the parameters that we're going to fit and their initial 'start' value
    paraValues = [6.77, 1.01, 1.26, 5.11]
    
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
    
    print 'Minimization finished. \n==== Result ====' 
    print getPluginResult(lm.plugin)
       
    # Get the experimental data as a numpy array
    experimentalData = experimentalData.toNumpy
    
    # Get the fitted and residual data
    fittedData = lm.getProperty ("FittedData").toNumpy
    residuals  = lm.getProperty ("Residuals").toNumpy
    
    telplugins.plot(experimentalData   [:,[0,1]], "blue",  "",     "*",  "y1 Data")
    telplugins.plot(experimentalData   [:,[0,2]], "red", "",     "*",    "y2 Data")
    telplugins.plot(fittedData         [:,[0,1]], "blue", "-",    "",    "y1 Fit")
    telplugins.plot(fittedData         [:,[0,2]], "red", "-",     "",    "y2 Fit")
    telplugins.plot(residuals          [:,[0,1]], "blue", "None", "x",   "y1 Residual")
    telplugins.plot(residuals          [:,[0,2]], "red",  "None", "x",   "y2 Residual")

    telplugins.plt.show()
    
except Exception as e:
    print 'Problem.. ' + `e`          