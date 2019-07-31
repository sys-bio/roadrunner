#-------------------------------------------------------------------------------
# Purpose: Example demonstrating how to setup the Nelder-Mead plugin
# and find parameter minimums. This example also shows how to setup a
# user defined event function that is called during plugin execution. 
#
# Author: Totte Karlsson (totte@dunescientific.com)
#-------------------------------------------------------------------------------
from teplugins import *

# Load Plugins
chiPlugin       = Plugin("tel_chisquare")
nm              = Plugin("tel_nelder_mead")
modelPlugin     = Plugin("tel_test_model")
addNoisePlugin  = Plugin("tel_add_noise")

try:    
    #========== EVENT FUNCTION SETUP ===========================
    def myEvent(dummy): #We are not capturing any data from the plugin, so just pass a dummy
        print 'Iteration, Norm = ' + `nm.getProperty("NrOfIter")`  + ', ' + `nm.getProperty("Norm")`

    #Setup progress event function
    progressEvent =  NotifyEventEx(myEvent)     
    assignOnProgressEvent(nm.plugin, progressEvent)
    #============================================================
    
    #Create model data, with and without noise using the test_model plugin
    modelPlugin.execute()     
    
    #Setup plugin properties.
    nm.SBML             = modelPlugin.Model 
    nm.ExperimentalData = modelPlugin.TestDataWithNoise
    
    # Add the parameters that we're going to fit and an initial 'start' value
    nm.setProperty("InputParameterList", ["k1", .3])
    nm.setProperty("FittedDataSelectionList", "[S1] [S2]")
    nm.setProperty("ExperimentalDataSelectionList", "[S1] [S2]")
    
    # Start minimization
    nm.execute()
    
    print 'Minimization finished. \n==== Result ===='        

    print 'Hessian Matrix'
    print nm.getProperty("Hessian")
    
    print 'Covariance  Matrix'
    print nm.getProperty("CovarianceMatrix")
             
    print 'ChiSquare = '            + `nm.getProperty("ChiSquare")`
    print 'Reduced ChiSquare = '    + `nm.getProperty("ReducedChiSquare")`
        
    #This is a list of parameters
    parameters = tpc.getPluginProperty (nm.plugin, "OutputParameterList")
    confLimits = tpc.getPluginProperty (nm.plugin, "ConfidenceLimits")    
    
    #Iterate trough list of parameters and confidence limits
    para  = getFirstProperty(parameters)
    limit = getFirstProperty(confLimits)     
    while para and limit:           
        print getPropertyName(para) + ' = ' + `getPropertyValue(para)` + ' +/- ' + `getPropertyValue(limit)`
        para  = getNextProperty(parameters)
        limit = getNextProperty(confLimits)                        
                                 
    
    # Get the fitted and residual data
    fittedData = nm.getProperty ("FittedData").toNumpy
    residuals  = nm.getProperty ("Residuals").toNumpy

    # Get the experimental data as a numpy array
    experimentalData = modelPlugin.TestDataWithNoise.toNumpy
    
    telplugins.plot(fittedData         [:,[0,1]], "blue", "-",    "",    "S1 Fitted")
    telplugins.plot(fittedData         [:,[0,2]], "blue", "-",    "",    "S2 Fitted")
    telplugins.plot(residuals          [:,[0,1]], "blue", "None", "x",   "S1 Residual")
    telplugins.plot(residuals          [:,[0,2]], "red",  "None", "x",   "S2 Residual")
    telplugins.plot(experimentalData   [:,[0,1]], "red",  "",     "*",   "S1 Data")
    telplugins.plot(experimentalData   [:,[0,2]], "blue", "",     "*",   "S2 Data")
    telplugins.plt.show()
    
    #Finally, view the manual and version
    #nm.viewManual()    
    print 'Plugin version: ' + `nm.getVersion()`
    
except Exception as e:
    print 'Problem.. ' + `e`         