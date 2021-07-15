#-------------------------------------------------------------------------------
# Purpose: Example demonstrating how to setup the the MonteCarlo plugin and
# retrieve parameter confidence limits for a models parameters.
#
# Author: Totte Karlsson (totte@dunescientific.com)
#-------------------------------------------------------------------------------
import numpy as np
import rrplugins
import matplotlib.pyplot as plt

try:    
    #Load plugins        
    modelP      = rrplugins.Plugin("tel_test_model")
    lmP         = rrplugins.Plugin("tel_levenberg_marquardt")
    mcP         = rrplugins.Plugin("tel_monte_carlo_bs") 

    #========== EVENT FUNCTION SETUP ===========================
    def myEventFunction(ignore1, ignore2):           
        # Get the fitted and residual data
        experimentalData    = lmP.getProperty ("ExperimentalData").toNumPy()
        fittedData          = lmP.getProperty ("FittedData").toNumPy()
        residuals           = lmP.getProperty ("Residuals").toNumPy()
        
        rrplugins.plot(fittedData         [:,[0,1]], "blue", "-",    "",    "S1 Fitted")
        rrplugins.plot(fittedData         [:,[0,2]], "blue", "-",    "",    "S2 Fitted")
        rrplugins.plot(residuals          [:,[0,1]], "blue", "None", "x",   "S1 Residual")
        rrplugins.plot(residuals          [:,[0,2]], "red",  "None", "x",   "S2 Residual")
        rrplugins.plot(experimentalData   [:,[0,1]], "red",  "",     "*",   "S1 Data")
        rrplugins.plot(experimentalData   [:,[0,2]], "blue", "",     "*",   "S2 Data")
        
        print('Minimization finished. \n==== Result ====')
        print(rrplugins.getPluginResult(lmP.plugin))
        rrplugins.plt.show()       
                     
    #Communicating event
    myEvent =  rrplugins.NotifyEventEx(myEventFunction)
    
    #Uncomment the event assignment below to plot each monte carlo data set
    # rrplugins.assignOnFinishedEvent(lmP.plugin, myEvent, None)                    
    
    #This will create test data with noise. We will use that as 'experimental' data
    modelP.execute() 
           
    #Setup Monte Carlo properties.
    mcP.SBML                             = modelP.Model
    mcP.ExperimentalData                 = modelP.TestDataWithNoise
    
    #Select what minimization plugin to use                     
    #mcP.MinimizerPlugin                  = "tel_nelder_mead"
    mcP.MinimizerPlugin                  = "tel_levenberg_marquardt"
    mcP.NrOfMCRuns                       = 100
    mcP.InputParameterList               = ["k1", 0.5]
    mcP.FittedDataSelectionList          = "[S1] [S2]"
    mcP.ExperimentalDataSelectionList    = "[S1] [S2]"
    
    # Start Monte Carlo
    mcP.execute()
    
    print('Monte Carlo Finished. \n==== Result ====')
    print(mcP.MonteCarloParameters.getColumnHeaders())
    paras = mcP.MonteCarloParameters.toNumPy()
    # print(paras)
       
    PropertyOfTypeListHandle = rrplugins.getPluginProperty(mcP.plugin, "Means")
    print("Optimized parameters:", rrplugins.getNamesFromPropertyList(PropertyOfTypeListHandle))
    aProperty = rrplugins.getFirstProperty(PropertyOfTypeListHandle)
    print("Means:", rrplugins.getPropertyValueAsString(aProperty))
    
    PropertyOfTypeListHandle = rrplugins.getPluginProperty(mcP.plugin, "ConfidenceIntervals")
    aProperty = rrplugins.getFirstProperty(PropertyOfTypeListHandle)
    print("Confidence intervals:", rrplugins.getPropertyValueAsString(aProperty))
    
    PropertyOfTypeListHandle = rrplugins.getPluginProperty(mcP.plugin, "Percentiles")           
    aProperty = rrplugins.getFirstProperty(PropertyOfTypeListHandle)
    while aProperty:
        print("Percentile:", rrplugins.getPropertyName(aProperty), "=", rrplugins.getPropertyValueAsString(aProperty))
        aProperty = rrplugins.getNextProperty(PropertyOfTypeListHandle)

    #Show Monte Carlo parameters as a histogram
    plt.hist(paras, 30)
    plt.show()

    #Plot Monte Carlo data sets        
    #dataSeries =  DataSeries.readDataSeries("MCDataSets.dat")    
    #dataSeries.plot()           

    #Finally, view the manual and version
    #mcP.viewManual()    
    print('Plugin version: ', mcP.getVersion())
             
    
except Exception as e:
    print('Problem.. ', e) 
