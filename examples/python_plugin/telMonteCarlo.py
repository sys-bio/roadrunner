#-------------------------------------------------------------------------------
# Purpose: Example demonstrating how to setup the the MonteCarlo plugin and
# retrieve parameter confidence limits for a models parameters.
#
# Author: Totte Karlsson (totte@dunescientific.com)
#-------------------------------------------------------------------------------
from rrplugins import *
import matplotlib.pyplot as plt

try:    
    #Load plugins        
    modelP      = Plugin("tel_test_model")
    nP          = Plugin("tel_add_noise")    
    chiP        = Plugin("tel_chisquare")
    lmP         = Plugin("tel_levenberg_marquardt")
    nmP         = Plugin("tel_nelder_mead")
    mcP         = Plugin("tel_monte_carlo_bs") 

    #========== EVENT FUNCTION SETUP ===========================
    def myEventFunction(ignore):           
        # Get the fitted and residual data
        experimentalData    = lmP.getProperty ("ExperimentalData").toNumpy
        fittedData          = lmP.getProperty ("FittedData").toNumpy
        residuals           = lmP.getProperty ("Residuals").toNumpy
        
        telplugins.plot(fittedData         [:,[0,1]], "blue", "-",    "",    "S1 Fitted")
        telplugins.plot(fittedData         [:,[0,2]], "blue", "-",    "",    "S2 Fitted")
        telplugins.plot(residuals          [:,[0,1]], "blue", "None", "x",   "S1 Residual")
        telplugins.plot(residuals          [:,[0,2]], "red",  "None", "x",   "S2 Residual")
        telplugins.plot(experimentalData   [:,[0,1]], "red",  "",     "*",   "S1 Data")
        telplugins.plot(experimentalData   [:,[0,2]], "blue", "",     "*",   "S2 Data")
        
        print 'Minimization finished. \n==== Result ====' 
        print getPluginResult(lmP.plugin)
        telplugins.plt.show()       
                     
    #Communicating event
    myEvent =  NotifyEventEx(myEventFunction)
    
    #Uncomment the event assignment below to plot each monte carlo data set
    #assignOnFinishedEvent(lmP.plugin, myEvent, None)                    
    
    #This will create test data with noise. We will use that as 'experimental' data
    modelP.execute() 
           
    #Setup Monte Carlo properties.
    mcP.SBML                             = modelP.Model
    mcP.ExperimentalData                 = modelP.TestDataWithNoise
    
    #Select what minimization plugin to use                     
    #mcP.MinimizerPlugin                  = "Nelder-Mead"
    mcP.MinimizerPlugin                  = "Levenberg-Marquardt"
    mcP.NrOfMCRuns                       = 100
    mcP.InputParameterList               = ["k1", 1.5]
    mcP.FittedDataSelectionList          = "[S1] [S2]"
    mcP.ExperimentalDataSelectionList    = "[S1] [S2]"
    
    # Start Monte Carlo
    mcP.execute()
    
    print 'Monte Carlo Finished. \n==== Result ===='
    print mcP.MonteCarloParameters.getColumnHeaders()  
    paras = mcP.MonteCarloParameters.toNumpy
    print paras
       
    #Get mean (assuming normal distribution).
    print "The mean: k1= " + `np.mean(paras)`      
    
     
    PropertyOfTypeListHandle = getPluginProperty(mcP.plugin, "ConfidenceLimits")           
    print `getNamesFromPropertyList(PropertyOfTypeListHandle)`            
    aProperty = getFirstProperty(PropertyOfTypeListHandle)
    if aProperty:
        print getPropertyValueAsString(aProperty)
    
    #Show MOnte Carlo parameters as a histogram
    plt.hist(paras, 50, normed=True)
    plt.show()

    #Plot Monte Carlo data sets        
    #dataSeries =  DataSeries.readDataSeries("MCDataSets.dat")    
    #dataSeries.plot()           

    #Finally, view the manual and version
    #mcP.viewManual()    
    print 'Plugin version: ' + `mcP.getVersion()`
             
    
except Exception as e:
    print 'Problem.. ' + `e`    