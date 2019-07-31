from teplugins import *

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

try:    
    #Load some plugins        
    modelP      = Plugin("tel_test_model")
    chiP        = Plugin("tel_chisquare")
    lmP         = Plugin("tel_levenberg_marquardt")
    lmP         = Plugin("tel_nelder_mead")
    mcP         = Plugin("tel_monte_carlo_bs") 
                     
    #Communicating event
    myEvent =  NotifyEventEx(myEventFunction)
#    assignOnFinishedEvent(lmP.plugin, myEvent, None)                    
        
    #Setup Monte Carlo properties.
    mcP.ExperimentalData                 = DataSeries.readDataSeries ("testData.dat")    
    mcP.SBML                             = modelP.Model             
    mcP.MinimizerPlugin                  = "Nelder-Mead"
    mcP.NrOfMCRuns                       = 10
    mcP.InputParameterList               = ["k1", 1.3]
    mcP.FittedDataSelectionList          = "[S1] [S2]"
    mcP.ExperimentalDataSelectionList    = "[S1] [S2]"
    
    # Start Monte Carlo
    mcP.execute()
    
    print 'Monte Carlo Finished. \n==== Result ===='
    print mcP.MonteCarloParameters.getColumnHeaders()  
    print mcP.MonteCarloParameters.toNumpy
    
    PropertyOfTypeListHandle = getPluginProperty(mcP.plugin, "ConfidenceLimits")           
    print `getNamesFromPropertyList(PropertyOfTypeListHandle)`            
    aProperty = getFirstProperty(PropertyOfTypeListHandle)
    if aProperty:
        print getPropertyValueAsString(aProperty)
                
    
except Exception as e:
    print 'Problem.. ' + `e`    