import rrplugins
import matplotlib.pyplot as plt
import numpy as np

#Load plugins        
modelP      = rrplugins.Plugin("tel_test_model")
nP          = rrplugins.Plugin("tel_add_noise")    
chiP        = rrplugins.Plugin("tel_chisquare")
lmP         = rrplugins.Plugin("tel_levenberg_marquardt")
nmP         = rrplugins.Plugin("tel_nelder_mead")
mcP         = rrplugins.Plugin("tel_monte_carlo_bs") 

#========== EVENT FUNCTION SETUP ===========================
def myEventFunction(ignore, ignore2):           
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

print('Monte Carlo Finished. \n==== Result ====')
print(mcP.MonteCarloParameters.getColumnHeaders())
paras = mcP.MonteCarloParameters.toNumPy()
print(paras)
   
#Get mean (assuming normal distribution).
print("The mean: k1= ", np.mean(paras))

 
PropertyOfTypeListHandle = rrplugins.getPluginProperty(mcP.plugin, "ConfidenceIntervals")
print(rrplugins.getNamesFromPropertyList(PropertyOfTypeListHandle))
aProperty = rrplugins.getFirstProperty(PropertyOfTypeListHandle)
if aProperty:
    print(rrplugins.getPropertyValueAsString(aProperty))

PercentileHandle = rrplugins.getPluginProperty(mcP.plugin, "Percentiles")
print(rrplugins.getNamesFromPropertyList(PercentileHandle))
aProperty = rrplugins.getFirstProperty(PercentileHandle)
while aProperty:
    print(rrplugins.getPropertyName(aProperty), ":", rrplugins.getPropertyValueAsString(aProperty))
    aProperty = rrplugins.getNextProperty(PercentileHandle)

#Show MOnte Carlo parameters as a histogram
plt.hist(paras, 50)
plt.show()

#Plot Monte Carlo data sets        
#dataSeries =  DataSeries.readDataSeries("MCDataSets.dat")    
#dataSeries.plot()           

#Finally, view the manual and version
# mcP.viewManual()    
print('Plugin version: ', mcP.getVersion())
         
