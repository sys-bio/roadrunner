import rrplugins
from rrplugins import tpc

# Load Plugins
chiPlugin       = rrplugins.Plugin("tel_chisquare")
nm              = rrplugins.Plugin("tel_nelder_mead")
modelPlugin     = rrplugins.Plugin("tel_test_model")
addNoisePlugin  = rrplugins.Plugin("tel_add_noise")

#========== EVENT FUNCTION SETUP ===========================
def myEvent(dummy, dummy2): #We are not capturing any data from the plugin, so just pass a dummy
    print('Iteration, Norm = ', nm.getProperty("NrOfIter") , ', ', nm.getProperty("Norm"))

#Setup progress event function
progressEvent =  rrplugins.NotifyEventEx(myEvent)     
rrplugins.assignOnProgressEvent(nm.plugin, progressEvent)
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

print('Minimization finished. \n==== Result ====')

print('ChiSquare = '           , nm.getProperty("ChiSquare"))
print('Reduced ChiSquare = '   , nm.getProperty("ReducedChiSquare"))
    
#This is a list of parameters
parameters = tpc.getPluginProperty (nm.plugin, "OutputParameterList")

#Iterate trough list of parameters and confidence limits
para  = rrplugins.getFirstProperty(parameters)
while para:           
    print(rrplugins.getPropertyName(para), ' = ', rrplugins.getPropertyValue(para))
    para  = rrplugins.getNextProperty(parameters)


# Get the fitted and residual data
fittedData = nm.getProperty ("FittedData").toNumPy()
residuals  = nm.getProperty ("Residuals").toNumPy()

# Get the experimental data as a numpy array
experimentalData = modelPlugin.TestDataWithNoise.toNumPy()

rrplugins.plot(fittedData         [:,[0,1]], "blue", "-",    "",    "S1 Fitted")
rrplugins.plot(fittedData         [:,[0,2]], "blue", "-",    "",    "S2 Fitted")
rrplugins.plot(residuals          [:,[0,1]], "blue", "None", "x",   "S1 Residual")
rrplugins.plot(residuals          [:,[0,2]], "red",  "None", "x",   "S2 Residual")
rrplugins.plot(experimentalData   [:,[0,1]], "red",  "",     "*",   "S1 Data")
rrplugins.plot(experimentalData   [:,[0,2]], "blue", "",     "*",   "S2 Data")
rrplugins.plt.show()

#Finally, view the manual and version
# nm.viewManual()    
print('Plugin version: ', nm.getVersion())
