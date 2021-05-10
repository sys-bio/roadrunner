import ctypes
import teplugins as tel

#Get a nmfit plugin object
chiPlugin   = tel.Plugin("tel_chisquare")
nm          = tel.Plugin("tel_nelder_mead")

#========== EVENT FUNCTION SETUP ===========================
def pluginIsProgressing(nmP):
    # The plugin don't know what a python object is.
    # We need to cast it here, to a proper python object
    nmObject = ctypes.cast(nmP, ctypes.py_object).value
    print 'Iterations = ' + `nmObject.getProperty("NrOfIter")` \
                + 'FuncIterations = ' + `nmObject.getProperty("NrOfFuncIter")` \
                + '\tNorm = ' + `nmObject.getProperty("Norm")`

try:   
    progressEvent =  tel.NotifyEventEx(pluginIsProgressing)
    
    #The ID of the plugin is passed as the last argument in the assignOnProgressEvent. 
    #The plugin ID is later on retrieved in the plugin Event handler, see above
    theId = id(nm)
    tel.assignOnProgressEvent(nm.plugin, progressEvent, theId)
    #============================================================
    #Retrieve a SBML model from plugin        
    modelPlugin= tel.Plugin("tel_test_model")        
    test_model = modelPlugin.Model
    
    #Setup nmfit properties.
    nm.SBML = test_model
    experimentalData = tel.DataSeries.readDataSeries ("testData.dat")
    nm.ExperimentalData = experimentalData
    
    # Add the parameters that we're going to fit and a initial 'start' value
    nm.setProperty("InputParameterList", ["k1", .03])
    nm.setProperty("FittedDataSelectionList", "[S1] [S2]")
    nm.setProperty("ExperimentalDataSelectionList", "[S1] [S2]")
    
    # Start minimization
    nm.execute()
    
    print 'Minimization finished. \n==== Result ====' 
    print tel.getPluginResult(nm.plugin)
    
    # Get the experimental data as a numpy array
    experimentalData = experimentalData.toNumpy
    
    # Get the fitted and residual data
    fittedData = nm.getProperty ("FittedData").toNumpy
    residuals  = nm.getProperty ("Residuals").toNumpy
    
    tel.telplugins.plot(fittedData         [:,[0,1]], "blue", "-",    "",    "S1 Fitted")
    tel.telplugins.plot(fittedData         [:,[0,2]], "blue", "-",    "",    "S2 Fitted")
    tel.telplugins.plot(residuals          [:,[0,1]], "blue", "None", "x",   "S1 Residual")
    tel.telplugins.plot(residuals          [:,[0,2]], "red",  "None", "x",   "S2 Residual")
    tel.telplugins.plot(experimentalData   [:,[0,1]], "red",  "",     "*",   "S1 Data")
    tel.telplugins.plot(experimentalData   [:,[0,2]], "blue", "",     "*",   "S2 Data")
    tel.telplugins.plt.show()
    
except Exception as e:
    print 'Problem.. ' + `e`          