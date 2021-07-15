import matplotlib.pyplot as plt
import scipy.stats as stats
import teplugins as tel
import ctypes as ct

try:
    def pluginIsProgressing(lm):
        lmObject = ct.cast(lm, ct.py_object).value
        print 'Iteration, Norm = ' + `lmObject.getProperty("NrOfIter")` + ',' + `lmObject.getProperty("Norm")` #Retrieve plugin parameters

    #Retrieve a SBML model from plugin        
    modelPlugin= tel.Plugin("tel_test_model")        
    test_model = modelPlugin.Model

    chiPlugin   = tel.Plugin("tel_chisquare")        
    lm          = tel.Plugin("tel_levenberg_marquardt")
    experimentalData = tel.DataSeries.readDataSeries ("testData.dat")   
    lm.ExperimentalData = experimentalData;
    lm.SBML = test_model
    
    # Add the parameters that we're going to fit and the initial value
    lm.InputParameterList = ["k1", 0.2]   
    lm.FittedDataSelectionList = "[S1] [S2]"
    lm.ExperimentalDataSelectionList = "[S1] [S2]"
    
    lm.OnProgress (pluginIsProgressing)
    
    # Execute lmfit plugin
    res = lm.execute()
    
    # Get the experimental data as a numpy array
    experimentalData = experimentalData.toNumpy
    
    # Get the fitted and residual data
    fittedData = lm.FittedData.toNumpy
    residuals  = lm.Residuals.toNumpy
    
    
    tel.telplugins.plot (fittedData[:,[0,1]], myColor="blue", myLinestyle="-", myMarker="", myLabel="S1 Fitted")
    tel.telplugins.plot (fittedData[:,[0,2]], myColor="blue", myLinestyle="-", myMarker="", myLabel="S2 Fitted")
    tel.telplugins.plot (residuals[:,[0,1]], myColor="blue", myLinestyle="None", myMarker="x", myLabel="S1 Residual")
    tel.telplugins.plot (residuals[:,[0,2]], myColor="red", myLinestyle="None", myMarker="x", myLabel="S2 Residual")
    tel.telplugins.plot (experimentalData[:,[0,1]], myColor="red", myLinestyle="", myMarker="*", myLabel="S1 Data")
    tel.telplugins.plot (experimentalData[:,[0,2]], myColor="blue", myLinestyle="", myMarker="*", myLabel="S2 Data")
    tel.plt.show()
       
except Exception as e:
    print 'Exception: ' + `e`

