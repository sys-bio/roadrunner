import roadrunner
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from teplugins import *
import ctypes as ct

try:
    def pluginIsProgressing(lm):
        lmObject = ct.cast(lm, ct.py_object).value
        print 'Iteration, Norm = ' + `lmObject.getProperty("NrOfIter")` + ',' + `lmObject.getProperty("Norm")` #Retrieve plugin parameters
        
    chiPlugin   = Plugin("tel_chisquare")
    #Retrieve a SBML model from plugin        
    modelPlugin= Plugin("tel_test_model")        
    sbmlModel= modelPlugin.Model   
     
    # Create a roadrunner instance and create some data
    rr = roadrunner.RoadRunner()
    rr.load(sbmlModel)    
    data = rr.simulate(0, 10, 5000)

    print data
    
    #Add noise to the data
    noisePlugin = Plugin ("tel_add_noise")

    # Get the dataseries from data returned by roadrunner
    d = getDataSeries (data)

    # Assign the dataseries to the plugin inputdata
    noisePlugin.InputData = d

    # Set parameter for the 'size' of the noise
    noisePlugin.Sigma = 3.e-1

    # Add the noise
    noisePlugin.execute()
    fName = 'testData3.dat'
    noisePlugin.InputData.writeDataSeries (fName)    
    
    #===================================================================
    lm = Plugin ("tel_levenberg_marquardt")
    experimentalData = DataSeries.readDataSeries (fName)    
       
    lm.ExperimentalData = experimentalData;
    lm.SBML = sbmlModel
    
    # Add the parameters that we're going to fit and the initial value
    lm.InputParameterList = ["k1", .1]   
    lm.FittedDataSelectionList = "[S1] [S2]"
    lm.ExperimentalDataSelectionList = "[S1] [S2]"
    
    lm.OnProgress (pluginIsProgressing)
    
    # Execute lmfit plugin
    res = lm.execute()
    
    #======== Use the stat plugin to calculate statistics
    stat = Plugin("tel_stat")
    
    #Add an offset to residuals to test QQ plot behaviour
    res = DataSeries(lm.Residuals)
    
    #print res.cols
    #for col in range(res.cols):
    #    for row in range(res.rows):
    #        res.setElement(row,col, res.getElement(row, col) + 1)
            
    stat.Residuals = lm.Residuals 
    stat.execute()
        
    # Get the residuals    
    residuals  = stat.Residuals.toNumpy
    residuals = residuals[:,[1,2]]
    
    #Plot as a histogram
    plt.hist(residuals, 50, normed=True)
    plt.show()
        
    stdResiduals = stat.StandardizedResiduals.toNumpy
    stdResiduals = stdResiduals[:,[1,2]]            
    plt.hist(stdResiduals, 50, normed=True)
    plt.show()
        
    #Plot normal probability plots
    probPlots = stat.NormalProbabilityOfResiduals.toNumpy
    print probPlots
    
    x1 = probPlots[:,[0]]
    y1 = probPlots[:,[1]]    
        
    x2 = probPlots[:,[2]]
    y2 = probPlots[:,[3]]    
    
    lineX = np.arange(-5, 5, .1)
    lineY = lineX
    plt.plot(lineX, lineY, '-r')         
    plt.plot(x1,y1, 'o')
    plt.plot(x2,y2, 'x')    
    #plt.scatter(x,y)    
    plt.show()
        
except Exception as e:
    print 'Exception: ' + `e`

