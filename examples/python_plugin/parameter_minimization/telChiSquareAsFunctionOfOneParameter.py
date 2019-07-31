#-------------------------------------------------------------------------------
# Purpose: Example demonstrating how to calculate the ChiSquare, using the 
# ChiSquare plugin, as a function of a model parameter. 
# Author: Totte Karlsson (totte@dunescientific.com)
#-------------------------------------------------------------------------------

import roadrunner
import numpy as np
import matplotlib.pyplot as plt
import teplugins as tel

try:       
#    Config.setValue(Config.LOADSBMLOPTIONS_CONSERVED_MOIETIES, False)    
#    Config.setValue(Config.SIMULATEOPTIONS_STRUCTURED_RESULT, True)    
    noisePlugin = tel.Plugin("tel_add_noise")
    modelPlugin = tel.Plugin("tel_test_model")   
    test_model  = modelPlugin.Model
    modelPlugin.execute()
    
    #Use the testModel plugins property TestDataWithNoise as the 'Experimental' data
    expData = modelPlugin.TestDataWithNoise
    
    # Create a roadrunner instance and create some MODEL data
    rr = roadrunner.RoadRunner()
    rr.load(test_model)
    
    #Create variables to be used in the loop
    x = np.array([])
    y = np.array([])         
    
    kStart = 0.1
    kEnd = 2.5
    kStep = .01
    
    #Sweep the k value
    for k in np.arange(kStart, kEnd, kStep):      
        
        rr.reset()
        rr.setValue("k1", k)   
        
        #Simulate using the same numbers as in the 'Experimental' data
        data = rr.simulate(0,10,15)
        #roadrunner.plot(data)
    
        #Get chi square plugin and set it up
        chiSquare =  tel.Plugin("tel_chisquare")    
        chiSquare.ExperimentalData = expData
        chiSquare.ModelData = tel.getDataSeries(data)
        chiSquare.NrOfModelParameters = 1
    
        #Calculate the Chi Square
        chiSquare.execute()        
               
        redChi = chiSquare.ReducedChiSquare               
        print  'k, Reduced Chisquare = ' + `k` +', '+  `redChi` 
        x = np.append(x, k) 
        y = np.append(y, redChi)         
    
    plt.plot(x,y, '-o', label="Reduced ChiSquare")
    plt.legend()
    plt.xlabel("k1")
    
    plt.show()
    print "done"

except Exception as e:
    print 'Problem: ' + `e`