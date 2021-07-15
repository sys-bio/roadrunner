import roadrunner
import numpy as np
import matplotlib.pyplot as plt
import teplugins as tel

try:
    #Read some 'experimental' data   
    expData = tel.DataSeries()
    
    #This does not look right.. but it works..
    expData = expData.readDataSeries('ExperimentalData.dat')   
        
    modelPlugin = tel.Plugin("tel_test_model")   
    test_model  = modelPlugin.Model
    
    # Create a roadrunner instance and create some MODEL data
    rr = roadrunner.RoadRunner()
    rr.load(test_model)
    
    #Simulate using the same numbers as in the 'Experimental data
    x = np.array([])
    y = np.array([])
         
    kStart = 0.1
    kEnd = 2.5
    kStep = .01
    for k in np.arange(kStart, kEnd, kStep):      
          
        rr.setValue("k1", k)   
        rr.reset()
        timeStart = 0
        timeEnd = 10
        nrPoints = 50         
        data = rr.simulate(timeStart, timeEnd, nrPoints - 1) # Want 512 points
        #roadrunner.plot(data)
    
        #Get chi square plugin and set it up
        chiSquare =  tel.Plugin("tel_chisquare")    
        chiSquare.ExperimentalData = expData
        chiSquare.ModelData = tel.getDataSeries(data)
        chiSquare.NrOfModelParameters = 1
    
        #Calculate the Chi Square
        chiSquare.execute()        
               
        redChi = chiSquare.ReducedChiSquare               
        print  'k, chisquare = ' + `k` +', '+  `redChi` 
        x = np.append(x, k) 
        y = np.append(y, redChi)         
    
    plt.plot(x,y, '-o', label="Reduced ChiSquare")
    plt.legend()
    plt.xlabel("k1")
    
    plt.show()
    print "done"

except Exception as e:
    print 'Problem: ' + `e`