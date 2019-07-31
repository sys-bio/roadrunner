import roadrunner
import numpy as np
import matplotlib.pyplot as plt
import teplugins as tel

try:
    #Read some 'experimental' data   
    expData = tel.DataSeries()
    
    #This does not look right.. but it works..
    expData = expData.readDataSeries('ExperimentalData.dat')   
            
    test_model = 'two_parameters.xml'            
    # Create a roadrunner instance and create some MODEL data
    rr = roadrunner.RoadRunner()
    rr.load(test_model)
    
    #Simulate using the same numbers as in the 'Experimental data
    x = np.array([])
    y = np.array([])
         
    kStart = 0.5
    kEnd = 6
    kStep = .1
    
    timeStart = 0
    timeEnd = 1.5
    nrPoints = 15         

    for k in np.arange(kStart, kEnd, kStep):                                 
        rr.reset()
        rr.setValue("k1", k)
        rr.setValue("k2", 2.5)          
        data = rr.simulate(timeStart, timeEnd, nrPoints - 1)        
    
        #Get chi square plugin and set it up
        chiSquare =  tel.Plugin("tel_chisquare")    
        chiSquare.ExperimentalData = expData
        chiSquare.ModelData = tel.getDataSeries(data)
        chiSquare.NrOfModelParameters = 2
    
        if not chiSquare.execute():
            raise Exception( tel.getLastError() )
               
        redChi = chiSquare.ReducedChiSquare               
        print  'k, chisquare = ' + `k` +', '+  `redChi` 
        x = np.append(x, k) 
        y = np.append(y, redChi)         
    
    plt.plot(x,y, '-o', label="Reduced ChiSquare")
    plt.legend()
    plt.xlabel("k2")
    
    plt.show()
    print "done"

except Exception as e:
    print 'Problem: ' + `e`