import roadrunner
import numpy as np
import matplotlib.pyplot as plt
import teplugins as tel

def firstDerivative(p2, p1, h):
    return (p2 - p1) / (2.0*h)

try:
    #Read some 'experimental' data   
    expData = tel.DataSeries()
    
    #This does not look right.. but it works..
    expData = expData.readDataSeries('ExperimentalData.dat')   
        
    #Get a model        
    modelPlugin = tel.Plugin("tel_test_model")   
    test_model  = modelPlugin.Model
    
    # Create a roadrunner instance and create some MODEL data
    rr = roadrunner.RoadRunner()
    rr.load(test_model)
    
    #Simulate using the same numbers as in the 'Experimental data
    x = np.array([])
    y = np.array([])     
    
    #Get chi square plugin and set it up
    chiSquare =  tel.Plugin("tel_chisquare")    
    chiSquare.ExperimentalData = expData
    chiSquare.NrOfModelParameters = 1
    
    ptn = 0        
    kStart = 0.1
    kEnd = 1.5
    kStep = .01
    for k in np.arange(kStart, kEnd, kStep):                
        rr.setValue("k1", k)   
        rr.reset()
        timeStart = 0
        timeEnd = 10
        nrPoints = 50         
         
        data = rr.simulate(timeStart, timeEnd, nrPoints - 1)
        chiSquare.ModelData = tel.getDataSeries(data)
        
        chiSquare.execute()            
               
        redChi = chiSquare.ReducedChiSquare               
        print  'ptn, k, chisquare = ' + `ptn` + ', '+ `k` +', '+  `redChi` 
        x = np.append(x, k) 
        y = np.append(y, redChi)
        ptn = ptn + 1         
    
    plt.subplot(1,3,1)
    plt.plot(x,y, '-o', label="Reduced ChiSquare")
    plt.legend()
    plt.xlabel("k1")
    plt.ylabel("ChiSquare")    
    

    #Calculate the derivative
    i = 1
    imax = len(x) -1
    yPrime = np.array([])
    xPrime = np.array([])
    for k in x:                                
        yp = firstDerivative(y[i], y[i-1], x[i] - x[i-1])
        
        xp = (x[i] + x[i-1]) /2.                 
        yPrime = np.append(yPrime, yp)
        xPrime = np.append(xPrime, xp)
        
        print 'k, chiSquarePrime = ' + `xp` + ', ' + `yp`               
        i = i + 1
        if i >= imax:
            print 'breaking'
            break
            
    plt.subplot(1,3,2)
    plt.plot(xPrime,yPrime, '-o', label="Reduced ChiSquare Derivative")
    plt.legend()
    plt.xlabel("k1")
    plt.ylabel("ChiSquare'")
    plt.hlines(0, kStart, kEnd)
    plt.vlines(0.57, -150, 150)        
        
    #Calculate the second derivative as the first derivative on first derivative
    i = 1
    imax = len(xPrime) -1
    yPPrime = np.array([])
    xPPrime = np.array([])
    for k in xPrime:                                
        yp = firstDerivative(yPrime[i], yPrime[i-1], xPrime[i] - xPrime[i-1])        
        xp = (xPrime[i] + xPrime[i-1]) /2.                 
        yPPrime = np.append(yPPrime, yp)
        xPPrime = np.append(xPPrime, xp)
        
        print 'k, chiSquarePrime = ' + `xp` + ', ' + `yp`               
        i = i + 1
        if i >= imax:
            print 'breaking'
            break           
    
    plt.subplot(1,3,3)
    plt.plot(xPPrime,yPPrime, '-o', label="Reduced ChiSquare Second Derivative")
    plt.legend()
    plt.xlabel("k1")
    plt.ylabel("ChiSquare'")
    plt.hlines(0, kStart, kEnd)
    plt.vlines(0.57, -150, 150)        
    plt.show()

    print "done"

except Exception as e:
    print 'Problem: ' + `e`