import numpy as np
#from numpy import matrix
#from numpy import linalg
import matplotlib.pyplot as plt
import roadrunner
import teplugins as tel


def getHessElement(r,c, k1, k2, eta):
    h1 = eta *abs(k1)
    h2 = eta *abs(k2)    
    f = chiFun
    if r == c:
        if r == 1:
            element = ( f(k1 + h1, k2) + f(k1 - h1, k2) -2.0* f(k1, k2) ) / (h1*h1)   
        else:
            element = ( f(k1, k2 + h2) + f(k1, k2 - h2) -2.0* f(k1, k2) ) / (h2*h2)            
    else:
        element = ( f(k1+h1, k2+h2) - f(k1+h1,k2) - f(k1,k2+h2) + f(k1,k2) ) / (h1*h2)
    return element         
                
def chiFun(k1,k2):
    rr.reset()
    rr.setValue("k1", k1)
    rr.setValue("k2", k2)          
    data = rr.simulate(timeStart, timeEnd, nrPoints - 1)            
    chiSquare.ModelData = tel.getDataSeries(data)
    
    if not chiSquare.execute():
        raise Exception( tel.getLastError() )
           
    return chiSquare.ReducedChiSquare                   
            
try:
    #Read some 'experimental' data   
    expData = tel.DataSeries()
    
    #This does not look right.. but it works..
    expData = expData.readDataSeries('ExperimentalData.dat')   
            
    test_model = 'two_parameters.xml'            

    # Create a roadrunner instance and create some MODEL data
    rr = roadrunner.RoadRunner()
    rr.load(test_model)

    #Get chi square plugin and set it up
    chiSquare =  tel.Plugin("tel_chisquare")    
    chiSquare.ExperimentalData = expData    
    chiSquare.NrOfModelParameters = 2  
             
    k1 = 1.3;  k2 = 2.5       
    timeStart = 0;  timeEnd = 1.5 ;  nrPoints = 15
             
    eta = 6.0e-6
    
    H = np.matrix('0.0 0.0; 0.0 0.0')
    #Diagonal elements
    H[0,0] = getHessElement(1, 1, k1, k2, eta)
    H[0,1] = getHessElement(1, 2, k1, k2, eta)    
    H[1,0] = getHessElement(2, 1, k1, k2, eta)
    H[1,1] = getHessElement(2, 2, k1, k2, eta)                                                                                         
    
    print 'The Hessian: \n' + `H`    
    print '\nInverse Hessian \nH.' + `H.I`
                                                    
    print "done"

except Exception as e:
    print 'Problem: ' + `e`