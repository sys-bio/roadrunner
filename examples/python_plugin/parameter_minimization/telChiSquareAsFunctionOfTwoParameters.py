#-------------------------------------------------------------------------------
# Purpose: Example demonstrating how to calculate the ChiSquare, using the
# ChiSquare plugin, as a function of a TWO model parameters.
#
# This example are using example data in the file TwoParameters.dat, which is
# generated using the model in 'two_parameters.xml'. Both of these files are
# needed for the example
#
# The generated contour plot will indicate the value of k1 and k2 that was
# used in generating the 'Experimental' data.
# Author: Totte Karlsson (totte@dunescientific.com)
#-------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import roadrunner
import teplugins as tel
from roadrunner import Config


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
    Config.setValue(Config.LOADSBMLOPTIONS_CONSERVED_MOIETIES, False)    
    Config.setValue(Config.SIMULATEOPTIONS_STRUCTURED_RESULT, True)
    #Read some 'experimental' data   
    expData = tel.DataSeries()
    
    #This does not look right.. but it works..
    expData = expData.readDataSeries('TwoParametersData.dat')   
            
    test_model = 'two_parameters.xml'            

    # Create a roadrunner instance and create some MODEL data
    rr = roadrunner.RoadRunner()
    rr.load(test_model)

    #Get chi square plugin and set it up
    chiSquare =  tel.Plugin("tel_chisquare")    
    chiSquare.ExperimentalData = expData    
    chiSquare.NrOfModelParameters = 2
    
    #Simulate using the same numbers as in the 'Experimental data             
    k1Start = 1 ;  k1End = 1.8    
    k2Start = 1 ;  k2End = 4.5
    timeStart = 0;  timeEnd = 1.5 ;  nrPoints = 16
             
    X = np.linspace(k1Start, k1End, 55)
    Y = np.linspace(k2Start, k2End, 55)
    nrX = len(X) ; nrY = len(Y)
    print 'nrPoints = ' + `nrX * nrY`
    
    Z = np.zeros([nrX, nrY])            
    for row in range(nrX):    
        for col in range(nrY):
            print 'row,col = ' + `row` + ", " + `col`
            Z[col,row] = chiFun(X[row], Y[col])                                                     
                                
    X, Y = np.meshgrid(X, Y)        
        
    CS = plt.contour(X, Y, Z, 400)
    plt.clabel(CS, inline=1, fontsize=10)        
    plt.title('Reduced ChiSquare for SBML model with two parameters')
    plt.xlabel('k1')
    plt.ylabel('k2')    
    plt.show()                                                
    print "done"

except Exception as e:
    print 'Problem: ' + `e`