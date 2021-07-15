import matplotlib.pyplot as plt
from teplugins import *

#========== EVENT FUNCTION SETUP ===========================
def myEventFunction(ignore):           
    myEventFunction.runNr += 1    
    print 'Monte Carlo Run: ' + `myEventFunction.runNr`
           

try:    
    #Load some plugins        
    modelP      = Plugin("tel_test_model")
    chiP        = Plugin("tel_chisquare")
    lmP         = Plugin("tel_levenberg_marquardt")
    mcP         = Plugin("tel_monte_carlo_bs") 
    myEventFunction.runNr = 0  
    #=== The above variables are available in the following callback
                         
    #Communicating event
    myEvent =  NotifyEventEx(myEventFunction)
    assignOnFinishedEvent(lmP.plugin, myEvent, None)                    
        
    #Setup Monte Carlo properties.
    mcP.ExperimentalData                 = DataSeries.readDataSeries ("testData.dat")    
    mcP.SBML                             = modelP.Model             
    mcP.NrOfMCRuns                       = 100
    mcP.InputParameterList               = ["k1", 1.3]
    mcP.FittedDataSelectionList          = "[S1] [S2]"
    mcP.ExperimentalDataSelectionList    = "[S1] [S2]"
    
    # Start Monte Carlo    
    mcP.execute()
    telplugins.plt.show()
        
    print 'Monte Carlo Finished. \n==== Result ===='
    print mcP.MonteCarloParameters.getColumnHeaders()  
    print mcP.MonteCarloParameters.toNumpy
    
    mcParaData = mcP.MonteCarloParameters.toNumpy         
    plt.hist(mcParaData, 50, normed=True)
    plt.show()
        
                          
except Exception as e:
    print 'Problem.. ' + `e`    