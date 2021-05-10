import matplotlib.pyplot as plt
import roadrunner
from teplugins import *

try:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    #Read some 'experimental' data   
    expData = DataSeries()
    
    #This does not look right..
    expData = expData.readDataSeries('ExperimentalData.dat')   
        
    modelPlugin = Plugin("tel_test_model")   
    test_model  = modelPlugin.Model
    
    # Create a roadrunner instance and create some MODEL data
    rr = roadrunner.RoadRunner()
    rr.load(test_model)
    rr.setValue('k1', 0.3)
    data = rr.simulate(0, 10, 49) 

    #Calculate Chi squares
    chiPlugin                       = Plugin("tel_chisquare")
    chiPlugin.NrOfModelParameters   = 1
    chiPlugin.ExperimentalData      = expData
    chiPlugin.ModelData             = getDataSeries(data)
        
    chiPlugin.execute()
    
    
    print 'Chi Square = ' + `chiPlugin.ChiSquare`
    print 'Reduced Chi Square = ' + `chiPlugin.ReducedChiSquare`        

    #Plot data sets
    exNP = chiPlugin.ExperimentalData.toNumpy
    x1 = exNP[:,[0]];    y1 = exNP[:,[1]];   y2 = exNP[:,[2]]    
    plt.plot(x1,y1, 'o', x1, y2, 'o')              
    
    #Simulated data
    model = chiPlugin.ModelData.toNumpy
    x1 = model[:,[0]];    y1 = model[:,[1]];   y2 = model[:,[2]]    
    plt.plot(x1,y1, '-b', x1, y2, '-g')
    
    Chi = round(chiPlugin.ChiSquare,4)
    redChi = round(chiPlugin.ReducedChiSquare,4)
      
    plt.text(6,0.00008, r"\begin{eqnarray*}\chi^2 &=" + `Chi` + r"\\ \chi^2_{reduced} &= " + `redChi` + r"\end{eqnarray*}", fontsize=21, color='red')            
    plt.show()                       
    print "done"

except Exception as e:
    print 'Problem: ' + `e`