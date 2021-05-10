from teplugins import *

try:    
    #Plot Monte Carlo data sets        
    dataSeries =  DataSeries.readDataSeries("MCDataSets.dat")    
    dataSeries.plot()

    print "done"
except Exception as e:
    print 'Exception: ' + `e`
