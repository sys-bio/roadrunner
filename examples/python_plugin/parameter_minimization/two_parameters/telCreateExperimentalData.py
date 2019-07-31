import roadrunner
from teplugins import *

try:
    # Create a roadrunner instance and create some data             
    rr = roadrunner.RoadRunner()
    rr.load('two_parameters.xml') 
    
    rr.setValue("k1", 1.3)   
    rr.setValue("k2", 2.5)
   
    timeStart = 0
    timeEnd = 1.5
    nrPoints  = 15
    data = rr.simulate(0, timeEnd, nrPoints  - 1)

    #Add some noise to the data
    noisePlugin = Plugin ("tel_add_noise")

    # Get the dataseries from data returned by roadrunner
    d = getDataSeries (data)

    # Assign the dataseries to the plugin inputdata
    noisePlugin.InputData = d

    # Set parameter for the 'size' of the noise
    sigma = 2.5e-3
    noisePlugin.Sigma = sigma
        
    # Add the noise
    noisePlugin.execute()

    # Get the data to plot
    theData = noisePlugin.InputData
    theData.plot()
            
    # Need to add weight stuff to DataSeries class?            
    dataHandle = theData._data
    if not hasWeights(dataHandle):
        allocateWeights(dataHandle)
                    
    #Populate weights, using the sigma above
    hdrs = list(theData.getColumnHeaders().split(','))
    
    for c in range(theData.cols):
        for r in range(theData.rows):    
            if hdrs[c].lower() != "Time".lower():
                weight = theData.getWeight(r,c)
                theData.setWeight(r,c, sigma * sigma)                
    
    noisePlugin.InputData.writeDataSeries("ExperimentalData.dat")
    print "done"

except Exception as e:
    print 'Problem: ' + `e`
