import roadrunner
from teplugins import *

try:
    # Create a roadrunner instance and create some data
    modelPlugin = Plugin("tel_test_model")     
    
    rr = roadrunner.RoadRunner()
    rr.load(modelPlugin.Model) 
    
    rr.setValue("k1", 1.57)   
    rr.reset()   
    timeStart = 0
    timeEnd = 10
    nrPoints  = 50
    data = rr.simulate(0, timeEnd, nrPoints  - 1)

    #Add some noise to the data
    noisePlugin = Plugin ("tel_add_noise")

    # Get the dataseries from data returned by roadrunner
    d = getDataSeries (data)

    # Assign the dataseries to the plugin inputdata
    noisePlugin.InputData = d

    # Set parameter for the 'size' of the noise
    sigma = 5.34e-6
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
    
    noisePlugin.InputData.writeDataSeries("testData.dat")
    print "done"

except Exception as e:
    print 'Problem: ' + `e`
