import roadrunner
import numpy as np
import matplotlib.pyplot as plt
import rrplugins

timeStart = 0
timeEnd = 4.5
nrPoints = 15         
k = 0.57

modelPlugin = rrplugins.Plugin("tel_test_model")   
test_model  = modelPlugin.Model    
rr = roadrunner.RoadRunner()
rr.load(test_model)

data = rr.simulate(timeStart, timeEnd, nrPoints -1)
noisePlugin = rrplugins.Plugin ("tel_add_noise")    
d = rrplugins.getDataSeries (data)    
noisePlugin.InputData = d    
noisePlugin.Sigma = 1.34e-6    
noisePlugin.execute()

# Need to add weight stuff to DataSeries class?
theData = noisePlugin.InputData                
dataHandle = theData._data
if not rrplugins.hasWeights(dataHandle):
    rrplugins.allocateWeights(dataHandle)
                
#Populate weights, using the sigma above
hdrs = theData.getColumnHeaders()

for c in range(theData.cols):
    for r in range(theData.rows):    
        if hdrs[c].lower() != "Time".lower():
            weight = theData.getWeight(r,c)
            theData.setWeight(r,c, noisePlugin.Sigma * noisePlugin.Sigma)                

expData = noisePlugin.InputData.toNumPy()
x = expData[:,0]
plt.plot(x, expData[:,1],'og')        
plt.plot(x, expData[:,2],'or')  
     
data = data.view(np.float).reshape(data.shape + (-1,))
plt.plot(x, data[:,1],'-g')
plt.plot(x, data[:,2],'-r')
    
plt.show()

#Calculate chi square
chiSquare =  rrplugins.Plugin("tel_chisquare")    
chiSquare.ExperimentalData = noisePlugin.InputData
chiSquare.ModelData = d
chiSquare.NrOfModelParameters = 1

if not chiSquare.execute():
    raise Exception( rrplugins.getLastError() )
       
redChi = chiSquare.ReducedChiSquare               
print('Reduced chisquare =', redChi) 


