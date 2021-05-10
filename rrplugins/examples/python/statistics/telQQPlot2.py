import roadrunner
import numpy as np
import matplotlib.pyplot as plt
import rrplugins
import ctypes as ct

def pluginIsProgressing(lm, ignore):
    lmObject = ct.cast(lm, ct.py_object).value
    print('Iteration, Norm =', lmObject.getProperty("NrOfIter"), ',', lmObject.getProperty("Norm")) #Retrieve plugin parameters
    
chiPlugin   = rrplugins.Plugin("tel_chisquare")
#Retrieve a SBML model from plugin        
modelPlugin= rrplugins.Plugin("tel_test_model")        
sbmlModel= modelPlugin.Model   
 
# Create a roadrunner instance and create some data
modelPlugin.TimeEnd = 10
modelPlugin.NumPoints = 51
modelPlugin.execute()

#Add noise to the data
noisePlugin = rrplugins.Plugin ("tel_add_noise")

# Assign the dataseries to the plugin inputdata
noisePlugin.InputData = modelPlugin.TestData

# Set parameter for the 'size' of the noise
noisePlugin.Sigma = 3.e-1

# Add the noise
noisePlugin.execute()

#===================================================================
lm = rrplugins.Plugin ("tel_levenberg_marquardt")
   
lm.ExperimentalData = noisePlugin.InputData;
lm.SBML = sbmlModel

# Add the parameters that we're going to fit and the initial value
lm.InputParameterList = ["k1", .1]   
lm.FittedDataSelectionList = "[S1] [S2]"
lm.ExperimentalDataSelectionList = "[S1] [S2]"

lm.OnProgress (pluginIsProgressing)

# Execute lmfit plugin
res = lm.execute()

#======== Use the stat plugin to calculate statistics

# The 'stat' plugin no longer exists; it can be replaced with normal Python statistics modules.
# stat = rrplugins.Plugin("tel_stat")

# #Add an offset to residuals to test QQ plot behaviour
# res = rrplugins.DataSeries(lm.Residuals)

# #print(res.cols
# #for col in range(res.cols):
# #    for row in range(res.rows):
# #        res.setElement(row,col, res.getElement(row, col), 1)
        
# stat.Residuals = lm.Residuals 
# stat.execute()
    
# # Get the residuals    
# residuals  = stat.Residuals.toNumpy
# residuals = residuals[:,[1,2]]

# #Plot as a histogram
# plt.hist(residuals, 50, normed=True)
# plt.show()
    
# stdResiduals = stat.StandardizedResiduals.toNumpy
# stdResiduals = stdResiduals[:,[1,2]]            
# plt.hist(stdResiduals, 50, normed=True)
# plt.show()
    
# #Plot normal probability plots
# probPlots = stat.NormalProbabilityOfResiduals.toNumpy
# print(probPlots)

# x1 = probPlots[:,[0]]
# y1 = probPlots[:,[1]]    
    
# x2 = probPlots[:,[2]]
# y2 = probPlots[:,[3]]    

# lineX = np.arange(-5, 5, .1)
# lineY = lineX
# plt.plot(lineX, lineY, '-r')         
# plt.plot(x1,y1, 'o')
# plt.plot(x2,y2, 'x')    
# #plt.scatter(x,y)    
# plt.show()
