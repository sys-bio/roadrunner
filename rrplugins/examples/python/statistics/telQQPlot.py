import roadrunner
import numpy as np
import matplotlib.pyplot as plt
import rrplugins as tel

chiPlugin   = tel.Plugin("tel_chisquare")

#Retrieve a SBML model from plugin        
modelPlugin= tel.Plugin("tel_test_model")        
sbmlModel= modelPlugin.Model   

# Create a roadrunner instance and create some data
# rr = roadrunner.RoadRunner()
# rr.load(sbmlModel)
# data = rr.simulate(0, 10, 501)

modelPlugin.TimeEnd = 10
modelPlugin.NumPoints = 501
modelPlugin.execute()

#Add noise to the data
noisePlugin = tel.Plugin ("tel_add_noise")

# Assign the dataseries to the plugin inputdata
noisePlugin.InputData = modelPlugin.TestData

# Set parameter for the 'size' of the noise
noisePlugin.Sigma = 8.e-6

# Add the noise
noisePlugin.execute()

#=========== FIT DATA =============================================

lm = tel.Plugin ("tel_levenberg_marquardt")

lm.ExperimentalData = noisePlugin.InputData;
lm.SBML = sbmlModel

# Add the parameters that we're going to fit and the initial value
lm.InputParameterList = ["k1", .1]
lm.FittedDataSelectionList = "[S1] [S2]"
lm.ExperimentalDataSelectionList = "[S1] [S2]"

# Execute lmfit plugin
res = lm.execute()

# Plot some data
experimentalData    = lm.ExperimentalData.toNumPy()
fittedData          = lm.FittedData.toNumPy()
residuals           = lm.Residuals.toNumPy()

tel.telplugins.plot (fittedData[:,[0,1]], myColor="blue", myLinestyle="-", myMarker="", myLabel="S1 Fitted")
tel.telplugins.plot (fittedData[:,[0,2]], myColor="blue", myLinestyle="-", myMarker="", myLabel="S2 Fitted")
tel.telplugins.plot (residuals[:,[0,1]], myColor="blue", myLinestyle="None", myMarker="x", myLabel="S1 Residual")
tel.telplugins.plot (residuals[:,[0,2]], myColor="red", myLinestyle="None", myMarker="x", myLabel="S2 Residual")
tel.telplugins.plot (experimentalData[:,[0,1]], myColor="red", myLinestyle="", myMarker="*", myLabel="S1 Data")
tel.telplugins.plot (experimentalData[:,[0,2]], myColor="blue", myLinestyle="", myMarker="*", myLabel="S2 Data")
tel.plt.show()

#Plot a histogram
bins = 100
stdResiduals = lm.StandardizedResiduals.toNumPy()
stdResiduals = stdResiduals[:,[1,2]]
stdResiduals = np.reshape(stdResiduals, 2*len(stdResiduals[:,[1]]))

plt.hist(stdResiduals, bins)
plt.show()

#Plot normal probability plots
probPlots = lm.NormalProbabilityOfResiduals.toNumPy()

x1 = probPlots[:,[0]]
y1 = probPlots[:,[1]]

lineX = np.arange(-5, 5, .1)
lineY = lineX
plt.plot(x1,y1, 'o')
plt.plot(lineX, lineY, '-r') #Straight line
plt.show()

