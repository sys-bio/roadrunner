#-------------------------------------------------------------------------------
# Purpose: Example demonstrating how to add noise to Tellurium data using the 
# AddNoise plugin. This example also shows how to save the data to a file.
#
# Author:      Totte Karlsson (totte@dunescientific.com)
#-------------------------------------------------------------------------------
import tellurium as te
import rrplugins as tel

noisePlugin = tel.Plugin("tel_add_noise")

r = te.loadSBMLModel(te.getTestModel("Test_1.xml"))
d = r.simulate()
test_data = tel.getDataSeries(d)
        
r.plot()
# Assign the dataseries to the plugin inputdata
noisePlugin.InputData = test_data

# Set parameter for the 'size' of the noise
noisePlugin.Sigma = .005

# Add the noise
noisePlugin.execute()

# Get the data to plot
noisePlugin.InputData.plot()    

#Write the data to file
noisePlugin.InputData.writeDataSeries("testData.dat")        


