#-------------------------------------------------------------------------------
# Purpose: Example demonstrating how to setup the add noise pluging and
# add noise to simulated data
#
# Authors: Totte Karlsson (totte@dunescientific.com), J Kyle Medley
#-------------------------------------------------------------------------------
from rrplugins import Plugin

modelPlugin = Plugin("tel_test_model")        
noisePlugin = Plugin("tel_add_noise")

#Generate internal test data
modelPlugin.execute()
test_data = modelPlugin.TestData
    
test_data.plot()
# Assign the dataseries to the plugin inputdata
noisePlugin.InputData = test_data

# Set parameter for the 'size' of the noise
noisePlugin.Sigma = 8.e-6

# Add the noise
noisePlugin.execute()

# Get the data to plot
noisePlugin.InputData.plot()            

#noisePlugin.viewManual()
print('Plugin version: {}'.format(noisePlugin.getVersion()))