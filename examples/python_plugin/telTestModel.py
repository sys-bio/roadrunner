#-------------------------------------------------------------------------------
# Purpose: Example demonstrating how to setup and use the TestModel plugin
# This example also shows how to plot data and how to view a plugins 
# embedded manual 
#
# Authors: Totte Karlsson (totte@dunescientific.com), J Kyle Medley
#-------------------------------------------------------------------------------
from rrplugins import Plugin

modelPlugin = Plugin("tel_test_model")            

#Test model plugin depends on the add_noise plugin
noisePlugin = Plugin("tel_add_noise")

#Generate internal test data
modelPlugin.execute()
test_data = modelPlugin.TestData
test_data_with_noise = modelPlugin.TestDataWithNoise
    
test_data.plot()
test_data_with_noise.plot()
    
#modelPlugin.viewManual()   
print('Plugin version: {}'.format(modelPlugin.getVersion()))