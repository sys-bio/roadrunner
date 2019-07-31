#-------------------------------------------------------------------------------
# Purpose: Example demonstrating how to add noise to Tellurium data using the 
# AddNoise plugin. This example also shows how to save the data to a file.
#
# Author:      Totte Karlsson (totte@dunescientific.com)
#-------------------------------------------------------------------------------
import teplugins as tel

try:
    modelPlugin = tel.Plugin("tel_test_model")        
    noisePlugin = tel.Plugin("tel_add_noise")
    
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

    #Write the data to file
    noisePlugin.InputData.writeDataSeries("testData.dat")        

except Exception as e:
    print 'Problem: ' + `e`