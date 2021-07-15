#-------------------------------------------------------------------------------
# Purpose: Example demonstrating how to read Tellurium data from file, and plot it.
#
# Author: Totte Karlsson
#-------------------------------------------------------------------------------
import rrplugins

#Make sure the file testData.dat exists 
#You can generate a testData.dat file by executing the telCreateNoisyData.py example script
dataSeries =  rrplugins.DataSeries.readDataSeries("testData.dat")    
dataSeries.plot()           

print("done")
