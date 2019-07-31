#-------------------------------------------------------------------------------
# Purpose: Example demonstrating creation of a new dataseries, 
# adding noise and plotting 
#
# Author: Herbert Sauro
#-------------------------------------------------------------------------------
import teplugins as tel
import numpy
import math

p = tel.Plugin ("tel_add_noise")

n = 200
timeSeries = numpy.zeros (shape=(n,3))
for i in range (n):
    timeSeries[i] = [i, math.sin (i/8.0), math.cos (i/8.0)]

data = tel.DataSeries.fromNumPy (timeSeries)

data.plot ()

p.InputData = data
p.Sigma = 0.1

p.execute()

p.InputData.plot()

