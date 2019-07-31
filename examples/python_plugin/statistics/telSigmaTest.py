# Show that add noise plugin correctly computes Sigma (standard deviation)

import teplugins as tel
import numpy

p = tel.Plugin ("tel_add_noise")

n = 80000
timeSeries = numpy.zeros (shape=(n,2))
for i in range (n):
    timeSeries[i] = [i, 1]

data = tel.DataSeries.fromNumPy (timeSeries)

p.InputData = data
p.Sigma = 0.25

p.execute()

values = p.InputData.toNumpy

print
print "Testing that Sigma is generating the expected data..."
print "The following data should roughly equal 1.0 and 0.25\n"

print "mean=", numpy.mean (values[:,1])
print "std=", numpy.std (values[:,1])



