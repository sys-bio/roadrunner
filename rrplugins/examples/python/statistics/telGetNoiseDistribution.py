# Show that add noise plugin correctly computes Sigma (standard deviation)
import matplotlib.pyplot as plt
import scipy.stats as stats
import rrplugins as tel
import numpy as np

p = tel.Plugin ("tel_add_noise")

value = 2.34    #This will be the mean
n = 8000
inputData  = np.zeros (shape=(1,2))
inputData[0] = [0, value]

data = tel.DataSeries.fromNumPy (inputData)
p.Sigma = 0.25

outArray = []
for i in range(n):
    p.InputData = data       
    p.execute()
    outValues = p.InputData.toNumPy()
    outArray.append(outValues[0][1])
    
plt.hist(outArray, 200)

# Overlay analytical solution
aRange = np.arange(min(outArray), max(outArray), 0.001)
plt.plot(aRange, stats.norm.pdf(aRange, value, p.Sigma), linestyle='--', linewidth='2', color='red')

plt.show()


