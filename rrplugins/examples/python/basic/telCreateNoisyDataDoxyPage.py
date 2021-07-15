import tellurium as te
import rrplugins as tel

noisePlugin = tel.Plugin ("tel_add_noise")

print(noisePlugin.name())
print(noisePlugin.hint())
print(noisePlugin.description())

print('==== List of properties ====')
print(noisePlugin.listOfProperties())

r = te.loadSBMLModel(te.getTestModel("test_1.xml"))
data = r.simulate()

# The plugin will need a handle to the underlying roadrunner data
d = tel.getDataSeries (data)

noisePlugin.InputData = d

# Get parameter for the 'size' of the noise
noisePlugin.Sigma = 0.005

noisePlugin.InputData.plot()
noisePlugin.execute()

noisePlugin.InputData.plot()
tel.show()
