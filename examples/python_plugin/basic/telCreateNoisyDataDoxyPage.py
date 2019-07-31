import roadrunner
import teplugins as tel

noisePlugin = tel.Plugin ("tel_add_noise")

print noisePlugin.name()
print noisePlugin.hint()
print noisePlugin.description()

print '==== List of properties ===='
print noisePlugin.listOfProperties()

modelPlugin= tel.Plugin("tel_test_model")        
test_model = modelPlugin.Model

# Create a roadrunner instance
rr = roadrunner.RoadRunner()
rr.load(test_model)

# Generate data
data = rr.simulate(0, 10, 50) 

# The plugin will need a handle to the underlying roadrunner data
d = tel.getDataSeries (data)

noisePlugin.InputData = d

# Get parameter for the 'size' of the noise
noisePlugin.Sigma = 5.e-6

noisePlugin.InputData.plot()
noisePlugin.execute()

noisePlugin.InputData.plot()
tel.show()

print "done"
