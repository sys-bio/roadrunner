import roadrunner

import rrplugins as tel

rr = roadrunner.RoadRunner() 

rr.load("path to any model. Eg. some models are provided in source code, like, sbml_test_0001") 

data = rr.simulate(0, 10, 511)

noisePlugin = tel.Plugin ("tel_add_noise") 

d = tel.getDataSeries (data) 

noisePlugin.InputData = d 

noisePlugin.Sigma = 3.e-6 

noisePlugin.execute()

noisePlugin.InputData.plot()

