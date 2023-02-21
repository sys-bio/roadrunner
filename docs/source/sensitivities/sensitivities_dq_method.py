from roadrunner import RoadRunner
from roadrunner.tests.TestModelFactory import TestModelFactory, getAvailableTestModels


# note: use getAvailableTestModels() to see options for test models.
sbml = TestModelFactory("OpenLinearFlux").str() # get the test model's sbml string

# print(sbml) # uncomment if you want to see the sbml

# Create a RoadRunner instance
model = RoadRunner(sbml)
sensSolver = model.getSensitivitySolver()



sensSolver.setValue("DQ_method", "forward")
time, sens, rownames, colnames = model.timeSeriesSensitivities(0, 10, 11, k=0)


print("using forward method of finite different approximation")
print("time: ", time)
print("rownames: ", rownames)
print("colnames: ", colnames)
print(sens)
sensSolver.setValue("DQ_method", "centered") # put back the default value
