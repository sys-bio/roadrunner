from roadrunner import RoadRunner
from roadrunner.tests.TestModelFactory import TestModelFactory, getAvailableTestModels


# note: use getAvailableTestModels() to see options for test models.
sbml = TestModelFactory("OpenLinearFlux").str() # get the test model's sbml string

# print(sbml) # uncomment if you want to see the sbml

# Create a RoadRunner instance
model = RoadRunner(sbml)
sensSolver = model.getSensitivitySolver()

time, sens, rownames, colnames = model.timeSeriesSensitivities(0, 10, 11, species=["S1"])

print("time: ", time)
print("rownames: ", rownames)
print("colnames: ", colnames)
print(sens)
