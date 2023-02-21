from roadrunner import RoadRunner
from roadrunner.tests.TestModelFactory import TestModelFactory, getAvailableTestModels


# note: use getAvailableTestModels() to see options for test models.
sbml = TestModelFactory("OpenLinearFlux").str() # get the test model's sbml string

# print(sbml) # uncomment if you want to see the sbml

# Create a RoadRunner instance
model = RoadRunner(sbml)
sensSolver = model.getSensitivitySolver()



# By default, RoadRunner uses the sundials "newton" nonlinear solver.
# This can be changed to the fixed_point nonlinear solver
sensSolver.setValue("nonlinear_solver", "fixed_point")
time, sens, rownames, colnames = model.timeSeriesSensitivities(0, 10, 11, k=0)

print("using the fixed point nonlinear solver")
print("time: ", time)
print("rownames: ", rownames)
print("colnames: ", colnames)
print(sens)
