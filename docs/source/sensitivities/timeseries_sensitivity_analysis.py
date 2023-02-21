############################################
# Time series sensitivities example
#

"""
In this example we show you how to use roadrunners time series sensitivities
feature. Roadrunner will integrate the model and compute sensitivities at
each time step. You can select which parameters you want sensitivities for
and you can have the return matrix (a 3D numpy.ndarray) as the kth order
derivative of the sensitivity matrix.
"""

import roadrunner
from roadrunner import RoadRunner
from roadrunner.tests.TestModelFactory import TestModelFactory, getAvailableTestModels

# For this demonstration we use a toy sbml model prepackaged with roadrunner

# note: use getAvailableTestModels() to see options for test models.
sbml = TestModelFactory("OpenLinearFlux").str() # get the test model's sbml string

# print(sbml) # uncomment if you want to see the sbml

# Create a RoadRunner instance
model = RoadRunner(sbml)
sensSolver = model.getSensitivitySolver()

time, sens, rownames, colnames = model.timeSeriesSensitivities(0, 10, 11)

print("time: ", time)
print("rownames: ", rownames)
print("colnames: ", colnames)

# Note that in this model, the parameter kb is not used in any
# rate law and therefore the sensitivities of model variables
# to kb is always 0
print(sens)

# By default, all model parameters are automatically selected.
# You can be more selective about which parameters to compute sensitivities for.
# This will have a performance advantage, because sensitivities are only computed
# for the parameters specified here.
time, sens, rownames, colnames = model.timeSeriesSensitivities(0, 10, 11, params=["kin"])

print("time: ", time)
print("rownames: ", rownames)
print("colnames: ", colnames)
print(sens)

# By default, all model species are returned in the results matrix. We can
# be more selective by using the species argument
time, sens, rownames, colnames = model.timeSeriesSensitivities(0, 10, 11, species=["S1"])

print("time: ", time)
print("rownames: ", rownames)
print("colnames: ", colnames)
print(sens)

# The return type, by default, returns the k=0th derivative of the
# sensitivity matrix at each time point. The kth order derivative
# be obtained instead by using the k parameter
time, sens, rownames, colnames = model.timeSeriesSensitivities(0, 10, 11, k=1)

print("time: ", time)
print("rownames: ", rownames)
print("colnames: ", colnames)
print(sens)

# RoadRunner time series sensitivities uses cvodes under the hood. By default,
# we use the "simultaneous" method, whereby sensitivity equations are solved at the same
# time as the variables in the ODE's. This can be changed so that the variables
# are solved first followed by the sensitivity equations; aka in a staggered approach.
sensSolver.setValue("sensitivity_method", "staggered")
time, sens, rownames, colnames = model.timeSeriesSensitivities(0, 10, 11, k=0)

print("using a staggered approach")
print("time: ", time)
print("rownames: ", rownames)
print("colnames: ", colnames)
print(sens)
sensSolver.setValue("sensitivity_method", "simultaneous") # return to simultaneous mode


# RoadRunner via cvodes uses finite differences for approximating gradients.
# By default we use the "centered" method, while the alternative is the "forward" method
# Both methods are implemented in cvodes and automatically figure out the best value of delta
# (perturbation amount) to use. Moreover, there is an adaptive switching mechanism in use
# that makes it very unlikely that users should need to modify this setting.
sensSolver.setValue("DQ_method", "forward")
time, sens, rownames, colnames = model.timeSeriesSensitivities(0, 10, 11, k=0)


print("using forward method of finite different approximation")
print("time: ", time)
print("rownames: ", rownames)
print("colnames: ", colnames)
print(sens)
sensSolver.setValue("DQ_method", "centered") # put back the default value

# By default, RoadRunner uses the sundials "newton" nonlinear solver.
# This can be changed to the fixed_point nonlinear solver
sensSolver.setValue("nonlinear_solver", "fixed_point")
time, sens, rownames, colnames = model.timeSeriesSensitivities(0, 10, 11, k=0)

print("using the fixed point nonlinear solver")
print("time: ", time)
print("rownames: ", rownames)
print("colnames: ", colnames)
print(sens)

