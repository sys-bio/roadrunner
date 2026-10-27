Running Simulations
===========================

Once a model is successfully loaded we can run a time course simulation. To run a simulation we use the
:meth:`~RoadRunner.simulate()` method::

   result = rr.simulate()

The output will be in a Python numpy array. The first column will contain time points and the remaining columns will include
all the floating species amounts/concentrations. In the simulate method we didn't specify how long to run the simulation or how many
points to generate.  By default the starting time is set to zero, ending time to 10 and the number of points to 51.
There are two ways to set these values. The easiest way is to change the positional arguments in :meth:`~RoadRunner.simulate()`
in the following manner::

   result = rr.simulate (0, 10, 100)

This will set the starting time to zero, the ending time to 10 and generate 100 points. This means that the result will
be out in time intervals of 1/99.

The simulate method also accepts the steps keyword argument instead of points::

  result = rr.simulate(0, 10, steps=99)

For more details of the simulate method see :meth:`~RoadRunner.simulate()`.
The following table summarizes the various options.

================  =============
 Option           Description
================  =============
start             Starting time for simulation
end               Ending time for simulation. Setting 'end' will automatically change 'duration'
points            Number of rows to include in the output matrix
selections        (Optional) A list of variables to include in the output, e.g. ``['time','A']`` for a model with species ``A``. More below.
steps             (Optional keyword argument) Number of steps at which the output is sampled where the samples are evenly spaced. Steps = points-1. Steps and points may not both be specified.
================  =============

One important point to note about :meth:`~RoadRunner.simulate()`: When :meth:`~RoadRunner.simulate()` is run, the concentration of
the floating species will naturally change. If :meth:`~RoadRunner.simulate()` is called a second time, the simulation
will start the simulation from the previous simulated values. This can be used to easily follow on
simulations. However there will be times when we wish to run the same simulation again but perhaps
with slightly different parameters values. For this we must reset the model to its initial conditions. To do that we run the command :meth:`~RoadRunner.reset()`::

   rr.reset()

RoadRunner also has two other reset methods: :meth:`~RoadRunner.resetAll()`, which resets everything to the *model's* copy of the initial values, and :meth:`~RoadRunner.resetToOrigin()`, which resets everything to *roadrunner's* copy of the initial values. The model's initial values are drawn from roadrunner when the model is created, and can be subsequently set with `rr.model.setValue('init(x)')`.  Roadrunner's initial values are drawn from the SBML model when that model is read, and can be subsequently set with `rr.setValue('init(x)')`.  This immediately regenerates the model, clearing any model initial values that might have been set previously.
