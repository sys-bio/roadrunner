**************
Basic Tutorial
**************

.. currentmodule:: RoadRunner

Import RoadRunner
-----------------

To startup RoadRunner use the commands at the Python prompt::

   import roadrunner
   rr = roadrunner.RoadRunner()

The variable ``rr`` is your reference to roadrunner. Anything you want to do with roadrunner must be
specified using the variable ``rr``.

To get the current version of libRoadRunner type the following::

   roadrunner.__version__

.. _loading-models:

Loading Models
--------------

RoadRunner reads models using the SBML format. If you have a SBML model stored on your hard drive, it is
possible to load that model either by giving he document contents or path to the Roadrunner
constructor, or later by using the method, :meth:`~RoadRunner.load()`. Let's assume you have a model called
mymodel.xml in ``C:\MyModels``. To load this model in **Windows** we would use the command::

   rr = roadrunner.RoadRunner("C:/MyModels/mymodel.xml")

Note, Windows typically used the back slash, '\\' to indicate a directory separator, however in
most languages including python, this is the escape character, therefore one can also enter Windows
paths using the forward slash '/' which does not cause issues. If one want to use the backslash,
these must be typed twice, i.e. ``'C:\\MyModels\\mymodel.xml'``.

On the **Mac or Linux** we might use::

   rr = roadrunner.RoadRunner("/home/MyModels/mymodel.xml")

RoadRunner can also load models directly from a URL via::

   rr = roadrunner.RoadRunner("http://www.ebi.ac.uk/biomodels-main/download?mid=BIOMD0000000010")

If the model was loaded successfully, the RoadRunner object is now ready to use,  otherwise an exception will
be raised that contains extended information detailing exactly what failed. If any warnings are
found in the SBML document, these will be displayed in the console error log.

It is also possible to load a model from a string containing a SBML model, for example:

   rr = roadrunner.RoadRunner (sbmlStr)

This is useful when one wishes to create a new roadrunner instance from an existing model, eg:

   sbmlStr = rr.getCurrentSBML()
   rrnew = roadrunner.RoadRunner (sbmlStr)

Additionally, there are a couple models **included with libRoadRunner**. The models  ``feedback.xml``
and ``Test_1.xml`` are available in the ``roadrunner.testing`` module. To access these use::

   import roadrunner.testing as test
   r = test.getRoadRunner('feedback.xml')

There are a few additional models in the ``models/`` directory of the distribution, where you installed libRoadRunner.

Running Simulations
-------------------

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

RoadRunner also has two other reset methods: :meth:`~RoadRunner.resetAll()` and :meth:`~RoadRunner.resetToOrigin()`. These are typically only used in advanced scenarios. This is because RoadRunner maintains its own copy of the "initial" value of every quantity, which can be set via ``rr_instance.setValue('init(quantity)',123)``. When :meth:`~RoadRunner.resetAll()` is called, it resets the quantity to RoadRunner's internal copy, not the SBML-defined value. :meth:`~RoadRunner.resetToOrigin()` completely reverts everything in the model back to the SBML-specified values, whereas :meth:`~RoadRunner.resetAll()` preserves any changes you have made to initial values.

Changing Parameters
-------------------

Often during a modeling experiment we will need to change parameter values, that is the values of the various
kinetic constants in the model. If a model has a kinetic constants k1, then we can change or inspect the value using
the following syntax::

   print rr.k1
   rr.k1 = 1.2


Selecting Simulation Output
---------------------------

RoadRunner supports a range of options for selecting what data a simulation should return. For more
detailed information on selections, see the :ref:`selecting-values` section.

The simulate method, by default returns an `structured array
<http://docs.scipy.org/doc/numpy/user/basics.rec.html>`_,
which are arrays that also contain column names. These can be plotted directly using the
built in :meth:`~RoadRunner.plot()` function.

The output selections default to time and the set of floating species.
It is possible to change the simulation result values by changing the selection list.
For example assume that a model has three species, S1, S2, and S3 but we only want :meth:`~RoadRunner.simulate()` to
return time in the first column and S2 in the second column. To specify this we would type::

   rr.timeCourseSelections = ['time', 'S2']
   result = rr.simulate (0, 10, 100)

In another example let say we wanted to plot a phase plot where S1 is plotted against S2. To do this we
type the following::

   rr.timeCourseSelections = ['S1', 'S2']
   result = rr.simulate(0, 10, 100)

Some additional examples include:

.. code-block:: Python

   # Select time and two rates of change (dS1/dt and dS2/dt)
   rr.timeCourseSelections = ["time", "S1'", "S2'"]

   # By default species names yield amounts, concentrations can be obtained
   # using square brackets, e.g.
   rr.timeCourseSelections = ['time', '[S1]', '[S2]']

.. seealso:: More details on :doc:`selecting_values`

.. _plotting-data:

Plotting Data
-------------

RoadRunner has a built in :meth:`~RoadRunner.plot()` method which can perform basic plotting.
Simply call::

   result = rr.simulate(0, 10, 100)
   rr.plot()

If one wants more control over the data plots, one may use matplotlib directly.  Assuming the
simulate returns an array called result, and that the first column represents the x axis and the
remaining columns the y axis, we type::

   import pylab
   pylab.plot (result[:,0],result[:,1:])
   pylab.show()

This will bring up a new window showing the plot. To clear the plot for next time, type the command::

   pylab.clf()

One may also override the built-in :meth:`~RoadRunner.plot()` method with a more more capable plotting routine.

Below is a simplified version of the :meth:`~RoadRunner.plot()` method. You may copy and write a
customized version and even attach it to the RoadRunner object. The first argument is a RoadRunner
object instance, and the second is a flag which tells the method to show the plot or not::

  def plot(r, show=True):

      import pylab as p

      result = self.getSimulationData()

      if result is None:
          raise Exception("no simulation result")

      # assume result is a standard numpy array

      selections = r.timeCourseSelections

      if len(result.shape) != 2 or result.shape[1] != len(selections):
          raise Exception("simulation result columns not equal to number of selections,"
                          "likely a simulation has not been run")

      times = result[:,0]

      for i in range(1, len(selections)):
          series = result[:,i]
          name = selections[i]
          p.plot(times, series, label=str(name))

          p.legend()

      if show:
          p.show()


You can attach your plotting function to the RoadRunner object by simply setting the plot
method::

  def my_plot(r, show):
      pass

  import roadrunner
  roadrunner.RoadRunner.plot = my_plot

Now, whenever the :meth:`~RoadRunner.plot()` method is called, your plot function will be the one that is
invoked.


Changing Initial Conditions
---------------------------


There are a number of methods to get and set the initial conditions of a loaded model. In order to
specify a given initial conditions we use the notation, ``init()``.  The values stored in the
initial conditions are applied to the model whenever it is reset. The list of all initial condition
symbols can be obtained by the methods, :meth:`~ExecutableModel.getFloatingSpeciesInitAmountIds()`
and :meth:`~ExecutableModel.getFloatingSpeciesInitConcentrationIds()` assuming ``r`` is a RoadRunner
instance. As with all other selection symbols, the :meth:`~ExecutableModel.keys()` returns all
available selection symbols:

  >>>  r.model.keys()
  [ 'S1', 'S2', '[S1]', '[S2]', 'compartment', 'k1', '_CSUM0',
    'reaction1',  'init([S1])',  'init([S2])', 'init(S1)',
    'init(S2)',  "S1'"]

Symbols for selecting initial values specifically for amounts and concentrations can be obtained
via:

  >>> r.model.getFloatingSpeciesInitAmountIds()
  ['init(S1)', 'init(S2)']

  >>> r.model.getFloatingSpeciesInitConcentrationIds()
  ['init([S1])', 'init([S2])']

Getting or setting initial values is easily accomplished using the array operator and the selection
symbols:

  >>> r.model["init(S1)"]
  0.00015

  >>> r.model["init([S1])"]
  2.9999999999999997e-05

  >>> r.model["init([S1])"] = 2

  >>> r.model["init(S1)"]
  10.0

The values for the initial conditions for all floating species can be obtained using the calls:

  >>> r.model.getFloatingSpeciesInitConcentrations()
  array([ 0.7,  5.6])

Initial conditions can be set using the two methods for all species in one call:

  >>> r.model.setFloatingSpeciesInitAmounts ([3.4, 5.6])


  >>> r.model.setFloatingSpeciesInitConcentrations ([6.7, 0.1])

.. _roadrunner-solver:

Solvers
---------------------------


RoadRunner has multiple types of solvers including integrators and steady-state solvers.
Integrators control numerical timecourse integration via the :meth:`~RoadRunner.simulate()` method.
By default, RoadRunner uses CVODE, a real differential equation solver from the
SUNDIALS suite. Internally, CVODE features an adaptive timestep. However, unless `variableStep`
is specified in the call to :meth:`~RoadRunner.simulate()`, the output will contain evenly spaced intervals.

  >>>  r.simulate(0, 10, 10)
  # Output will contain evenly spaced intervals
  >>>  r.simulate(variableStep=True)
  # Intervals will vary according to CVODE step size

To use basic 4th-order Runge-Kutta integrator ('rk4'), call :meth:`~RoadRunner.setIntegrator()`:

  >>>  r.setIntegrator('rk4')

Runge-Kutta always uses a fixed step size, and does not support events.
RoadRunner supports Runge-Kutta-Fehlberg Method ('rkf45') as well as a stochastic integrator based on Gillespie algorithm ('gilliespie'). To get a list of all available integrators, run:

  >>>  roadrunner.integrators
  ['cvode', 'gillespie', 'rk4', 'rk45']

Some integrators, such as CVODE, have parameters which can be set by the user.
To see a list of these settings, use :meth:`~roadrunner.Solver.getSettings()` on an integrator instance:

  >>>  r.getIntegrator().getSettings()
  ('relative_tolerance',
  'absolute_tolerance',
  'stiff',
  'maximum_bdf_order',
  'maximum_adams_order',
  'maximum_num_steps',
  'maximum_time_step',
  'minimum_time_step',
  'initial_time_step',
  'multiple_steps',
  'variable_step_size')

To set a parameter, you can use both methods described below:

  >>>  r.getIntegrator().relative_tolerance = 1e-10
  >>>  r.getIntegrator().setValue('relative_tolerance', 1e-10)

Be sure to set the parameter to the correct type, which can be obtained from
the parameter's hint or description:

  >>>  r.getIntegrator().getHint('relative_tolerance')
  'Specifies the scalar relative tolerance (double).'
  >>>  r.getIntegrator().getDescription('relative_tolerance')
  '(double) CVODE calculates a vector of error weights which is used in all error and convergence tests. The weighted RMS norm for the relative tolerance should not become smaller than this value.'

Parameters also have a display name:

  >>>  r.getIntegrator().getDisplayName('relative_tolerance')
  'Relative Tolerance'

If you prefer to change settings on integrators without switching the current integrator,
you can use :meth:`~RoadRunner.getIntegratorByName()` as follows:

  >>>  r.getIntegratorByName('gillespie').seed = 12345

Also, if you find yourself switching back and forth between integrators a lot, you can use
:meth:`~RoadRunner.setIntegratorSetting()`.

  >>>  r.setIntegratorSetting('gillespie', 'seed', 12345)

The other type of solver is a steady-state solver, which works in essentially the same way:

  >>>  r.getSteadyStateSolver().getSettings()
  ('maximum_iterations',
  'minimum_damping',
  'relative_tolerance')
  >>>  r.getSteadyStateSolver().getHint('maximum_iterations')
  'The maximum number of iterations the solver is allowed to use (int)'
  >>>  r.getSteadyStateSolver().getDescription('maximum_iterations')
  '(int) Iteration caps off at the maximum, regardless of whether a solution has been reached'

The steady state solver is invoked by a call to :meth:`~RoadRunner.steadyState()`.
Currently, RoadRunner only has a single steady state solver (NLEQ).
