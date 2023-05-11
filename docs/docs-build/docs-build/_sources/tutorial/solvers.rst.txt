Roadrunner Solvers
===========================


RoadRunner has multiple types of solvers including integrators and steady-state solvers.
Integrators control numerical timecourse integration via the :meth:`~RoadRunner.simulate()` method.
By default, RoadRunner uses CVODE, a real differential equation solver from the
SUNDIALS suite. Internally, CVODE features an adaptive timestep. However, unless `variableStep`
is specified in the call to :meth:`~RoadRunner.simulate()`, the output will contain evenly spaced intervals.

  >>>  rr.simulate(0, 10, 10)
  # Output will contain evenly spaced intervals
  >>>  rr.simulate(variableStep=True)
  # Intervals will vary according to CVODE step size

To use basic 4th-order Runge-Kutta integrator ('rk4'), call :meth:`~RoadRunner.setIntegrator()`:

  >>>  rr.setIntegrator('rk4')

Runge-Kutta always uses a fixed step size, and does not support events.
RoadRunner supports Runge-Kutta-Fehlberg Method ('rkf45') as well as a stochastic integrator based on Gillespie algorithm ('gilliespie'). To get a list of all available integrators, run:

  >>>  roadrunner.integrators
  ['cvode', 'gillespie', 'rk4', 'rk45']

Some integrators, such as CVODE, have parameters which can be set by the user.
To see a list of these settings, use :meth:`~roadrunner.Solver.getSettings()` on an integrator instance:

  >>>  rr.getIntegrator().getSettings()
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

To set or get a parameter, you can use the methods described below:

  >>>  rr.getIntegrator().setValue('relative_tolerance', 1e-10)
  >>>  rr.getIntegrator().getValue('initial_time_step')

Be sure to set the parameter to the correct type, which can be obtained from
the parameter's hint or description:

  >>>  rr.getIntegrator().getHint('relative_tolerance')
  'Specifies the scalar relative tolerance (double).'
  >>>  rr.getIntegrator().getDescription('relative_tolerance')
  '(double) CVODE calculates a vector of error weights which is used in all error and convergence tests. The weighted RMS norm for the relative tolerance should not become smaller than this value.'

Parameters also have a display name:

  >>>  rr.getIntegrator().getDisplayName('relative_tolerance')
  'Relative Tolerance'

If you prefer to change settings on integrators without switching the current integrator,
you can use :meth:`~RoadRunner.getIntegratorByName()` as follows:

  >>>  rr.getIntegratorByName('gillespie').setValue("seed", 12345)

Also, if you find yourself switching back and forth between integrators a lot, you can use
:meth:`~RoadRunner.setIntegratorSetting()`.

  >>>  rr.setIntegratorSetting('gillespie', 'seed', 12345)

The other type of solver is a steady-state solver, which works in essentially the same way:

  >>>  rr.getSteadyStateSolver().getSettings()
  ('maximum_iterations',
  'minimum_damping',
  'relative_tolerance')
  >>>  rr.getSteadyStateSolver().getHint('maximum_iterations')
  'The maximum number of iterations the solver is allowed to use (int)'
  >>>  rr.getSteadyStateSolver().getDescription('maximum_iterations')
  '(int) Iteration caps off at the maximum, regardless of whether a solution has been reached'

The steady state solver is invoked by a call to :meth:`~RoadRunner.steadyState()`.
Currently, RoadRunner only has a single steady state solver (NLEQ).
