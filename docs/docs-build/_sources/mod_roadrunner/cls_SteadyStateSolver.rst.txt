Steady State Solver Class
_________________________

.. class:: roadrunner.SteadyStateSolver

    A Steady State Solver gives the steady states of involved species. Currently, libRoadRunner supports NLEQ2 solver. For solver related methods, refer to http://sys-bio.github.io/roadrunner/python_docs/api_reference.html#solver-class.


NLEQ2
-----

   NLEQ2 is a non-linear equation solver which uses a global Newton method with adaptive damping strategies (see http://www.zib.de/weimann/NewtonLib/index.html).

.. attribute:: SteadyStateSolver.allow_presimulation

    Flag for starting steady state analysis with simulation. This flag does not affect the usage of approximation routine when the default steaty state solver fails. Default value is false.

    To enable, type:

    >>> r.getSteadyStateSolver().allow_presimulation = True


.. attribute:: SteadyStateSolver.presimulation_tolerance


    Tolerance for presimulation before steady state analysis. Absolute tolerance used by presimulation routine. Only used when allow_presimulation is True. Default value is 1e-3.

    >>> r.getSteadyStateSolver().presimulation_tolerance = 1e-2


.. attribute:: SteadyStateSolver.presimulation_maximum_steps

    Maximum number of steps that can be taken for presimulation before steady state analysis. Takes priority over presimulation_time. Only used when allow_presimulation is True. Default value is 100.

    >>> r.getSteadyStateSolver().presimulation_maximum_steps = 50


.. attribute:: SteadyStateSolver.presimulation_time

    End time for presimulation steady state analysis. Only used when allow_presimulation is True. Default value is 100.

    >>> r.getSteadyStateSolver().presimulation_time = 50


.. attribute:: SteadyStateSolver.allow_approx

    Flag for using steady state approximation routine when steady state solver fails. Approximation routine will run only when the default solver fails to fine a solution. This flag does not affect usage of approximation routine for pre-simulation. Default is True.

    To disable, type:

    >>> r.getSteadyStateSolver().allow_approx = False


.. attribute:: SteadyStateSolver.approx_tolerance

    Tolerance for steady state approximation routine. Absolute tolerance used by steady state approximation routine. Only used when steady state approximation routine is used. Default value is 1e-12.

    >>> r.getSteadyStateSolver().approx_tolerance = 1e-6


.. attribute:: SteadyStateSolver.approx_maximum_steps

    Maximum number of steps that can be taken for steady state approximation routine. Takes priority over approx_time. Only used when steady state approximation routine is used. Default value is 10000.

    >>> r.getSteadyStateSolver().approx_maximum_steps = 5000


.. attribute:: SteadyStateSolver.approx_time

    End time for steady state approximation routine. approx_maximum_steps takes priority. Only used when steady state approximation routine is used. Default value is 10000.

    >>> r.getSteadyStateSolver().approx_time = 5000


.. attribute:: SteadyStateSolver.broyden_method

    Switches on Broyden method, which is a quasi-Newton approximation for rank-1 updates. Default value is 0.

    To enable, type:
   
   >>> r.getSteadyStateSolver().broyden_method = 1


.. attribute:: SteadyStateSolver.linearity

    Specifies linearity of the problem. 1 is for linear problem and 4 is for extremly nonlinear problem. Default value is 3.

    >>> r.getSteadyStateSolver().linearity = 1


.. attribute:: SteadyStateSolver.maximum_iterations

    The maximum number of iterations the solver is allowed to use. Iteration caps off at the maximum, regardless of whether a solution has been reached. Default value is 100.

    >>> r.getSteadyStateSolver().maximum_iterations = 50


.. attribute:: SteadyStateSolver.minimum_damping

    The minimum damping factor used by the algorithm. Default value is 1e-4.

    >>> r.getSteadyStateSolver().minimum_damping = 1e-20


.. attribute:: SteadyStateSolver.relative_tolerance

    Specifies the relative tolerance used by the solver. Default value is 1e-16.

    >>> r.getSteadyStateSolver().relative_tolerance = 1e-15


.. method:: SteadyStateSolver.solve()

    Main solver routine