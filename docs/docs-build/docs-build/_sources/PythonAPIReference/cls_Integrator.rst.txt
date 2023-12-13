Integrator Class
________________

.. class:: roadrunner.Integrator

    An integrator is used to describe the method of carrying out a mathematical integration. Currently, libRoadRunner supports cvode, gillespie, rk4 and rk45 solvers. For solver related methods, refer to http://sys-bio.github.io/roadrunner/python_docs/api_reference.html#solver-class. The following methods are supported by all the Integrators.


.. method:: Integrator.getIntegrationMethod()

    Gets the method of integration as an integer. (0-Deterministic, 1-Stochastic, 2-Hybrid, 3-Other).


.. method:: Integrator.getListener()

    Gets the integrator listener.


.. method:: Integrator.integrate(t0, hstep)

    Main integration routine.

    :param double t0: start time
    :param double hstep: time step size


.. method:: Integrator.loadSBMLSettings(filename)

    Load an SBML settings file and apply the configuration options.
    Note: Can assign relative and absolute tolerances.

    :param string filename: A string which may be a local path, URI or contents of an SBML document.


.. method:: Integrator.restart(t0)

    Reset time to zero and reinitialize model. Applies events which occur before time zero. Reinitializes CVODE and the executable model.

    :param double t0: start time


.. method:: Integrator.setListener(listener)

    Sets the integrator listener.

    :param listener: Swig Object of type 'rr::PyIntegratorListenerPtr* '


.. method:: Integrator.tweakTolerances()

    Fix tolerances for SBML tests. In order to ensure that the results of the SBML test suite remain valid, this method enforces a lower bound on tolerance values. Sets minimum absolute and relative tolerances to Config::CVODE_MIN_ABSOLUTE and Config::CVODE_MIN_RELATIVE respectively.
    
.. method:: Integrator.setIndividualTolerance(sid, value)

    Sets absolute tolerance for individual floating species or variable that has a rate rule. Only used for CVODE Integrator.
    Note that this tolerance is based on the amount of species, and will be stored in absolute_tolerance.
    
    :param str sid: identifier of individual species or varible that has a rate rule
    :param double value: tolerance value to set
 
    
CVODE
-----

   CVODE is a deterministic ODE solver from the SUNDIALS suite of timecourse integrators. It implements an Adams-Moulton solver for non-stiff problems and a backward differentiation formula (BDF) solver for stiff problems.

.. currentmodule:: CVODE

.. attribute:: Integrator.absolute_tolerance

    Specifies the scalar or vector absolute tolerance. As of roadrunner version 2.4.0, if a scalar absolute tolerance is used, it is multiplied by the value of every element in the state vector before being used.  If an element of the state vector is zero, the scalar is multiplied by 1.0 for non-species values, and by the size of the species compartment for species values (i.e. the amount of species present if its concentration was 1.0).

    If desired, the full absolute tolerance vector may be used instead of a scalar.  In this case, it is assumed that the user has already done any appropriate scaling for individual state vector levels, and no additional scaling is performed: the values are used in Sundials as-is.  Calling 'getValue('absolute_tolerance') will return the scalar or vector set by the user, or the default Config::CVODE_MIN_ABSOLUTE value.  To obtain the full vector obtained from the scalar, use 'getAbsoluteToleranceVector'.

    In Sundials, the CVODE algorithm calculates a vector of error weights which is used in all error and convergence tests. The weighted RMS norm for the absolute tolerance should not become smaller than this value. Default value is Config::CVODE_MIN_ABSOLUTE (a scalar).
    
    >>> rr.integrator.absolute_tolerance = 1e-10
    >>> rr.integrator.absolute_tolerance = [1, 0.1, 0.01, 0.001] // setting various tolerances for each species.

    >>> rr.integrator.absolute_tolerance //Gets the scalar or vector absolute tolerance, as set by the user or the default value.
    >>> rr.integrator.getAbsoluteToleranceVector()   //Gets the absolute tolerance vector, as derived from the scalar value, or as set by the user directly.
    

.. attribute:: Integrator.initial_time_step

    Specifies the initial time step size. If inappropriate, CVODE will attempt to estimate a better initial time step. Default value is 0.0

    >>> rr.integrator.initial_time_step = 1
    >>> rr.integrator.initial_time_step

.. attribute:: Integrator.maximum_adams_order

    Specifies the maximum order for Adams-Moulton intergration. This integration method is used for non-stiff problems. Default value is 12.

    >>> rr.integrator.maximum_adams_order = 20
    >>> rr.integrator."maximum_adams_order

.. attribute:: Integrator.maximum_bdf_order

    Specifies the maximum order for Backward Differentiation Formula integration. This integration method is used for stiff problems. Default value is 5.


.. attribute:: Integrator.maximum_num_steps

    Specifies the maximum number of steps to be taken by the CVODE solver in its attempt to reach tout. Default value is 20000.


.. attribute:: Integrator.maximum_time_step

    Specifies the maximum absolute value of step size allowed. If inappropriate, CVODE will attempt to estimate a better maximum time step. Default value is 0.0.


.. attribute:: Integrator.minimum_time_step

    Specifies the minimum absolute value of step size allowed. If inappropriate, CVODE will attempt to estimate a better maximum time step. Default value is 0.0.


.. attribute:: Integrator.multiple_steps

    Perform a multiple time step simulation. Default value is false.

    >>> rr.integrator.multiple_steps = True
    >>> rr.integrator.multiple_steps

.. attribute:: Integrator.relative_tolerance

    Specifies the scalar relative tolerance. CVODE calculates a vector of error weights which is used in all error and convergence tests. The weighted RMS norm for the relative tolerance should not become smaller than this value. Default value is Config::CVODE_MIN_RELATIVE.


.. attribute:: Integrator.stiff

    Specifies whether the integrator attempts to solve stiff equations. Ensure the integrator can solver stiff differential equations by setting this value to true. Default value is true.


.. attribute:: Integrator.variable_step_size

    Perform a variable time step simulation. Enabling this setting will allow the integrator to adapt the size of each time step. This will result in a non-uniform time column. Default value is false.


Gillespie
---------

    RoadRunner's implementation of the standard Gillespie Direct Method SSA. The granularity of this simulator is individual molecules and kinetic processes are stochastic. Results will, in general, be different in each run, but a sufficiently large ensemble of runs should be statistically correct.

    Can be used with the `rr.gillespie function <https://libroadrunner.readthedocs.io/en/latest/PythonAPIReference/cls_RoadRunner.html#RoadRunner.RoadRunner.gillespie>`_ or by setting integrator to gillespie (see below)

.. currentmodule:: Gillespie
.. attribute:: Integrator.initial_time_step

    Specifies the initial time step size. If inappropriate, CVODE will attempt to estimate a better initial time step. Default value is 0.0

    >>> rr.setIntegrator('gillespie') # set integrator first
    >>> rr.integrator.setValue("initial_time_step", 2)


.. attribute:: Integrator.maximum_time_step

    Specifies the maximum absolute value of step size allowed. If inappropriate, CVODE will attempt to estimate a better maximum time step. Default value is 0.0.


.. attribute:: Integrator.minimum_time_step

    Specifies the minimum absolute value of step size allowed. If inappropriate, CVODE will attempt to estimate a better maximum time step. Default value is 0.0.


.. attribute:: Integrator.nonnegative

    Prevents species amounts from going negative during a simulation. Default value is false.


.. attribute:: Integrator.seed

   Set the seed into the random engine.  By default, this value is set by the system clock in microseconds.


.. attribute:: Integrator.variable_step_size

    Perform a variable time step simulation. Enabling this setting will allow the integrator to adapt the size of each time step. This will result in a non-uniform time column. Default value is true.

.. attribute:: Integrator.max_output_rows

    Sets the maximum number of output rows for variable step size integration.  This may truncate some simulations that may not reach the desired end time, but prevents massive output for simulations where the variable step size ends up decreasing too much.  This setting is ignored when the variable_step_size is false, and is also ignored when the output is being written directly to a file.  Default value is 100,000.


Euler
-----

    The Euler method is one of the simplest approaches to solving a first order ODE. Given the rate of change of function f at time t, it computes the new value of f as ``f(t+h) = f(t) + h*f'(t)``, where h is the time step. Euler's method is rarely used in practice due to poor numerical robustness. Can be used with:

    >>> rr.setIntegrator('euler')

RK4
---
    
    Runge-Kutta methods are a family of algorithms for solving ODEs. They have considerably better accuracy than the Euler method. This integrator is a standard 4th order Runge-Kutta solver. Can be used with:

    >>> rr.setIntegrator('rk4')


RK45
----

    The Runge-Kutte-Fehlberg algorithm (RK45) is a variant of the Runge-Kutta method that estimates the values using a fourth-order method, and estimtes the error using a fifth-order method, allowing adaptive step sizes to be calculated automaticaly.  It can be chosen with:

    >>> rr.setIntegrator('rk45')


.. currentmodule:: Euler-and-Runge-Kutta
.. attribute:: Integrator.epsilon

    Specifies the maximum error tolerance allowed. Default value is 1e-12.

    >>> rr.setIntegrator('rk45') # set integrator first
    >>> rr.integrator.setValue("epsilon", 1e-10)

.. attribute:: Integrator.maximum_time_step

    Specifies the maximum absolute value of step size allowed. If inappropriate, CVODE will attempt to estimate a better maximum time step. Default value is 0.0.


.. attribute:: Integrator.minimum_time_step

    Specifies the minimum absolute value of step size allowed. If inappropriate, CVODE will attempt to estimate a better maximum time step. Default value is 0.0.


.. attribute:: Integrator.variable_step_size

    Perform a variable time step simulation. Enabling this setting will allow the integrator to adapt the size of each time step. This will result in a non-uniform time column. Default value is false.

