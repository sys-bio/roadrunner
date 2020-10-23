*********************
Stochastic Simulation
*********************

.. currentmodule:: RoadRunner

RoadRunner supports stochastic simulation through the use of Gillespie algorithm, which is a variation of
Monte Carlo method.

To set the current instance of RoadRunner to use the Gillespie solver, try:

     >>>  rr.setIntegrator('gillespie')
	 
RoadRunner also supports a simplified method to run Gillespie solver through :meth:`~RoadRunner.gillespie`. 

One of the important component of stochastic simulation is setting the seed. Random number genetration in 
computers are known to be 'pseudo-random', meaning it can only 'approximate' randomness. Seed is an initial 
key value for generating a sequence of numbers. This means that when a seed is given, it is possible to 
regenerate identical sequence of numbers from random number generator. This might be desireable for reproducibility
perposes but undesireable for multiple iterations of random number creation.

By using :meth:`~roadrunner.Solver.getSettings`, you can check settings accepted by Gillespie solver.

     >>>  rr.getIntegrator().getSettings()
     ('seed',
      'variable_step_size',
      'initial_time_step',
      'minimum_time_step',
      'maximum_time_step',
      'nonnegative')

For information on what the settings represent, try :meth:`~roadrunner.Solver.getDescription`.
Check :ref:`roadrunner-solver` for additional information.
	  
The following methods deal with stochastic simulation:

.. autosummary::

   RoadRunner.gillespie

   
