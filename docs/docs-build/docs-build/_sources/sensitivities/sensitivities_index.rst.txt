Time Series Sensitivities
=========================

.. note::

    For these demonstrations we use a toy
    sbml model prepackaged with roadrunner


Using time series sensitivities
-------------------------------

In this example we show you how to use roadrunners time series sensitivities
feature. Roadrunner will integrate the model and compute sensitivities at
each time step.

Start integration with sensitivities at 0, end at 10 and collect 11 data points
The return type is a 4-Tuple, a time vector and a 3D sensitivity matrix with
dimensions time x parameters x variables - that is, an array of matrices corresponding
to the time points in the time vector with parameters down the rows and model variables (species)
accross the columns. The third and forth elements are row and column names respectively.


.. literalinclude:: sensitivities.py
    :linenos:
    :language: python

Selecting parameters
----------------------
By default, all model parameters are automatically selected.
You can be more selective about which parameters to compute sensitivities for.
This will have a performance advantage, because sensitivities are only computed
for the parameters specified here.

.. literalinclude:: sensitivities_selective_parameters.py
    :linenos:
    :language: python


Selecting Species
------------------
By default, all model species are returned in the results matrix. We can
be more selective by using the species argument. This does not have
any additional performance advantage and in fact cost additional resources
because the full results matrix is sliced to select the selected columns.

.. literalinclude:: sensitivities_selective_species.py
    :linenos:
    :language: python

Sensitivity Method
--------------------
RoadRunner time series sensitivities uses cvodes under the hood. By default,
we use the "simultaneous" method, whereby sensitivity equations are solved at the same
time as the variables in the ODE's. This can be changed so that the variables
are solved first followed by the sensitivity equations; aka in a staggered approach.


.. literalinclude:: sensitivities_method.py
    :linenos:
    :language: python

Sensitivity Difference Quotient Method
---------------------------------------
RoadRunner via cvodes uses finite differences for approximating gradients.
By default we use the "centered" method, while the alternative is the "forward" method
Both methods are implemented in cvodes and automatically figure out the best value of delta
(perturbation amount) to use. Moreover, there is an adaptive switching mechanism in use
that makes it unlikely that users should need to modify this setting.


.. literalinclude:: sensitivities_dq_method.py
    :linenos:
    :language: python

Sensitivity Solver Type
------------------------
By default, RoadRunner uses the sundials "newton" nonlinear solver.
This can be changed to the "fixed_point" nonlinear solver

.. literalinclude:: sensitivities_solver_type.py
    :linenos:
    :language: python



