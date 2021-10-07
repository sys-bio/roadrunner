Using RoadRunner with Python's parallel libraries
======================================================

The model
---------
For these simulations we use a simple model that we can access easily from one of RoadRunners
test suites. Its called 'BatchImmigrationDeath03' and is model number 00039 from the
stochastic sbml test suite.

In each of the following sections we use RoadRunner to generate 1e6 stochastic simulations.

Serial code
------------
Performing the simulations in serial gives us a way to measure the performance enhancements
of using parallel code.

.. literalinclude:: gillespie_simulations_serial.py
    :linenos:
    :language: python


Multiprocessing library
------------------------

The simplest way in Python to use multiple cores simultaneously for running RoadRunner simulations
is to use the `multiprocessing <https://docs.python.org/3/library/multiprocessing.html>`_ library.
Further, the simplest multiprocessing tool that one can use is a `multiprocessing.Pool`. This leads
to about a x3 speed up for this example.

.. literalinclude:: gillespie_simulations_multiprocessing_pool.py
    :linenos:
    :language: python


Threading library
-----------------

.. literalinclude:: gillespie_simulations_threading.py
    :linenos:
    :language: python



Ray library
------------

.. literalinclude:: gillespie_simulations_ray.py
    :linenos:
    :language: python


MPI library
------------

.. note::

    coming soon...









































