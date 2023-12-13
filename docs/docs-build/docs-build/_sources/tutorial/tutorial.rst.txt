*********
Tutorial
*********


.. toctree::
    :maxdepth: 1
    :hidden:

    what_is_sbml
    what_is_rr
    import_roadrunner
    loading_models
    running_simulations
    changing_parameters
    selecting_output
    plotting_data
    changing_initial_conditions
    solvers
    jit_engines

Quickstart
==========

To get you started here is a very simple script that will load an example SBML model and run a time course
simulation and plot the results.

.. code-block:: python

   import roadrunner

   # load an SBML model
   rr = roadrunner.RoadRunner("mymodel.xml")

   # simulate from 0 to 10 time units with 100 output rows
   result = rr.simulate(0,10,100)

   rr.plot()


Looking for a model to start with? We included a couple with libRoadRunner, see :ref:`Loading Models`.



