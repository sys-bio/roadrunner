########################
RoadRunner Documentation
########################

The Roadrunner Simulatin Engine is a C++ library for simulating and analysing
systems of differential equations in [SBML]() format. Roadrunner was designed
with performance as a priority and is an exceptionally fast simulator.

.. todo::

    todo link to figure comparing speed of roadrunner with other simulators. We want to sell it to people afterall.

Installation
=============
For instructions on how to install roadrunner please see the :ref:`installation section <Installing RoadRunner>`.

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


Looking for a model to start with? We included a couple with libRoadRunner, see :ref:`loading-models`.

Features
========

.. todo::

    list all the things that you can do with roadrunner. Be inclusive.
    Eventually I want hyperlinks here directing people to the relevant example or section.

* Construct SBML models

* Run time series tasks on SBML models with the following integrators:

    * CVODE

    * RK4

    * RK5

* Compute steady state of SBML models using the following methods:

    * [insert methods here]

.. toctree::
    :maxdepth: 1
    :hidden:

    self
    introduction
    Installation/installation
    read_write_functions
    selecting_values
    steady_state
    stochastic
    metabolic
    stability
    stoichiometric
    bifurcation
    integration
    utility_functions
    accessing_model
    api_reference






Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. highlight:: python
