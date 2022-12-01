########################
RoadRunner Documentation
########################

The Roadrunner Simulation Engine is a C++ library for simulating and analysing
systems of differential equations. Roadrunner was designed
with performance as a priority and is an exceptionally fast simulator.

For the C API documentation, please find the doxygen generated webpage here:

    * `C API Docs are here <http://sys-bio.github.io/roadrunner/OriginalDoxygenStyleDocs/html/index.html>`_

Features
========

* Time Dependent Simulation (with optional conservation law reduction) using CVODE from the sundials suite.
* Steady State evaluation using NLEQ2
* Supports SBML Level 2 to 3 but currently excludes algebraic rules and delay differential equations
* Uses latest libSBML distribution
* Defaults to LLVM code generation on the backend, resulting is very fast simulation times
* Optional generation of model C code and linking at run-time
* Add plugins, distribution comes with Levenberg-Marquardt optimizer plugin
* Supports Metabolic Control Analysis
* Supports Frequency Domain Analysis
* Additional Plugin API (C and C++)
* APIs support fast access to model quantities
* Access to other items such as:
* Eigenvalues and Eigenvectors
* Jacobian, full and reduced
* Structural Matrices of the stoichiometry matrix

.. todo::

    todo link to figure comparing speed of roadrunner with other simulators. We want to sell it to people afterall.

Installation
=============
For instructions on how to install roadrunner please see the :ref:`installation section <Installing RoadRunner>`.


.. toctree::
    :maxdepth: 1
    :hidden:

    self
    Installation/installation
    tutorial/tutorial
    Examples/examples_index
    read_write_functions
    selecting_values
    steady_state
    stochastic
    metabolic
    stability
    stoichiometric
    bifurcation
    simulation_and_integration
    sensitivities/sensitivities_index
    parallel/parallel_index
    utility_functions
    accessing_model
    UsingRoadRunnerFromCxx/using_roadrunner_from_cxx
    rrplugins/introduction
    PythonAPIReference/api_reference
    C++APIReference/index
    developers_docs/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. highlight:: python
