Accessing the SBML Model Variables
__________________________________

All of the SBML model variables are accessed via the ``RoadRunner.model`` object. This is an instance of
the ExecutableModel class described here. One always access the model object that belongs to the top
RoadRunner object, i.e. ``r.model``, where ``r`` is an instance of the ``RoadRunner`` class.

The ExecutableModel class also implements the Python dictionary protocol, so that it can be used as
a dictionary. The dictionary keys are all of the symbols specified in the original model as well as
a number of selection strings described in the Selections section.

.. method:: ExecutableModel.keys()
   :module: RoadRunner

   Get a list of all the keys that this model has. This is a very good way of looking at all the
   available symbols and selection strings:

   >>> r.model.keys()
   [ 'S1', 'S2', '[S1]', '[S2]', 'compartment', 'k1', 'cm0',
     'reaction1',  'init([S1])',  'init([S2])', 'init(S1)',
     'init(S2)',  "S1'"]


.. method:: ExecutableModel.items()
   :module: RoadRunner

   Get a list of key / value pairs of all the selections / values in the model.

   >>> r.model.items()
   [('S1', 0.5), ('S2', 9.99954570660308), ('[S1]', 0.5), ('[S2]', 9.99954570660308),
   ('default_compartment', 1.0), ('k1', 1.0), ('init(k1)', 1.0), ('_J0', 0.5), ('init([S1])', 10.0),
   ('init([S2])', 0.0), ('init(S1)', 10.0), ('init(S2)', 0.0), ("S1'", -0.5), ("S2'", 0.5)]


.. method:: ExecutableModel.__getitem__
   :module: RoadRunner

   Implements the python ``[]`` indexing operator, so the model values can be accessed like::

     >>> r.model["S1"]
     0.0

   Following notation is also accepted::

     >>> r.S1
     0.0


.. method:: ExecutableModel.__setitem__
   :module: RoadRunner

   Implements the python ``[]`` indexing operator for setting values::

	   >>> r.model["S1]
	   0.0
	   >>> r.model["S1"] = 1.3
	   >>> r.model["S1"]
	   1.3

   Following notation is also accepted::

   >>> r.S1 = 1.0

   Note, some keys are read only such as values defined by rules, or calculated values such as
   reaction rates. If one attempts to set the value of a read-only symbol,
   an exception is raised indicating the error, and no harm done.



Floating Species
----------------

.. method:: ExecutableModel.getFloatingSpeciesIds()
   :module: RoadRunner

   Return a list of all floating species SBML ids.

   >>> r.getFloatingSpeciesIds()
   ['S1', 'S2', 'S3', 'S4']


.. method:: ExecutableModel.getDependentFloatingSpeciesIds()
   :module: RoadRunner

   Return a list of dependent floating species SBML ids.

   >>> r.getDependentFloatingSpeciesIds()
   ['S4']


.. method:: ExecutableModel.getIndependentFloatingSpeciesIds()
   :module: RoadRunner

   Return a list of independent floating species SBML ids.

   >>> r.getIndependentFloatingSpeciesIds()
   ['S1', 'S2', 'S3']


.. method:: ExecutableModel.getFloatingSpeciesConcentrationIds()
   :module: RoadRunner

   Return a list of all floating species concentration ids.

   >>> r.getFloatingSpeciesConcentrationIds()
   ['[S1]', '[S2]', '[S3]', '[S4]']
   

.. method:: ExecutableModel.getNumFloatingSpecies()
   :module: RoadRunner

   Return the number of floating species in the model.

   >>> r.getNumFloatingSpecies()
   2


.. method:: ExecutableModel.getFloatingSpeciesAmounts([index])
   :module: RoadRunner

   Get the list of floating species amounts. If no arguments are given, this
   returns all floating species amounts.

   :param numpy.ndarray index: (optional) an optional array of desired floating species indices.
   :retuns: an array of floating species amounts.
   :rtype: numpy.ndarray

   To get all the amounts::

     >>> r.model.getFloatingSpeciesAmounts()
     array([ 0.97390578,  1.56331018,  1.15301155,  1.22717548])

   To get amounts from index 0 and 1::

     >>> r.model.getFloatingSpeciesAmounts([0,1])
     array([ 0.97390578,  1.56331018])


.. method:: ExecutableModel.setFloatingSpeciesAmounts([index], values)
   :module: RoadRunner

   Use this to set the entire set of floating species amounts in one call.
   The order of species is given by the order of Ids returned by getFloatingSpeciesIds()

   :param numpy.ndarray index: (optional) an index array indicating which items to set,
                               or if no index array is given, the first param should be
                               an array of all the values to set.
   :param numpy.ndarray values: the values to set.

   >>> r.model.getFloatingSpeciesAmounts([0,1])
   array([ 0.97390578,  1.56331018])
   >>> r.model.setFloatingSpeciesAmounts([0,1], [1.0, 1.5])
   >>> r.model.getFloatingSpeciesAmounts([0,1])
   array([ 1. ,  1.5])


.. method:: ExecutableModel.getFloatingSpeciesConcentrations([index])
   :module: RoadRunner

   Return a vector of floating species concentrations. The order of species is
   given by the order of Ids returned by getFloatingSpeciesIds()

   :param numpy.ndarray index: (optional) an index array indicating which items to return.
   :returns: an array of floating species concentrations.
   :rtype: numpy.ndarray

   >>> r.model.getFloatingSpeciesConcentrations()
   array([  4.54293397e-04,   9.99954571e+00])


.. method:: ExecutableModel.setFloatingSpeciesConcentrations([index], values)
   :module: RoadRunner

   Use this to set the entire set of floating species concentrations in one call.
   The order of species is given by the order of Ids returned by getFloatingSpeciesIds()


   :param numpy.ndarray index: (optional) an index array indicating which items to set,
                               or if no index array is given, the first param should be an
                               array of all the  values to set.
   :param numpy.ndarray values: the values to set.

   >>> r.model.getFloatingSpeciesConcentrations()
   array([  4.54293397e-04,   9.99954571e+00])
   >>> r.model.setFloatingSpeciesConcentrations([0],[0.5])
   >>> r.model.getFloatingSpeciesConcentrations()
   array([ 0.5       ,  9.99954571])

   

Floating Species Initial Conditions
-----------------------------------
RoadRunner stores all initial conditions separately from the model state variables. This means that
you can update the initial conditions at any time, and it does not affect the current state of the
model. To reset the model, that is, reset it to its original state, or a new original state where
what has changed the initial conditions use the :meth:`~ExecutableModel.reset()` method.

The following methods allow access to the floating species initial condition values:

.. method:: ExecutableModel.getFloatingSpeciesInitAmountIds()
   :module: RoadRunner

   Return a list of the floating species amount initial amount selection symbols.

   >>> r.model.getFloatingSpeciesInitAmountIds()
   ['init(S1)', 'init(S2)']


.. method:: ExecutableModel.getFloatingSpeciesInitConcentrationIds()
   :module: RoadRunner

   Return a list of the floating species amount initial concentration selection symbols.

   >>> r.model.getFloatingSpeciesInitConcentrationIds()
   ['init([S1])', 'init([S2])']


.. method:: ExecutableModel.getFloatingSpeciesInitConcentrations([index])
   :module: RoadRunner

   Return a vector of floating species initial concentrations. The order of species is
   given by the order of Ids returned by getFloatingSpeciesInitialConcentrationIds()

   :param numpy.ndarray index: (optional) an index array indicating which items to return.
   :returns: an array of floating species initial concentrations.
   :rtype: numpy.ndarray


   >>> r.model.getFloatingSpeciesInitConcentrations()
   array([ 10.,   0.])


.. method:: ExecutableModel.setFloatingSpeciesInitConcentrations([index], values)
   :module: RoadRunner

   Set a vector of floating species initial concentrations. The order of species is
   given by the order of Ids returned by getFloatingSpeciesInitialAmountIds()

   :param numpy.ndarray index: (optional) an index array indicating which items to return.


   >>> r.model.setFloatingSpeciesInitConcentrations([0], [1])
   >>> r.model.getFloatingSpeciesInitConcentrations()
   array([ 1.,  0.])


.. method:: ExecutableModel.getFloatingSpeciesInitAmounts([index])
   :module: RoadRunner

   Return a vector of floating species initial amounts. The order of species is
   given by the order of Ids returned by getFloatingSpeciesInitialConcentrationIds()

   :param numpy.ndarray index: (optional) an index array indicating which items to return.
   :returns: an array of floating species initial amounts.
   :rtype: numpy.ndarray


   >>> r.model.getFloatingSpeciesInitAmounts()
   array([ 10.,   0.])


.. method:: ExecutableModel.setFloatingSpeciesInitAmounts([index], values)
   :module: RoadRunner

   Set a vector of floating species initial amounts. The order of species is
   given by the order of Ids returned by getFloatingSpeciesInitialAmountIds()

   :param numpy.ndarray index: (optional) an index array indicating which items to return.


   >>> r.model.setFloatingSpeciesInitAmounts([0], [0.1])
   >>> r.model.getFloatingSpeciesInitAmounts()
   array([ 0.1,  0. ])


Boundary Species
----------------

.. method:: ExecutableModel.getBoundarySpeciesAmounts([index])
   :module: RoadRunner

   Return a vector of boundary species amounts. The order of species is
   given by the order of Ids returned by getBoundarySpeciesIds()

   :param numpy.ndarray index: (optional) an index array indicating which items to return.
   :returns: an array of the boundary species amounts.
   :rtype: numpy.ndarray

   >>> r.model.getBoundarySpeciesAmounts()
   array([ 15.,   0.])


.. method:: ExecutableModel.getBoundarySpeciesConcentrations([index])
   :module: RoadRunner

   Return a vector of boundary species concentrations. The order of species is
   given by the order of Ids returned by getBoundarySpeciesIds()

   :param numpy.ndarray index: (optional) an index array indicating which items to return.
   :returns: an array of the boundary species concentrations.
   :rtype: numpy.ndarray

   >>> r.getBoundarySpeciesConcentrations()
   array([ 0.5,   0.])


.. method:: ExecutableModel.getBoundarySpeciesIds()
   :module: RoadRunner

   Return a vector of boundary species Ids.

   :param numpy.ndarray index: (optional) an index array indicating which items to return.
   :returns: a list of boundary species ids.


   >>> r.getBoundarySpeciesIds()
   ['X0', 'X1']


.. method:: ExecutableModel.getBoundarySpeciesConcentrationIds()
   :module: RoadRunner

   Return a vector of boundary species concentration Ids.

   :param numpy.ndarray index: (optional) an index array indicating which items to return.
   :returns: a list of boundary species concentration ids.


   >>> r.getBoundarySpeciesConcentrationIds()
   ['[X0]', '[X1]']   
   
   
.. method:: ExecutableModel.getNumBoundarySpecies()
   :module: RoadRunner

   Return the number of boundary species in the model.


   >>> r.getNumBoundarySpecies()
   2


.. method:: ExecutableModel.setBoundarySpeciesConcentrations([index], values)
   :module: RoadRunner

   Use this to set the entire set of boundary species concentrations in one call.
   The order of species is given by the order of boundary species returned by getBoundarySpeciesIds()

   :param numpy.ndarray index: (optional) an index array indicating which items to set,
                               or if no index array is given, the first param should be an
                               array of all the  values to set.

   :param numpy.ndarray values: the values to set.


   >>> r.model.setBoundarySpeciesConcentrations([0], [1])
   >>> r.getBoundarySpeciesConcentrations()
   array([ 1.,  0.])


Compartments
------------

.. method:: ExecutableModel.getCompartmentIds([index])
   :module: RoadRunner

   Return a vector of compartment identifier symbols.

   :param index: A array of compartment indices indicating which compartment ids to return.
   :type index: None or numpy.ndarray
   :returns: a list of compartment ids.


   >>> r.getCompartmentIds()
   ['compartment1']


.. method:: ExecutableModel.getCompartmentVolumes([index])
   :module: RoadRunner

   Return a vector of compartment volumes. The order of volumes is
   given by the order of Ids returned by getCompartmentIds()

   :param numpy.ndarray index: (optional) an index array indicating which items to return.
   :returns: an array of compartment volumes.
   :rtype: numpy.ndarray.


   >>> r.getCompartmentVolumes()
   array([ 1.])


.. method:: ExecutableModel.getNumCompartments()
   :module: RoadRunner

   Return the number of compartments in the model.

   :rtype: int


   >>> r.getNumCompartments()
   1


.. method:: ExecutableModel.setCompartmentVolumes([index], values)
   :module: RoadRunner

   Set a vector of compartment volumes.

   If the index vector is not give, then the values vector treated as a vector of all
   compartment volumes to set. If index is given, then  values should have the same
   length as index.

   :param numpy.ndarray index: (optional) an index array indicating which items to set,
                               or if no index array is given, the first param should be an
                               array of all the  values to set.

   :param numpy.ndarray values: the values to set.


   >>> r.model.setCompartmentVolumes([0], [2.5])
   >>> r.getCompartmentVolumes()
   array([ 2.5])


Global Parameters
-----------------


.. method:: ExecutableModel.getGlobalParameterIds()
   :module: RoadRunner

   Return a list of global parameter ids.

   :returns: a list of global parameter ids.


.. method:: ExecutableModel.getGlobalParameterValues([index])
   :module: RoadRunner

   Returns a vector of global parameter values. The order of species is
   given by the order of Ids returned by getGlobalParameterIds()

   :param numpy.ndarray index: (optional) an index array indicating which items to return.
   :returns: an array of global parameter values.
   :rtype: numpy.ndarray.


   >>> r.getGlobalParameterValues()
   array([ 10. ,  10. ,  10. ,   2.5,   0.5])


.. method:: ExecutableModel.getNumGlobalParameters()
   :module: RoadRunner


   Returns the number of global parameters in the model.

   >>> r.getNumGlobalParameters()
   5


.. method:: ExecutableModel.setGlobalParameterValues([index], values)
   :module: RoadRunner

   Sets the entire set of global parameters.
   The order of parameters is given by the order of Ids returned by getGlobalParameterIds()


   :param numpy.ndarray index: (optional) an index array indicating which items to set,
                               or if no index array is given, the first param should be an
                               array of all the values to set.
   :param numpy.ndarray values: the values to set.


   >>> r.model.setGlobalParameterValues([0], [1.5])
   >>> r.getGlobalParameterValues()
   array([  1.5,  10. ,  10. ,   2.5,   0.5])


Reactions
---------

.. method:: ExecutableModel.getNumReactions()
   :module: RoadRunner

   Return the number of reactions in the model.


   >>> r.getNumReactions()
   5


.. method:: ExecutableModel.getReactionIds()
   :module: RoadRunner

   Return a vector of reaction Ids.

   :param numpy.ndarray index: (optional) an index array indicating which items to return.
   :returns: a list of reaction ids.


   >>> r.getReactionIds()
   ['J0', 'J1', 'J2', 'J3', 'J4']


.. method:: ExecutableModel.getReactionRates([index])
   :module: RoadRunner

   Return a vector of reaction rates (reaction velocity) for the current state of the model. The
   order of reaction rates is given by the order of Ids returned by getReactionIds()

   :param numpy.ndarray index: (optional) an index array indicating which items to return.
   :returns: an array of reaction rates.
   :rtype: numpy.ndarray


   >>> r.getReactionRates()
   array([ 0.14979613,  2.37711263,  2.68498886,  2.41265507,  1.89417737])

Events
------

.. method:: ExecutableModel.getNumEvents()
   :module: RoadRunner

   Returns the number of events.

   >>> r.getNumEvents()
   1


Rate Rules
----------

.. method:: ExecutableModel.getNumRateRules()
   :module: RoadRunner

   Returns the number of rate rules.

   >>> r.getNumRateRules()
   1

   
.. method:: ExecutableModel.getEventIds()
   :module: RoadRunner

   Returns a list of event ids.

   :returns: a list of event ids.
   
   >>> r.model.getEventIds()
   ['E1']
   
   
   
Stoichiometry
-------------

.. method:: ExecutableModel.getStoichiometry(speciesIndex, reactionIndex)
   :module: RoadRunner

   Return the stochiometric coefficient for the given species index and reaction index.

   Frequently one does not need the full stochiometrix matrix, particularly if the system is
   large and only a single coefficient is needed.


   :param speciesIndex: a floating species index from :meth:`getFloatingSpeciesIds`
   :param reactionIndex: a reaction index from :meth:`getReactionIds`


   >>> r.model.getStoichiometry(1, 3)
   1.0


Conserved Moieties
------------------

Refer to :attr:`RoadRunner.conservedMoietyAnalysis` and :attr:`Config.LOADSBMLOPTIONS_CONSERVED_MOIETIES` for more information.

.. method:: ExecutableModel.getNumConservedMoieties()
   :module: RoadRunner

   Return the number of conserved moieties in the model.

   :rtype: int


   >>> r.getNumConservedMoieties()
   1


.. method:: ExecutableModel.getConservedMoietyIds([index])
   :module: RoadRunner

   Return a vector of conserved moiety identifier symbols.

   :param index: A array of compartment indices indicating which compartment ids to return.
   :type index: None or numpy.ndarray
   :returns: a list of compartment ids.


   >>> r.getConservedMoietyIds()
   ['_CSUM0']


.. method:: ExecutableModel.getConservedMoietyValues([index])
   :module: RoadRunner

   Return a vector of conserved moiety volumes. The order of values is
   given by the order of Ids returned by getConservedMoietyIds()

   :param numpy.ndarray index: (optional) an index array indicating which items to return.
   :returns: an array of conserved moiety values.
   :rtype: numpy.ndarray.


   >>> r.getConservedMoietyValues()
   array([ 2.])


.. method:: ExecutableModel.setConservedMoietyValues([index], values)
   :module: RoadRunner

   Set a vector of conserved moiety values.

   *Note* This method currently only updates the conserved moiety values, it does
   not update the initial species condition from which the values were calculated.

   If the index vector is not given, then the values vector treated as a vector of all
   values to set. If index is given, then  values should have the same
   length as index.

   :param numpy.ndarray index: (optional) an index array indicating which items to set,
                               or if no index array is given, the first param should be an
                               array of all the  values to set.
   :param numpy.ndarray values: the values to set.


   >>> r.model.setConservedMoietyValues([0], [5])
   >>> r.getConservedMoietyValues()
   array([ 5.])


Misc
----

.. method:: ExecutableModel.getAllTimeCourseComponentIds()
   :module: RoadRunner

   Return a list of all component ids.  The list includes ids of amount/concentration of 
   floating species, boundary species, global parameters, compartments, and reactions, 
   as well as `time`.

   :returns: a list of all component ids widely used in time course selections.
   >>> r.model.getAllTimeCourseComponentIds()
   ['time', 'S1', 'S2', 'S3', 'k1', 'k2', 'default_compartment', '_J0', '_J1']



.. method:: ExecutableModel.getInfo()
   :module: RoadRunner

   Get various info about the model.


   >>> print(r.getInfo())
   <roadrunner.RoadRunner() {
   'this' : 13DEF5F8
   'modelLoaded' : true
   'modelName' : feedback
   'libSBMLVersion' : LibSBML Version: 5.12.0
   'jacobianStepSize' : 1e-005
   'conservedMoietyAnalysis' : false
   'simulateOptions' :
   < roadrunner.SimulateOptions()
   {
   'this' : 0068A7F0,
   'reset' : 0,
   'structuredResult' : 0,
   'copyResult' : 1,
   'steps' : 50,
   'start' : 0,
   'duration' : 40
   }>,
   'integrator' :
   < roadrunner.Integrator() >
     name: cvode
     settings:
         relative_tolerance: 0.00001
         absolute_tolerance: 0.0000000001
	 absolute_concentration_tolerance: 0.0000000001
                      stiff: true
          maximum_bdf_order: 5
        maximum_adams_order: 12
          maximum_num_steps: 20000
          maximum_time_step: 0
             minimum_time_step: 0
          initial_time_step: 0
             multiple_steps: false
         variable_step_size: false
   }>


.. method:: ExecutableModel.getModelName()
   :module: RoadRunner

   Get the model name specified in the SBML.


   >>> r.model.getModelName()
   'feedback'


.. method:: ExecutableModel.getTime()
   :module: RoadRunner

   Get the model time. The model originally start at time t=0 and is advaced forward in time by the
   integrator. So, if one ran a simulation from time = 0 to time = 10, the model will then have it's
   time = 10.

   >>> r.model.getTime()
   40.0


.. method:: ExecutableModel.setTime(time)
   :module: RoadRunner

   Set the model time variable.

   :param time: time the time value to set.


   >>> rr.model.setTime(20.)
   >>> rr.model.getTime()
   20.0


.. method:: ExecutableModel.reset()
   :module: RoadRunner

   Resets all the floating species concentrations to their initial values.


.. method:: ExecutableModel.resetAll()
   :module: RoadRunner

   Resets all variables, species, etc. to the CURRENT initial values.
   It also resets all parameter back to the values they had when the model was first loaded


.. method:: ExecutableModel.resetToOrigin()
   :module: RoadRunner

   Resets the model back to the state is was when it was FIRST loaded.
   The scope of reset includes all initial values and parameters, etc.
