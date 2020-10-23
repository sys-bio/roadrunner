**********************************
Accessing the SBML Model Variables
**********************************

.. currentmodule:: RoadRunner

The following methods allow users to obtain information on the model


Compartments
------------

.. autosummary::

   ExecutableModel.getCompartmentVolumes
   ExecutableModel.getNumCompartments
   ExecutableModel.getCompartmentIds
   ExecutableModel.setCompartmentVolumes

   
Boundary Species
----------------

.. autosummary::

   ExecutableModel.getBoundarySpeciesConcentrations
   ExecutableModel.setBoundarySpeciesConcentrations
   ExecutableModel.getNumBoundarySpecies
   ExecutableModel.getBoundarySpeciesIds


Floating Species
----------------

.. autosummary::

   ExecutableModel.getFloatingSpeciesConcentrations
   ExecutableModel.setFloatingSpeciesConcentrations
   ExecutableModel.getNumFloatingSpecies
   ExecutableModel.getFloatingSpeciesIds


Initial Conditions
------------------

.. autosummary::

   ExecutableModel.getFloatingSpeciesInitAmountIds
   ExecutableModel.getFloatingSpeciesInitConcentrationIds
   ExecutableModel.getFloatingSpeciesInitAmounts
   ExecutableModel.setFloatingSpeciesInitAmounts
   ExecutableModel.getFloatingSpeciesInitConcentrations
   ExecutableModel.setFloatingSpeciesInitConcentrations

Reactions
---------

.. autosummary::

   ExecutableModel.getNumReactions
   ExecutableModel.getReactionRates
   ExecutableModel.getReactionIds


Reaction Rates
---------------

.. autosummary::

   ExecutableModel.getFloatingSpeciesAmountRates
   ExecutableModel.getReactionRates
   ExecutableModel.getNumReactions
   ExecutableModel.getReactionIds


Global Parameters
-----------------

.. autosummary::

   ExecutableModel.setGlobalParameterValues
   ExecutableModel.getGlobalParameterValues
   ExecutableModel.getNumGlobalParameters
   ExecutableModel.getNumGlobalParameters
   


