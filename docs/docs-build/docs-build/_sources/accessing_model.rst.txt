**********************************
Accessing the SBML Model Variables
**********************************

.. currentmodule:: RoadRunner

The following methods allow users to obtain information on the model


Compartments
------------

   * :cpp:func:`rr::ExecutableModel::getCompartmentVolumes`
   * :cpp:func:`rr::ExecutableModel::getNumCompartments`
   * :cpp:func:`rr::ExecutableModel::getCompartmentIds`
   * :cpp:func:`rr::ExecutableModel::setCompartmentVolumes`

   
Boundary Species
----------------

   * :cpp:func:`rr::ExecutableModel::getBoundarySpeciesConcentrations`
   * :cpp:func:`rr::ExecutableModel::setBoundarySpeciesConcentrations`
   * :cpp:func:`rr::ExecutableModel::getNumBoundarySpecies`
   * :cpp:func:`rr::ExecutableModel::getBoundarySpeciesIds`


Floating Species
----------------

   * :cpp:func:`rr::ExecutableModel::getFloatingSpeciesConcentrations`
   * :cpp:func:`rr::ExecutableModel::setFloatingSpeciesConcentrations`
   * :cpp:func:`rr::ExecutableModel::getNumFloatingSpecies`
   * :cpp:func:`rr::ExecutableModel::getFloatingSpeciesIds`


Initial Conditions
------------------

   * :cpp:func:`rr::ExecutableModel::getFloatingSpeciesInitAmountIds`
   * :cpp:func:`rr::ExecutableModel::getFloatingSpeciesInitConcentrationIds`
   * :cpp:func:`rr::ExecutableModel::getFloatingSpeciesInitAmounts`
   * :cpp:func:`rr::ExecutableModel::setFloatingSpeciesInitAmounts`
   * :cpp:func:`rr::ExecutableModel::getFloatingSpeciesInitConcentrations`
   * :cpp:func:`rr::ExecutableModel::setFloatingSpeciesInitConcentrations`

Reactions
---------

   * :cpp:func:`rr::ExecutableModel::getNumReactions`
   * :cpp:func:`rr::ExecutableModel::getReactionRates`
   * :cpp:func:`rr::ExecutableModel::getReactionIds`


Reaction Rates
---------------

   * :cpp:func:`rr::ExecutableModel::getFloatingSpeciesAmountRates`
   * :cpp:func:`rr::ExecutableModel::getReactionRates`
   * :cpp:func:`rr::ExecutableModel::getNumReactions`
   * :cpp:func:`rr::ExecutableModel::getReactionIds`


Global Parameters
-----------------

   * :cpp:func:`rr::ExecutableModel::setGlobalParameterValues`
   * :cpp:func:`rr::ExecutableModel::getGlobalParameterValues`
   * :cpp:func:`rr::ExecutableModel::getNumGlobalParameters`
   * :cpp:func:`rr::ExecutableModel::getNumGlobalParameters`
   


